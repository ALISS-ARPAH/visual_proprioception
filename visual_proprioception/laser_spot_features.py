"""
Laser spot segmentation and contact features shared by training and runtime.

This module has no ROS dependency. The offline dataset builder, the trainer
and laser_contact_model_node all call it, so a model sees the same features in
training and on the robot.

Segmentation changes relative to laser_segmentation_node, found on the
bladder-resection recordings (lesion 1):

* Hue is capped at 78. The node's h_max=90 overlaps the blue glow at the
  fiber tip (hue ~85-128), which was sometimes measured as the spot.
* A pixel must be greener than its neighbourhood (local contrast). On red
  tissue the spot looks yellow and fails a plain colour threshold.
* Long, thin blobs (glare streaks on the tool shaft) are rejected.

Parameters were tuned on 540x540 rectified endoscope frames.
"""

from collections import deque
import math

import cv2
import numpy as np

SPOT_PARAMS = {
    'h_min': 25,             # OpenCV hue, 0-179
    'h_max': 78,             # stays below the blue tip glow
    's_min': 45,
    'v_min': 70,
    'contrast_min': 18,      # green minus local median green, 0-255 scale
    'bg_ksize': 41,          # median filter size for the local green background
    'min_area_px': 6.0,
    'max_elongation': 4.0,   # major/minor above this is a glare streak
    'tip_gate_px': 120.0,    # with a projected tip: ignore blobs farther than this from it
}

# Defaults of laser_segmentation_node + laser_spot_analysis_node, used only to
# reproduce the current area-threshold rule as a baseline.
LEGACY_PARAMS = {
    'h_min': 35, 'h_max': 90, 's_min': 12, 's_max': 255, 'v_min': 95, 'v_max': 255,
    'green_excess_min': 9.0, 'blur_size': 5, 'kernel_size': 7, 'min_spot_area_px': 30.0,
}
LEGACY_AREA_THRESHOLD_PX2 = 1000.0

WINDOW_PARAMS = {
    'short_window_s': 0.5,   # trend features
    'long_window_s': 1.5,    # shrink ratio and crop-centre memory
    'max_gap_s': 1.0,        # longer gaps (or time going backwards) reset history
}

# Model input order. Missing-spot frames use 0 for the size/brightness entries;
# 'detected' tells the model which case it is.
FEATURE_NAMES = (
    'detected',         # 1 if a spot was found in this frame
    'sqrt_area',        # sqrt(spot area), px
    'minor_axis',       # fitted-ellipse minor axis, px
    'elongation',       # major / minor axis
    'peak_contrast',    # brightest green contrast inside the spot
    'log_energy',       # log(1 + summed green contrast inside the spot)
    'n_blobs',          # number of compact green candidates
    'sqrt_area2',       # sqrt(area) of the second-largest candidate
    'detected_frac',    # fraction of frames with a spot, short window
    'sqrt_area_slope',  # d sqrt(area) / dt over the short window, px/s
    'shrink_ratio',     # sqrt_area / max sqrt_area over the long window
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def spot_mask(image, params=SPOT_PARAMS):
    """Return the binary spot-candidate mask and the local green-contrast image."""
    smoothed = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    green = np.ascontiguousarray(smoothed[..., 1])
    background = cv2.medianBlur(green, int(params['bg_ksize'])).astype(np.int16)
    contrast = green.astype(np.int16) - background
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((hue >= params['h_min']) & (hue <= params['h_max'])
            & (sat >= params['s_min']) & (val >= params['v_min'])
            & (contrast >= params['contrast_min'])
            & (smoothed[..., 1] > smoothed[..., 0]))
    mask = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _K5)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K5), contrast


def _axes(contour):
    if len(contour) < 5:
        d = 2.0 * math.sqrt(cv2.contourArea(contour) / math.pi)
        return d, d
    _, axes, _ = cv2.fitEllipse(contour)
    return max(axes), min(axes)


class Rectifier:
    """
    Undistort raw camera frames with CameraInfo (k, d, r, p), like image_proc, then resize.

    The models were trained on rectified 540x540 frames. Raw /ves_camera/image frames
    (1080x1080, plumb_bob distortion) must go through this first. With zero distortion
    and the target size, frames pass through unchanged.
    """

    def __init__(self, k, d, r, p, width, height, size=540):
        self.size = int(size)
        self.width, self.height = int(width), int(height)
        d = np.asarray(d, dtype=float)
        self.needed = bool(np.any(np.abs(d) > 0)) or (self.width, self.height) != (size, size)
        self.maps = None
        if np.any(np.abs(d) > 0):
            self.maps = cv2.initUndistortRectifyMap(
                np.asarray(k, dtype=float).reshape(3, 3), d,
                np.asarray(r, dtype=float).reshape(3, 3),
                np.asarray(p, dtype=float).reshape(3, 4)[:, :3],
                (self.width, self.height), cv2.CV_16SC2)

    @classmethod
    def from_camera_info(cls, info, size=540):
        return cls(info.k, info.d, info.r, info.p, info.width, info.height, size)

    def __call__(self, image):
        if not self.needed:
            return image
        if self.maps is not None:
            image = cv2.remap(image, *self.maps, cv2.INTER_LINEAR)
        if image.shape[:2] != (self.size, self.size):
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return image


class TipCheck:
    """
    Reject tip poses that are frozen or implausible, so a stale robot state can't steer the crop.

    Frozen: moved less than still_m for more than frozen_s (a stale robot state only
    changes by rounding noise; in training data the tip never stayed within 20 um
    for longer than 2.3 s). Implausible: the tip is closer than z_min_m or farther
    than z_max_m in front of the camera (training: 6-40 mm).
    """

    def __init__(self, frozen_s=3.0, still_m=1e-6, z_min_m=0.005, z_max_m=0.08):
        self.frozen_s, self.still_m = float(frozen_s), float(still_m)
        self.z_min_m, self.z_max_m = float(z_min_m), float(z_max_m)
        self.reset()

    def reset(self):
        self.ref = None
        self.since = None

    def update(self, t, xyz):
        """Return True if the tip pose xyz (m, camera frame) seen at time t is usable."""
        xyz = np.asarray(xyz, dtype=float)
        if (self.ref is None or t < self.since
                or np.linalg.norm(xyz - self.ref) > self.still_m):
            self.ref, self.since = xyz, t
        frozen = t - self.since > self.frozen_s
        return bool(not frozen and self.z_min_m <= xyz[2] <= self.z_max_m)


def project_tip(xyz, p, r=None, width=540, height=540, image_size=540):
    """
    Project a tip position (camera frame, m) to pixels of the image_size square frame.

    p and r are CameraInfo.p (12 values) and CameraInfo.r (9 values, identity if None);
    width and height are the CameraInfo image size. Returns (u, v) or None if the
    point is behind the camera, off the image or not finite.
    """
    xyz = np.asarray(xyz, dtype=float)
    if r is not None:
        xyz = np.asarray(r, dtype=float).reshape(3, 3) @ xyz
    x, y, z = xyz
    fx, cx, fy, cy = (float(p[i]) for i in (0, 2, 5, 6))
    if not np.isfinite([x, y, z, fx, fy, cx, cy]).all() or z <= 0 or width <= 0 or height <= 0:
        return None
    u = (fx * x / z + cx) * image_size / width
    v = (fy * y / z + cy) * image_size / height
    return (u, v) if 0 <= u < image_size and 0 <= v < image_size else None


def measure_spot(image, params=SPOT_PARAMS, tip=None):
    """
    Segment the spot in a BGR frame; return (measurement dict, contour or None).

    With tip (projected fiber tip, pixels), blobs farther than params['tip_gate_px']
    from it are ignored: they are glare or the other tool, not this laser's spot.
    """
    mask, contrast = spot_mask(image, params)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gate = params.get('tip_gate_px', 0.0)
    compact = []
    for c in contours:
        major, minor = _axes(c)
        if major > params['max_elongation'] * max(minor, 1.0):
            continue
        if tip is not None and gate > 0:
            m = cv2.moments(c)
            if m['m00'] <= 0 or math.hypot(m['m10'] / m['m00'] - tip[0],
                                           m['m01'] / m['m00'] - tip[1]) > gate:
                continue
        compact.append(c)
    compact.sort(key=cv2.contourArea, reverse=True)
    nan = float('nan')
    spot = {
        'detected': 0.0, 'area': nan, 'major': nan, 'minor': nan, 'u': nan, 'v': nan,
        'peak': nan, 'energy': nan, 'n_blobs': float(len(compact)),
        'area2': float(cv2.contourArea(compact[1])) if len(compact) > 1 else 0.0,
    }
    if not compact or cv2.contourArea(compact[0]) < params['min_area_px']:
        return spot, None
    contour = compact[0]
    moments = cv2.moments(contour)
    if moments['m00'] <= 0:
        return spot, None
    inside = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(inside, [contour], -1, 255, cv2.FILLED)
    values = contrast[inside > 0]
    major, minor = _axes(contour)
    spot.update(detected=1.0, area=float(cv2.contourArea(contour)), major=major, minor=minor,
                u=moments['m10'] / moments['m00'], v=moments['m01'] / moments['m00'],
                peak=float(values.max()), energy=float(np.clip(values, 0, None).sum()))
    return spot, contour


def legacy_spot_area(image, params=LEGACY_PARAMS):
    """Spot area in px^2 as the current segmentation/analysis nodes compute it, NaN if none."""
    blur = int(params['blur_size'])
    smoothed = cv2.GaussianBlur(image, (blur, blur), 0)
    hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
    lower = np.array([params['h_min'], params['s_min'], params['v_min']], np.uint8)
    upper = np.array([params['h_max'], params['s_max'], params['v_max']], np.uint8)
    binary = cv2.inRange(hsv, lower, upper)
    b, g, r = cv2.split(smoothed.astype(np.float32))
    excess = (g - 0.5 * (r + b)) >= params['green_excess_min']
    binary &= excess.astype(np.uint8) * 255
    k = int(params['kernel_size'])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = max((cv2.contourArea(c) for c in contours), default=0.0)
    return float(area) if area >= params['min_spot_area_px'] else float('nan')


class SpotHistory:
    """Causal features over recent spot measurements, windowed by time in seconds."""

    def __init__(self, short_window_s=0.5, long_window_s=1.5, max_gap_s=1.0):
        self.short_window_s = float(short_window_s)
        self.long_window_s = float(long_window_s)
        self.max_gap_s = float(max_gap_s)
        self._buf = deque()

    def reset(self):
        self._buf.clear()

    def update(self, t, spot, tip=None):
        """
        Add the measurement taken at time t; return (features, crop centre or None).

        The crop centre is the current spot, else the projected tip (if given), else
        the last spot seen in the long window, else None (image centre).
        """
        if self._buf and not 0.0 <= t - self._buf[-1][0] <= self.max_gap_s:
            self._buf.clear()
        self._buf.append((t, spot))
        while t - self._buf[0][0] > self.long_window_s:
            self._buf.popleft()

        short = [(ti, s) for ti, s in self._buf if t - ti <= self.short_window_s]
        seen = [(ti, math.sqrt(s['area'])) for ti, s in short if s['detected']]
        slope = 0.0
        if len(seen) >= 2 and seen[-1][0] - seen[0][0] >= 0.1:
            ts, sizes = np.array(seen).T
            slope = float(np.polyfit(ts - ts[-1], sizes, 1)[0])

        found = bool(spot['detected'])
        size = math.sqrt(spot['area']) if found else 0.0
        largest = max((math.sqrt(s['area']) for _, s in self._buf if s['detected']), default=0.0)
        major, minor = (spot['major'], spot['minor']) if found else (0.0, 0.0)
        feats = {
            'detected': float(found),
            'sqrt_area': size,
            'minor_axis': minor,
            'elongation': major / minor if found and minor > 0 else 0.0,
            'peak_contrast': spot['peak'] if found else 0.0,
            'log_energy': math.log1p(spot['energy']) if found else 0.0,
            'n_blobs': spot['n_blobs'],
            'sqrt_area2': math.sqrt(spot['area2']),
            'detected_frac': sum(s['detected'] for _, s in short) / len(short),
            'sqrt_area_slope': slope,
            'shrink_ratio': size / largest if found and largest > 0 else 0.0,
        }
        center = (spot['u'], spot['v']) if found else tip
        for _, s in reversed(self._buf):
            if center is not None:
                break
            if s['detected']:
                center = (s['u'], s['v'])
        return feats, center


def feature_vector(feats):
    """Return the model input vector (float32) in FEATURE_NAMES order."""
    return np.array([feats[name] for name in FEATURE_NAMES], np.float32)


def crop_square(image, center, size):
    """
    Return a size x size crop centred on center=(u, v), zero-padded at borders.

    The crop has a fixed pixel size so the spot's apparent size, a distance
    cue, survives. Cropping tightly to the spot and resizing would erase it.
    A centre of None uses the image centre.
    """
    h, w = image.shape[:2]
    cu, cv = (w / 2.0, h / 2.0) if center is None else center
    x0 = int(round(cu)) - size // 2
    y0 = int(round(cv)) - size // 2
    out = np.zeros((size, size) + image.shape[2:], image.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + size, w), min(y0 + size, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def crop_to_tensor(crop_bgr, input_size):
    """Resize a BGR crop; return a 1x3xHxW float32 array normalised as ImageNet RGB."""
    resized = cv2.resize(crop_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].copy()
