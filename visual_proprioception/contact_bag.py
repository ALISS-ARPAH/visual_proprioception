r"""
Score a ROS 2 bag with a trained cutting-depth / contact model and render a video.

Usage:
    ros2 run visual_proprioception score_contact_bag \
        --bag ~/aliss_runs/test_data/cut_1 \
        --model ~/aliss_runs/contact_full/runD_seed0/combined/model.json \
        --out ~/aliss_runs/test_data/cut_1_prediction.mp4

Frames are processed in recording order exactly as laser_contact_model_node does:
rectify the raw frame with CameraInfo (the models were trained on rectified
540x540 frames), find the spot near the projected fiber tip, update the
time-window features, crop, score, and average P over the model's smoothing
window. A frame only uses messages recorded before it, as it would live. A tip
pose that is frozen or implausible (TipCheck) is ignored.

--bag may also be a folder of split files without metadata.yaml (e.g. copied
from a backup drive) or a single .mcap file; split files are read in order and
empty ones are skipped.

Writes <out> (real-time video; pauses in the recording are shortened to
--max-hold s) and <out without .mp4>.csv (one row per frame):

    top left   recording name, time, laser state (laser_topic)
    top right  model decision and P
    bottom     P over the last --window s, the thresholds, and laser-on periods
"""

import argparse
from collections import deque
import csv
import math
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .contact_model import CausalMean, ContactModel
from .contact_video import BG, font, MUTED, STATE_COLORS, TEXT
from .laser_spot_features import measure_spot, project_tip, Rectifier, SpotHistory, TipCheck

PANEL_H = 170
LASER_COLOR = (235, 87, 87)


def bag_parts(bag):
    """
    Return [(uri, storage_id)] to read in order.

    A folder with metadata.yaml is one bag. Without it (or with an empty one, as a
    failed copy leaves), each .mcap / .db3 split
    file is opened on its own in split-number order; empty files (failed copies)
    are skipped with a warning. A single file is read as it is.
    """
    if os.path.isfile(bag):
        return [(bag, 'mcap' if bag.endswith('.mcap') else 'sqlite3')]
    names = os.listdir(bag)
    storage = 'mcap' if any(n.endswith('.mcap') for n in names) else 'sqlite3'
    if 'metadata.yaml' in names and os.path.getsize(os.path.join(bag, 'metadata.yaml')) > 0:
        return [(bag, storage)]

    def split_number(name):
        m = re.search(r'_(\d+)\.[^.]+$', name)
        return (int(m.group(1)) if m else -1, name)

    ext = '.mcap' if storage == 'mcap' else '.db3'
    parts = []
    for name in sorted((n for n in names if n.endswith(ext)), key=split_number):
        path = os.path.join(bag, name)
        if os.path.getsize(path) == 0:
            print(f'WARNING: skipping empty {path}')
        else:
            parts.append((path, storage))
    return parts


def image_to_bgr(msg):
    """Convert a sensor_msgs/Image (rgb8, bgr8, mono8) to a BGR array."""
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported image encoding {msg.encoding!r}')
    data = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    image = data[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding == 'rgb8':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == 'mono8':
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image)


def stamp_seconds(header):
    return header.stamp.sec + 1e-9 * header.stamp.nanosec


class BagRenderer:
    """Draws one video frame: the camera image, the decision box and a scrolling P trace."""

    def __init__(self, width, thresholds, name, window_s, depth_labels):
        self.w = width
        self.thresholds = thresholds
        self.name = name
        self.window_s = window_s
        self.names = ({'CONTACT': 'AT CUT DEPTH', 'SUSPECTED': 'CLOSE',
                       'NO CONTACT': 'TOO SHALLOW'} if depth_labels
                      else {k: k for k in STATE_COLORS})
        self.p_name = 'P(at cut depth)' if depth_labels else 'P(contact)'
        self.f = {k: font(s, b) for k, (s, b) in {
            'state': (30, True), 'prob': (17, False), 'label': (16, True), 'small': (13, False),
            'laser': (16, True)}.items()}

    def state(self, p):
        if p >= self.thresholds['confirm']:
            return 'CONTACT'
        return 'SUSPECTED' if p >= self.thresholds['suspect'] else 'NO CONTACT'

    def frame(self, image, trace, t_rel, p, laser, overlay):
        if overlay is not None:
            image = image.copy()
            contour, center, tip, crop_size = overlay
            if center is not None:
                x0 = int(round(center[0])) - crop_size // 2
                y0 = int(round(center[1])) - crop_size // 2
                cv2.rectangle(image, (x0, y0), (x0 + crop_size, y0 + crop_size), (255, 200, 0), 1)
            if contour is not None:
                cv2.drawContours(image, [contour], -1, (0, 0, 255), 1)
            if tip is not None:
                cv2.drawMarker(image, (int(round(tip[0])), int(round(tip[1]))), (255, 0, 255),
                               cv2.MARKER_CROSS, 12, 1)
        rgb = cv2.cvtColor(cv2.resize(image, (self.w, self.w), interpolation=cv2.INTER_CUBIC),
                           cv2.COLOR_BGR2RGB)
        canvas = Image.new('RGB', (self.w, self.w + PANEL_H), BG)
        canvas.paste(Image.fromarray(rgb), (0, 0))
        d = ImageDraw.Draw(canvas, 'RGBA')

        # top left: recording, time and laser state
        title = f'{self.name}   {int(t_rel // 60):02d}:{t_rel % 60:04.1f}'
        tw = d.textlength(title, font=self.f['label'])
        d.rounded_rectangle((12, 12, 32 + tw, 44), 8, fill=(0, 0, 0, 150))
        d.text((22, 17), title, font=self.f['label'], fill=TEXT)
        if laser:
            lw = d.textlength('LASER ON', font=self.f['laser'])
            d.rounded_rectangle((12, 50, 32 + lw, 80), 8, fill=LASER_COLOR + (230,))
            d.text((22, 55), 'LASER ON', font=self.f['laser'], fill=(255, 255, 255))

        # top right: model decision
        if math.isfinite(p):
            key = self.state(p)
            bg, fg = STATE_COLORS[key]
            shown, ptxt = self.names[key], f'{self.p_name} = {p:.2f}'
        else:
            bg, fg, shown, ptxt = (58, 63, 75), TEXT, 'NO DATA', 'waiting for calibration'
        bw, bh = 250, 80
        x0, y0 = self.w - bw - 14, 14
        d.rounded_rectangle((x0, y0, x0 + bw, y0 + bh), 12, fill=bg + (235,))
        sf = self.f['state']
        while d.textlength(shown, font=sf) > bw - 16 and sf.size > 12:
            sf = sf.font_variant(size=sf.size - 1)
        d.text((x0 + (bw - d.textlength(shown, font=sf)) / 2, y0 + 7), shown, font=sf, fill=fg)
        d.text((x0 + (bw - d.textlength(ptxt, font=self.f['prob'])) / 2, y0 + 50), ptxt,
               font=self.f['prob'], fill=fg)

        # bottom: scrolling trace of P with laser-on periods
        py = self.w
        left, right, top, bottom = 52, self.w - 18, py + 34, py + 140
        d.text((left, py + 10), f'{self.p_name}, last {self.window_s:g} s', font=self.f['small'],
               fill=MUTED)
        lx = right - d.textlength('laser on', font=self.f['small'])
        d.rectangle((lx - 16, py + 13, lx - 6, py + 23), fill=LASER_COLOR + (110,))
        d.text((lx, py + 10), 'laser on', font=self.f['small'], fill=MUTED)
        t0 = t_rel - self.window_s

        def x(t):
            return left + (t - t0) / self.window_s * (right - left)

        def y(v):
            return bottom - v * (bottom - top)

        points = [(tt, pp, ll) for tt, pp, ll in trace if tt >= t0]
        for (ta, _, la), (tb, _, _) in zip(points, points[1:] + [(t_rel, 0, 0)]):
            if la:
                d.rectangle((x(ta), top, x(tb), bottom), fill=LASER_COLOR + (70,))
        d.rectangle((left, top, right, bottom), outline=(60, 64, 72))
        for name, color in (('confirm', STATE_COLORS['CONTACT'][0]),
                            ('suspect', STATE_COLORS['SUSPECTED'][0])):
            thr = self.thresholds[name]
            for xs in range(left, right, 10):
                d.line((xs, y(thr), min(xs + 5, right), y(thr)), fill=color, width=1)
            d.text((16, y(thr) - 9), f'{thr:.1f}', font=self.f['small'], fill=color)
        line = [(x(tt), y(pp)) for tt, pp, _ in points if math.isfinite(pp)]
        if len(line) > 1:
            d.line(line, fill=TEXT, width=2, joint='curve')
        if line:
            cx, cy = line[-1]
            d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=bg, outline=TEXT)
        d.text((left, bottom + 6), f'-{self.window_s:g} s', font=self.f['small'], fill=MUTED)
        d.text((right - d.textlength('now', font=self.f['small']), bottom + 6), 'now',
               font=self.f['small'], fill=MUTED)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def summarize(rows, confirm):
    """Print how P behaves with the laser on/off and before each laser-on edge."""
    p = np.array([r['p'] for r in rows], float)
    laser = np.array([r['laser'] for r in rows], float) > 0.5
    t = np.array([r['t_rel'] for r in rows], float)
    ok = np.isfinite(p)
    print(f'{len(rows)} frames over {t[-1] - t[0]:.0f} s; '
          f'laser on in {laser[ok].mean():.0%} of frames')
    for name, sel in (('laser on', ok & laser), ('laser off', ok & ~laser)):
        if sel.any():
            print(f'  {name:9s}: mean P {p[sel].mean():.2f}, '
                  f'P >= {confirm:g} in {np.mean(p[sel] >= confirm):.0%}')
    edges = np.flatnonzero(laser[1:] & ~laser[:-1]) + 1
    flagged = [bool(np.any(p[(t >= t[e] - 1.0) & (t < t[e]) & ok] >= confirm)) for e in edges]
    if edges.size:
        print(f'  {len(edges)} laser-on events; P >= {confirm:g} in the second before '
              f'{sum(flagged)} of them ({np.mean(flagged):.0%})')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bag', required=True, help='rosbag2 directory (mcap or sqlite3)')
    parser.add_argument('--model', required=True, help='model.json from train_contact_model')
    parser.add_argument('--out', help='output .mp4 (default <bag>_prediction.mp4)')
    parser.add_argument('--name', help='recording name shown in the video (default: bag folder)')
    parser.add_argument('--image-topic', default='/ves_camera/image')
    parser.add_argument('--camera-info-topic', default='/ves_camera/camera_info')
    parser.add_argument('--tip-pose-topic', default='/ves_smoother/right/tip_pose')
    parser.add_argument('--laser-topic', default='/ves/right/measured_tool')
    parser.add_argument('--tip-max-age', type=float, default=0.1,
                        help='use the tip pose only if within this many s of the frame')
    parser.add_argument('--smooth', type=float,
                        help='mean of P over this many s (default: the value in model.json, '
                             'as the node does)')
    parser.add_argument('--window', type=float, default=10.0, help='trace length in s')
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--max-hold', type=float, default=0.5,
                        help='longest time one frame stays on screen (s)')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='stop after this many s of recording (0: whole bag)')
    parser.add_argument('--no-overlay', dest='overlay', action='store_false',
                        help='do not draw the spot, crop box and projected tip')
    args = parser.parse_args(argv)

    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        sys.exit('score_contact_bag needs ROS 2 (source /opt/ros/<distro>/setup.bash).')

    bag = os.path.expanduser(args.bag).rstrip('/')
    model = ContactModel(args.model)
    if model.target == 'depth':
        sys.exit('This tool renders P(contact / cutting depth); '
                 'distance models are not supported.')
    depth_labels = (model.meta.get('labels') or {}).get('scheme') == 'depth'
    out = os.path.expanduser(args.out or f'{bag}_prediction.mp4')
    csv_path = os.path.splitext(out)[0] + '.csv'

    parts = bag_parts(bag)
    if not parts:
        sys.exit(f'No bag files in {bag}')
    if len(parts) > 1:
        print(f'{bag}: no metadata.yaml; reading {len(parts)} split files in order')

    def open_reader(uri, storage):
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage),
                    rosbag2_py.ConverterOptions('cdr', 'cdr'))
        return reader

    reader = open_reader(*parts[0])
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    topics = [args.image_topic, args.camera_info_topic, args.tip_pose_topic, args.laser_topic]
    missing = [t for t in topics if t not in types]
    if args.image_topic in missing or args.camera_info_topic in missing:
        sys.exit(f'{bag} lacks {missing}')
    if missing:
        print(f'WARNING: {bag} lacks {missing}; continuing without them')
    wanted = rosbag2_py.StorageFilter(topics=[t for t in topics if t in types])
    msg_types = {t: get_message(types[t]) for t in topics if t in types}

    def messages():
        for k, part in enumerate(parts):
            r = reader if k == 0 else open_reader(*part)
            r.set_filter(wanted)
            while r.has_next():
                yield r.read_next()

    history = SpotHistory(**model.window)
    tip_check = TipCheck()
    smoother = CausalMean(model.smooth_sec if args.smooth is None else args.smooth)
    name = args.name or os.path.basename(bag)
    renderer = BagRenderer(args.width // 2 * 2, model.thresholds, name, args.window, depth_labels)
    poses = deque(maxlen=300)
    info = rectifier = None
    laser = math.nan
    trace = deque()
    rows = []
    writer = pending = None
    t_first = None
    clock = 0.0
    started = time.time()

    def emit(frame, seconds):
        nonlocal clock
        n = round((clock + seconds) * args.fps) - round(clock * args.fps)
        for _ in range(max(n, 1)):
            writer.write(frame)
        clock += seconds

    for topic, data, t_rec in messages():
        msg = deserialize_message(data, msg_types[topic])
        if topic == args.camera_info_topic:
            if info is None or tuple(msg.d) != tuple(info.d) or tuple(msg.p) != tuple(info.p):
                info, rectifier = msg, Rectifier.from_camera_info(msg)
            continue
        if topic == args.tip_pose_topic:
            p = msg.pose.pose.position
            poses.append((stamp_seconds(msg.header), (p.x, p.y, p.z)))
            continue
        if topic == args.laser_topic:
            laser = float(msg.data)
            continue
        if info is None:
            continue
        t = stamp_seconds(msg.header) or t_rec * 1e-9
        t_first = t if t_first is None else t_first
        if args.duration > 0 and t - t_first > args.duration:
            break
        image = rectifier(image_to_bgr(msg))
        tip = None
        if poses:
            stamp, xyz = min(poses, key=lambda e: abs(e[0] - t))
            if abs(stamp - t) <= args.tip_max_age and tip_check.update(t, xyz):
                tip = project_tip(xyz, info.p, info.r, info.width, info.height, image.shape[1])
        spot, contour = measure_spot(image, model.spot_params, tip)
        feats, center = history.update(t, spot, tip)
        p_raw = model.predict(image, feats, center, t)
        prob = smoother.update(t, p_raw)
        t_rel = t - t_first
        laser_on = bool(laser > 0.5)
        trace.append((t_rel, prob, laser_on))
        while trace and trace[0][0] < t_rel - args.window - 1.0:
            trace.popleft()
        rows.append({'t': t, 't_rel': t_rel, 'laser': laser, 'p_raw': p_raw, 'p': prob,
                     'decision': renderer.names[renderer.state(prob)],
                     'spot_detected': int(spot['detected']),
                     'crop_source': ('spot' if spot['detected'] else 'tip' if tip is not None
                                     else 'last spot' if center is not None else 'image centre'),
                     'tip_u': tip[0] if tip else math.nan, 'tip_v': tip[1] if tip else math.nan})
        crop = getattr(model, 'crop_size', 192)
        frame = renderer.frame(image, trace, t_rel, prob, laser_on,
                               (contour, center, tip, crop) if args.overlay else None)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'avc1'), args.fps, (w, h))
            if not writer.isOpened():
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
        if pending is not None:
            emit(pending[0], min(max(t - pending[1], 1.0 / args.fps), args.max_hold))
        pending = (frame, t)
        if len(rows) % 1000 == 0:
            print(f'  {len(rows)} frames, {t_rel:.0f} s of recording, '
                  f'{time.time() - started:.0f} s elapsed', flush=True)
    if writer is None:
        sys.exit(f'No {args.image_topic} frames after a CameraInfo in {bag}')
    emit(pending[0], 0.5)
    writer.release()
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{out} ({clock:.0f} s video) and {csv_path}')
    summarize(rows, model.thresholds['confirm'])


if __name__ == '__main__':
    main()
