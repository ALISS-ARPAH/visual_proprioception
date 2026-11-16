"""
Score a bag with our model and the colleague's model on the same frames.

Ours runs exactly as score_contact_bag / the live node: TipCheck on the tip pose, spot
gated near the projected tip, SpotHistory, CausalMean smoothing. Theirs follows their
pipeline (re-implemented): a 270x203 crop on the raw projected tip (no TipCheck),
resized to 224x168, padded to 224x224, ImageNet normalisation, softmax p_touch; frames
whose tip does not project into the image get no score. Each row also logs the nearest
tip pose (position + quaternion) so evaluate_bag.py can build depth labels.

Usage: score_bag.py <bag> <our model.json> <their .onnx from export_onnx.py> <out.csv>
       [--max-frames N]
"""
import argparse
from collections import deque
import csv
import math
import os
import sys
import time

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
import rosbag2_py
from rosidl_runtime_py.utilities import get_message

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from visual_proprioception.contact_bag import (bag_parts, image_to_bgr,  # noqa: E402
                                               stamp_seconds)
from visual_proprioception.contact_model import CausalMean, ContactModel  # noqa: E402
from visual_proprioception.laser_spot_features import (measure_spot,  # noqa: E402
                                                       project_tip, Rectifier, SpotHistory,
                                                       TipCheck)

IMG, INFO, POSE, LASER = ('/ves_camera/image', '/ves_camera/camera_info',
                          '/ves_smoother/right/tip_pose', '/ves/right/measured_tool')
CW, CH = 270, 203
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def their_input(image, tip):
    """Crop, resize and pad as BladderTouchDynamicCrop does; returns a 1x3x224x224 blob."""
    u, v = tip
    h, w = image.shape[:2]
    left, top = round(u - CW / 2), round(v - CH / 2)
    pl, pt = max(0, -left), max(0, -top)
    pr, pb = max(0, left + CW - w), max(0, top + CH - h)
    if pl or pt or pr or pb:
        image = cv2.copyMakeBorder(image, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        left, top = left + pl, top + pt
    crop = image[top:top + CH, left:left + CW]
    crop = cv2.resize(crop, (224, 168), interpolation=cv2.INTER_AREA)
    crop = cv2.copyMakeBorder(crop, 28, 28, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def messages(parts, topics):
    types = {}
    readers = []
    for uri, storage in parts:
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage),
               rosbag2_py.ConverterOptions('cdr', 'cdr'))
        types.update({t.name: t.type for t in r.get_all_topics_and_types()})
        readers.append(r)
    msg_types = {t: get_message(types[t]) for t in topics}
    for r in readers:
        r.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
        while r.has_next():
            topic, data, t_rec = r.read_next()
            yield topic, deserialize_message(data, msg_types[topic]), t_rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    parser.add_argument('bag', help='bag folder, folder of split .mcap files, or one .mcap')
    parser.add_argument('model', help='our model.json')
    parser.add_argument('their_onnx', help="colleague's model from export_onnx.py")
    parser.add_argument('out', help='output CSV')
    parser.add_argument('--max-frames', type=int, default=0, help='stop early (testing)')
    args = parser.parse_args()

    model = ContactModel(args.model)
    theirs = cv2.dnn.readNetFromONNX(args.their_onnx)
    parts = bag_parts(args.bag)
    if args.max_frames:
        parts = parts[:1]
    history, tip_check = SpotHistory(**model.window), TipCheck()
    smoother = CausalMean(model.smooth_sec)
    poses = deque(maxlen=300)
    info = rectifier = None
    laser = math.nan
    rows, started = [], time.time()
    for topic, msg, t_rec in messages(parts, (IMG, INFO, POSE, LASER)):
        if topic == INFO:
            if info is None or tuple(msg.d) != tuple(info.d) or tuple(msg.p) != tuple(info.p):
                info, rectifier = msg, Rectifier.from_camera_info(msg)
            continue
        if topic == POSE:
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            poses.append((stamp_seconds(msg.header), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))
            continue
        if topic == LASER:
            laser = float(msg.data)
            continue
        if info is None:
            continue
        t = stamp_seconds(msg.header) or t_rec * 1e-9
        image = rectifier(image_to_bgr(msg))
        xyz = quat = tip = raw_tip = None
        if poses:
            stamp, xyz_, quat_ = min(poses, key=lambda e: abs(e[0] - t))
            if abs(stamp - t) <= 0.1:
                xyz, quat = xyz_, quat_
                raw_tip = project_tip(xyz, info.p, info.r, info.width, info.height,
                                      image.shape[1])
                if tip_check.update(t, xyz):
                    tip = raw_tip
        spot, _ = measure_spot(image, model.spot_params, tip)
        feats, center = history.update(t, spot, tip)
        p_raw = model.predict(image, feats, center, t)
        p_ours = smoother.update(t, p_raw)
        p_theirs = math.nan
        if raw_tip is not None:
            theirs.setInput(their_input(image, raw_tip))
            z = theirs.forward()[0]
            p_theirs = float(1.0 / (1.0 + math.exp(z[0] - z[1])))
        rows.append({'t': t, 'laser': laser, 'ours_raw': p_raw, 'ours': p_ours,
                     'theirs': p_theirs,
                     **{k: (xyz[i] if xyz else math.nan) for i, k in enumerate('xyz')},
                     **{k: (quat[i] if quat else math.nan)
                        for i, k in enumerate(('qx', 'qy', 'qz', 'qw'))},
                     'tip_ok': int(tip is not None), 'spot': int(spot['detected'])})
        if args.max_frames and len(rows) >= args.max_frames:
            break
        if len(rows) % 2000 == 0:
            print(f'  {len(rows)} frames, {time.time() - started:.0f} s', flush=True)
    if not rows:
        sys.exit('no camera frames scored')
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} frames -> {args.out}')


if __name__ == '__main__':
    main()
