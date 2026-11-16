"""Tests for spot features, the spot model export round trip and the metrics."""

from collections import deque
import json
import math
import os

import cv2
import numpy as np

from visual_proprioception.contact_bag import bag_parts
from visual_proprioception.contact_model import CausalMean, ContactModel
from visual_proprioception.contact_train import (auc, causal_mean, fit_logistic, fit_ridge,
                                                 past_index, predict_linear, predict_logistic)
from visual_proprioception.laser_spot_features import (crop_square, FEATURE_NAMES,
                                                       feature_vector, measure_spot,
                                                       project_tip, Rectifier, SpotHistory,
                                                       TipCheck)


def tissue(size=540):
    """Red, slightly noisy background like bladder wall."""
    rng = np.random.default_rng(0)
    image = np.full((size, size, 3), (40, 45, 150), np.uint8)
    return cv2.add(image, rng.integers(0, 8, image.shape, dtype=np.uint8))


def test_green_spot_found_on_red_tissue():
    image = tissue()
    cv2.circle(image, (300, 200), 10, (90, 220, 60), -1)
    spot, contour = measure_spot(image)
    assert spot['detected'] == 1.0 and contour is not None
    assert abs(spot['u'] - 300) < 2 and abs(spot['v'] - 200) < 2
    assert 14 < math.sqrt(spot['area']) < 21


def test_blue_tip_glow_and_glare_streak_rejected():
    image = tissue()
    cv2.circle(image, (150, 150), 12, (220, 150, 40), -1)            # blue tip glow
    cv2.rectangle(image, (300, 400), (390, 405), (90, 220, 60), -1)  # thin glare streak
    spot, contour = measure_spot(image)
    assert spot['detected'] == 0.0 and contour is None


def spot_at(radius, u=270.0, v=270.0):
    if radius is None:
        return {'detected': 0.0, 'area': math.nan, 'major': math.nan, 'minor': math.nan,
                'u': math.nan, 'v': math.nan, 'peak': math.nan, 'energy': math.nan,
                'n_blobs': 0.0, 'area2': 0.0}
    area = math.pi * radius ** 2
    return {'detected': 1.0, 'area': area, 'major': 2 * radius, 'minor': 2 * radius,
            'u': u, 'v': v, 'peak': 80.0, 'energy': 50.0 * area, 'n_blobs': 1.0, 'area2': 0.0}


def test_history_trend_shrink_and_centre_memory():
    history = SpotHistory(short_window_s=0.5, long_window_s=1.5, max_gap_s=1.0)
    for i, r in enumerate([12, 11, 10, 9, 8, 7]):
        feats, center = history.update(0.1 * i, spot_at(r, u=100 + i))
    assert feats['sqrt_area_slope'] < 0
    assert 0.5 < feats['shrink_ratio'] < 0.7
    feats, center = history.update(0.6, spot_at(None))
    assert feats['detected'] == 0.0 and feats['sqrt_area'] == 0.0
    assert center == (105, 270.0)              # last seen spot
    assert 0.8 < feats['detected_frac'] < 0.9  # 5 of 6 frames in the 0.5 s window
    feats, center = history.update(5.0, spot_at(None))
    assert center is None                      # gap > max_gap_s resets


def test_crop_is_fixed_size_and_zero_padded():
    image = np.full((100, 100, 3), 255, np.uint8)
    crop = crop_square(image, (5.0, 5.0), 40)
    assert crop.shape == (40, 40, 3)
    assert crop[0, 0].sum() == 0 and crop[39, 39].sum() == 255 * 3


def test_spot_model_export_matches_training(tmp_path):
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    x = rng.normal(size=(400, len(FEATURE_NAMES))).astype(np.float32)
    x[:, 1] -= 2.0 * y   # contact frames have smaller spots
    params = fit_logistic(x, y)
    prob = predict_logistic(params, x)
    assert auc(prob, y == 1, y == 0) > 0.8
    mean, std, w, b = params
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({'kind': 'spot', 'feature_names': list(FEATURE_NAMES),
                                'mean': mean.tolist(), 'std': std.tolist(),
                                'weights': w.tolist(), 'bias': b}))
    model = ContactModel(str(path))
    for i in range(5):
        feats = dict(zip(FEATURE_NAMES, x[i].tolist()))
        assert abs(model.predict(None, feats, None) - prob[i]) < 1e-5
        assert np.allclose(feature_vector(feats), x[i])


def test_auc_handles_ties():
    score = np.array([0.0, 0.0, 1.0, 1.0])
    label = np.array([0, 1, 0, 1])
    assert auc(score, label == 1, label == 0) == 0.5


def irregular_stream(seed=1):
    """Two approaches with uneven frame spacing (4-30 Hz) and random probabilities."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.cumsum(rng.choice([0.033, 0.1, 0.26], 40)),
                        100 + np.cumsum(rng.choice([0.033, 0.1, 0.26], 40))])
    seq = np.array(['a'] * 40 + ['b'] * 40)
    return {'t': t, 'sequence': seq}, rng.random(80)


def test_live_smoothing_matches_training():
    data, prob = irregular_stream()
    offline = causal_mean(prob, data, 0.3)
    for seq in ('a', 'b'):
        live = CausalMean(0.3)
        for i in np.flatnonzero(data['sequence'] == seq):
            assert abs(live.update(data['t'][i], prob[i]) - offline[i]) < 1e-12


def test_live_past_frame_matches_training():
    """The live buffer picks the same past frame as past_index does offline."""
    data, _ = irregular_stream()
    lag = {'min': 0.2, 'target': 0.4, 'max': 0.8}
    offline = past_index(data, lag)
    model = ContactModel.__new__(ContactModel)
    model.lag, model.recent = lag, deque()
    for seq in ('a', 'b'):
        model.reset()
        for i in np.flatnonzero(data['sequence'] == seq):
            past = model._past_embedding(data['t'][i])
            assert (i if past is None else int(past[0])) == offline[i]
            model.recent.append((data['t'][i], np.array([i])))


def test_project_tip_matches_pinhole_and_rejects_points_behind():
    p = [370.0, 0, 310.0, 0, 0, 380.0, 290.0, 0, 0, 0, 1, 0]
    u, v = project_tip([0.01, -0.005, 0.025], p)
    assert abs(u - (370 * 0.4 + 310)) < 1e-9 and abs(v - (380 * -0.2 + 290)) < 1e-9
    half = project_tip([0.01, -0.005, 0.025], p, width=1080, height=1080)
    assert abs(half[0] - u / 2) < 1e-9
    assert project_tip([0.0, 0.0, -0.02], p) is None
    assert project_tip([1.0, 0.0, 0.01], p) is None   # off the image


def test_spot_far_from_projected_tip_is_ignored():
    image = tissue()
    cv2.circle(image, (60, 270), 8, (90, 220, 60), -1)    # glint at the scope edge
    cv2.circle(image, (300, 200), 6, (90, 220, 60), -1)   # real spot near the tip
    spot, _ = measure_spot(image, tip=(310.0, 215.0))
    assert spot['detected'] == 1.0 and abs(spot['u'] - 300) < 2
    spot, _ = measure_spot(image, tip=(450.0, 450.0))
    assert spot['detected'] == 0.0


def test_crop_centre_prefers_spot_then_tip_then_last_spot():
    history = SpotHistory()
    _, center = history.update(0.0, spot_at(5.0, 100.0, 120.0), tip=(90.0, 110.0))
    assert center == (100.0, 120.0)
    _, center = history.update(0.1, spot_at(None), tip=(95.0, 115.0))
    assert center == (95.0, 115.0)
    _, center = history.update(0.2, spot_at(None))
    assert center == (100.0, 120.0)


def test_depth_spot_model_outputs_mm_and_flags_below_threshold(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_NAMES))).astype(np.float32)
    y = 2.0 + 1.5 * x[:, 1] - 0.5 * x[:, 3]
    params = fit_ridge(x, y, l2=1e-3)
    assert np.abs(predict_linear(params, x) - y).max() < 0.05
    mean, std, w, b = params
    path = tmp_path / 'model.json'
    path.write_text(json.dumps({
        'kind': 'spot', 'target': 'depth', 'feature_names': list(FEATURE_NAMES),
        'thresholds': {'suspect': 2.0, 'confirm': 1.0},
        'mean': mean.tolist(), 'std': std.tolist(), 'weights': w.tolist(), 'bias': b}))
    model = ContactModel(str(path))
    out = model.predict(None, dict(zip(FEATURE_NAMES, x[0].tolist())), None)
    assert abs(out - predict_linear(params, x[:1])[0]) < 1e-4
    assert model.at_threshold(0.8, 1.0) and not model.at_threshold(1.2, 1.0)


def test_rectifier_passes_rectified_frames_and_undistorts_raw_ones():
    k = [400.0, 0, 270, 0, 400.0, 270, 0, 0, 1]
    r = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    p = [400.0, 0, 270, 0, 0, 400.0, 270, 0, 0, 0, 1, 0]
    image = tissue()
    same = Rectifier(k, [0.0] * 5, r, p, 540, 540)
    assert not same.needed and same(image) is image
    raw = cv2.resize(image, (1080, 1080))
    k2 = [800.0, 0, 540, 0, 800.0, 540, 0, 0, 1]
    p2 = [800.0, 0, 540, 0, 0, 800.0, 540, 0, 0, 0, 1, 0]
    out = Rectifier(k2, [-0.3, 0.1, 0, 0, 0], r, p2, 1080, 1080)(raw)
    assert out.shape == (540, 540, 3)


def test_tip_check_rejects_frozen_and_implausible_poses():
    check = TipCheck(frozen_s=3.0)
    assert check.update(0.0, (0.001, 0.002, 0.02))
    assert check.update(2.9, (0.001, 0.002, 0.02))         # still for 2.9 s: fine
    assert not check.update(3.5, (0.001, 0.002, 0.02))     # unchanged for 3.5 s: frozen
    assert check.update(3.6, (0.0011, 0.002, 0.02))        # moves again: usable
    assert not check.update(7.0, (0.0011 + 1e-17, 0.002, 0.02))  # rounding noise only
    assert not check.update(3.7, (0.0018, 0.0012, 0.0035))  # 3.5 mm from the camera


def test_bag_parts_reads_split_files_in_order_without_metadata(tmp_path):
    for name, size in (('cut_1_10.mcap', 5), ('cut_1_2.mcap', 5), ('cut_1_0.mcap', 5),
                       ('cut_1_11.mcap', 0), ('notes.txt', 5)):
        (tmp_path / name).write_bytes(b'x' * size)
    parts = bag_parts(str(tmp_path))
    names = [p.rsplit('/', 1)[1] for p, _ in parts]
    assert names == ['cut_1_0.mcap', 'cut_1_2.mcap', 'cut_1_10.mcap']
    assert {s for _, s in parts} == {'mcap'}
    (tmp_path / 'metadata.yaml').write_text('')                    # failed copy
    assert bag_parts(str(tmp_path)) == parts
    (tmp_path / 'metadata.yaml').write_text('rosbag2_bagfile_information: {}')
    assert bag_parts(str(tmp_path)) == [(str(tmp_path), 'mcap')]


def test_shipped_model_loads_and_scores_a_frame():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'cut_depth_combined',
                        'model.json')
    model = ContactModel(path)
    assert model.kind == 'combined'
    assert model.meta['thresholds'] == {'suspect': 0.6, 'confirm': 0.8}
    image = tissue()
    cv2.circle(image, (270, 270), 8, (90, 220, 60), -1)
    spot, _ = measure_spot(image, model.spot_params, (270.0, 270.0))
    feats, center = SpotHistory(**model.window).update(0.0, spot, (270.0, 270.0))
    p = model.predict(image, feats, center, 0.0)
    assert 0.0 <= p <= 1.0
