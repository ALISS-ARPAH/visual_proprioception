r"""
Train and evaluate laser contact models on a table from build_contact_dataset.

Usage:
    # spot-feature model (numpy only)
    ros2 run visual_proprioception train_contact_model \
        --frames ~/aliss_runs/contact/frames.csv --out-dir ~/aliss_runs/contact/run1

    # image and combined models need PyTorch + torchvision
    ~/aliss_runs/contact_venv/bin/python -m visual_proprioception.contact_train \
        --frames ~/aliss_runs/contact/frames.csv --out-dir ~/aliss_runs/contact/run1 \
        --models spot image combined

Models:
    spot      logistic regression on the SpotHistory features
    image     EfficientNet-B0 on a fixed-size crop around the spot
    combined  EfficientNet-B0 crop features joined with the spot features
    temporal  combined, plus the change in the crop's CNN features since the
              frame about --lag s earlier (the same network embeds both frames)

--target depth (needs a table from build_contact_dataset --labels depth) trains
every model to predict how many mm the fiber is short of the cutting depth
(depth_to_cut_mm, clipped to [-1, --depth-clip]) instead of a probability, so the
cutting range is a runtime setting (--suspect-mm / --confirm-mm) rather than a
training choice. All models are also scored per cutoff c: AUC for "within c mm"
vs "c + 1 mm or more short", for c in 0.5 / 1 / 1.5 / 2 mm.

Evaluation is grouped cross-validation. --group-by auto holds out whole
sessions (recording folders, e.g. one bladder each) when there are at least
--min-groups, else whole lesions, else whole approaches (optimistic: approaches
in one lesion share tissue and lighting). Every model is reported next to the
current rule (legacy spot area <= 1000 px^2), raw and after a causal mean over
the last --smooth s:

    AUC pre-fire  contact in the last frames before laser-on vs approaching
    AUC cutting   first second of cutting vs approaching
    false alarm   share of approaches with any approaching frame >= threshold
    before fire   share detected before laser-on, out of the approaches that
                  have a frame in the pre-fire window (others cannot count)
    by +1 s       share detected no later than 1 s after laser-on
    latency       median time of first detection relative to laser-on

After evaluation each model is retrained on all labelled frames and exported to
<out-dir>/<model>/model.json (+ model.onnx) for laser_contact_model_node.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from .contact_model import ContactModel
from .laser_spot_features import (crop_square, crop_to_tensor, FEATURE_NAMES,
                                  LEGACY_AREA_THRESHOLD_PX2, SPOT_PARAMS, WINDOW_PARAMS)

STR_COLS = {'session', 'lesion', 'cut', 'approach', 'sequence', 'cut_segment', 'part', 'image',
            'kind', 'crop_source', 'laser_kind'}
MODELS = ('spot', 'image', 'combined', 'temporal')


def load_frames(path):
    """Return ({column: array}, dataset settings) for a build_contact_dataset CSV."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f'{path} has no rows')
    data = {}
    for col in rows[0]:
        values = [r[col] for r in rows]
        if col in STR_COLS:
            data[col] = np.array(values)
        else:
            data[col] = np.array([float(v) if v != '' else math.nan for v in values])
    data['label'] = data['label'].astype(int)
    meta = {}
    if os.path.exists(path + '.json'):
        with open(path + '.json') as f:
            meta = json.load(f)
    if meta.get('feature_names', list(FEATURE_NAMES)) != list(FEATURE_NAMES):
        sys.exit(f'{path} was built with different features; rebuild it with this code.')
    return data, meta


def make_folds(groups, n_folds, seed):
    uniq = np.array(sorted(set(groups)))
    np.random.default_rng(seed).shuffle(uniq)
    k = min(n_folds, len(uniq))
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups]), k


def _rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def auc(score, pos, neg):
    """Area under the ROC curve (Mann-Whitney), NaN when a class is empty."""
    s1, s0 = score[pos], score[neg]
    if not len(s1) or not len(s0):
        return math.nan
    ranks = _rankdata(np.concatenate([s1, s0]))
    return float((ranks[:len(s1)].sum() - len(s1) * (len(s1) + 1) / 2) / (len(s1) * len(s0)))


def approach_metrics(prob, data, threshold, labels):
    """Per-approach alarm behaviour at one threshold, as deployment would see it."""
    false_alarm, latency, measurable = [], [], []
    for seq in sorted(set(data['sequence'])):
        idx = np.flatnonzero(data['sequence'] == seq)
        idx = idx[np.argsort(data['t'][idx])]
        p, rel, kind = prob[idx], data['t_rel'][idx], data['kind'][idx]
        false_alarm.append(bool(np.any((kind == 'approaching') & (p >= threshold))))
        # A detection counts once the fiber is labelled at contact / cutting depth, or after
        # laser-on (laser labels: the last pos_pre s before firing; depth labels: within
        # depth_pos_mm of the cutting depth, however early).
        hits = np.flatnonzero(((kind == 'pre-fire') | (rel >= 0)) & (p >= threshold))
        latency.append(float(rel[hits[0]]) if len(hits) else math.nan)
        measurable.append(bool(np.any((kind == 'pre-fire') & (rel < 0))))
    latency = np.array(latency)
    measurable = np.array(measurable)
    found = np.isfinite(latency)
    return {
        'approaches': len(latency),
        'false_alarm': float(np.mean(false_alarm)),
        'before_fire': (float(np.mean((found & (latency < 0))[measurable]))
                        if measurable.any() else math.nan),
        'before_fire_measurable': int(measurable.sum()),
        'by_1s': float(np.mean(found & (latency <= 1.0))),
        'median_latency_s': float(np.median(latency[found])) if found.any() else math.nan,
    }


DEPTH_CUTOFFS_MM = (0.5, 1.0, 1.5, 2.0)


def depth_aucs(score, data, cutoffs=DEPTH_CUTOFFS_MM, lateral_mm=2.0, margin_mm=1.0):
    """AUC of score for 'within c mm of the cutting depth' vs 'c + margin mm or more short'."""
    dep = data['depth_to_cut_mm']
    before = (data['t_rel'] < 0) & (data['laser'] != 1) & np.isfinite(dep)
    near = before & (data['lateral_mm'] <= lateral_mm)
    return {f'{c:g}mm': auc(score, near & (dep <= c), before & (dep >= c + margin_mm))
            for c in cutoffs}


def summarize(prob, data, thresholds, labels):
    kind = data['kind']
    neg = kind == 'approaching'
    extra = {}
    if 'depth_to_cut_mm' in data:
        extra['auc_depth'] = depth_aucs(prob, data)
    if 'laser_kind' in data:   # the same frames scored with the laser-timing labels
        lk = data['laser_kind']
        extra['auc_prefire_laser'] = auc(prob, lk == 'pre-fire', lk == 'approaching')
    return {
        'auc_prefire': auc(prob, kind == 'pre-fire', neg),
        'auc_cutting': auc(prob, kind == 'cutting', neg),
        'auc_contact': auc(prob, data['label'] == 1, neg),
        **extra,
        **{name: approach_metrics(prob, data, thr, labels) for name, thr in thresholds.items()},
    }


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def class_weights(kind, prefire_share=0.0):
    """
    Per-frame loss weights (mean 1): contact and no-contact weigh the same in total.

    With prefire_share > 0, pre-fire frames get that share of the contact weight and
    cutting frames the rest; otherwise every contact frame weighs the same.
    """
    w = np.zeros(len(kind))
    neg, pre, cut = kind == 'approaching', kind == 'pre-fire', kind == 'cutting'
    w[neg] = 0.5 / max(neg.sum(), 1)
    if prefire_share > 0 and pre.any() and cut.any():
        w[pre] = 0.5 * prefire_share / pre.sum()
        w[cut] = 0.5 * (1 - prefire_share) / cut.sum()
    else:
        w[pre | cut] = 0.5 / max((pre | cut).sum(), 1)
    return w * len(kind)


def fit_logistic(x, y, l2=1.0, iters=100, weight=None):
    """L2-regularised logistic regression (Newton), class-balanced unless weight is given."""
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-6] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    if weight is None:
        n_pos = max(y.sum(), 1)
        n_neg = max(len(y) - y.sum(), 1)
        weight = np.where(y == 1, len(y) / (2.0 * n_pos), len(y) / (2.0 * n_neg))
    reg = np.full(z.shape[1], float(l2))
    reg[-1] = 0.0
    beta = np.zeros(z.shape[1])
    for _ in range(iters):
        p = _sigmoid(z @ beta)
        grad = z.T @ (weight * (p - y)) + reg * beta
        hess = (z * (weight * p * (1 - p))[:, None]).T @ z + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return mean, std, beta[:-1], float(beta[-1])


def predict_linear(params, x):
    mean, std, w, b = params
    return ((x - mean) / std) @ w + b


def predict_logistic(params, x):
    return _sigmoid(predict_linear(params, x))


def fit_ridge(x, y, l2=1.0):
    """L2-regularised least squares on standardised features; return (mean, std, w, b)."""
    mean = x.mean(0)
    std = x.std(0)
    std[std < 1e-6] = 1.0
    z = np.hstack([(x - mean) / std, np.ones((len(x), 1))])
    reg = np.full(z.shape[1], float(l2))
    reg[-1] = 0.0
    beta = np.linalg.solve(z.T @ z + np.diag(reg + 1e-9), z.T @ y)
    return mean, std, beta[:-1], float(beta[-1])


def causal_mean(prob, data, window):
    """Mean of each frame's probability over the last `window` s of its approach."""
    out = prob.copy()
    if window <= 0:
        return out
    for seq in set(data['sequence']):
        idx = np.flatnonzero(data['sequence'] == seq)
        idx = idx[np.argsort(data['t'][idx], kind='stable')]
        t, p = data['t'][idx], prob[idx]
        start = np.searchsorted(t, t - window, side='right')
        csum = np.concatenate([[0.0], np.cumsum(p)])
        out[idx] = (csum[np.arange(1, len(idx) + 1)] - csum[start]) / (np.arange(1, len(idx) + 1)
                                                                       - start)
    return out


def past_index(data, lag):
    """
    For every frame, the earlier frame of its approach that the temporal model compares to.

    Picks the frame closest to t - lag['target'] within [t - lag['max'], t - lag['min']]
    (the same rule ContactModel applies to its live buffer); the frame itself if none.
    """
    past = np.arange(len(data['t']))
    for seq in set(data['sequence']):
        idx = np.flatnonzero(data['sequence'] == seq)
        idx = idx[np.argsort(data['t'][idx], kind='stable')]
        t = data['t'][idx]
        for k, ti in enumerate(t):
            ok = np.flatnonzero((t[:k] >= ti - lag['max']) & (t[:k] <= ti - lag['min']))
            if len(ok):
                past[idx[k]] = idx[ok[np.argmin(np.abs(t[ok] - (ti - lag['target'])))]]
    return past


# --------------------------------------------------------------------------- image models

def _import_torch():
    try:
        import torch
        import torchvision
    except ImportError:
        sys.exit('The image and combined models need PyTorch + torchvision. Run with\n'
                 '  ~/aliss_runs/contact_venv/bin/python -m visual_proprioception.contact_train')
    return torch, torchvision


def load_crops(data, size, cache=None):
    """
    Return uint8 BGR crops (n, size, size, 3) around each frame's crop centre.

    With cache (a .npy path) the crops are written to disk once and memory-mapped,
    so tables larger than RAM work and later runs skip the image reads. The cache is
    rebuilt when the frame table's image list or crop centres change.
    """
    n = len(data['image'])
    digest = hashlib.sha1('\n'.join(data['image']).encode())
    digest.update(np.round(np.nan_to_num(np.stack([data['crop_u'], data['crop_v']])), 3).tobytes())
    key = f'{n}:{size}:{digest.hexdigest()}'
    if cache and os.path.exists(cache) and os.path.exists(cache + '.key'):
        with open(cache + '.key') as f:
            if f.read() == key:
                return np.load(cache, mmap_mode='r')
    if cache:
        out = np.lib.format.open_memmap(cache, mode='w+', dtype=np.uint8, shape=(n, size, size, 3))
    else:
        out = np.empty((n, size, size, 3), np.uint8)

    def one(i):
        u, v = data['crop_u'][i], data['crop_v'][i]
        center = None if math.isnan(u) else (u, v)
        image = cv2.imread(data['image'][i])
        if image is None:
            raise FileNotFoundError(data['image'][i])
        out[i] = crop_square(image, center, size)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, range(n)))
    if not cache:
        return out
    out.flush()
    with open(cache + '.key', 'w') as f:
        f.write(key)
    return np.load(cache, mmap_mode='r')


def make_batch(crops, idx, cfg, rng=None, flips=None):
    """
    Cut crop_size windows out of the padded crops (random shift + flip if rng given).

    flips (one bool per sample) overrides the random flip, so a frame and its
    past frame can be flipped together.
    """
    j, s = cfg.jitter, cfg.crop_size
    out = np.empty((len(idx), 3, cfg.input_size, cfg.input_size), np.float32)
    if flips is None:
        flips = rng.random(len(idx)) < 0.5 if rng is not None else np.zeros(len(idx), bool)
    for k, i in enumerate(idx):
        dx, dy = rng.integers(-j, j + 1, 2) if rng is not None else (0, 0)
        crop = crops[i][j + dy:j + dy + s, j + dx:j + dx + s]
        if flips[k]:
            crop = crop[:, ::-1]
        out[k] = crop_to_tensor(np.ascontiguousarray(crop), cfg.input_size)[0]
    return out


EMBED_DIM = 32
SPOT_DIM = 16


def build_net(torch, torchvision, n_feat, mean, std, use_spot, temporal, pretrained):
    nn = torch.nn

    class ContactNet(nn.Module):
        """CNN crop embedding [+ its change since a past frame] [+ spot features] -> logit."""

        def __init__(self):
            super().__init__()
            weights = 'IMAGENET1K_V1' if pretrained else None
            base = torchvision.models.efficientnet_b0(weights=weights)
            self.backbone = nn.Sequential(base.features, base.avgpool, nn.Flatten())
            self.drop = nn.Dropout(0.3)
            self.proj = nn.Sequential(nn.Linear(1280, EMBED_DIM), nn.ReLU())
            self.use_spot = use_spot
            self.temporal = temporal
            self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
            self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
            if use_spot:
                self.spot_head = nn.Sequential(nn.Linear(n_feat, SPOT_DIM), nn.ReLU())
            n_in = EMBED_DIM * (2 if temporal else 1) + (SPOT_DIM if use_spot else 0)
            self.out = nn.Linear(n_in, 1)

        def embed(self, image):
            return self.proj(self.drop(self.backbone(image)))

        def head(self, emb, emb_past, spot):
            h = [emb]
            if self.temporal:
                h.append(emb - emb_past)
            if self.use_spot:
                h.append(self.spot_head((spot - self.mean) / self.std))
            return self.out(torch.cat(h, 1)).squeeze(1)

        def forward(self, image, spot, past=None):
            if not self.temporal:
                return self.head(self.embed(image), None, spot)
            # One backbone pass for both frames; one dropout mask for both, so the
            # difference reflects the images and not the dropout noise.
            f = self.backbone(torch.cat([image, past], 0))
            n = len(image)
            if self.training:
                keep = (torch.rand_like(f[:n]) >= self.drop.p).float() / (1 - self.drop.p)
                f = f * torch.cat([keep, keep], 0)
            e = self.proj(f)
            return self.head(e[:n], e[n:], spot)

    return ContactNet()


def _arrays(crops, idx, past, cfg, rng=None):
    """Input arrays for frames idx (and their past frames if past is given)."""
    flips = rng.random(len(idx)) < 0.5 if rng is not None else np.zeros(len(idx), bool)
    image = make_batch(crops, idx, cfg, rng, flips)
    return image, (None if past is None else make_batch(crops, past[idx], cfg, rng, flips))


def _prefetched(pool, fn, jobs, depth=3):
    """Yield fn(*job) for each job in order, keeping up to depth jobs running in pool."""
    pending = []
    for job in jobs:
        pending.append(pool.submit(fn, *job))
        if len(pending) > depth:
            yield pending.pop(0).result()
    for future in pending:
        yield future.result()


def train_net(torch, torchvision, crops, x, y, weight, train_idx, model, cfg, device, seed,
              past=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = x[train_idx].mean(0)
    std = x[train_idx].std(0)
    std[std < 1e-6] = 1.0
    net = build_net(torch, torchvision, x.shape[1], mean, std, model != 'image',
                    model == 'temporal', cfg.pretrained).to(device)
    backbone = list(net.backbone.parameters())
    head = [p for n, p in net.named_parameters() if not n.startswith('backbone.')]
    opt = torch.optim.AdamW([{'params': backbone, 'lr': cfg.lr_backbone},
                             {'params': head, 'lr': cfg.lr_head}], weight_decay=1e-4)
    if cfg.target == 'depth':   # mm short of the cutting depth; Huber is robust to pose noise
        def loss_fn(out, target):
            return torch.nn.functional.smooth_l1_loss(out, target, beta=0.5, reduction='none')
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
    bs = min(cfg.batch_size, len(train_idx))
    per_epoch = len(train_idx) if cfg.epoch_frames <= 0 else min(cfg.epoch_frames, len(train_idx))
    steps = cfg.epochs * (per_epoch // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lr_backbone, cfg.lr_head], total_steps=max(steps, 1))
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    wt = torch.from_numpy(weight.astype(np.float32)).to(device)
    amp = cfg.amp and str(device).startswith('cuda')
    with ThreadPoolExecutor(3) as pool:   # prepare upcoming batches while the GPU works
        for _ in range(cfg.epochs):
            net.train()
            perm = rng.permutation(train_idx)[:per_epoch]
            batches = [perm[s:s + bs] for s in range(0, len(perm) - bs + 1, bs)]
            # one generator per batch, so worker threads never share one
            jobs = [(crops, b, past, cfg, np.random.default_rng(rng.integers(2 ** 63)))
                    for b in batches]
            for b, arrays in zip(batches, _prefetched(pool, _arrays, jobs)):
                image, past_image = (None if a is None else torch.from_numpy(a).to(device)
                                     for a in arrays)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    logits = net(image, xt[b], past_image)
                loss = (loss_fn(logits.float(), yt[b]) * wt[b]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
    return net


def predict_net(torch, net, crops, x, idx, cfg, device, past=None):
    net.eval()
    out = []
    batches = [idx[s:s + 128] for s in range(0, len(idx), 128)]
    with torch.no_grad(), ThreadPoolExecutor(3) as pool:
        jobs = [(crops, b, past, cfg) for b in batches]
        for b, arrays in zip(batches, _prefetched(pool, _arrays, jobs)):
            image, past_image = (None if a is None else torch.from_numpy(a).to(device)
                                 for a in arrays)
            spot = torch.from_numpy(x[b]).to(device)
            raw = net(image, spot, past_image)
            out.append((raw if cfg.target == 'depth' else torch.sigmoid(raw)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def export_temporal(torch, net, cfg, path):
    """
    Write the embedding network to ONNX; return the (tiny) head as numpy-able lists.

    At runtime ContactModel embeds each frame once and keeps the embeddings of
    recent frames, so the backbone runs once per frame as for the other models.
    """

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, image):
            return self.net.embed(image)

    model = Embed().cpu().eval()
    torch.onnx.export(model, (torch.zeros(1, 3, cfg.input_size, cfg.input_size),), path,
                      input_names=['image'], output_names=['embedding'],
                      opset_version=12, dynamo=False)
    with torch.no_grad():
        return {
            'spot_mean': net.mean.tolist(), 'spot_std': net.std.tolist(),
            'spot_weight': net.spot_head[0].weight.tolist(),
            'spot_bias': net.spot_head[0].bias.tolist(),
            'out_weight': net.out.weight[0].tolist(), 'out_bias': float(net.out.bias[0]),
        }


def export_onnx(torch, net, use_spot, n_feat, cfg, path):
    nn = torch.nn

    class Exported(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, image, spot=None):
            if spot is None:
                spot = torch.zeros(image.shape[0], n_feat, device=image.device)
            raw = self.net(image, spot)
            return (raw if cfg.target == 'depth' else torch.sigmoid(raw)).reshape(-1, 1)

    model = Exported().cpu().eval()
    image = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
    args = (image, torch.zeros(1, n_feat)) if use_spot else (image,)
    names = ['image', 'spot'] if use_spot else ['image']
    torch.onnx.export(model, args, path, input_names=names, output_names=['prob'],
                      opset_version=12, dynamo=False)


# --------------------------------------------------------------------------- main

def model_meta(kind, cfg, dataset_meta):
    return {
        'kind': kind, 'feature_names': list(FEATURE_NAMES), 'target': cfg.target,
        'thresholds': ({'suspect': cfg.suspect_mm, 'confirm': cfg.confirm_mm}
                       if cfg.target == 'depth'
                       else {'suspect': cfg.suspect, 'confirm': cfg.confirm}),
        'depth_clip_mm': cfg.depth_clip if cfg.target == 'depth' else None,
        'spot_params': dataset_meta.get('spot_params', SPOT_PARAMS),
        'crop_center': dataset_meta.get('crop_center', 'spot, else last spot'),
        'window': dataset_meta.get('window', WINDOW_PARAMS),
        'labels': dataset_meta.get('labels'),
        'smooth_sec': cfg.smooth,
        'training': {'epochs': cfg.epochs, 'epoch_frames': cfg.epoch_frames,
                     'prefire_share': cfg.prefire_share, 'seed': cfg.seed},
        'trained_on': {'frames': os.path.abspath(cfg.frames),
                       'sessions': dataset_meta.get('sessions'),
                       'lesions': dataset_meta.get('lesions'),
                       'created': time.strftime('%Y-%m-%dT%H:%M:%S')},
    }


def check_export(model_json, data, x, idx, reference, past=None):
    """
    Score frames through the exported files (as the ROS node does); return max |diff|.

    For the temporal model the past frame is fed first, so the runtime buffer
    holds what it would hold live.
    """
    runtime = ContactModel(model_json)

    def score(i):
        feats = dict(zip(FEATURE_NAMES, x[i].tolist()))
        center = None if math.isnan(data['crop_u'][i]) else (data['crop_u'][i], data['crop_v'][i])
        image = cv2.imread(data['image'][i]) if runtime.kind != 'spot' else None
        return runtime.predict(image, feats, center, t=float(data['t'][i]))

    diffs = []
    for i, ref in zip(idx, reference):
        runtime.reset()
        if past is not None and past[i] != i:
            score(past[i])
        diffs.append(abs(score(i) - ref))
    return float(max(diffs))


def write_predictions(path, data, prob):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence', 't', 't_rel', 'kind', 'label', 'prob'])
        for row in zip(data['sequence'], data['t'], data['t_rel'], data['kind'],
                       data['label'], prob):
            w.writerow(row)


def print_table(results, thresholds, unit=''):
    name = 'confirm' if 'confirm' in thresholds else next(iter(thresholds))
    first = next(iter(results.values()))[name]
    print(f'\nper-approach columns use the {name} threshold ({thresholds[name]}{unit}); '
          f'before fire is out of the {first["before_fire_measurable"]} of '
          f'{first["approaches"]} approaches with a pre-fire frame')
    print(f'{"model":20s} {"AUC pre-fire":>12s} {"AUC cutting":>11s} {"false alarm":>11s} '
          f'{"before fire":>11s} {"by +1 s":>8s} {"latency":>8s}')
    for model, m in results.items():
        a = m[name]
        latency = a['median_latency_s']
        latency = f'{latency:+7.2f}s' if math.isfinite(latency) else '     n/a'
        print(f'{model:20s} {m["auc_prefire"]:12.2f} {m["auc_cutting"]:11.2f} '
              f'{a["false_alarm"]:11.0%} {a["before_fire"]:11.0%} {a["by_1s"]:8.0%} {latency}')
    if 'auc_depth' in next(iter(results.values())):
        cuts = list(next(iter(results.values()))['auc_depth'])
        print('\nAUC by cutting-range cutoff c (within c mm vs c + 1 mm or more short)')
        print(f'{"model":20s} ' + ' '.join(f'{c:>7s}' for c in cuts) + '   MAE (near) mm')
        for model, m in results.items():
            mae = (f'   {m["mae_mm"]:.2f} ({m["mae_near_mm"]:.2f})' if 'mae_mm' in m else '')
            print(f'{model:20s} ' + ' '.join(f'{m["auc_depth"][c]:7.2f}' for c in cuts) + mae)


def plot_timelines(path, data, probs, thresholds, depth_clip=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, len(probs), figsize=(4.2 * len(probs), 3.6), squeeze=False)
    colors = {'approaching': 'tab:blue', 'excluded': '0.7', 'pre-fire': 'tab:red',
              'cutting': 'tab:orange'}
    for ax, (model, prob) in zip(axes[0], probs.items()):
        for kind, color in colors.items():
            sel = data['kind'] == kind
            ax.scatter(data['t_rel'][sel], prob[sel], s=6, color=color, alpha=0.6, label=kind)
        for thr in thresholds.values():
            ax.axhline(thr, color='k', lw=0.6, ls=':')
        ax.axvline(0, color='k', lw=0.8)
        if depth_clip is None:
            ax.set(title=model, xlabel='time from laser-on (s)', xlim=(-4, 1.5),
                   ylim=(-0.03, 1.03))
        else:
            ax.set(title=model, xlabel='time from laser-on (s)', xlim=(-4, 1.5),
                   ylim=(-1.2, depth_clip + 0.2))
    axes[0][0].set_ylabel('P(contact), held-out' if depth_clip is None
                          else 'predicted mm short of cutting depth')
    axes[0][0].legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                     epilog=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--frames', required=True, help='CSV from build_contact_dataset')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=['spot'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--group-by', choices=('auto', 'session', 'lesion', 'approach'),
                        default='auto')
    parser.add_argument('--min-groups', type=int, default=3,
                        help='auto grouping holds out sessions (else lesions) once there '
                             'are this many')
    parser.add_argument('--smooth', type=float, default=0.3,
                        help='also report a causal mean of P over this many s (0: off); '
                             'stored in model.json for the node')
    parser.add_argument('--lag', type=float, nargs=3, default=(0.2, 0.4, 0.8),
                        metavar=('MIN', 'TARGET', 'MAX'),
                        help='temporal model: past frame closest to TARGET s back, '
                             'within [MIN, MAX] s')
    parser.add_argument('--target', choices=('label', 'depth'), default='label',
                        help='label: P(contact) from the labels; depth: mm short of the '
                             'cutting depth (table built with --labels depth)')
    parser.add_argument('--depth-clip', type=float, default=6.0,
                        help='depth target: clip to [-1, this] mm (farther is just "far")')
    parser.add_argument('--reg-lateral', type=float, default=3.0,
                        help='depth target: train on frames within this many mm sideways '
                             'of the firing position')
    parser.add_argument('--suspect-mm', type=float, default=2.0,
                        help='depth target: flag "close" at or below this many mm')
    parser.add_argument('--confirm-mm', type=float, default=1.0,
                        help='depth target: flag "at cutting depth" at or below this')
    parser.add_argument('--suspect', type=float, default=0.6, help='slow down / stop threshold')
    parser.add_argument('--confirm', type=float, default=0.8, help='contact confirmed threshold')
    parser.add_argument('--l2', type=float, default=1.0, help='spot model regularisation')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--epoch-frames', type=int, default=0,
                        help='CNN models: random training frames per epoch (0: all). '
                             'Consecutive 30 Hz frames are near-duplicates.')
    parser.add_argument('--prefire-share', type=float, default=0.0,
                        help='share of the contact loss weight given to pre-fire frames '
                             '(0: all contact frames weigh the same)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='train in float32 instead of bfloat16 mixed precision')
    parser.add_argument('--crop-cache', default='auto',
                        help="crop cache .npy ('auto': next to --frames, 'none': in memory)")
    parser.add_argument('--no-export', dest='export', action='store_false',
                        help='cross-validate only; skip the final fit and export')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--crop-size', type=int, default=192, help='crop side in image px')
    parser.add_argument('--input-size', type=int, default=224, help='CNN input side')
    parser.add_argument('--jitter', type=int, default=12, help='max random crop shift, px')
    parser.add_argument('--lr-backbone', type=float, default=1e-4)
    parser.add_argument('--lr-head', type=float, default=1e-3)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='random CNN init (no weight download; for testing only)')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=0)
    cfg = parser.parse_args(argv)
    cfg.frames = os.path.expanduser(cfg.frames)
    out_dir = os.path.expanduser(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    data, dataset_meta = load_frames(cfg.frames)
    labels = dataset_meta.get('labels', {'pos_pre': 0.15})
    thresholds = {'suspect': cfg.suspect, 'confirm': cfg.confirm}
    x = np.stack([data[n] for n in FEATURE_NAMES], 1).astype(np.float32)
    depth_target = cfg.target == 'depth'
    if depth_target:
        if 'depth_to_cut_mm' not in data:
            sys.exit('--target depth needs a frame table built with --labels depth')
        dep = data['depth_to_cut_mm']
        trainable = ((data['t_rel'] < 0) & (data['laser'] != 1) & np.isfinite(dep)
                     & (data['lateral_mm'] <= cfg.reg_lateral))
        y = np.where(trainable, np.clip(dep, -1.0, cfg.depth_clip), np.nan).astype(np.float32)
        labelled = np.flatnonzero(trainable)
        thresholds = {'suspect': cfg.suspect_mm, 'confirm': cfg.confirm_mm}
    else:
        y = data['label']
        labelled = np.flatnonzero(y >= 0)
    # evaluation scores: higher = closer to contact / cutting depth
    eval_thresholds = {k: -v for k, v in thresholds.items()} if depth_target else thresholds

    def score_of(pred):
        return -pred if depth_target else pred

    def weights_for(idx):
        w = np.zeros(len(y))
        w[idx] = 1.0 if depth_target else class_weights(data['kind'][idx], cfg.prefire_share)
        return w

    if 'session' not in data:
        data['session'] = np.full(len(y), '')
    n_sessions = len(set(data['session']) - {''})
    n_lesions = len(set(data['lesion']))
    group_by = cfg.group_by
    if group_by == 'auto':
        group_by = ('session' if n_sessions >= cfg.min_groups else
                    'lesion' if n_lesions >= cfg.min_groups else 'approach')
    groups = data[{'approach': 'sequence'}.get(group_by, group_by)]
    fold, k = make_folds(groups, cfg.folds, cfg.seed)
    print(f'{len(data["t"])} frames, {len(set(data["sequence"]))} approaches, '
          f'{n_lesions} lesion(s), {n_sessions} session(s); '
          f'{k}-fold CV holding out whole {group_by}s')
    if group_by == 'approach':
        print('  (approach-level CV within the same lesions is optimistic; '
              'hold out lesions once more are available)')
    lag = dict(zip(('min', 'target', 'max'), cfg.lag))
    past = past_index(data, lag) if 'temporal' in cfg.models else None

    legacy = data['legacy_area']
    legacy_prob = (np.isfinite(legacy) & (legacy <= LEGACY_AREA_THRESHOLD_PX2)).astype(float)
    probs = {} if depth_target else {'current rule': legacy_prob}
    results = {'current rule': summarize(legacy_prob, data, {'suspect': 0.5, 'confirm': 0.5},
                                         labels)}

    def evaluate(model, pred):
        probs[model] = pred
        results[model] = summarize(score_of(pred), data, eval_thresholds, labels)
        if depth_target:
            err = np.abs(pred[labelled] - y[labelled])
            near = y[labelled] <= 3.0
            results[model].update(mae_mm=float(np.mean(err)),
                                  mae_near_mm=float(np.mean(err[near])))
        if cfg.smooth > 0:
            smoothed = causal_mean(pred, data, cfg.smooth)
            results[f'{model} ~{cfg.smooth:g}s'] = summarize(score_of(smoothed), data,
                                                             eval_thresholds, labels)

    torch = torchvision = crops = device = None
    if any(m != 'spot' for m in cfg.models):
        torch, torchvision = _import_torch()
        device = cfg.device if cfg.device != 'auto' else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f'loading {len(data["image"])} crops; training on {device}')
        side = cfg.crop_size + 2 * cfg.jitter
        caches = {'none': None, 'auto': f'{cfg.frames}.crops{side}.npy'}
        cache = caches.get(cfg.crop_cache, cfg.crop_cache)
        crops = load_crops(data, side, cache)

    for model in cfg.models:
        started = time.time()
        oof = np.full(len(y), math.nan)
        for f in range(k):
            train = labelled[fold[labelled] != f]
            test = np.flatnonzero(fold == f)
            weight = weights_for(train)
            if model == 'spot' and depth_target:
                oof[test] = predict_linear(fit_ridge(x[train], y[train], cfg.l2), x[test])
            elif model == 'spot':
                params = fit_logistic(x[train], y[train], cfg.l2, weight=weight[train])
                oof[test] = predict_logistic(params, x[test])
            else:
                p = past if model == 'temporal' else None
                net = train_net(torch, torchvision, crops, x, y, weight, train, model, cfg,
                                device, cfg.seed + f, p)
                oof[test] = predict_net(torch, net, crops, x, test, cfg, device, p)
            print(f'  {model} fold {f + 1}/{k}: {time.time() - started:.0f} s', flush=True)
        evaluate(model, oof)

        model_dir = os.path.join(out_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        write_predictions(os.path.join(model_dir, 'oof_predictions.csv'), data, oof)
        if not cfg.export:
            print(f'{model}: cross-validated in {time.time() - started:.0f} s (no export)')
            continue
        meta = model_meta(model, cfg, dataset_meta)
        check = labelled[:: max(1, len(labelled) // 20)]
        if model == 'spot':
            if depth_target:
                params = fit_ridge(x[labelled], y[labelled], cfg.l2)
                reference = predict_linear(params, x[check])
            else:
                params = fit_logistic(x[labelled], y[labelled], cfg.l2,
                                      weight=weights_for(labelled)[labelled])
                reference = predict_logistic(params, x[check])
            mean, std, w, b = params
            meta.update(mean=mean.tolist(), std=std.tolist(), weights=w.tolist(), bias=b)
        else:
            p = past if model == 'temporal' else None
            net = train_net(torch, torchvision, crops, x, y, weights_for(labelled), labelled,
                            model, cfg, device, cfg.seed + 1000, p)
            reference = predict_net(torch, net, crops, x, check, cfg, device, p)
            meta.update(onnx='model.onnx', crop_size=cfg.crop_size, input_size=cfg.input_size)
            onnx_path = os.path.join(model_dir, 'model.onnx')
            if model == 'temporal':
                meta.update(inputs=['image'], outputs=['embedding'], lag=lag,
                            head=export_temporal(torch, net, cfg, onnx_path))
            else:
                export_onnx(torch, net, model == 'combined', x.shape[1], cfg, onnx_path)
                meta.update(inputs=['image', 'spot'] if model == 'combined' else ['image'])
        model_json = os.path.join(model_dir, 'model.json')
        with open(model_json, 'w') as fh:
            json.dump(meta, fh, indent=2)
        diff = check_export(model_json, data, x, check, reference,
                            past if model == 'temporal' else None)
        results[model]['export_max_abs_diff'] = diff
        print(f'{model}: done in {time.time() - started:.0f} s -> {model_json} '
              f'(runtime vs training output, max diff {diff:.1e})')

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as fh:
        json.dump({'group_by': group_by, 'folds': k, 'thresholds': thresholds,
                   'target': cfg.target,
                   'smooth_sec': cfg.smooth, 'lag': lag, 'seed': cfg.seed,
                   'prefire_share': cfg.prefire_share, 'epochs': cfg.epochs,
                   'epoch_frames': cfg.epoch_frames, 'results': results}, fh, indent=2)
    print_table(results, eval_thresholds if not depth_target else thresholds,
                ' mm' if depth_target else '')
    plot_timelines(os.path.join(out_dir, 'timelines.png'), data, probs, thresholds,
                   cfg.depth_clip if depth_target else None)
    print(f'\nmetrics, predictions and timelines.png in {out_dir}')


if __name__ == '__main__':
    main()
