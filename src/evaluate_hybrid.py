"""Hybrid (detector + InceptionV3) reproduction, ablation and bootstrap confidence intervals, all from saved artifacts."""
import os, sys, json, pickle, argparse
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np, pandas as pd, cv2, torch, torchvision
import tensorflow as tf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--artifacts', required=True)
ap.add_argument('--data', default=os.path.join(REPO, 'data'))
ap.add_argument('--boot', type=int, default=100, help='bootstrap resamples')
args = ap.parse_args()
ART = find_root(args.artifacts).replace('\\', '/'); DATA = args.data.replace('\\', '/')
SEED_BOOT = 0
print(f'SEEDS: split={SEED_SPLIT}, training(original runs)={SEED_TRAIN}, bootstrap={SEED_BOOT}, resamples={args.boot}', flush=True)

paths = test_images(f'{DATA}/taco_yolo'); assert len(paths) == 113, f'expected 113 test images, found {len(paths)}'
gt22 = load_gt(paths)
gt6 = [np.c_[SUP22[g[:, 0].astype(int)], g[:, 1:]] if len(g) else g for g in gt22]
inc = tf.keras.models.load_model(f'{ART}/inception/inception_best.keras')

def to6(d): return np.c_[d[:, :5], SUP22[d[:, 5].astype(int)]] if len(d) else np.zeros((0, 6))
def agnostic_nms(d):
    if not len(d): return d
    keep = torchvision.ops.nms(torch.tensor(d[:, :4], dtype=torch.float32), torch.tensor(d[:, 4], dtype=torch.float32), 0.6).numpy()
    return d[keep]

def hybrid(det):
    out_det, out_dc = [], []
    for p, d in zip(paths, det):
        d = agnostic_nms(d)
        if not len(d): out_det.append(np.zeros((0, 6))); out_dc.append(np.zeros((0, 6))); continue
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); Hh, Ww = img.shape[:2]; crops = []
        for x1, y1, x2, y2 in d[:, :4]:
            x1 = int(np.clip(x1, 0, Ww - 1)); y1 = int(np.clip(y1, 0, Hh - 1))
            x2 = int(np.clip(x2, x1 + 1, Ww)); y2 = int(np.clip(y2, y1 + 1, Hh))
            crops.append(cv2.resize(img[y1:y2, x1:x2], (CLS_IMG, CLS_IMG)))
        prob = inc.predict(np.array(crops, dtype='float32'), batch_size=256, verbose=0)
        out_det.append(np.c_[d[:, :4], d[:, 4], prob.argmax(1)])
        out_dc.append(np.c_[d[:, :4], d[:, 4] * prob.max(1), prob.argmax(1)])
    return out_det, out_dc

rows, boot_rows, worst = [], [], 0.0
rng = np.random.default_rng(SEED_BOOT)
for key, eid in EXPS.items():
    print(f'--- {key}', flush=True)
    det = pickle.load(open(f'{ART}/exp{eid}_{key}/test_detections.pkl', 'rb'))
    base = [to6(d) for d in det]
    nms_only = [to6(agnostic_nms(d)) for d in det]
    hyb, hyb_dc = hybrid(det)
    cfg = {'A baseline': base, 'B baseline + agnostic NMS (no Inception)': nms_only,
           'C hybrid, score=det (reported)': hyb, 'D hybrid, score=det*cls (ablation)': hyb_dc}
    for name, preds in cfg.items():
        m = compute_metrics(preds, gt6, 6)
        rows.append(dict(detector=key, config=name, P=m['P'], R=m['R'], mAP50=m['mAP50'], mAP50_95=m['mAP50_95']))
        print(f"  {name:45s} P={m['P']:.4f} R={m['R']:.4f} mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50_95']:.4f}", flush=True)
    saved = json.load(open(os.path.join(REPO, 'results', 'tables', f'exp{eid+4}_{key}_inception_metrics_test.json')))['final_6class']
    mC = compute_metrics(hyb, gt6, 6)
    d = max(abs(mC[k] - saved[k]) for k in ('mAP50', 'mAP50_95')); worst = max(worst, d); print('  per-metric diff vs saved:', {k: round(mC[k] - saved[k], 4) for k in ('P', 'R', 'mAP50', 'mAP50_95')})
    print(f'  hybrid (C) vs saved json: max diff {d:.2e}', flush=True)
    # paired bootstrap over test images
    res = {k: [] for k in 'ABC'}
    for b in range(args.boot):
        idx = rng.integers(0, len(paths), len(paths)); g = [gt6[i] for i in idx]
        for k, preds in zip('ABC', (base, nms_only, hyb)):
            res[k].append(compute_metrics([preds[i] for i in idx], g, 6)['mAP50'])
    A, B, C = (np.array(res[k]) for k in 'ABC')
    ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))
    for lab, x in (('A baseline', A), ('B baseline + agnostic NMS', B), ('C hybrid', C), ('B - A (effect of NMS alone)', B - A), ('C - B (effect of Inception)', C - B), ('C - A (total)', C - A)):
        lo, hi = ci(x); boot_rows.append(dict(detector=key, quantity=lab, mean=x.mean(), ci_low=lo, ci_high=hi))
        print(f'  bootstrap mAP50 {lab:30s} mean={x.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]', flush=True)

out = os.path.join(REPO, 'results', 'tables')
pd.DataFrame(rows).round(4).to_csv(f'{out}/ablation_hybrid.csv', index=False)
pd.DataFrame(boot_rows).round(4).to_csv(f'{out}/bootstrap_ci.csv', index=False)
print('hybrid vs saved json worst diff:', f'{worst:.2e}', '-> REPRODUCED' if worst < 5e-3 else '-> MISMATCH, investigate')