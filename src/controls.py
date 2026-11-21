"""Negative controls promised in the proposal (Gate 1): oracle, shuffle, no-leakage, annotation audit (TACO test only)."""
import os, sys, json, pickle, glob, hashlib, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--artifacts', required=True); ap.add_argument('--data', default=os.path.join(REPO, 'data'))
args = ap.parse_args(); ART = find_root(args.artifacts).replace('\\', '/'); DATA = args.data.replace('\\', '/')
print(f'SEEDS: split={SEED_SPLIT}, audit sample seed=0')
paths = test_images(f'{DATA}/taco_yolo'); gt22 = load_gt(paths)
gt6 = [np.c_[SUP22[g[:, 0].astype(int)], g[:, 1:]] if len(g) else g for g in gt22]

def as_preds(gts): return [np.c_[g[:, 1:5], np.ones(len(g)), g[:, 0]] if len(g) else np.zeros((0, 6)) for g in gts]
o22, o6 = compute_metrics(as_preds(gt22), gt22, 22), compute_metrics(as_preds(gt6), gt6, 6)
ok_o = all(abs(m[k] - 1) < 1e-6 for m in (o22, o6) for k in ('P', 'R')) and all(m[k] >= 0.995 - 1e-6 for m in (o22, o6) for k in ('mAP50', 'mAP50_95'))  # 101-point AP is capped at 0.995 (same as Ultralytics)
print(f"ORACLE  22-class mAP50={o22['mAP50']:.4f} mAP50-95={o22['mAP50_95']:.4f} | 6-class mAP50={o6['mAP50']:.4f} P={o6['P']:.4f} R={o6['R']:.4f} -> {'PASS' if ok_o else 'FAIL'}")

ok_s = True
for key, eid in EXPS.items():
    det = pickle.load(open(f'{ART}/exp{eid}_{key}/test_detections.pkl', 'rb'))
    det = det[37:] + det[:37]
    m = compute_metrics([np.c_[d[:, :5], SUP22[d[:, 5].astype(int)]] if len(d) else d for d in det], gt6, 6)
    ok_s &= m['mAP50'] < 0.05
    print(f"SHUFFLE {key:8s} 6-class mAP50={m['mAP50']:.4f} (unshuffled is ~0.16-0.19)")
print('SHUFFLE ->', 'PASS' if ok_s else 'FAIL')

h = {}
for sp in ('train', 'val', 'test'):
    for f in glob.glob(f'{DATA}/taco_yolo/images/{sp}/*.jpg'):
        h.setdefault(hashlib.md5(open(f, 'rb').read()).hexdigest(), []).append((sp, os.path.basename(f)))
dups = [v for v in h.values() if len({sp for sp, _ in v}) > 1]; same = [v for v in h.values() if len(v) > 1 and len({sp for sp, _ in v}) == 1]
print(f'LEAKAGE exact-duplicate files shared between different splits: {len(dups)} -> {"PASS" if not dups else "FAIL"}; duplicates inside one split (harmless): {len(same)} {same} (near-duplicate scenes are NOT checked by hash)')

names = json.load(open(os.path.join(REPO, 'results', 'tables', 'split_and_counts.json')))['classes22']
out = os.path.join(REPO, 'results', 'figures', 'annotation_audit'); os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
for i in rng.choice(len(paths), 20, replace=False):
    im = cv2.imread(paths[i])
    for c, x1, y1, x2, y2 in gt22[i]:
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(im, names[str(int(c))], (int(x1), max(int(y1) - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    s = 640 / max(im.shape[:2]); im = cv2.resize(im, None, fx=s, fy=s)
    cv2.imwrite(f'{out}/{os.path.basename(paths[i])}', im, [cv2.IMWRITE_JPEG_QUALITY, 80])
print('AUDIT 20 annotated test images written to results/figures/annotation_audit')