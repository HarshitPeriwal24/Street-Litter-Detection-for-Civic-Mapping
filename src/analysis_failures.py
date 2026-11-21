"""Fixed-threshold precision/recall, per-class AP, localisation-vs-classification and size breakdown, failure images. Saved artifacts only."""
import os, sys, pickle, argparse
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0'); os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np, pandas as pd, cv2, torch, torchvision
import tensorflow as tf
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument('--artifacts', required=True); ap.add_argument('--data', default=os.path.join(REPO, 'data'))
args = ap.parse_args(); ART = find_root(args.artifacts).replace('\\', '/'); DATA = args.data.replace('\\', '/')
print('SEEDS: split=%d, training(original runs)=%d (no randomness in this script)' % (SEED_SPLIT, SEED_TRAIN))
paths = test_images(f'{DATA}/taco_yolo'); assert len(paths) == 113
gt22 = load_gt(paths); gt6 = [np.c_[SUP22[g[:, 0].astype(int)], g[:, 1:]] if len(g) else g for g in gt22]
ALLC = np.concatenate([g[:, 0] for g in gt6])
sizes = [Image.open(p).size for p in paths]
inc = tf.keras.models.load_model(f'{ART}/inception/inception_best.keras')
THRS, BINS = (0.25, 0.50, 0.75), (0.01, 0.05)       # fixed in advance

def to6(d): return np.c_[d[:, :5], SUP22[d[:, 5].astype(int)]] if len(d) else np.zeros((0, 6))
def agn_nms(d):
    if not len(d): return d
    return d[torchvision.ops.nms(torch.tensor(d[:, :4], dtype=torch.float32), torch.tensor(d[:, 4], dtype=torch.float32), 0.6).numpy()]
def hybrid(det):
    out = []
    for p, d in zip(paths, det):
        d = agn_nms(d)
        if not len(d): out.append(np.zeros((0, 6))); continue
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); H, W = img.shape[:2]; crops = []
        for x1, y1, x2, y2 in d[:, :4]:
            x1 = int(np.clip(x1, 0, W - 1)); y1 = int(np.clip(y1, 0, H - 1)); x2 = int(np.clip(x2, x1 + 1, W)); y2 = int(np.clip(y2, y1 + 1, H))
            crops.append(cv2.resize(img[y1:y2, x1:x2], (CLS_IMG, CLS_IMG)))
        prob = inc.predict(np.array(crops, dtype='float32'), batch_size=256, verbose=0)
        out.append(np.c_[d[:, :5], prob.argmax(1)])
    return out

def prf(preds, gts, t):
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        p = p[p[:, 4] >= t] if len(p) else p; used = np.zeros(len(g), bool)
        if len(p):
            p = p[np.argsort(-p[:, 4])]; iou = box_iou(p[:, :4], g[:, 1:]) if len(g) else None
            for i in range(len(p)):
                if iou is None: fp += 1; continue
                cand = np.where((~used) & (g[:, 0] == p[i, 5]), iou[i], -1); j = cand.argmax()
                if cand[j] >= 0.5: used[j] = True; tp += 1
                else: fp += 1
        fn += len(g) - int(used.sum())
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), tp, fp, fn

def loc_cls(preds, t=0.25):
    """per GT object: (class, relative area, located by any pred IoU>=.5, located by a same-class pred)."""
    rows = []
    for k, (p, g) in enumerate(zip(preds, gt6)):
        p = p[p[:, 4] >= t] if len(p) else p
        for j in range(len(g)):
            area = (g[j, 3] - g[j, 1]) * (g[j, 4] - g[j, 2]) / (sizes[k][0] * sizes[k][1])
            loc = cor = False
            if len(p):
                m = box_iou(g[j:j+1, 1:], p[:, :4])[0] >= 0.5
                loc = bool(m.any()); cor = bool((m & (p[:, 5] == g[j, 0])).any())
            rows.append((int(g[j, 0]), area, loc, cor))
    return rows

tab_t, tab_c, tab_l, tab_s = [], [], [], []
hyb_all = {}
for key, eid in EXPS.items():
    det = pickle.load(open(f'{ART}/exp{eid}_{key}/test_detections.pkl', 'rb'))
    systems = {'baseline': [to6(d) for d in det], 'hybrid': hybrid(det)}
    hyb_all[key] = systems['hybrid']
    for name, preds in systems.items():
        for t in THRS:
            P, R, tp, fp, fn = prf(preds, gt6, t); tab_t.append(dict(detector=key, system=name, conf=t, precision=P, recall=R, TP=tp, FP=fp, FN=fn))
        m = compute_metrics(preds, gt6, 6)
        for c in range(6): tab_c.append(dict(detector=key, system=name, cls=SUPER[c], n_test_objects=int((ALLC == c).sum()), AP50=m['per_class_AP50'].get(c, np.nan)))
        rows = loc_cls(preds); n = len(rows)
        tab_l.append(dict(detector=key, system=name, objects=n, located_pct=100 * sum(r[2] for r in rows) / n, located_and_correct_class_pct=100 * sum(r[3] for r in rows) / n))
        for lab, lo, hi in (('<1% of image', 0, BINS[0]), ('1-5%', BINS[0], BINS[1]), ('>5%', BINS[1], 9)):
            sel = [r for r in rows if lo <= r[1] < hi]
            tab_s.append(dict(detector=key, system=name, size=lab, objects=len(sel), recall_class_correct_pct=100 * sum(r[3] for r in sel) / max(len(sel), 1)))
out = os.path.join(REPO, 'results', 'tables')
for nm, tb in (('thresholds', tab_t), ('per_class_ap', tab_c), ('localisation_vs_classification', tab_l), ('recall_by_size', tab_s)):
    df = pd.DataFrame(tb).round(4); df.to_csv(f'{out}/{nm}.csv', index=False); print(f'\n== {nm} ==\n' + df.to_string(index=False))

# failure images: yolov5l + Inception, conf>=0.25, the 6 images with most GT objects not found with the right class
preds = hyb_all['yolov5l']; miss = []
for k, (p, g) in enumerate(zip(preds, gt6)):
    p = p[p[:, 4] >= 0.25] if len(p) else p; n_miss = 0
    for j in range(len(g)):
        ok = len(p) > 0 and bool(((box_iou(g[j:j+1, 1:], p[:, :4])[0] >= 0.5) & (p[:, 5] == g[j, 0])).any()); n_miss += (not ok)
    miss.append((n_miss, k))
fd = os.path.join(REPO, 'results', 'figures', 'failure_cases'); os.makedirs(fd, exist_ok=True)
for n_miss, k in sorted(miss, key=lambda x: (-x[0], x[1]))[:6]:
    im = cv2.imread(paths[k]); p = preds[k]; p = p[p[:, 4] >= 0.25] if len(p) else p
    for c, x1, y1, x2, y2 in gt6[k]: cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4); cv2.putText(im, 'GT ' + SUPER[int(c)], (int(x1), max(int(y1) - 8, 14)), 0, 1.2, (0, 200, 0), 3)
    for x1, y1, x2, y2, s, c in p: cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3); cv2.putText(im, f'{SUPER[int(c)]} {s:.2f}', (int(x1), int(y2) + 28), 0, 1.0, (0, 0, 255), 3)
    s = 800 / max(im.shape[:2]); cv2.imwrite(f'{fd}/{os.path.basename(paths[k])}', cv2.resize(im, None, fx=s, fy=s), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(f'failure image {os.path.basename(paths[k])}: {n_miss} GT objects missed or wrong class (green = ground truth, red = prediction)')