"""Shared constants and metric code.
load_gt, box_iou and compute_metrics are copied verbatim from the original Colab notebook
(notebooks/training_colab_original.ipynb) so that the numbers are computed by the same code."""
import os, json, glob, pickle
import numpy as np
from PIL import Image

SEED_SPLIT = 42          # train/val/test split seed
SEED_TRAIN = 0           # detector and classifier training seed (used in the Colab runs)
SPLIT = (0.8, 0.1, 0.1)
IMG, CLS_IMG = 416, 224

SUPER = ['cardboard', 'paper', 'plastic', 'metal', 'glass', 'trash']
TACO_SEL = [(4,'plastic'),(5,'plastic'),(29,'plastic'),(36,'plastic'),(39,'plastic'),
            (6,'glass'),(23,'glass'),(26,'glass'),
            (12,'metal'),(52,'metal'),(8,'metal'),(28,'metal'),
            (14,'cardboard'),(16,'cardboard'),(17,'cardboard'),(18,'cardboard'),
            (31,'paper'),(32,'paper'),(33,'paper'),(34,'paper'),
            (25,'trash'),(1,'trash')]
ID2IDX = {tid: i for i, (tid, _) in enumerate(TACO_SEL)}
SUP22 = np.array([SUPER.index(s) for _, s in TACO_SEL])

EXPS = {'yolov5s': 1, 'yolov5l': 2, 'yolov8s': 3, 'yolo26s': 4}   # model -> experiment id (hybrid = id + 4)


def find_root(path):
    """Return the folder that directly contains exp1_yolov5s (works even if the zip added an extra level)."""
    for dp, dn, _ in os.walk(path):
        if 'exp1_yolov5s' in dn:
            return dp
    raise FileNotFoundError(f'exp1_yolov5s not found under {path}')


def test_images(data_dir):
    return sorted(glob.glob(f'{data_dir}/images/test/*.jpg'))


def load_gt(paths):
    gts = []
    for p in paths:
        W, H = Image.open(p).size
        lf = p.replace('/images/', '/labels/').replace('\\images\\', '\\labels\\').replace('.jpg', '.txt'); rows = []
        if os.path.exists(lf):
            for l in open(lf).read().split('\n'):
                if l.strip():
                    c, xc, yc, w, h = l.split(); xc, yc, w, h = float(xc)*W, float(yc)*H, float(w)*W, float(h)*H
                    rows.append([int(c), xc-w/2, yc-h/2, xc+w/2, yc+h/2])
        gts.append(np.array(rows).reshape(-1, 5))
    return gts


def box_iou(a, b):
    ix1, iy1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    ix2, iy2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2-ix1, 0, None) * np.clip(iy2-iy1, 0, None)
    aa, ab = (a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]), (b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


_trapz = getattr(np, 'trapezoid', None) or np.trapz
def compute_metrics(preds, gts, nc):
    """P, R (at max-mean-F1 conf, as in YOLO), mAP@.5, mAP@.5:.95 - COCO 101-point AP. preds:[(n,6) x1y1x2y2 conf cls]  gts:[(m,5) cls x1y1x2y2]"""
    thrs = np.linspace(0.5, 0.95, 10); rec = [[] for _ in range(nc)]   # per class: (conf, tp[10])
    n_gt = np.zeros(nc)
    for p, g in zip(preds, gts):
        for c in range(nc):
            gc = g[g[:, 0] == c][:, 1:] if len(g) else np.zeros((0, 4)); n_gt[c] += len(gc)
            pc = p[p[:, 5] == c] if len(p) else np.zeros((0, 6))
            if not len(pc): continue
            pc = pc[np.argsort(-pc[:, 4])]; tp = np.zeros((len(pc), 10), bool)
            if len(gc):
                iou = box_iou(pc[:, :4], gc)
                for ti, t in enumerate(thrs):
                    used = np.zeros(len(gc), bool)
                    for i in range(len(pc)):
                        cand = np.where(used, -1, iou[i]); j = cand.argmax()
                        if cand[j] >= t: used[j] = True; tp[i, ti] = True
            rec[c] += [(pc[i, 4], tp[i]) for i in range(len(pc))]
    ap = np.zeros((nc, 10)); px = np.linspace(0, 1, 1000); P, R = [], []
    valid = [c for c in range(nc) if n_gt[c] > 0]
    for c in valid:
        if not rec[c]: P.append(np.zeros(1000)); R.append(np.zeros(1000)); continue
        conf = np.array([r[0] for r in rec[c]]); tp = np.array([r[1] for r in rec[c]], float)
        o = np.argsort(-conf); conf, tp = conf[o], tp[o]
        for ti in range(10):
            tpc = np.cumsum(tp[:, ti]); fpc = np.cumsum(1 - tp[:, ti])
            r_, p_ = tpc / n_gt[c], tpc / (tpc + fpc + 1e-16)
            mrec, mpre = np.r_[0, r_, 1], np.r_[1, p_, 0]
            mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
            x = np.linspace(0, 1, 101); ap[c, ti] = _trapz(np.interp(x, mrec, mpre), x)
            if ti == 0: P.append(np.interp(-px, -conf, p_, left=0)); R.append(np.interp(-px, -conf, r_, left=0))
    P, R = np.mean(P, 0), np.mean(R, 0); F1 = 2*P*R / (P+R+1e-16); i = F1.argmax()
    return dict(P=float(P[i]), R=float(R[i]), mAP50=float(ap[valid, 0].mean()), mAP50_95=float(ap[valid].mean()),
                n_images=len(preds), n_gt=int(n_gt.sum()), classes_evaluated=len(valid),
                per_class_AP50={int(c): float(ap[c, 0]) for c in valid})