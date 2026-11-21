"""Recompute the 4 baseline rows from the saved test detections and compare with the metrics saved by the original run."""
import os, sys, json, pickle, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--artifacts', required=True, help='folder that contains exp1_yolov5s, exp2_yolov5l, ...')
ap.add_argument('--data', default=os.path.join(REPO, 'data'))
args = ap.parse_args()
ART = find_root(args.artifacts).replace('\\', '/')
DATA = args.data.replace('\\', '/')
print(f'SEEDS: split seed={SEED_SPLIT}, training seed (original Colab runs)={SEED_TRAIN}')

paths = test_images(f'{DATA}/taco_yolo')
gt22 = load_gt(paths)
gt6 = [np.c_[SUP22[g[:, 0].astype(int)], g[:, 1:]] if len(g) else g for g in gt22]
print(f'test images: {len(paths)}, ground-truth boxes: {sum(len(g) for g in gt22)}')

rows, worst = [], 0.0
for key, eid in EXPS.items():
    det = pickle.load(open(f'{ART}/exp{eid}_{key}/test_detections.pkl', 'rb'))
    assert len(det) == len(paths), f'{key}: {len(det)} detections vs {len(paths)} images'
    det6 = [np.c_[d[:, :5], SUP22[d[:, 5].astype(int)]] if len(d) else d for d in det]
    m22, m6 = compute_metrics(det, gt22, 22), compute_metrics(det6, gt6, 6)
    saved = json.load(open(os.path.join(REPO, 'results', 'tables', f'exp{eid}_{key}_metrics_test.json')))
    diffs = [abs(m[k] - saved[s][k]) for m, s in ((m22, 'detector_22class'), (m6, 'final_6class')) for k in ('P', 'R', 'mAP50', 'mAP50_95')]
    worst = max(worst, max(diffs))
    print(f"{key:8s} 6-class  P={m6['P']:.4f} R={m6['R']:.4f} mAP50={m6['mAP50']:.4f} mAP50-95={m6['mAP50_95']:.4f} | "
          f"22-class mAP50={m22['mAP50']:.4f} | max diff vs saved json = {max(diffs):.2e}")
    rows.append(dict(model=key, P6=m6['P'], R6=m6['R'], mAP50_6=m6['mAP50'], mAP50_95_6=m6['mAP50_95'], mAP50_22=m22['mAP50'], mAP50_95_22=m22['mAP50_95']))

import pandas as pd
os.makedirs(os.path.join(REPO, 'results', 'tables'), exist_ok=True)
pd.DataFrame(rows).round(4).to_csv(os.path.join(REPO, 'results', 'tables', 'reproduced_baselines.csv'), index=False)
print('REPRODUCED' if worst < 1e-6 else f'MISMATCH (worst diff {worst:.2e}) - investigate before using these numbers')