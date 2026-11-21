"""Rebuild the TACO subset (22 classes, 6 super-categories), the train/val/test split and the Inception crops.
Logic copied from cells 3-4 of notebooks/training_colab_original.ipynb. Then verify the split against the one saved
from the original A100 run (results/tables/split_and_counts.json)."""
import os, sys, json, time, random, argparse
import numpy as np, requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.path.join(REPO, 'data'))
args = ap.parse_args()
DATA = args.data.replace('\\', '/')
RAW, YDIR, CDIR = f'{DATA}/taco_raw', f'{DATA}/taco_yolo', f'{DATA}/taco_cls'
import shutil
for d in (YDIR, CDIR): shutil.rmtree(d, ignore_errors=True)   # always rebuild from scratch, never reuse files from an earlier run
for d in (RAW, YDIR, CDIR): os.makedirs(d, exist_ok=True)
print(f'SEEDS: split seed={SEED_SPLIT}, training seed (original Colab runs)={SEED_TRAIN}', flush=True)

ANN_URL = 'https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json'
def get_json(url, tries=6):
    import time
    for i in range(tries):
        try: return requests.get(url, timeout=60).json()
        except Exception as e:
            print(f'retry {i+1}/{tries} for {url}: {type(e).__name__}', flush=True); time.sleep(5 * (i + 1))
    raise SystemExit(f'could not download {url}')
ann = get_json(ANN_URL)
print(len(ann['images']), 'images,', len(ann['annotations']), 'annotations')

def fetch(img):
    out = f"{RAW}/{img['file_name'].replace('/', '_')}"
    if os.path.exists(out) and os.path.getsize(out) > 1000: return True
    for url in (img.get('flickr_640_url'), img.get('flickr_url')):
        if not url: continue
        for _ in range(3):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 200 and len(r.content) > 1000:
                    open(out, 'wb').write(r.content); return True
            except Exception: time.sleep(1)
    return False
with ThreadPoolExecutor(32) as ex: ok = list(ex.map(fetch, ann['images']))
print(f'downloaded {sum(ok)}/{len(ok)} images', flush=True)
if not all(ok): sys.exit(f'ERROR: {len(ok)-sum(ok)} images could not be downloaded - run again to retry (finished downloads are kept)')

cats = {c['id']: c['name'] for c in ann['categories']}
sel = set(ID2IDX)
by_img = {}
for a in ann['annotations']:
    if a['category_id'] in sel: by_img.setdefault(a['image_id'], []).append(a)
imgs = {i['id']: i for i in ann['images']}
usable = sorted(i for i in by_img if os.path.exists(f"{RAW}/{imgs[i]['file_name'].replace('/', '_')}"))
rng = random.Random(SEED_SPLIT); rng.shuffle(usable)
n = len(usable); n_tr, n_va = int(SPLIT[0]*n), int(SPLIT[1]*n)
split_of = {i: 'train' for i in usable[:n_tr]}
split_of.update({i: 'val' for i in usable[n_tr:n_tr+n_va]}); split_of.update({i: 'test' for i in usable[n_tr+n_va:]})
print(f'images with selected categories: {n}  (train {n_tr} / val {n_va} / test {n-n_tr-n_va})')

def load_fixed(path, W, H):
    im = Image.open(path); t = ImageOps.exif_transpose(im).convert('RGB'); r = im.convert('RGB')
    return min([t, r], key=lambda x: abs(x.width / x.height - W / H))

for sp in ('train', 'val', 'test'):
    for sub in ('images', 'labels'): os.makedirs(f'{YDIR}/{sub}/{sp}', exist_ok=True)
    for s in SUPER: os.makedirs(f'{CDIR}/{sp}/{s}', exist_ok=True)

counts22 = {tid: 0 for tid in ID2IDX}; n_crops = {sp: 0 for sp in ('train', 'val', 'test')}
for iid in usable:
    sp, info = split_of[iid], imgs[iid]; W, H = info['width'], info['height']
    im = load_fixed(f"{RAW}/{info['file_name'].replace('/', '_')}", W, H)
    stem = f"{iid:05d}"; im.save(f'{YDIR}/images/{sp}/{stem}.jpg', quality=95)
    sx, sy = im.width / W, im.height / H
    lines = []
    for a in by_img[iid]:
        x, y, w, h = a['bbox']
        if w < 2 or h < 2: continue
        k = ID2IDX[a['category_id']]; counts22[a['category_id']] += 1
        xc, yc = min(max((x + w/2)/W, 0), 1), min(max((y + h/2)/H, 0), 1)
        lines.append(f"{k} {xc:.6f} {yc:.6f} {min(w/W,1):.6f} {min(h/H,1):.6f}")
        if w*sx >= 8 and h*sy >= 8:
            im.crop((int(x*sx), int(y*sy), int((x+w)*sx), int((y+h)*sy))).save(f"{CDIR}/{sp}/{SUPER[SUP22[k]]}/{stem}_{a['id']}.jpg", quality=95)
            n_crops[sp] += 1
    open(f'{YDIR}/labels/{sp}/{stem}.txt', 'w').write('\n'.join(lines))

names = {i: cats[tid] for i, (tid, _) in enumerate(TACO_SEL)}
open(f'{YDIR}/taco.yaml', 'w').write(f"path: {YDIR}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 22\nnames:\n" +
                                      ''.join(f"  {i}: {n_}\n" for i, n_ in names.items()))
print('classifier crops:', n_crops)

# ---- verification against the split saved from the original run ----
saved = json.load(open(os.path.join(REPO, 'results', 'tables', 'split_and_counts.json')))
same_split = saved['split'] == {str(k): v for k, v in split_of.items()}
same_counts = saved['counts22'] == {str(k): v for k, v in counts22.items()}
same_crops = saved['crops'] == n_crops
print(f'VERIFY split identical to original run: {same_split} | class counts identical: {same_counts} | crop counts identical: {same_crops}')
if not (same_split and same_counts and same_crops):
    print('WARNING: rebuilt data differs from the original run - do NOT use these numbers until this is explained.')
    sys.exit(1)
print('OK: test set is identical to the one used for the reported results.')