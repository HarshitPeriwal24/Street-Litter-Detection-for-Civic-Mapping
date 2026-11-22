# Street-Litter Detection for Civic Mapping (UCS532 P59)

Team: Harshit Periwal (1024031053), Nipun Gupta (1024031169)

## Problem
Detect street litter in TACO images and test whether a second-stage classifier improves a standard detector. Baseline paper: Pang & Huang, *A Two-Stage Deep Learning Framework for Enhanced Waste Detection and Classification* (ICMLA 2023): a YOLO detector plus an InceptionV3 classifier on the detected crops. We implement it with YOLOv5s, YOLOv5l, YOLOv8s and YOLO26s, and add an ablation, bootstrap confidence intervals and a failure analysis. The original proposal (18 Oct) is kept unchanged in `docs/proposal/`; what changed is described in the report.

## Dataset
TACO (Proenca & Simoes, 2020), official `annotations.json`, 1,500 images: https://github.com/pedropro/TACO . Subset: the 22 categories of the baseline paper's Table 1 (1,116 images; 892 / 111 / 113 train/val/test, split seed 42), grouped into 6 super-categories (cardboard, paper, plastic, metal, glass, trash). `src/prepare_data.py` downloads the images from Flickr and verifies that the rebuilt split and class counts are identical to the original run. No dataset is stored in this repo.

## Requirements.txt
[View Requirements.txt](requirements.txt)

## Reproduce the main table
Requirements: git, bash (Git Bash on Windows; Linux, macOS, WSL or Colab also work), Python 3.12, internet access and about 6 GB of free disk. No GPU is needed, and nothing is retrained (training was done once on Colab, see below).

**Windows (Git Bash):**
```
git clone https://github.com/nipun1015/Street-Litter-Detection-for-Civic-Mapping.git
cd Street-Litter-Detection-for-Civic-Mapping
py -3.12 -m venv venv
source venv/Scripts/activate
python --version                  # should print 3.12.x
bash run_all.sh
```

**Linux / macOS / WSL:**
```
git clone https://github.com/nipun1015/Street-Litter-Detection-for-Civic-Mapping.git
cd Street-Litter-Detection-for-Civic-Mapping
python3 -m venv venv
source venv/bin/activate
python --version                  # should print 3.12.x
bash run_all.sh
```
In Colab, skip the venv lines and run `!bash run_all.sh`.

`run_all.sh` installs `requirements.txt`, downloads the released detections and InceptionV3 model, rebuilds the data, runs the controls, recomputes all rows and writes `results/main_table.md`. Every log is saved in `results/logs/`. Seeds are printed at the top of each log: split seed 42, training seed 0 (original Colab runs), bootstrap seed 0.

## Artifacts (download links, not in the repo)
- Detections + InceptionV3 model used by `run_all.sh` (77 MB): https://drive.google.com/file/d/12CViLk8V_4WIqX6KzsYZF5SNH9h3Rnib/view?usp=drive_link
- Trained detector weights (optional, for the demo; 132 MB): https://drive.google.com/file/d/1H6FospOdjIEYlh6ub5V59lFGK_jSpmNu/view?usp=sharing

## Hardware and runtime
- Training: Colab, NVIDIA A100 40 GB, run on a friend's laptop/account (Aarush). Detectors took 9.7 min (YOLOv5s), 19.4 (YOLOv5l), 7.3 (YOLOv8s), 11.8 (YOLO26s) for 100 epochs, batch 16, image size 416.
- Evaluation (`run_all.sh`): Intel Core i5-1235U, 16 GB RAM, Windows, Python 3.12, no GPU. Measured full run of `run_all.sh`: 21 min 49 s (Windows, Git Bash, Python 3.12.10, no GPU), including a 2.5 min download of the artifacts; the TACO images were already cached. A first run on a clean machine also installs the packages and downloads the 1,500 images, so expect longer. Full output: `results/logs/run_all_full_run.txt`.
- Training was run once in Colab from `notebooks/training_colab_original.ipynb` (committed unchanged) and was not re-run from this repo. GPU training is not bit-reproducible (`deterministic=False`), so a retrain gives close but not identical numbers. The evaluation from the released detections is deterministic.

## Results (TACO test, 113 images, 196 objects, 6 super-categories)
| Detector | Configuration | P | R | mAP@.5 | mAP@.5:.95 |
|---|---|---|---|---|---|
| yolov5s | A baseline | 0.301 | 0.238 | 0.187 | 0.126 |
| yolov5s | B baseline + agnostic NMS (no Inception) | 0.353 | 0.232 | 0.185 | 0.125 |
| yolov5s | C hybrid, score=det (reported) | 0.343 | 0.340 | 0.246 | 0.160 |
| yolov5s | D hybrid, score=det*cls (ablation) | 0.368 | 0.337 | 0.263 | 0.172 |
| yolov5l | A baseline | 0.278 | 0.243 | 0.159 | 0.119 |
| yolov5l | B baseline + agnostic NMS (no Inception) | 0.323 | 0.194 | 0.147 | 0.112 |
| yolov5l | C hybrid, score=det (reported) | 0.310 | 0.327 | 0.254 | 0.187 |
| yolov5l | D hybrid, score=det*cls (ablation) | 0.356 | 0.283 | 0.277 | 0.202 |
| yolov8s | A baseline | 0.157 | 0.241 | 0.177 | 0.134 |
| yolov8s | B baseline + agnostic NMS (no Inception) | 0.158 | 0.197 | 0.113 | 0.089 |
| yolov8s | C hybrid, score=det (reported) | 0.342 | 0.316 | 0.237 | 0.185 |
| yolov8s | D hybrid, score=det*cls (ablation) | 0.381 | 0.299 | 0.256 | 0.201 |
| yolo26s | A baseline | 0.179 | 0.371 | 0.193 | 0.161 |
| yolo26s | B baseline + agnostic NMS (no Inception) | 0.232 | 0.221 | 0.174 | 0.146 |
| yolo26s | C hybrid, score=det (reported) | 0.268 | 0.302 | 0.202 | 0.163 |
| yolo26s | D hybrid, score=det*cls (ablation) | 0.282 | 0.299 | 0.213 | 0.173 |

(Copied from `results/main_table.md`, which `run_all.sh` regenerates. P and R are at the confidence with maximum F1.)

| Detector | Configuration | P | R | mAP@.5 | mAP@.5:.95 |
|---|---|---|---|---|---|
| yolov5s | A baseline | 0.301 | 0.238 | 0.187 | 0.126 |
| yolov5s | B baseline + agnostic NMS (no Inception) | 0.353 | 0.232 | 0.185 | 0.125 |
| yolov5s | C hybrid, score=det (reported) | 0.343 | 0.340 | 0.246 | 0.160 |
| yolov5s | D hybrid, score=det*cls (ablation) | 0.368 | 0.337 | 0.263 | 0.172 |
| yolov5l | A baseline | 0.278 | 0.243 | 0.159 | 0.119 |
| yolov5l | B baseline + agnostic NMS (no Inception) | 0.323 | 0.194 | 0.147 | 0.112 |
| yolov5l | C hybrid, score=det (reported) | 0.310 | 0.327 | 0.254 | 0.187 |
| yolov5l | D hybrid, score=det*cls (ablation) | 0.356 | 0.283 | 0.277 | 0.202 |
| yolov8s | A baseline | 0.157 | 0.241 | 0.177 | 0.134 |
| yolov8s | B baseline + agnostic NMS (no Inception) | 0.158 | 0.197 | 0.113 | 0.089 |
| yolov8s | C hybrid, score=det (reported) | 0.342 | 0.316 | 0.237 | 0.185 |
| yolov8s | D hybrid, score=det*cls (ablation) | 0.381 | 0.299 | 0.256 | 0.201 |
| yolo26s | A baseline | 0.179 | 0.371 | 0.193 | 0.161 |
| yolo26s | B baseline + agnostic NMS (no Inception) | 0.232 | 0.221 | 0.174 | 0.146 |
| yolo26s | C hybrid, score=det (reported) | 0.268 | 0.302 | 0.202 | 0.163 |
| yolo26s | D hybrid, score=det*cls (ablation) | 0.282 | 0.299 | 0.213 | 0.173 |

(Copied from `results/main_table.md`, which `run_all.sh` regenerates. P and R are at the confidence with maximum F1.)

See `results/main_table.md` (generated by `run_all.sh`), `results/tables/` and `results/figures/`. Single training run per model; differences between hybrids are within the bootstrap intervals in `results/tables/bootstrap_ci.csv`. The reported hybrid (row C) uses the detector confidence as score, fixed before the test results were looked at; scoring by detector x classifier (row D) is shown only as an ablation. Known reproduction difference: the YOLOv8s + Inception precision recomputed on CPU is 0.342 against 0.325 saved from the GPU run (mAP differs by less than 1e-3).

## Controls (from the proposal's Gate 1)
`src/controls.py`: oracle (P = R = 1, mAP at the metric's 0.995 ceiling), shuffle (mAP 0.000), no leakage between splits (one exact duplicate pair, both in train), and a 20-image annotation audit in `results/figures/annotation_audit/`.

## Credits
- Pang & Huang, ICMLA 2023 (baseline method). TACO dataset and toolkit (Proenca & Simoes).
- Ultralytics YOLOv5 and YOLOv8 / YOLO26 (AGPL-3.0), used for training in Colab, with COCO-pretrained weights from Ultralytics.
- InceptionV3 ImageNet weights (Keras Applications), TensorFlow, PyTorch / torchvision (NMS).
- Metric code (`src/common.py`) is taken from our own Colab notebook; it follows the COCO 101-point AP definition used by YOLO.

## AI declaration
We used AI assistance in this project. In detail:

- Ideation and decisions (ours): the choice of problem and baseline paper, the
  decision to evaluate four detectors with the two-stage method, the scope of
  what to keep or drop, and interpretation of the results.
- Training code (notebooks/training_colab_original.ipynb): extracted from repo but used ai  assistance also .
- Evaluation code (src/*.py, run_all.sh): written with AI assistance.
  We ran every script on our own machine, checked that the outputs reproduce the
  numbers saved from training, and reviewed the control and ablation results.
- README and analysis design: drafted with AI assistance (Claude), then reviewed and edited by us.
- Report and slides: our ideation for slides but took help of ai in report

Every number in the report comes from the logs in this repo. Both team members
have read the code and can explain it.

## Layout
`src/` scripts - `notebooks/` original Colab training notebook - `results/` tables, figures, logs - `docs/proposal/` original proposal (unchanged) - `requirements.txt` - `run_all.sh`
