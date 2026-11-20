# Comparative analysis (TACO held-out test split, 6 super-categories)

- Best P (6-class): **yolov5s + Inception** (0.343)
- Best R (6-class): **yolo26s** (0.371)
- Best mAP@.5 (6-class): **yolov5l + Inception** (0.254)
- Best mAP@.5:.95 (6-class): **yolov5l + Inception** (0.187)

## Effect of adding the Inception second stage (hybrid minus baseline)

- yolov5s: mAP@.5 +0.059, mAP@.5:.95 +0.034, P +0.042, R +0.102
- yolov5l: mAP@.5 +0.095, mAP@.5:.95 +0.068, P +0.032, R +0.084
- yolov8s: mAP@.5 +0.060, mAP@.5:.95 +0.051, P +0.168, R +0.076
- yolo26s: mAP@.5 +0.009, mAP@.5:.95 +0.002, P +0.089, R -0.068

Inception classifier on TACO crops (test acc): 0.705