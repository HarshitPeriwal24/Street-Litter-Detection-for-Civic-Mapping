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
