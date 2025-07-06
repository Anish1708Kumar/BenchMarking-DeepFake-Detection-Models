# Deepfake Detection Benchmarking

This repository implements and evaluates multiple deepfake detection models across standard datasets to benchmark performance using the AUC (Area Under the Curve) metric.

---

## Project Title
**Towards Benchmarking and Evaluating Deepfake Detection**

---

## Models Used
- **Xception Model** (CNN-based baseline)
- **HeadPose Estimation Model**
- **FFD (Fake Feature Detector)**

---

## Datasets Used
- [UADFV](https://github.com/nddudn/Dataset-UADFV)
- [DFDC (Preview)](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- [FaceForensics++ (FF++)](https://github.com/ondyari/FaceForensics)
- [DF-TIMIT](https://github.com/grip-unina/DF-TIMIT)

*Note:* Only subsets from each dataset were used for practical evaluation.

---

## Evaluation Metric
- **AUC (Area Under the ROC Curve)** is used to evaluate each model’s classification performance.

---

## Project Breakdown
| Task                     | Description                                                  |
|--------------------------|--------------------------------------------------------------|
| Dataset Preparation      | Frame extraction, resizing, and preprocessing                |
| Model Training           | Xception, HeadPose, and FFD models                           |
| Testing and Inference    | Evaluate on test sets with label comparison                  |
| Performance Evaluation   | AUC computation and result visualization                     |

---

## Results Snapshot
| Model      | Dataset     | AUC Score |
|------------|-------------|-----------|
| Xception   | UADFV       | 0.91      |
| HeadPose   | DFDC        | 0.75      |
| FFD        | FaceForensics++ | 0.88 |

---

## Requirements
```bash
pip install -r requirements.txt
