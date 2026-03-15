# AcrossWeather — Performance Tracker

> Track all experiment results for the AcrossWeather project (SUES-200 dataset).

---

## Model Architecture

| Component | Detail |
|---|---|
| Backbone | DINOv2 ViT-S/14 |
| Part Discovery | Altitude-Aware (N=8, dim=256) |
| Embedding | 512-d fused (part + CLS) |
| Image Size | 336×336 |
| Total Params | 24.70M |
| Trainable Params | 13.30M |
| GFLOPs | 12.46 |
| Inference | 4.8 ms/query |

---

## Experiment Index

| EXP | Description | Training Data | Status |
|---|---|---|---|
| EXP35 | SPDGeo-DPEA-MAR (baseline, no weather) | SUES-200 clean | ✅ Done |
| EXP48 | Weather synthetic data generation | imgaug 10 conditions | ✅ Done |
| EXP49 | Weather robustness benchmark (zero-shot) | Eval only (EXP35 ckpt) | ✅ Done |
| EXP50 | Weather-augmented training + eval | SUES-200 + weather augment | 🔄 Planned |

---

## EXP35 — Baseline (No Weather Training)

**Config:** 120 epochs, P=16, K=4, LR=3e-4, backbone_LR=3e-5, 12 loss components

### Normal Evaluation (Drone → Satellite)

| Altitude | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| 150m | 92.50% | 98.75% | 100.00% | 95.15% |
| 200m | 95.00% | 100.00% | 100.00% | 97.50% |
| 250m | 97.50% | 100.00% | 100.00% | 98.54% |
| 300m | 98.75% | 100.00% | 100.00% | 99.38% |
| **Overall** | **96.32%** | **99.69%** | **100.00%** | **97.78%** |

---

## EXP49 — Weather Robustness Benchmark (Zero-Shot)

**Setup:** Load EXP35 checkpoint → evaluate on 10 weather conditions (no weather training)

### Drone → Satellite Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 | ΔmAP |
|---|---|---|---|---|---|---|
| normal | 95.94% | 99.69% | 100.00% | 97.64% | — | — |
| fog | 86.88% | 99.38% | 100.00% | 92.68% | -9.06% | -4.96% |
| rain | 43.75% | 71.25% | 82.81% | 56.99% | -52.19% | -40.65% |
| snow | 40.62% | 71.25% | 80.00% | 54.48% | -55.31% | -43.16% |
| dark | 66.88% | 87.81% | 92.50% | 76.74% | -29.06% | -20.90% |
| light | 83.44% | 98.75% | 99.69% | 89.97% | -12.50% | -7.67% |
| fog_rain | 42.19% | 67.50% | 79.69% | 54.92% | -53.75% | -42.72% |
| fog_snow | 22.19% | 50.00% | 61.25% | 35.01% | -73.75% | -62.63% |
| rain_snow | 34.69% | 58.44% | 72.19% | 46.85% | -61.25% | -50.79% |
| wind | 81.25% | 98.44% | 99.06% | 88.76% | -14.69% | -8.88% |
| **Avg(all)** | **59.78%** | **80.25%** | **86.72%** | **69.40%** | | |
| **Avg(adverse)** | **55.76%** | **78.09%** | **85.24%** | **66.27%** | **-40.17%** | **-31.38%** |

### R@1 — Weather × Altitude

| Weather | 150m | 200m | 250m | 300m | Avg |
|---|---|---|---|---|---|
| normal | 92.50% | 95.00% | 97.50% | 98.75% | 95.94% |
| fog | 82.50% | 86.25% | 90.00% | 88.75% | 86.88% |
| rain | 36.25% | 46.25% | 46.25% | 46.25% | 43.75% |
| snow | 37.50% | 40.00% | 42.50% | 42.50% | 40.62% |
| dark | 61.25% | 68.75% | 72.50% | 65.00% | 66.88% |
| light | 76.25% | 86.25% | 85.00% | 86.25% | 83.44% |
| fog_rain | 40.00% | 41.25% | 42.50% | 45.00% | 42.19% |
| fog_snow | 22.50% | 25.00% | 23.75% | 17.50% | 22.19% |
| rain_snow | 31.25% | 31.25% | 41.25% | 35.00% | 34.69% |
| wind | 80.00% | 78.75% | 82.50% | 83.75% | 81.25% |

### Key Findings (EXP49)
- **Worst weather:** fog_snow → R@1 = 22.19%
- **Severe degradation** on precipitation: rain (-52.19%), snow (-55.31%), combinations worse
- **Mild degradation** on lighting: fog (-9.06%), light (-12.50%), wind (-14.69%)
- **Conclusion:** Model needs weather-augmented training to generalize

---

## EXP50 — Weather-Augmented Training + Evaluation

**Setup:** Fine-tune from EXP35 checkpoint on weather-augmented SUES-200 train data, then evaluate on all 10 weather conditions

### Results

> _Pending — run on Kaggle_

---

## Comparison Summary

| Method | Normal R@1 | Avg(all) R@1 | Avg(adverse) R@1 | Worst R@1 |
|---|---|---|---|---|
| EXP35 (no weather) | 95.94% | 59.78% | 55.76% | 22.19% (fog_snow) |
| EXP50 (weather-trained) | _pending_ | _pending_ | _pending_ | _pending_ |

---

*Last updated: 2026-03-15*
