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
| EXP50 | Weather-augmented fine-tune + eval | SUES-200 + weather augment (from EXP35) | ✅ Done |
| EXP51 | Weather-augmented train from scratch + eval | SUES-200 + weather augment (from scratch) | 🔄 Planned |

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

## EXP50 — Weather-Augmented Fine-Tune + Evaluation

**Setup:** Fine-tune from EXP35 checkpoint on weather-augmented SUES-200 train data (60 epochs, LR=1e-4, backbone_LR=1e-5), then evaluate on all 10 weather conditions. Best model saved at epoch 25 by avg weather R@1.

**Training:** 4800 samples (120 locs × 4 alts × 10 weathers), P=16, K=4, 12 loss components, cosine LR with 3-epoch warmup. Duration: 727.6s.

### Drone → Satellite Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 (from normal) | ΔmAP (from normal) |
|---|---|---|---|---|---|---|
| normal | 94.38% | 100.00% | 100.00% | 96.78% | — | — |
| fog | 90.94% | 100.00% | 100.00% | 95.00% | -3.44% | -1.78% |
| rain | 85.31% | 98.44% | 99.69% | 91.07% | -9.06% | -5.71% |
| snow | 88.12% | 99.06% | 100.00% | 92.86% | -6.25% | -3.92% |
| dark | 80.00% | 95.31% | 97.81% | 86.58% | -14.38% | -10.20% |
| light | 86.25% | 98.44% | 99.38% | 91.42% | -8.12% | -5.36% |
| fog_rain | 78.12% | 97.19% | 98.12% | 86.61% | -16.25% | -10.17% |
| fog_snow | 73.44% | 94.69% | 95.94% | 82.36% | -20.94% | -14.43% |
| rain_snow | 84.69% | 98.12% | 99.06% | 90.36% | -9.69% | -6.42% |
| wind | 85.62% | 97.50% | 99.38% | 90.78% | -8.75% | -6.00% |
| **Avg(all)** | **84.69%** | **97.88%** | **98.94%** | **90.38%** | | |
| **Avg(adverse)** | **83.61%** | | | **89.67%** | **-10.76%** | **-7.11%** |

### R@1 — Weather × Altitude

| Weather | 150m | 200m | 250m | 300m | Avg |
|---|---|---|---|---|---|
| normal | 87.50% | 95.00% | 97.50% | 97.50% | 94.38% |
| fog | 85.00% | 92.50% | 93.75% | 92.50% | 90.94% |
| rain | 81.25% | 81.25% | 90.00% | 88.75% | 85.31% |
| snow | 81.25% | 88.75% | 91.25% | 91.25% | 88.12% |
| dark | 72.50% | 81.25% | 88.75% | 77.50% | 80.00% |
| light | 76.25% | 86.25% | 90.00% | 92.50% | 86.25% |
| fog_rain | 70.00% | 72.50% | 88.75% | 81.25% | 78.12% |
| fog_snow | 66.25% | 76.25% | 81.25% | 70.00% | 73.44% |
| rain_snow | 75.00% | 91.25% | 87.50% | 85.00% | 84.69% |
| wind | 82.50% | 85.00% | 90.00% | 85.00% | 85.62% |

### EXP50 vs EXP49 (Zero-Shot) Comparison

| Weather | EXP49 R@1 | EXP50 R@1 | Δ R@1 | EXP49 mAP | EXP50 mAP | Δ mAP |
|---|---|---|---|---|---|---|
| normal | 95.94% | 94.38% | -1.56% | 97.64% | 96.78% | -0.86% |
| fog | 86.88% | 90.94% | **+4.06%** | 92.68% | 95.00% | +2.32% |
| rain | 43.75% | 85.31% | **+41.56%** | 56.99% | 91.07% | +34.08% |
| snow | 40.62% | 88.12% | **+47.51%** | 54.48% | 92.86% | +38.38% |
| dark | 66.88% | 80.00% | **+13.12%** | 76.74% | 86.58% | +9.84% |
| light | 83.44% | 86.25% | **+2.81%** | 89.97% | 91.42% | +1.45% |
| fog_rain | 42.19% | 78.12% | **+35.94%** | 54.92% | 86.61% | +31.69% |
| fog_snow | 22.19% | 73.44% | **+51.25%** | 35.01% | 82.36% | +47.35% |
| rain_snow | 34.69% | 84.69% | **+50.00%** | 46.85% | 90.36% | +43.51% |
| wind | 81.25% | 85.62% | **+4.38%** | 88.76% | 90.78% | +2.02% |
| **Avg(adverse)** | **55.77%** | **83.61%** | **+27.85%** | **66.27%** | **89.67%** | **+23.40%** |
| **Avg(all)** | **59.78%** | **84.69%** | **+24.90%** | **69.40%** | **90.38%** | **+20.98%** |

### Key Findings (EXP50)
- **Massive improvement** on worst weathers: fog_snow +51.25%, rain_snow +50.00%, snow +47.51%
- **Slight normal degradation**: -1.56% R@1 (trade-off for weather robustness)
- **Best epoch:** 25 (avg weather R@1 = 85.31%)
- **Worst still:** fog_snow (73.44%) but hugely improved from 22.19%
- **Training duration:** 727.6s (~12 min)

---

## EXP51 — Weather-Augmented Train from Scratch + Evaluation

**Setup:** Train from scratch (random init + pretrained DINOv2 backbone) on weather-augmented SUES-200 train data, then evaluate on all 10 weather conditions. Same architecture and loss pipeline, but no EXP35 checkpoint — full 120 epochs.

### Results

> _Pending — run on Kaggle_

---

## Comparison Summary

| Method | Normal R@1 | Avg(all) R@1 | Avg(adverse) R@1 | Worst R@1 |
|---|---|---|---|---|
| EXP49 — Zero-shot (EXP35 ckpt) | 95.94% | 59.78% | 55.76% | 22.19% (fog_snow) |
| EXP50 — Fine-tune from EXP35 | 94.38% | 84.69% | 83.61% | 73.44% (fog_snow) |
| EXP51 — Train from scratch | _pending_ | _pending_ | _pending_ | _pending_ |

---

*Last updated: 2026-03-15*
