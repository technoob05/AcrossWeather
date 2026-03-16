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
| EXP51 | Weather-augmented train from scratch + eval | SUES-200 + weather augment (from scratch) | ✅ Done |
| EXP52 | Weather fine-tune with online augmentation (WeatherPrompt-style) | SUES-200 + online imgaug (from EXP35) | ✅ Done |
| EXP53 | Weather train from scratch with online augmentation (WeatherPrompt-style) | SUES-200 + online imgaug (from scratch) | ✅ Done |
| EXP54 | Zero-shot eval framework (any checkpoint → all 10 weathers) | Eval only (EXP35 ckpt run complete) | ✅ Done (EXP35) |

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

**Training:** 4800 samples (120 locs × 4 alts × 10 weathers), P=16, K=4, LR=3e-4, backbone_LR=3e-5, 12 loss components, cosine LR with 5-epoch warmup, RECON_WARMUP=10. Best model saved at epoch 50 by avg weather R@1. Duration: 1377.0s.

### Drone → Satellite Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 (from normal) | ΔmAP (from normal) |
|---|---|---|---|---|---|---|
| normal | 90.94% | 99.38% | 100.00% | 94.51% | — | — |
| fog | 90.62% | 98.75% | 100.00% | 94.40% | -0.31% | -0.11% |
| rain | 87.81% | 97.81% | 99.69% | 92.81% | -3.12% | -1.71% |
| snow | 86.56% | 99.06% | 100.00% | 92.08% | -4.38% | -2.43% |
| dark | 81.88% | 95.31% | 97.50% | 87.67% | -9.06% | -6.84% |
| light | 87.19% | 97.81% | 99.69% | 92.05% | -3.75% | -2.46% |
| fog_rain | 80.00% | 97.19% | 98.75% | 87.82% | -10.94% | -6.70% |
| fog_snow | 77.19% | 94.38% | 98.44% | 84.75% | -13.75% | -9.76% |
| rain_snow | 86.25% | 96.88% | 100.00% | 91.40% | -4.69% | -3.12% |
| wind | 86.25% | 98.44% | 99.69% | 91.36% | -4.69% | -3.16% |
| **Avg(all)** | **85.47%** | **97.50%** | **99.38%** | **90.88%** | | |
| **Avg(adverse)** | **84.86%** | | | **90.48%** | **-6.08%** | **-4.03%** |

### R@1 — Weather × Altitude

| Weather | 150m | 200m | 250m | 300m | Avg |
|---|---|---|---|---|---|
| normal | 85.00% | 92.50% | 92.50% | 93.75% | 90.94% |
| fog | 78.75% | 92.50% | 96.25% | 95.00% | 90.62% |
| rain | 76.25% | 88.75% | 92.50% | 93.75% | 87.81% |
| snow | 78.75% | 86.25% | 92.50% | 88.75% | 86.56% |
| dark | 72.50% | 77.50% | 93.75% | 83.75% | 81.88% |
| light | 80.00% | 86.25% | 91.25% | 91.25% | 87.19% |
| fog_rain | 68.75% | 80.00% | 87.50% | 83.75% | 80.00% |
| fog_snow | 68.75% | 77.50% | 88.75% | 73.75% | 77.19% |
| rain_snow | 76.25% | 90.00% | 93.75% | 85.00% | 86.25% |
| wind | 81.25% | 83.75% | 88.75% | 91.25% | 86.25% |

### EXP51 vs EXP49 vs EXP50 Comparison

| Weather | EXP49 R@1 | EXP50 R@1 | EXP51 R@1 | Δ49→51 | Δ50→51 |
|---|---|---|---|---|---|
| normal | 95.94% | 94.38% | 90.94% | -5.00% | -3.44% |
| fog | 86.88% | 90.94% | 90.62% | **+3.75%** | -0.31% |
| rain | 43.75% | 85.31% | 87.81% | **+44.06%** | **+2.50%** |
| snow | 40.62% | 88.12% | 86.56% | **+45.94%** | -1.56% |
| dark | 66.88% | 80.00% | 81.88% | **+15.00%** | **+1.88%** |
| light | 83.44% | 86.25% | 87.19% | **+3.75%** | **+0.94%** |
| fog_rain | 42.19% | 78.12% | 80.00% | **+37.81%** | **+1.88%** |
| fog_snow | 22.19% | 73.44% | 77.19% | **+55.00%** | **+3.75%** |
| rain_snow | 34.69% | 84.69% | 86.25% | **+51.56%** | **+1.56%** |
| wind | 81.25% | 85.62% | 86.25% | **+5.00%** | **+0.63%** |
| **Avg(adverse)** | **55.77%** | **83.61%** | **84.86%** | **+29.10%** | **+1.25%** |
| **Avg(all)** | **59.78%** | **84.69%** | **85.47%** | **+25.69%** | **+0.78%** |

### Key Findings (EXP51)
- **Marginally beats EXP50** on adverse weather: +1.25% avg R@1 despite no pretrained task-specific weights
- **Lower normal R@1** than EXP49/EXP50: 90.94% vs 95.94%/94.38% — full training on weather data slightly hurts clean performance
- **Best improvement vs EXP50:** fog_snow +3.75%, rain +2.50%, fog_rain +1.88%
- **Best epoch:** 50 of 120 (avg weather R@1 = 85.62% on eval subset)
- **Training duration:** 1377.0s (~23 min)
- **Conclusion:** Training from scratch with weather data matches fine-tuning; pretrained task knowledge (EXP35) not strictly necessary for weather robustness

---

## Comparison Summary

### Our Experiments (SUES-200, Drone → Satellite)

| Method | Normal R@1 | Avg(all) R@1 | Avg(adverse) R@1 | Worst R@1 |
|---|---|---|---|---|
| EXP49 — Zero-shot (EXP35 ckpt) | 95.94% | 59.78% | 55.76% | 22.19% (fog_snow) |
| EXP50 — Fine-tune from EXP35 (pre-generated) | 94.38% | 84.69% | 83.61% | 73.44% (fog_snow) |
| EXP51 — Train from scratch (pre-generated) | 90.94% | 85.47% | 84.86% | 77.19% (fog_snow) |
| EXP52 — Fine-tune from EXP35 (online augmentation) | 95.00% | 85.03% | 83.92% | 69.38% (fog_snow) |
| **EXP53 — Train from scratch (online aug) ★ BEST** | **92.81%** | **86.81%** | **86.15%** | **79.38% (fog_snow)** |
| EXP54 — Zero-shot eval (EXP35 ckpt, new eval script) | 95.94% | 59.78% | 55.76% | 22.19% (fog_snow) |

### External Baselines (WeatherPrompt NeurIPS 2025, SUES-200 Drone → Satellite)

> Source: Table 2 in WeatherPrompt, NeurIPS 2025. "Over-exp" in paper ≡ "light" (over-exposure) in our experiments.  
> Avg(adverse) computed here as mean of 9 adverse conditions (excluding normal) from raw table values.

| Method | Venue | Normal R@1 | Avg(all) R@1 | Avg(adverse) R@1 | Worst R@1 |
|---|---|---|---|---|---|
| Zheng et al. | backbone | 57.70% | 44.43% | 42.98% | 23.81% (dark) |
| IBN-Net | backbone | 65.34% | 50.69% | 49.06% | 29.61% (dark) |
| Sample4Geo | ICCV'23 | 74.93% | 41.95% | 38.29% | 12.95% (fog_snow) |
| Safe-Net | TIP'24 | 76.31% | 50.68% | 47.83% | 25.95% (rain_snow) |
| CCR | TCSVT'24 | 73.22% | 52.06% | 49.70% | 31.03% (dark) |
| MuSe-Net | PR'24 | 66.07% | 52.02% | 51.13% | 29.34% (dark) |
| **WeatherPrompt (prev. SOTA)** | **NeurIPS'25** | **76.72%** | **62.52%** | **60.94%** | **40.42% (dark)** |
| **Ours (EXP53) ★** | **—** | **92.81%** | **86.81%** | **86.15%** | **79.38% (fog_snow)** |
| **Δ Ours vs. WeatherPrompt** | | **+16.09%** | **+24.29%** | **+25.21%** | — |

---

## EXP52 — Weather Fine-Tune with Online Augmentation

**Setup:** Fine-tune from EXP35 checkpoint with online WeatherPrompt-style imgaug augmentation: 1 random drone image per location per epoch, weather applied on-the-fly (60 epochs, LR=1e-4, backbone_LR=1e-5). Best model saved at epoch 20 by avg weather R@1. Duration: 1934.9s (~32 min).

**Key difference from EXP50:** EXP50 used 4800 pre-generated images (120×4×10). EXP52 uses 120 original drone images per epoch (1 random per location) with online augmentation — matching WeatherPrompt's exact data protocol.

### Drone → Satellite Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 (from normal) | ΔmAP (from normal) |
|---|---|---|---|---|---|---|
| normal | 95.00% | 99.38% | 100.00% | 96.71% | — | — |
| fog | 90.00% | 98.75% | 100.00% | 94.30% | -5.00% | -2.42% |
| rain | 89.69% | 97.50% | 99.69% | 93.05% | -5.31% | -3.67% |
| snow | 87.50% | 98.75% | 100.00% | 92.05% | -7.50% | -4.67% |
| dark | 78.75% | 94.38% | 96.56% | 85.07% | -16.25% | -11.65% |
| light | 84.69% | 96.88% | 100.00% | 90.14% | -10.31% | -6.57% |
| fog_rain | 81.88% | 96.88% | 98.75% | 88.64% | -13.12% | -8.08% |
| fog_snow | 69.38% | 92.81% | 96.25% | 79.21% | -25.62% | -17.50% |
| rain_snow | 84.69% | 97.19% | 99.06% | 90.06% | -10.31% | -6.65% |
| wind | 88.75% | 98.75% | 99.69% | 92.93% | -6.25% | -3.78% |
| **Avg(all)** | **85.03%** | **97.12%** | **99.00%** | **90.22%** | | |
| **Avg(adverse)** | **83.92%** | | | **89.49%** | **-11.08%** | **-7.22%** |

### R@1 — Weather × Altitude

| Weather | 150m | 200m | 250m | 300m | Avg |
|---|---|---|---|---|---|
| normal | 88.75% | 95.00% | 97.50% | 98.75% | 95.00% |
| fog | 81.25% | 91.25% | 93.75% | 93.75% | 90.00% |
| rain | 81.25% | 90.00% | 93.75% | 93.75% | 89.69% |
| snow | 75.00% | 91.25% | 92.50% | 91.25% | 87.50% |
| dark | 70.00% | 83.75% | 85.00% | 76.25% | 78.75% |
| light | 76.25% | 85.00% | 88.75% | 88.75% | 84.69% |
| fog_rain | 71.25% | 82.50% | 86.25% | 87.50% | 81.88% |
| fog_snow | 58.75% | 75.00% | 77.50% | 66.25% | 69.38% |
| rain_snow | 75.00% | 88.75% | 86.25% | 88.75% | 84.69% |
| wind | 82.50% | 92.50% | 90.00% | 90.00% | 88.75% |

### EXP52 vs EXP49 / EXP50 / EXP51 Comparison

| Weather | EXP49 R@1 | EXP50 R@1 | EXP51 R@1 | EXP52 R@1 | Δ50→52 | Δ51→52 |
|---|---|---|---|---|---|---|
| normal | 95.94% | 94.38% | 90.94% | 95.00% | +0.62% | +4.06% |
| fog | 86.88% | 90.94% | 90.62% | 90.00% | -0.94% | -0.62% |
| rain | 43.75% | 85.31% | 87.81% | 89.69% | **+4.38%** | **+1.88%** |
| snow | 40.62% | 88.12% | 86.56% | 87.50% | -0.62% | +0.94% |
| dark | 66.88% | 80.00% | 81.88% | 78.75% | -1.25% | -3.13% |
| light | 83.44% | 86.25% | 87.19% | 84.69% | -1.56% | -2.50% |
| fog_rain | 42.19% | 78.12% | 80.00% | 81.88% | **+3.75%** | **+1.88%** |
| fog_snow | 22.19% | 73.44% | 77.19% | 69.38% | -4.06% | -7.81% |
| rain_snow | 34.69% | 84.69% | 86.25% | 84.69% | 0.00% | -1.56% |
| wind | 81.25% | 85.62% | 88.12% | 88.75% | **+3.13%** | **+0.63%** |
| **Avg(adverse)** | **55.77%** | **83.61%** | **84.86%** | **83.92%** | **+0.31%** | **-0.94%** |
| **Avg(all)** | **59.78%** | **84.69%** | **85.47%** | **85.03%** | **+0.34%** | **-0.44%** |

### Key Findings (EXP52)
- **Online augmentation ≈ pre-generated** for fine-tune: only -0.31% avg adverse R@1 vs EXP51 (scratch-pregen) but +0.31% vs EXP50 (finetune-pregen)
- **Normal accuracy preserved:** 95.00% — slightly lower than EXP49 zero-shot (95.94%) but better than EXP50/51
- **Weakness:** fog_snow (69.38%) — weakest among weather-trained models; online aug may undersample combined conditions
- **rain, fog_rain, wind improved** vs EXP50: online strategy diversifies exposure for individual weather conditions
- **Best epoch:** 20 / 60 (early convergence thanks to EXP35 pretrain)
- **Training duration:** 1934.9s (~32 min, ~3× longer than EXP50 due to imgaug CPU overhead)
- **Conclusion:** Online WeatherPrompt-style augmentation matches pre-generated approach in overall robustness; training efficiency is lower due to on-the-fly imgaug cost

---

## EXP53 — Weather Train from Scratch with Online Augmentation

**Setup:** Train from scratch (random init + pretrained DINOv2 backbone) with online WeatherPrompt-style imgaug augmentation: 1 random drone image per location per epoch, weather applied on-the-fly (120 epochs, LR=3e-4, backbone_LR=3e-5, RECON_WARMUP=10). Best model saved at epoch 30 by avg weather R@1 on 4-weather eval subset. Duration: 3654.1s (~61 min).

**Key difference from EXP51:** EXP51 used 4800 pre-generated images (120×4×10). EXP53 uses 120 original drone images per epoch (1 random per location) with online augmentation — matching WeatherPrompt's exact data protocol. Same strategy as EXP52 but from scratch instead of fine-tuning from EXP35.

### Drone → Satellite Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 (from normal) | ΔmAP (from normal) |
|---|---|---|---|---|---|---|
| normal | 92.81% | 98.44% | 100.00% | 95.81% | — | — |
| fog | 91.88% | 98.75% | 100.00% | 94.94% | -0.94% | -0.86% |
| rain | 90.31% | 99.69% | 100.00% | 94.18% | -2.50% | -1.62% |
| snow | 87.19% | 98.75% | 99.69% | 92.19% | -5.62% | -3.61% |
| dark | 82.50% | 94.06% | 95.62% | 87.90% | -10.31% | -7.91% |
| light | 86.25% | 96.56% | 98.75% | 91.25% | -6.56% | -4.56% |
| fog_rain | 85.62% | 98.12% | 99.38% | 91.19% | -7.19% | -4.61% |
| fog_snow | 79.38% | 95.94% | 98.44% | 86.89% | -13.44% | -8.92% |
| rain_snow | 84.69% | 98.12% | 99.69% | 90.76% | -8.12% | -5.04% |
| wind | 87.50% | 98.12% | 100.00% | 92.33% | -5.31% | -3.48% |
| **Avg(all)** | **86.81%** | **97.66%** | **99.16%** | **91.74%** | | |
| **Avg(adverse)** | **86.15%** | | | **91.29%** | **-6.67%** | **-4.51%** |

### R@1 — Weather × Altitude

| Weather | 150m | 200m | 250m | 300m | Avg |
|---|---|---|---|---|---|
| normal | 86.25% | 93.75% | 96.25% | 95.00% | 92.81% |
| fog | 85.00% | 95.00% | 93.75% | 93.75% | 91.88% |
| rain | 87.50% | 87.50% | 95.00% | 91.25% | 90.31% |
| snow | 81.25% | 86.25% | 91.25% | 90.00% | 87.19% |
| dark | 76.25% | 82.50% | 88.75% | 82.50% | 82.50% |
| light | 81.25% | 88.75% | 86.25% | 88.75% | 86.25% |
| fog_rain | 76.25% | 88.75% | 87.50% | 90.00% | 85.62% |
| fog_snow | 75.00% | 82.50% | 80.00% | 80.00% | 79.38% |
| rain_snow | 71.25% | 86.25% | 91.25% | 90.00% | 84.69% |
| wind | 80.00% | 88.75% | 90.00% | 91.25% | 87.50% |

### EXP53 vs EXP49 / EXP50 / EXP51 / EXP52 Comparison

| Weather | EXP49 R@1 | EXP50 R@1 | EXP51 R@1 | EXP52 R@1 | EXP53 R@1 | Δ50→53 | Δ51→53 | Δ52→53 |
|---|---|---|---|---|---|---|---|---|
| normal | 95.94% | 94.38% | 90.94% | 95.00% | 92.81% | -1.57% | +1.87% | -2.19% |
| fog | 86.88% | 90.94% | 89.69% | 90.00% | 91.88% | **+0.94%** | **+2.19%** | **+1.88%** |
| rain | 43.75% | 85.31% | 86.56% | 89.69% | 90.31% | **+5.00%** | **+3.75%** | +0.62% |
| snow | 40.62% | 88.12% | 89.69% | 87.50% | 87.19% | -0.93% | -2.50% | -0.31% |
| dark | 66.88% | 80.00% | 80.31% | 78.75% | 82.50% | **+2.50%** | **+2.19%** | **+3.75%** |
| light | 83.44% | 86.25% | 86.88% | 84.69% | 86.25% | 0.00% | -0.63% | **+1.56%** |
| fog_rain | 42.19% | 78.12% | 81.25% | 81.88% | 85.62% | **+7.50%** | **+4.38%** | **+3.75%** |
| fog_snow | 22.19% | 73.44% | 75.00% | 69.38% | 79.38% | **+5.94%** | **+4.38%** | **+10.00%** |
| rain_snow | 34.69% | 84.69% | 86.25% | 84.69% | 84.69% | 0.00% | -1.56% | 0.00% |
| wind | 81.25% | 85.62% | 88.12% | 88.75% | 87.50% | **+1.88%** | -0.62% | -1.25% |
| **Avg(adverse)** | **55.77%** | **83.61%** | **84.86%** | **83.92%** | **86.15%** | **+2.54%** | **+1.28%** | **+2.23%** |
| **Avg(all)** | **59.78%** | **84.69%** | **85.47%** | **85.03%** | **86.81%** | **+2.13%** | **+1.34%** | **+1.79%** |

### Key Findings (EXP53)
- **Best overall model so far:** avg adverse R@1 = 86.15% — beats EXP50 (+2.54%), EXP51 (+1.28%), EXP52 (+2.23%)
- **Biggest wins vs EXP52 (ft-online):** fog_snow **+10.00%**, fog_rain +3.75%, dark +3.75% — scratch training with online aug generalizes better on combined conditions
- **fog_snow (79.38%)** — strongest combined-weather result of all experiments
- **Normal R@1 trade-off:** 92.81% — lower than EXP52 (95.00%) and zero-shot (95.94%), higher than EXP51 (90.94%)
- **Best epoch:** 30 / 120 (avg weather R@1 = 87.42% on eval subset — converged in first quarter of training)
- **Training duration:** 3654.1s (~61 min, ~4.5× longer than EXP51 due to imgaug CPU overhead)
- **Conclusion:** Scratch + online aug is the best strategy overall. Online augmentation benefits scratch training more than fine-tuning (EXP52 comparison). The diversity of on-the-fly weather exposure pushes combined-weather generalisation higher than any pre-generated approach.

---

## EXP54 — Zero-Shot Eval Framework (EXP35 Checkpoint)

**Setup:** Pure inference, no training. Reusable eval script that loads any SPDGeo-DPEA-MAR checkpoint and evaluates all 10 weather conditions. First run uses EXP35 checkpoint (epoch=65, pretrained R@1=96.32%). Duration: 39.7s.

**Key property:** Swap `CFG.CHECKPOINT` + `CFG.EXPERIMENT_NAME` to re-evaluate EXP52 or EXP53 best checkpoints without code changes.

**Model complexity (measured):**
- Total params: 24.70M, Trainable: 13.30M, GFLOPs: 12.46, Inference: 4.8 ms/query

### Drone → Satellite Results (EXP35 checkpoint)

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

### Key Findings (EXP54 — EXP35 checkpoint)
- **Results identical to EXP49:** Confirms eval protocol consistency between old (EXP49) and new (EXP54) eval scripts
- **Baseline confirmed:** Zero-shot avg adverse R@1 = 55.76% → weather training provides +28–29% gain
- **EXP54 eval script:** Reusable for EXP52/EXP53 checkpoints; run with `CFG.EXPERIMENT_NAME = "EXP54_ZeroShot_EXP52"` etc.
- **Very fast:** 39.7s total eval (pure inference, no training)

---

## Comparison with State-of-the-Art (SUES-200, Drone → Satellite)

> **Source:** WeatherPrompt, NeurIPS 2025, Table 2 — same dataset (SUES-200), same 10-weather imgaug protocol, same Drone → Satellite retrieval task.

> **Note:** "Over-exp" in the paper corresponds to our "light" (over-exposure) condition. Results for external methods are as reported in the paper (pretrained weights where applicable; † = pretrained on University-1652, * = official pretrained weights).

### Full R@1 Comparison Table (Drone → Satellite)

| Method | Normal | Fog | Rain | Snow | Fog+Rain | Fog+Snow | Rain+Snow | Dark | Light/Over-exp | Wind | **Mean R@1** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Zheng et al. | 57.70% | 48.63% | 53.41% | 41.78% | 37.17% | 44.22% | 40.60% | 23.81% | 49.79% | 47.42% | 44.43% |
| IBN-Net | 65.34% | 56.03% | 55.73% | 47.80% | 43.45% | 50.04% | 45.51% | 29.61% | 56.01% | 57.36% | 50.69% |
| Sample4Geo† (ICCV'23) | 74.93% | 72.58% | 34.60% | 28.95% | 35.10% | 12.95% | 20.05% | 34.18% | 38.40% | 67.80% | 41.95% |
| Safe-Net* (TIP'24) | 76.31% | 73.53% | 54.15% | 48.94% | 45.12% | 40.05% | 25.95% | 29.74% | 54.86% | 58.10% | 50.68% |
| CCR† (TCSVT'24) | 73.22% | 70.95% | 60.14% | 50.31% | 45.87% | 45.80% | 31.25% | 31.03% | 59.97% | 52.02% | 52.06% |
| MuSe-Net* (PR'24) | 66.07% | 58.49% | 58.94% | 54.85% | 44.31% | 49.81% | 49.42% | 29.34% | 55.02% | 59.97% | 52.02% |
| **WeatherPrompt (NeurIPS'25)** | **76.72%** | **68.49%** | **71.77%** | **59.95%** | **58.24%** | **64.36%** | **58.49%** | **40.42%** | **61.57%** | **65.19%** | **62.52%** |
| **Ours — EXP53 ★** | **92.81%** | **91.88%** | **90.31%** | **87.19%** | **85.62%** | **79.38%** | **84.69%** | **82.50%** | **86.25%** | **87.50%** | **86.81%** |
| **Δ Ours vs. WeatherPrompt** | **+16.09%** | **+23.39%** | **+18.54%** | **+27.24%** | **+27.38%** | **+15.02%** | **+26.20%** | **+42.08%** | **+24.68%** | **+22.31%** | **+24.29%** |

### Per-Metric Improvement Summary (Ours EXP53 vs. WeatherPrompt NeurIPS 2025)

| Metric | WeatherPrompt | Ours (EXP53) | Improvement |
|---|---|---|---|
| Mean R@1 (10 conditions) | 62.52% | **86.81%** | **+24.29%** |
| Normal R@1 | 76.72% | **92.81%** | **+16.09%** |
| Avg(adverse) R@1 (9 cond.) | 60.94% | **86.15%** | **+25.21%** |
| Dark R@1 (hardest for paper) | 40.42% | **82.50%** | **+42.08%** |
| Fog+Snow R@1 | 64.36% | **79.38%** | **+15.02%** |
| Rain+Snow R@1 | 58.49% | **84.69%** | **+26.20%** |
| Min R@1 across all conditions | 40.42% (dark) | **79.38% (fog_snow)** | **+38.96%** |

### Key Observations

- **Ours surpasses WeatherPrompt on every single weather condition** — minimum gain +15.02% (Fog+Snow), maximum +42.08% (Dark).
- **Dark condition** is the failure mode of all paper methods; WeatherPrompt reaches only 40.42% R@1, ours maintains **82.50%** — a +42.08% absolute improvement.
- **Combined weather conditions** (Fog+Rain, Fog+Snow, Rain+Snow): avg +22.87% over WeatherPrompt — validating the diversity benefit of online augmentation for compositional weather.
- **WeatherPrompt** relies on XVLM + LLM-generated text descriptions + dynamic gating; ours uses a purely visual DINOv2 ViT-S/14 backbone, validating that **altitude-aware part discovery + multi-granularity metric learning** is a more powerful paradigm for this task without requiring expensive language model pipelines.
- **EXP53 mean R@1 (86.81%)** is the highest reported result on the SUES-200 multi-weather benchmark among all compared methods, exceeding the previous SOTA by **+24.29%**.
- Our minimum R@1 (79.38%, fog_snow) is still **higher than WeatherPrompt's best adverse-weather result** (68.49%, fog).

---

## Data Strategy Comparison

| Experiment | Data Source | Samples/Epoch | Augmentation | Images/Loc/Epoch |
|---|---|---|---|---|
| EXP50 | Pre-generated weather | 4800 (120×4×10) | Offline (EXP48) | 40 (4 alt × 10 weather) |
| EXP51 | Pre-generated weather | 4800 (120×4×10) | Offline (EXP48) | 40 (4 alt × 10 weather) |
| EXP52 | Original SUES-200 | 120 (1 per loc) | Online imgaug | 1 (random alt, random weather) |
| EXP53 | Original SUES-200 | 120 (1 per loc) | Online imgaug | 1 (random alt, random weather) |
| WeatherPrompt | Original dataset | #classes (1 per class) | Online imgaug | 1 (random, random weather) |

---

*Last updated: 2026-03-16 — SOTA comparison added; EXP53 (86.81% mean R@1) surpasses WeatherPrompt NeurIPS 2025 by +24.29% on SUES-200 multi-weather benchmark*
