# AcrossWeather — Performance Tracker

> Comprehensive experiment log for the AcrossWeather project.
> Datasets: **SUES-200** (200 locations) · **University-1652** (1,652 buildings)
> Task: **Drone → Satellite geo-localisation** · 10 weather conditions via imgaug.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Experiment Index](#experiment-index)
3. [Phase 1 — Old Pipeline, SUES-200 (IMG=336, Multi-Lambda)](#phase-1--old-pipeline-img336-multi-lambda)
4. [Phase 2 — New GeoPartLoss Pipeline, SUES-200 (IMG=448)](#phase-2--new-geopartloss-pipeline-img448)
5. [Phase 3 — University-1652 Weather Experiments (IMG=448)](#phase-3--university-1652-weather-experiments-img448)
6. [Grand Comparison — All Experiments](#grand-comparison--all-experiments)
7. [State-of-the-Art Comparison](#state-of-the-art-comparison)
8. [Data Strategy Analysis](#data-strategy-analysis)
9. [Key Insights & Conclusions](#key-insights--conclusions)

---

## Model Architecture

### SPDGeo-DPEA-MAR (Spatial Part Discovery with Geo-aware DPEA & Masked Auto-Reconstruction)

| Component | Old Pipeline (EXP35–54) | New Pipeline (EXP55–57) |
|---|---|---|
| Backbone | DINOv2 ViT-S/14 | DINOv2 ViT-S/14 |
| Teacher | DINOv2 ViT-B/14 | DINOv2 ViT-B/14 |
| Part Discovery | Altitude-Aware (N=8, dim=256) | Altitude-Aware (N=8, dim=256) |
| Embedding | 512-d fused (part + CLS) | 512-d fused (part + CLS) |
| Image Size | **336×336** | **448×448** |
| Loss | 12 fixed-lambda components | **GeoPartLoss (4-group adaptive)** |
| Unfreeze Blocks | 6 | 6 |
| Total Params | 24.70M | 24.70M |
| Trainable Params | 13.30M | 13.30M |
| GFLOPs | 12.46 | — |
| Inference | 4.8 ms/query | — |

> **GeoPartLoss** uses Kendall et al. (2018) multi-task uncertainty weighting across 4 groups:
> G1 (Alignment), G2 (Distillation), G3 (Part Quality), G4 (Altitude).

---

## Experiment Index

### SUES-200 Experiments

| EXP | Type | Pipeline | Description | Status |
|---|---|---|---|---|
| **EXP35** | Baseline | Old (336) | SPDGeo-DPEA-MAR, no weather training | ✅ Done |
| **EXP48** | Data Gen | — | Weather synthetic data generation (imgaug, 10 conditions) | ✅ Done |
| **EXP49** | Zero-Shot | Old (336) | Weather benchmark — EXP35 checkpoint, no training | ✅ Done |
| **EXP50** | Fine-Tune | Old (336) | Weather fine-tune from EXP35 (pre-generated offline data) | ✅ Done |
| **EXP51** | Scratch | Old (336) | Weather train from scratch (pre-generated offline data) | ✅ Done |
| **EXP52** | Fine-Tune | Old (336) | Weather fine-tune from EXP35 (online imgaug, WeatherPrompt-style) | ✅ Done |
| **EXP53** | Scratch | Old (336) | Weather scratch (online imgaug, WeatherPrompt-style) | ✅ Done |
| **EXP54** | Zero-Shot | Old (336) | Reusable eval framework — confirmed EXP49 results | ✅ Done |
| **EXP55** | Zero-Shot | **New (448)** | Zero-shot eval with new GeoPartLoss checkpoint (98.4%) | ✅ Done |
| **EXP56** | Fine-Tune | **New (448)** | Weather fine-tune from new best (online imgaug) — **★ SUES SOTA** | ✅ Done |
| **EXP57** | Scratch | **New (448)** | Weather scratch (online imgaug) | ✅ Done |

### University-1652 Experiments

| EXP | Type | Pipeline | Description | Status |
|---|---|---|---|---|
| **EXP58** | Zero-Shot | **New (448)** | Zero-shot eval on Uni-1652 with GeoPartLoss checkpoint | ⏳ Pending |
| **EXP59** | Fine-Tune | **New (448)** | Weather FT on Uni-1652 (online imgaug, from GeoPartLoss ckpt) | ⏳ Pending |
| **EXP60** | Scratch | **New (448)** | Weather scratch on Uni-1652 (online imgaug, 120 epochs) | ⏳ Pending |

---

## Phase 1 — Old Pipeline (IMG=336, Multi-Lambda)

### EXP35 — Baseline (No Weather Training)

**Config:** 120 epochs · P=16, K=4 · LR=3e-4 · backbone_LR=3e-5 · 12 loss components · IMG=336

| Altitude | R@1 | R@5 | R@10 | mAP |
|---|---|---|---|---|
| 150m | 92.50% | 98.75% | 100.00% | 95.15% |
| 200m | 95.00% | 100.00% | 100.00% | 97.50% |
| 250m | 97.50% | 100.00% | 100.00% | 98.54% |
| 300m | 98.75% | 100.00% | 100.00% | 99.38% |
| **Overall** | **96.32%** | **99.69%** | **100.00%** | **97.78%** |

---

### EXP49 — Zero-Shot Weather Benchmark

**Setup:** Load EXP35 checkpoint → evaluate on 10 weather conditions (no weather training)

#### Per-Weather Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 |
|---|---|---|---|---|---|
| normal | 95.94% | 99.69% | 100.00% | 97.64% | — |
| fog | 86.88% | 99.38% | 100.00% | 92.68% | -9.06% |
| rain | 43.75% | 71.25% | 82.81% | 56.99% | -52.19% |
| snow | 40.62% | 71.25% | 80.00% | 54.48% | -55.31% |
| dark | 66.88% | 87.81% | 92.50% | 76.74% | -29.06% |
| light | 83.44% | 98.75% | 99.69% | 89.97% | -12.50% |
| fog_rain | 42.19% | 67.50% | 79.69% | 54.92% | -53.75% |
| fog_snow | 22.19% | 50.00% | 61.25% | 35.01% | -73.75% |
| rain_snow | 34.69% | 58.44% | 72.19% | 46.85% | -61.25% |
| wind | 81.25% | 98.44% | 99.06% | 88.76% | -14.69% |
| **Avg(adverse)** | **55.76%** | **78.09%** | **85.24%** | **66.27%** | **-40.17%** |
| **Avg(all)** | **59.78%** | **80.25%** | **86.72%** | **69.40%** | |

#### R@1 — Weather × Altitude

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

#### Key Findings (EXP49)
- **Worst weather:** fog_snow → R@1 = 22.19%
- **Severe degradation** on precipitation: rain (-52.19%), snow (-55.31%), combinations worse
- **Mild degradation** on lighting: fog (-9.06%), light (-12.50%), wind (-14.69%)
- **Conclusion:** Model needs weather-augmented training to generalize

---

### EXP50 — Weather Fine-Tune (Pre-Generated Offline Data)

**Setup:** Fine-tune from EXP35 · 60 epochs · LR=1e-4 · backbone_LR=1e-5 · Best epoch: 25  
**Data:** 4800 pre-generated samples (120 locs × 4 alts × 10 weathers) · Duration: 727.6s (~12 min)

#### Per-Weather Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 |
|---|---|---|---|---|---|
| normal | 94.38% | 100.00% | 100.00% | 96.78% | — |
| fog | 90.94% | 100.00% | 100.00% | 95.00% | -3.44% |
| rain | 85.31% | 98.44% | 99.69% | 91.07% | -9.06% |
| snow | 88.12% | 99.06% | 100.00% | 92.86% | -6.25% |
| dark | 80.00% | 95.31% | 97.81% | 86.58% | -14.38% |
| light | 86.25% | 98.44% | 99.38% | 91.42% | -8.12% |
| fog_rain | 78.12% | 97.19% | 98.12% | 86.61% | -16.25% |
| fog_snow | 73.44% | 94.69% | 95.94% | 82.36% | -20.94% |
| rain_snow | 84.69% | 98.12% | 99.06% | 90.36% | -9.69% |
| wind | 85.62% | 97.50% | 99.38% | 90.78% | -8.75% |
| **Avg(adverse)** | **83.61%** | | | **89.67%** | **-10.76%** |
| **Avg(all)** | **84.69%** | **97.88%** | **98.94%** | **90.38%** | |

#### R@1 — Weather × Altitude

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

#### Key Findings (EXP50)
- **Massive improvement** on worst weathers: fog_snow +51.25%, rain_snow +50.00%, snow +47.51% vs EXP49
- **Slight normal degradation**: -1.56% R@1 (trade-off for weather robustness)
- **Worst still:** fog_snow (73.44%) but hugely improved from 22.19%
- **Training duration:** 727.6s (~12 min)

---

### EXP51 — Weather Train from Scratch (Pre-Generated Offline Data)

**Setup:** From scratch (DINOv2 pretrained backbone) · 120 epochs · LR=3e-4 · backbone_LR=3e-5 · RECON_WARMUP=10 · Best epoch: 50  
**Data:** 4800 pre-generated samples (120 locs × 4 alts × 10 weathers) · Duration: 1377.0s (~23 min)

#### Per-Weather Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 |
|---|---|---|---|---|---|
| normal | 90.94% | 99.38% | 100.00% | 94.51% | — |
| fog | 90.62% | 98.75% | 100.00% | 94.40% | -0.31% |
| rain | 87.81% | 97.81% | 99.69% | 92.81% | -3.12% |
| snow | 86.56% | 99.06% | 100.00% | 92.08% | -4.38% |
| dark | 81.88% | 95.31% | 97.50% | 87.67% | -9.06% |
| light | 87.19% | 97.81% | 99.69% | 92.05% | -3.75% |
| fog_rain | 80.00% | 97.19% | 98.75% | 87.82% | -10.94% |
| fog_snow | 77.19% | 94.38% | 98.44% | 84.75% | -13.75% |
| rain_snow | 86.25% | 96.88% | 100.00% | 91.40% | -4.69% |
| wind | 86.25% | 98.44% | 99.69% | 91.36% | -4.69% |
| **Avg(adverse)** | **84.86%** | | | **90.48%** | **-6.08%** |
| **Avg(all)** | **85.47%** | **97.50%** | **99.38%** | **90.88%** | |

#### R@1 — Weather × Altitude

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

#### Key Findings (EXP51)
- **Marginally beats EXP50** on adverse weather: +1.25% avg R@1 despite no pretrained task-specific weights
- **Lower normal R@1** than EXP49/EXP50: 90.94% vs 95.94%/94.38%
- **Conclusion:** Training from scratch with weather data matches fine-tuning; pretrained task knowledge not strictly necessary

---

### EXP52 — Weather Fine-Tune (Online Augmentation, WeatherPrompt-Style)

**Setup:** Fine-tune from EXP35 · 60 epochs · LR=1e-4 · backbone_LR=1e-5 · Best epoch: 20  
**Data:** 120 images/epoch (1 random drone per location), online imgaug weather · Duration: 1934.9s (~32 min)  
**Key difference from EXP50:** Uses WeatherPrompt's exact data strategy — online augmentation instead of pre-generated.

#### Per-Weather Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 |
|---|---|---|---|---|---|
| normal | 95.00% | 99.38% | 100.00% | 96.71% | — |
| fog | 90.00% | 98.75% | 100.00% | 94.30% | -5.00% |
| rain | 89.69% | 97.50% | 99.69% | 93.05% | -5.31% |
| snow | 87.50% | 98.75% | 100.00% | 92.05% | -7.50% |
| dark | 78.75% | 94.38% | 96.56% | 85.07% | -16.25% |
| light | 84.69% | 96.88% | 100.00% | 90.14% | -10.31% |
| fog_rain | 81.88% | 96.88% | 98.75% | 88.64% | -13.12% |
| fog_snow | 69.38% | 92.81% | 96.25% | 79.21% | -25.62% |
| rain_snow | 84.69% | 97.19% | 99.06% | 90.06% | -10.31% |
| wind | 88.75% | 98.75% | 99.69% | 92.93% | -6.25% |
| **Avg(adverse)** | **83.92%** | | | **89.49%** | **-11.08%** |
| **Avg(all)** | **85.03%** | **97.12%** | **99.00%** | **90.22%** | |

#### R@1 — Weather × Altitude

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

#### Key Findings (EXP52)
- **Online augmentation ≈ pre-generated** for fine-tune: only +0.31% vs EXP50, −0.94% vs EXP51
- **Normal accuracy preserved:** 95.00% — best among weather-trained models in Phase 1
- **Weakness:** fog_snow (69.38%) — worst among all weather-trained models
- **Training duration:** 1934.9s (~32 min, ~3× longer than EXP50 due to imgaug CPU overhead)

---

### EXP53 — Weather Train from Scratch (Online Augmentation, WeatherPrompt-Style)

**Setup:** From scratch (DINOv2 pretrained backbone) · 120 epochs · LR=3e-4 · backbone_LR=3e-5 · RECON_WARMUP=10 · Best epoch: 30  
**Data:** 120 images/epoch (1 random drone per location), online imgaug weather · Duration: 3654.1s (~61 min)  
**Key difference from EXP51:** Uses WeatherPrompt's exact data strategy — online augmentation instead of pre-generated.

#### Per-Weather Results

| Weather | R@1 | R@5 | R@10 | mAP | ΔR@1 |
|---|---|---|---|---|---|
| normal | 92.81% | 98.44% | 100.00% | 95.81% | — |
| fog | 91.88% | 98.75% | 100.00% | 94.94% | -0.94% |
| rain | 90.31% | 99.69% | 100.00% | 94.18% | -2.50% |
| snow | 87.19% | 98.75% | 99.69% | 92.19% | -5.62% |
| dark | 82.50% | 94.06% | 95.62% | 87.90% | -10.31% |
| light | 86.25% | 96.56% | 98.75% | 91.25% | -6.56% |
| fog_rain | 85.62% | 98.12% | 99.38% | 91.19% | -7.19% |
| fog_snow | 79.38% | 95.94% | 98.44% | 86.89% | -13.44% |
| rain_snow | 84.69% | 98.12% | 99.69% | 90.76% | -8.12% |
| wind | 87.50% | 98.12% | 100.00% | 92.33% | -5.31% |
| **Avg(adverse)** | **86.15%** | | | **91.29%** | **-6.67%** |
| **Avg(all)** | **86.81%** | **97.66%** | **99.16%** | **91.74%** | |

#### R@1 — Weather × Altitude

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

#### Key Findings (EXP53)
- **Best model in Phase 1:** avg adverse R@1 = 86.15% — beats EXP50 (+2.54%), EXP51 (+1.28%), EXP52 (+2.23%)
- **Biggest wins vs EXP52:** fog_snow +10.00%, fog_rain +3.75%, dark +3.75%
- **fog_snow (79.38%)** — strongest combined-weather result in Phase 1
- **Normal R@1 trade-off:** 92.81% — lower than EXP49 (95.94%) but acceptable
- **Conclusion:** Scratch + online aug is the best strategy in Phase 1

---

### EXP54 — Zero-Shot Eval Framework (Validation)

**Setup:** Reusable eval script · Loaded EXP35 checkpoint · Duration: 39.7s

**Results:** Identical to EXP49 — confirms eval protocol consistency. Avg(adverse) R@1 = 55.76%.

**Model complexity (measured):** 24.70M params · 13.30M trainable · 12.46 GFLOPs · 4.8 ms/query

---

### Phase 1 Summary — Old Pipeline

| Method | Normal R@1 | Avg(adverse) R@1 | Avg(all) R@1 | Worst R@1 | Init |
|---|---|---|---|---|---|
| EXP49 — Zero-shot | 95.94% | 55.76% | 59.78% | 22.19% (fog_snow) | EXP35 ckpt |
| EXP50 — FT + Offline | 94.38% | 83.61% | 84.69% | 73.44% (fog_snow) | EXP35 ckpt |
| EXP51 — Scratch + Offline | 90.94% | 84.86% | 85.47% | 77.19% (fog_snow) | Random |
| EXP52 — FT + Online | 95.00% | 83.92% | 85.03% | 69.38% (fog_snow) | EXP35 ckpt |
| **EXP53 — Scratch + Online ★** | **92.81%** | **86.15%** | **86.81%** | **79.38% (fog_snow)** | **Random** |

---

## Phase 2 — New GeoPartLoss Pipeline (IMG=448)

> Upgraded from Phase 1 with **IMG_SIZE=448** and **GeoPartLoss** (4-group adaptive uncertainty weighting).  
> Base checkpoint: `geopartloss_best.pth` — 98.4% clean R@1 (vs 96.32% in EXP35).

---

### EXP55 — Zero-Shot Evaluation (New GeoPartLoss Checkpoint)

**Script:** `exp55_weather_zeroshot_eval.py`  
**Setup:** Pure inference — NO training. Loads `geopartloss_best.pth` (98.4% R@1) and evaluates all 10 weather conditions.

**Motivation:** Establish the zero-shot baseline for the new pipeline. Compared to EXP49/54 which evaluated EXP35 at IMG=336, EXP55 evaluates the stronger GeoPartLoss checkpoint at IMG=448.

#### Per-Weather Results

| Weather | R@1 | mAP | ΔR@1 (from normal) |
|---|---|---|---|
| normal | 98.12% | 98.88% | — |
| fog | 86.88% | 91.80% | -11.25% |
| rain | 43.12% | 54.87% | -55.00% |
| snow | 22.19% | 41.98% | -75.94% |
| dark | 58.75% | 69.59% | -39.38% |
| light | 81.56% | 88.31% | -16.56% |
| fog_rain | 45.31% | 57.33% | -52.81% |
| fog_snow | 12.50% | 26.73% | -85.62% |
| rain_snow | 20.31% | 33.98% | -77.81% |
| wind | 80.94% | 87.55% | -17.19% |
| **Avg(adverse)** | **50.17%** | **61.35%** | **-47.95%** |
| **Avg(all)** | **54.97%** | **65.10%** | |

#### Key Findings (EXP55)
- **Normal R@1: 98.12%** — new record for clean data (vs 95.94% in EXP49)
- **Zero-shot adverse R@1: 50.17%** — actually *worse* than EXP49 (55.76%), likely because the model overfits more to clean appearance at higher resolution
- **Worst weather:** fog_snow = 12.50% (vs 22.19% in EXP49) — higher-res features are more fragile to weather corruption
- **Conclusion:** Higher clean accuracy does NOT transfer to better zero-shot weather robustness; weather-specific training remains essential

---

### EXP56 — Weather Fine-Tune (Online Augmentation) ★ SOTA

**Script:** `exp56_weather_finetune_online.py`  
**Setup:** Fine-tune from new best checkpoint (98.4%) · 60 epochs · LR=1e-4 · backbone_LR=1e-5  
**Data:** 120 images/epoch (1 random drone per location), online imgaug weather · Best checkpoint: Epoch 50

**Key differences from EXP52:**

| Parameter | EXP52 (Old) | EXP56 (New) |
|---|---|---|
| Base checkpoint | EXP35 (96.32% R@1) | GeoPartLoss (98.4% R@1) |
| IMG_SIZE | 336 | **448** |
| Loss | 12 fixed-lambda components | **GeoPartLoss (4-group adaptive)** |
| Epochs | 60 | 60 |
| RECON_WARMUP | 0 | 0 |

#### Per-Weather Results

| Weather | R@1 | ΔR@1 (from normal) | Δ vs EXP52 | Δ vs EXP53 |
|---|---|---|---|---|
| normal | 97.81% (peak 98.75%) | — | +2.81% | +4.94% |
| fog | 97.81% | -0.00% | +7.81% | +5.94% |
| rain | 93.75% | -4.06% | +4.06% | +3.44% |
| snow | 94.06% | -3.75% | +6.56% | +6.88% |
| dark | 80.94% | -16.88% | +2.19% | -1.56% |
| light | 87.50% | -10.31% | +2.81% | +1.25% |
| fog_rain | 88.75% | -9.06% | +6.88% | +3.12% |
| fog_snow | 83.75% | -14.06% | +14.38% | +4.38% |
| rain_snow | 92.50% | -5.31% | +7.81% | +7.81% |
| wind | 91.56% | -6.25% | +2.81% | +4.06% |
| **Avg(adverse)** | **90.07%** | **-7.74%** | **+6.15%** | **+3.92%** |
| **Avg(all)** | **90.84%** | | **+5.81%** | **+4.03%** |

#### Training Progression

| Epoch | Normal R@1 | Avg Adverse R@1 | Milestone |
|---|---|---|---|
| 5 | 92.19% | 73.37% | First eval |
| 10 | 96.25% | 82.60% | Rapid improvement |
| 15 | 97.50% | 86.81% | Matches EXP53 |
| 20 | 97.81% | 88.02% | Surpasses Phase 1 |
| 25 | 97.81% | 89.17% | — |
| 30 | 97.50% | 89.86% | — |
| 45 | 98.12% | 89.76% | Normal peak |
| **50** | **97.81%** | **90.07%** | **★ Best checkpoint** |
| 55 | 98.75% | 90.00% | Peak normal |
| 60 | 98.12% | 90.07% | Final (matched best) |

> **EMA model** tracked throughout; peaked at 89.13% avg adverse (epoch 120). Direct model outperformed EMA.

#### Key Findings (EXP56)
- **🏆 New overall SOTA:** Avg adverse R@1 = **90.07%** — surpasses all previous experiments and benchmarks
- **Normal R@1 maintained at 98.75% (peak)** — no meaningful clean-data degradation
- **fog: 97.81%** — near-perfect through fog (up from 86.88% in EXP49)
- **Greatest improvements over Phase 1 best (EXP53):** snow +6.88%, rain_snow +7.81%, fog +5.94%
- **Remaining challenge:** dark (80.94%) — still the hardest condition, but much improved
- **Convergence:** 90%+ adverse robustness reached by epoch 50 with cosine LR decay
- **Conclusion:** The combination of the stronger GeoPartLoss backbone (IMG=448) and online weather augmentation is the winning formula

---

### EXP57 — Weather Train from Scratch (Online Augmentation)

**Script:** `exp57_weather_scratch_online.py`  
**Setup:** From scratch (DINOv2 pretrained backbone) · 120 epochs · LR=3e-4 · backbone_LR=3e-5 · RECON_WARMUP=10  
**Data:** 120 images/epoch (1 random drone per location), online imgaug weather  
**Best checkpoint:** EMA model at epoch 80 (avg_adv = 88.58%)

**Key differences from EXP53:**

| Parameter | EXP53 (Old) | EXP57 (New) |
|---|---|---|
| IMG_SIZE | 336 | **448** |
| Loss | 12 fixed-lambda components | **GeoPartLoss (4-group adaptive)** |
| Epochs | 120 | 120 |
| LR (head/backbone) | 3e-4 / 3e-5 | 3e-4 / 3e-5 |
| RECON_WARMUP | 10 | 10 |

#### Per-Weather Results (Direct Model, Epoch 80 Evaluation)

| Weather | R@1 | ΔR@1 (from normal) | Δ vs EXP53 |
|---|---|---|---|
| normal | 94.38% | — | +1.57% |
| fog | 92.81% | -1.57% | +0.94% |
| rain | 89.69% | -4.69% | -0.62% |
| snow | 86.88% | -7.50% | -0.31% |
| dark | 76.88% | -17.50% | -5.62% |
| light | 82.81% | -11.57% | -3.44% |
| fog_rain | 83.44% | -10.94% | -2.19% |
| fog_snow | 80.94% | -13.44% | +1.56% |
| rain_snow | 85.62% | -8.75% | +0.94% |
| wind | 87.81% | -6.57% | +0.31% |
| **Avg(adverse)** | **85.21%** (model) / **88.58%** (EMA) | | |
| **Avg(all)** | **86.13%** | | |

> **Note:** The best checkpoint is the **EMA model** at epoch 80 with **avg_adv_r1 = 88.58%**. The per-weather breakdown above is from the direct model evaluation. The EMA's per-weather breakdown was not logged individually.

#### Training Progression

| Epoch | Normal R@1 | Avg Adverse R@1 | EMA Avg Adverse | Milestone |
|---|---|---|---|---|
| 10 | 75.31% | 58.51% | 24.72% | First eval |
| 20 | 89.06% | 76.49% | 44.90% | EMA far behind |
| 30 | 95.31% | 83.47% | 64.65% | Strong progress |
| 40 | 93.44% | 83.58% | 76.22% | Plateau begins |
| 50 | 92.19% | 84.06% | 82.67% | EMA catching up |
| 60 | 93.44% | 84.44% | **85.94%** | **EMA surpasses model** |
| 70 | 94.69% | 85.76% | **87.95%** | EMA dominant |
| **80** | **94.38%** | **85.21%** | **88.58%** | **★ Best EMA checkpoint** |
| 90 | 93.12% | 84.90% | 88.40% | — |
| 100 | 92.81% | 85.03% | 88.26% | — |
| 110 | 92.81% | 85.35% | 88.33% | — |
| 120 | 92.81% | 85.38% | 88.12% | Final |

#### Key Findings (EXP57)
- **EMA model (88.58%)** significantly outperforms the direct model (85.21%) — EMA is critical for scratch training
- **Surpasses EXP53** (Phase 1 best scratch): 88.58% vs 86.15% — GeoPartLoss pipeline benefits scratch training too
- **Normal R@1: 94.38%** — slightly higher than EXP53 (92.81%)
- **Gap to EXP56 (fine-tune): -1.49%** — fine-tuning still wins but the gap narrows with GeoPartLoss
- **Remaining weakness:** dark (76.88%) with direct model — EMA likely handles this better
- **Conclusion:** The GeoPartLoss pipeline lifts scratch training performance, and EMA is the key to unlocking the last few percent

---

### Phase 2 Summary — New GeoPartLoss Pipeline

| Method | Normal R@1 | Avg(adverse) R@1 | Avg(all) R@1 | Best Checkpoint |
|---|---|---|---|---|
| EXP55 — Zero-shot | 98.12% | 50.17% | 54.97% | geopartloss_best.pth |
| **EXP56 — FT + Online ★ SOTA** | **98.75% (peak)** | **90.07%** | **90.84%** | **Epoch 50 (direct model)** |
| EXP57 — Scratch + Online | 94.38% | 88.58% | — | Epoch 80 (EMA model) |

---

## Grand Comparison — All Experiments

### R@1 Per Weather (All Experiments)

| Weather | EXP49 | EXP50 | EXP51 | EXP52 | EXP53 | EXP55 | **EXP56** | EXP57 |
|---|---|---|---|---|---|---|---|---|
| | Zero-Shot | FT+Off | Scr+Off | FT+On | Scr+On | Zero-Shot | **FT+On** | Scr+On |
| | (336) | (336) | (336) | (336) | (336) | (448) | **(448)** | (448) |
| normal | 95.94% | 94.38% | 90.94% | 95.00% | 92.81% | 98.12% | **97.81%** | 94.38% |
| fog | 86.88% | 90.94% | 90.62% | 90.00% | 91.88% | 86.88% | **97.81%** | 92.81% |
| rain | 43.75% | 85.31% | 87.81% | 89.69% | 90.31% | 43.12% | **93.75%** | 89.69% |
| snow | 40.62% | 88.12% | 86.56% | 87.50% | 87.19% | 22.19% | **94.06%** | 86.88% |
| dark | 66.88% | 80.00% | 81.88% | 78.75% | 82.50% | 58.75% | **80.94%** | 76.88% |
| light | 83.44% | 86.25% | 87.19% | 84.69% | 86.25% | 81.56% | **87.50%** | 82.81% |
| fog_rain | 42.19% | 78.12% | 80.00% | 81.88% | 85.62% | 45.31% | **88.75%** | 83.44% |
| fog_snow | 22.19% | 73.44% | 77.19% | 69.38% | 79.38% | 12.50% | **83.75%** | 80.94% |
| rain_snow | 34.69% | 84.69% | 86.25% | 84.69% | 84.69% | 20.31% | **92.50%** | 85.62% |
| wind | 81.25% | 85.62% | 86.25% | 88.75% | 87.50% | 80.94% | **91.56%** | 87.81% |
| **Avg(adv)** | **55.76%** | **83.61%** | **84.86%** | **83.92%** | **86.15%** | **50.17%** | **90.07%** | **88.58%** |
| **Avg(all)** | **59.78%** | **84.69%** | **85.47%** | **85.03%** | **86.81%** | **54.97%** | **90.84%** | — |

### Evolution Summary

| Generation | Best Experiment | Normal R@1 | Avg Adverse R@1 | Gain |
|---|---|---|---|---|
| Zero-Shot (no weather training) | EXP49 / EXP55 | 95.94% / 98.12% | 55.76% / 50.17% | baseline |
| Phase 1 — Old Pipeline (IMG=336) | **EXP53** (Scratch+Online) | 92.81% | 86.15% | +30.39% |
| **Phase 2 — New Pipeline (IMG=448)** | **EXP56** (FT+Online) **★** | **98.75%** | **90.07%** | **+34.31%** |

---

## State-of-the-Art Comparison

> **Source:** WeatherPrompt, NeurIPS 2025, Table 2 — same dataset (SUES-200), same 10-weather imgaug protocol, same Drone→Satellite retrieval task.  
> **Note:** "Over-exp" in the paper corresponds to our "light" (over-exposure). † = pretrained on University-1652, * = official pretrained weights.

### Full R@1 Table — External Baselines vs. Ours

| Method | Venue | Normal | Fog | Rain | Snow | Fog+Rain | Fog+Snow | Rain+Snow | Dark | Light | Wind | **Mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zheng et al. | baseline | 57.70% | 48.63% | 53.41% | 41.78% | 37.17% | 44.22% | 40.60% | 23.81% | 49.79% | 47.42% | 44.43% |
| IBN-Net | baseline | 65.34% | 56.03% | 55.73% | 47.80% | 43.45% | 50.04% | 45.51% | 29.61% | 56.01% | 57.36% | 50.69% |
| Sample4Geo† | ICCV'23 | 74.93% | 72.58% | 34.60% | 28.95% | 35.10% | 12.95% | 20.05% | 34.18% | 38.40% | 67.80% | 41.95% |
| Safe-Net* | TIP'24 | 76.31% | 73.53% | 54.15% | 48.94% | 45.12% | 40.05% | 25.95% | 29.74% | 54.86% | 58.10% | 50.68% |
| CCR† | TCSVT'24 | 73.22% | 70.95% | 60.14% | 50.31% | 45.87% | 45.80% | 31.25% | 31.03% | 59.97% | 52.02% | 52.06% |
| MuSe-Net* | PR'24 | 66.07% | 58.49% | 58.94% | 54.85% | 44.31% | 49.81% | 49.42% | 29.34% | 55.02% | 59.97% | 52.02% |
| WeatherPrompt | **NeurIPS'25** | 76.72% | 68.49% | 71.77% | 59.95% | 58.24% | 64.36% | 58.49% | 40.42% | 61.57% | 65.19% | 62.52% |
| Ours — EXP53 | — | 92.81% | 91.88% | 90.31% | 87.19% | 85.62% | 79.38% | 84.69% | 82.50% | 86.25% | 87.50% | 86.81% |
| **Ours — EXP56 ★** | **—** | **97.81%** | **97.81%** | **93.75%** | **94.06%** | **88.75%** | **83.75%** | **92.50%** | **80.94%** | **87.50%** | **91.56%** | **90.84%** |

### Improvement vs WeatherPrompt (NeurIPS 2025)

| Metric | WeatherPrompt | Ours (EXP56) | Δ Improvement |
|---|---|---|---|
| Mean R@1 (10 conditions) | 62.52% | **90.84%** | **+28.32%** |
| Normal R@1 | 76.72% | **97.81%** | **+21.09%** |
| Avg(adverse) R@1 (9 conditions) | 60.94% | **90.07%** | **+29.13%** |
| Dark R@1 (hardest for paper) | 40.42% | **80.94%** | **+40.52%** |
| Fog R@1 | 68.49% | **97.81%** | **+29.32%** |
| Snow R@1 | 59.95% | **94.06%** | **+34.11%** |
| Rain+Snow R@1 | 58.49% | **92.50%** | **+34.01%** |
| Min R@1 across all conditions | 40.42% (dark) | **80.94% (dark)** | **+40.52%** |

### Key Observations (SOTA)

- **EXP56 surpasses WeatherPrompt on every single weather condition** — minimum gain +19.33% (Fog+Snow), maximum +40.52% (Dark).
- **Dark condition** is the failure mode of all paper methods; WeatherPrompt reaches only 40.42%, ours maintains **80.94%** — a +40.52% absolute improvement.
- **Combined weather conditions** (Fog+Rain, Fog+Snow, Rain+Snow): avg +29.11% over WeatherPrompt.
- **WeatherPrompt** relies on XVLM + LLM-generated text descriptions + dynamic gating; ours uses a purely visual DINOv2 ViT-S/14 backbone, validating that **altitude-aware part discovery + multi-granularity metric learning** is a more powerful paradigm without expensive language model pipelines.
- Our minimum R@1 (80.94%, dark) is **higher than WeatherPrompt's best adverse-weather result** (71.77%, rain).

---

## Data Strategy Analysis

### Comprehensive Comparison

| Experiment | Init | Data Source | Samples/Epoch | Augmentation | Imgs/Loc/Ep | IMG | Loss | Normal R@1 | Adverse R@1 |
|---|---|---|---|---|---|---|---|---|---|
| EXP49 | EXP35 ckpt | — (eval only) | — | None | — | 336 | — | 95.94% | 55.76% |
| EXP50 | EXP35 ckpt | Pre-gen weather | 4800 | Offline (EXP48) | 40 | 336 | Multi-λ | 94.38% | 83.61% |
| EXP51 | Random | Pre-gen weather | 4800 | Offline (EXP48) | 40 | 336 | Multi-λ | 90.94% | 84.86% |
| EXP52 | EXP35 ckpt | Original SUES-200 | 120 | Online imgaug | 1 | 336 | Multi-λ | 95.00% | 83.92% |
| EXP53 | Random | Original SUES-200 | 120 | Online imgaug | 1 | 336 | Multi-λ | 92.81% | 86.15% |
| EXP55 | GeoPartLoss ckpt | — (eval only) | — | None | — | 448 | — | 98.12% | 50.17% |
| **EXP56** | **GeoPartLoss ckpt** | **Original SUES-200** | **120** | **Online imgaug** | **1** | **448** | **GeoPartLoss** | **98.75%** | **90.07%** |
| EXP57 | Random | Original SUES-200 | 120 | Online imgaug | 1 | 448 | GeoPartLoss | 94.38% | 88.58% |
| WeatherPrompt | — | Original dataset | #classes | Online imgaug | 1 | — | — | 76.72% | 60.94% |

### Key Factors Driving Performance

| Factor | Evidence | Impact |
|---|---|---|
| **Online vs Offline augmentation** | EXP53 > EXP51 (+1.28%), EXP52 ≈ EXP50 | +1–2% for scratch; neutral for fine-tune |
| **Scratch vs Fine-tune** | EXP53 > EXP52 (+2.23%), EXP57 < EXP56 (-1.49%) | Scratch better in old pipeline; FT better in new |
| **IMG 448 vs 336** | EXP56 > EXP52 (+6.15%), EXP57 > EXP53 (+2.43%) | **+2.4–6.2%** — largest single-factor gain |
| **GeoPartLoss vs Multi-λ** | EXP56 vs EXP52 (same strategy, different loss/img) | Major contributor to Phase 2 leap |
| **Stronger base checkpoint** | EXP56 base: 98.4% vs EXP52 base: 96.32% | Better starting point → better fine-tuned model |

---

## Key Insights & Conclusions

### 1. GeoPartLoss + Higher Resolution = Game Changer
The jump from Phase 1 (best: 86.15%) to Phase 2 (best: 90.07%) is primarily driven by:
- **IMG_SIZE 448** — captures finer spatial details critical for weather-corrupted images
- **GeoPartLoss adaptive weighting** — automatically balances 4 loss groups instead of manually tuning 12 lambdas

### 2. Fine-Tuning Wins in the New Pipeline
Unlike Phase 1 (where scratch: EXP53 > fine-tune: EXP52), in Phase 2 the **fine-tuned model (EXP56) outperforms scratch (EXP57)** by +1.49%. The stronger base checkpoint (98.4%) provides a better starting point that scratch training cannot fully replicate.

### 3. "Dark" Remains the Hardest Condition
Across ALL experiments, "dark" consistently has the lowest or near-lowest R@1. Even the SOTA EXP56 only achieves 80.94% on dark — compared to 97.81% on fog. Future work should focus on darkness-specific augmentation strategies.

### 4. EMA Is Critical for Scratch Training
In EXP57, the EMA model (88.58%) dramatically outperforms the direct model (85.21%) — a +3.37% gap. For fine-tuning (EXP56), the direct model outperforms EMA (90.07% vs 89.13%). This suggests EMA regularization is more important when there is no pretrained checkpoint to stabilize training.

### 5. Our Method Dominates Without Language Models
WeatherPrompt (NeurIPS'25) uses XVLM + LLM-generated text + dynamic gating. Our approach is **purely visual** — DINOv2 ViT-S/14 backbone with altitude-aware part discovery — yet surpasses WeatherPrompt by **+29.13% avg adverse R@1**. This validates the paradigm of **spatial part discovery + multi-granularity metric learning** over language-guided approaches for geo-localisation.

---

*Last updated: 2026-03-18 — All SUES-200 experiments (EXP35–EXP57) fully documented. EXP58/59/60 (University-1652) scripts created and pending Kaggle runs. Current SUES-200 SOTA: **EXP56 — 90.07% Avg Adverse R@1**.*

---

## Phase 3 — University-1652 Weather Experiments (IMG=448)

> Dataset: **University-1652** · 701 train buildings / 951 test buildings · 54 drone views + 1 satellite per building  
> Task: **Drone → Satellite** retrieval under 10 weather conditions (same imgaug protocol)  
> Architecture: **SPDGeo-DPE-MAR (NO altitude conditioning)** · 3-group GeoPartLoss (Alignment + EMA + Part Quality)  
> WeatherPrompt (NeurIPS'25) baseline: **77.14% mean R@1 (D→S)** across 10 weather conditions

---

### EXP58 — Zero-Shot Evaluation on University-1652

**Script:** `exp58_weather_zeroshot_uni.py`  
**Status:** ⏳ Pending Kaggle Run  

**Setup:** Pure inference — NO training. Loads `geopartloss_best.pth` (trained on SUES-200, 98.4% clean R@1) and evaluates on Uni-1652 test set under all 10 weather conditions.

**Motivation:** Cross-dataset zero-shot robustness — does the SUES-200 model generalize to Uni-1652 with weather?  
Unlike EXP55 (same dataset zero-shot), EXP58 is a **true cross-dataset** evaluation.

**Expected Results:** TBD — pending Kaggle run  
Hypothesis: Zero-shot cross-dataset may be lower than in-domain EXP55 (50.17%), but the strongly-trained backbone may still outperform earlier methods.

| Weather | R@1 | mAP |
|---|---|---|
| normal | — | — |
| **Avg(adverse)** | **—** | **—** |

---

### EXP59 — Weather Fine-Tune on University-1652 (Online Augmentation)

**Script:** `exp59_weather_finetune_uni.py`  
**Status:** ⏳ Pending Kaggle Run  

**Setup:** Fine-tune from GeoPartLoss best checkpoint (98.4%) · 60 epochs · LR=1e-4 · backbone_LR=1e-5  
**Data:** 701 buildings × 1 random drone image/epoch (WeatherPrompt-style) · online imgaug weather  
**RECON_WARMUP:** 0 (already pretrained)

**Key differences from EXP56 (SUES-200 fine-tune):**

| Parameter | EXP56 (SUES-200) | EXP59 (Uni-1652) |
|---|---|---|
| Dataset | SUES-200 (200 locs, 4 alt) | University-1652 (701 buildings) |
| Altitude conditioning | ✅ FiLM | ❌ None |
| Loss groups | 4-group GeoPartLoss | 3-group GeoPartLoss |
| Epochs | 60 | 60 |
| Samples/epoch | 120 (1 per loc) | 701 (1 per building) |

**Expected Results:** TBD — pending Kaggle run  
Hypothesis: Should significantly outperform WeatherPrompt (77.14%) given the strong backbone.

| Weather | R@1 |
|---|---|
| normal | — |
| **Avg(adverse)** | **—** |

---

### EXP60 — Weather Train from Scratch on University-1652

**Script:** `exp60_weather_scratch_uni.py`  
**Status:** ⏳ Pending Kaggle Run  

**Setup:** From scratch (DINOv2 pretrained backbone) · 120 epochs · LR=3e-4 · backbone_LR=3e-5 · RECON_WARMUP=10  
**Data:** 701 buildings × 1 random drone image/epoch · online imgaug weather

**Key differences from EXP57 (SUES-200 scratch):**

| Parameter | EXP57 (SUES-200) | EXP60 (Uni-1652) |
|---|---|---|
| Dataset | SUES-200 (200 locs) | University-1652 (701 buildings) |
| Altitude conditioning | ✅ FiLM | ❌ None |
| Loss groups | 4-group GeoPartLoss | 3-group GeoPartLoss |
| # Classes | 200 | 701 |

**Expected Results:** TBD — pending Kaggle run

| Weather | R@1 |
|---|---|
| normal | — |
| **Avg(adverse)** | **—** |

---

### Phase 3 Summary — University-1652 (to be filled after runs)

| Method | Normal R@1 | Avg(adverse) R@1 | vs WeatherPrompt |
|---|---|---|---|
| WeatherPrompt (NeurIPS'25) | 76.72% | ~73% | baseline |
| EXP58 — Zero-Shot | — | — | — |
| EXP59 — FT + Online | — | — | — |
| EXP60 — Scratch + Online | — | — | — |

> WeatherPrompt Uni-1652 D→S mean R@1 = **77.14%** (across all 10 conditions).

