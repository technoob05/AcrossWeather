# AcrossWeather: All-Weather Drone-View Geo-Localization

Robust drone-to-satellite geo-localization under diverse weather conditions.

## Project Status: **Early Stage**

---

## Motivation

Existing cross-view geo-localization methods assume clear-weather drone imagery. Real-world UAV operations face fog, rain, snow, darkness, wind blur, and compound conditions. This project develops weather-robust geo-localization methods.

## Weather Conditions (10 types)

| # | Condition | Description |
|---|-----------|-------------|
| 0 | normal | Clean, no augmentation |
| 1 | fog | Cloud/fog layer overlay |
| 2 | rain | 5 stacked rain layers (varying drop sizes) |
| 3 | snow | 5 stacked snowflake layers |
| 4 | dark | Darkening + reduced brightness |
| 5 | light | Overexposure / bright sunlight |
| 6 | fog_rain | Fog + rain composite |
| 7 | fog_snow | Fog + snow composite |
| 8 | rain_snow | Mixed rain + snow precipitation |
| 9 | wind | Motion blur (simulates wind) |

## Directory Structure

```
AcrossWeather/
├── reference/
│   └── WeatherPrompt/          # Reference: WeatherPrompt (NeurIPS 2025)
│       ├── model.py            # XVLM + dynamic channel gating
│       ├── train.py            # Training pipeline (4-view: drone, sat, weather-text, scene-text)
│       ├── weather.py          # imgaug weather synthesis definitions
│       ├── image_folder.py     # Multi-weather dataset loaders
│       ├── modules.py          # AdaptiveInstanceNorm, AdaIBN blocks
│       └── dataset/            # Pre-generated Qwen2.5-VL weather captions (JSON)
│
├── exp/
│   └── exp48_weather_synthetic_gen.py   # Standalone synthetic data generator (Kaggle-ready)
│
├── src/                        # [TODO] Model source code
├── docs/                       # [TODO] Research notes & literature review
└── configs/                    # [TODO] Training configurations
```

## Getting Started

### Step 1: Generate Synthetic Weather Data

```bash
# Quick test (3 classes, verify pipeline)
python exp/exp48_weather_synthetic_gen.py --quick-test \
    --uni1652-root /path/to/University-1652/University-Release

# Full generation
python exp/exp48_weather_synthetic_gen.py \
    --dataset university1652 \
    --uni1652-root /path/to/University-1652/University-Release \
    --output-root ./data/weather_synthetic
```

### Step 2: Generate Weather Captions (Optional, needs GPU)

```bash
python exp/exp48_weather_synthetic_gen.py --generate-captions \
    --caption-model Qwen/Qwen2.5-VL-7B-Instruct
```

## Reference Paper

> **WeatherPrompt: All-Weather Drone-View Geo-Localization via Multimodal Prompting**
> Wen et al., NeurIPS 2025
> - XVLM backbone (Swin-B + BERT) with dynamic channel gating
> - CoT weather captioning via Qwen2.5-VL-32B
> - 10 synthetic weather conditions via imgaug
> - Losses: L_ITC + L_ITM + L_LA + L_CE

## Planned Contributions (differentiation from WeatherPrompt)

1. **[ ]** Weather-aware backbone adaptation (vs. text-gated channel modulation)
2. **[ ]** Weather-robust part discovery (extending SPARC's semantic parts)
3. **[ ]** Cross-weather contrastive learning
4. **[ ]** Altitude × Weather joint conditioning
5. **[ ]** Real-world weather evaluation (beyond synthetic)

## Datasets

- **University-1652**: 1652 locations, 701 train / 951 test, drone + satellite views
- **SUES-200**: 200 locations, 4 altitudes (150-300m), 120 train / 80 test
