# =============================================================================
# EXP49: AcrossWeather Robustness Benchmark — SUES-200
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Loads pretrained SPDGeo-DPEA-MAR checkpoint (no weather training),
# evaluates drone→satellite retrieval under 10 weather conditions.
# Outputs everything needed for a research paper:
#   - Per-weather R@1/R@5/R@10/mAP
#   - Per-altitude R@1/R@5/R@10/mAP
#   - Per-weather × per-altitude breakdown
#   - Model complexity: params, GFLOPs, ms/query
#   - Weather robustness drop (Δ from normal baseline)
#   - JSON results + LaTeX table
#
# Kaggle Data Sources:
#   1. SUES-200 original: satellite gallery
#   2. Weather synthetic:  weather-augmented drone queries
#   3. Model checkpoint:   pretrained .pth
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, math, json, gc, random, copy, time, datetime
import numpy as np
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Paths (update if your Kaggle dataset slugs differ) ---
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT    = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth"
    OUTPUT_DIR      = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120                      # train classes (for model init)

    # --- Model architecture (must match training) ---
    IMG_SIZE        = 336
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6

    # --- Weather conditions ---
    WEATHER_NAMES   = ["normal", "fog", "rain", "snow", "dark", "light",
                       "fog_rain", "fog_snow", "rain_snow", "wind"]

    # --- Eval ---
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    SEED            = 42
    USE_AMP         = True


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# MODEL DEFINITION (copy from training — inference only)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta  = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            gamma = self.gamma[alt_idx]
            beta  = self.beta[alt_idx]
            return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07,
                 num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience,
                'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, 0.30)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)


# =============================================================================
# WEATHER QUERY DATASET
# =============================================================================
class WeatherQueryDataset(Dataset):
    """
    Loads weather-augmented drone images for evaluation.

    Expected structure (from exp48):
        {weather_root}/sues200_weather_test/{loc_id}_{alt}/{stem}-{weather}.jpg
    """
    def __init__(self, weather_root, sues_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name
        self.sat_dir = os.path.join(sues_root, CFG.SAT_DIR)

        # Scan weather test images
        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        self.samples = []  # (drone_path, loc_label_idx, altitude_str)

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        if not os.path.isdir(weather_test_dir):
            print(f"  [WARN] Weather test dir not found: {weather_test_dir}")
            return

        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            # class_dir_name = "0121_150"
            parts = class_dir_name.split("_")
            if len(parts) != 2:
                continue
            loc_id, alt = parts[0], parts[1]
            if loc_id not in loc_to_idx:
                continue

            label_idx = loc_to_idx[loc_id]

            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                # Filter by weather: "{stem}-{weather}.jpg"
                stem_no_ext = os.path.splitext(fname)[0]
                if '-' not in stem_no_ext:
                    continue
                w = stem_no_ext.rsplit('-', 1)[-1]
                if w == weather_name:
                    self.samples.append((
                        os.path.join(class_path, fname),
                        label_idx,
                        alt
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, alt = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        return {'drone': img, 'label': label, 'altitude': int(alt), 'alt_idx': alt_idx}


# =============================================================================
# SATELLITE GALLERY BUILDER
# =============================================================================
def build_satellite_gallery(model, sues_root, device, transform):
    """Build gallery from ALL 200 satellite locations (with distractors)."""
    sat_dir = os.path.join(sues_root, CFG.SAT_DIR)
    test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}

    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_imgs, sat_labels = [], []
    distractor_cnt = 0

    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_imgs.append(transform(Image.open(sp).convert('RGB')))
        if loc in loc_to_idx:
            sat_labels.append(loc_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    # Extract embeddings
    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i+CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch, alt_idx=None)
            sat_feats.append(feats.cpu())

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)
    print(f"  Satellite gallery: {len(sat_feats)} images ({distractor_cnt} distractors)")
    return sat_feats, sat_labels


# =============================================================================
# RETRIEVAL EVALUATION
# =============================================================================
@torch.no_grad()
def evaluate_retrieval(model, query_ds, sat_feats, sat_labels, device):
    """
    Drone→Satellite retrieval.
    Returns: overall_metrics, per_altitude_metrics
    """
    model.eval()
    loader = DataLoader(query_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    t0 = time.time()
    for batch in loader:
        imgs = batch['drone'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            feats = model.extract_embedding(imgs, alt_idx=alt_idx)
        drone_feats.append(feats.cpu())
        drone_labels.append(batch['label'])
        drone_alts.append(batch['altitude'])

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)
    t_extract = time.time() - t0

    # Compute similarity & rank
    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0)

    # Overall metrics
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)

    overall = {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N, 'extract_time_s': t_extract,
    }

    # Per-altitude metrics
    per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1:  a1 += 1
            if f < 5:  a5 += 1
            if f < 10: a10 += 1
            aap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100,
                             'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    return overall, per_alt


# =============================================================================
# SATELLITE→DRONE EVALUATION (S2D)
# =============================================================================
def build_drone_gallery_s2d(model, weather_root, sues_root, weather_name, device, transform):
    """
    Builds drone gallery for Satellite→Drone (S2D) retrieval.
    Gallery = weather-augmented test drones (labeled 0..79) +
              clean train drones as distractors (labeled -1000...).
    SUES-200 has 200 classes total; 80 test + 120 train. Hardcoded here
    since EXP49 has no CFG.TRAIN_LOCS.
    """
    test_locs  = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    train_locs = [f"{l:04d}" for l in range(1, 121)]

    drone_imgs, drone_lbls, drone_alt_idxs = [], [], []
    distractor_cnt = 0

    # 1. Weather-augmented TEST drone images (labeled)
    weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
    if os.path.isdir(weather_test_dir):
        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts_split = class_dir_name.split("_")
            if len(parts_split) != 2:
                continue
            loc_id, alt = parts_split[0], parts_split[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w != weather_name:
                    continue
                try:
                    img = transform(Image.open(os.path.join(class_path, fname)).convert('RGB'))
                    drone_imgs.append(img)
                    drone_lbls.append(label_idx)
                    drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                except Exception:
                    pass

    # 2. Clean TRAIN drone images (distractors)
    drone_dir = os.path.join(sues_root, CFG.DRONE_DIR)
    for loc in train_locs:
        for alt in CFG.ALTITUDES:
            alt_dir = os.path.join(drone_dir, loc, alt)
            if not os.path.isdir(alt_dir):
                continue
            imgs_in_dir = sorted(f for f in os.listdir(alt_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if not imgs_in_dir:
                continue
            try:
                img = transform(Image.open(os.path.join(alt_dir, imgs_in_dir[0])).convert('RGB'))
                drone_imgs.append(img)
                drone_lbls.append(-1000 - distractor_cnt)
                drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                distractor_cnt += 1
            except Exception:
                pass

    # Embed all drone images
    drone_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(drone_imgs), CFG.BATCH_SIZE):
            bimgs = torch.stack(drone_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            balts = torch.tensor(drone_alt_idxs[i:i + CFG.BATCH_SIZE], device=device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(bimgs, alt_idx=balts)
            drone_feats.append(feats.cpu())

    drone_feats = torch.cat(drone_feats)
    drone_lbls  = torch.tensor(drone_lbls)
    test_cnt    = len(drone_feats) - distractor_cnt
    print(f"  Drone gallery S2D ({weather_name}): {len(drone_feats)} "
          f"(test={test_cnt}, distractors={distractor_cnt})")
    return drone_feats, drone_lbls


@torch.no_grad()
def evaluate_s2d(sat_feats, sat_labels, drone_feats, drone_labels):
    """
    Satellite→Drone retrieval.
    Queries  = test satellite images (sat_labels >= 0, 80 queries).
    Gallery  = weather-augmented test drones + clean distractor train drones.
    Each query has up to 4 matches (one per altitude of same location).
    """
    q_mask   = sat_labels >= 0
    q_feats  = sat_feats[q_mask]
    q_labels = sat_labels[q_mask]
    sim = q_feats @ drone_feats.T
    _, rank = sim.sort(1, descending=True)
    N = q_feats.size(0)
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(drone_labels[rank[i]] == q_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)
    return {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }


def print_s2d_weather_table(all_s2d_results, normal_s2d):
    """Print Satellite→Drone per-weather results table."""
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Satellite→Drone Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'\u0394R@1':>7s}  {'\u0394mAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_s2d['R@1']; nmap = normal_s2d['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_s2d_results[w_name]
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {dr1:+6.2f}%  {dmap:+6.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_s2d_results[w]['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_s2d_results[w]['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_s2d_results[w]['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_s2d_results[w]['mAP']  for w in CFG.WEATHER_NAMES])
    avg_adv_r1  = np.mean([all_s2d_results[w]['R@1']  for w in adv])
    avg_adv_map = np.mean([all_s2d_results[w]['mAP']  for w in adv])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


# =============================================================================
# MODEL COMPLEXITY
# =============================================================================
def measure_model_complexity(model, device):
    sz = CFG.IMG_SIZE
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    dummy = torch.randn(1, 3, sz, sz).to(device)

    gflops = None
    try:
        from thop import profile as thop_profile
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m.extract_embedding(x)
        macs, _ = thop_profile(_Wrap(model).to(device), inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
    except Exception:
        pass

    model.eval()
    with torch.no_grad():
        for _ in range(10): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000

    return {
        'total_params': total_params,
        'total_params_M': total_params / 1e6,
        'trainable_params': trainable_params,
        'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops,
        'ms_per_query': ms_query,
        'img_size': sz,
    }


# =============================================================================
# PRETTY PRINTERS
# =============================================================================
def print_weather_table(all_results, normal_results):
    """Print per-weather results table with Δ from normal."""
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'ΔR@1':>7s}  {'ΔmAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")

    nr1 = normal_results['R@1']
    nmap = normal_results['mAP']

    for w_name in CFG.WEATHER_NAMES:
        r = all_results[w_name]['overall']
        dr1 = r['R@1'] - nr1
        dmap = r['mAP'] - nmap
        sign_r1 = '+' if dr1 >= 0 else ''
        sign_map = '+' if dmap >= 0 else ''
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {sign_r1}{dr1:5.2f}%  {sign_map}{dmap:5.2f}%  "
              f"{r['num_queries']:>5d}")

    print(f"  {'-'*78}")

    # Average across all weathers
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in CFG.WEATHER_NAMES])
    # Average excluding normal
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1']  for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP']  for w in adv_weathers])

    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def print_altitude_breakdown(all_results, weather_name):
    """Print per-altitude breakdown for one weather."""
    r = all_results[weather_name]
    print(f"\n  [{weather_name}] Per-Altitude:")
    print(f"  {'Alt':>6s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*45}")
    for alt in CFG.ALTITUDES:
        a = r['per_alt'].get(int(alt), {})
        if not a: continue
        print(f"  {alt+'m':>6s}  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  "
              f"{a['mAP']:6.2f}%  {a['n']:>5d}")
    o = r['overall']
    print(f"  {'All':>6s}  {o['R@1']:6.2f}%  {o['R@5']:6.2f}%  {o['R@10']:6.2f}%  "
          f"{o['mAP']:6.2f}%  {o['num_queries']:>5d}")


def print_cross_table(all_results):
    """Print Weather × Altitude R@1 table."""
    print(f"\n{'='*75}")
    print(f"  R@1 (%) — Weather × Altitude")
    print(f"{'='*75}")
    header = f"  {'Weather':>12s}"
    for alt in CFG.ALTITUDES:
        header += f"  {alt+'m':>7s}"
    header += f"  {'Avg':>7s}"
    print(header)
    print(f"  {'-'*60}")

    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        vals = []
        for alt in CFG.ALTITUDES:
            v = all_results[w]['per_alt'].get(int(alt), {}).get('R@1', 0)
            row += f"  {v:6.2f}%"
            vals.append(v)
        row += f"  {np.mean(vals):6.2f}%"
        print(row)
    print(f"{'='*75}")


def print_complexity(stats):
    print(f"\n  ┌─ Model Complexity ─────────────────────────────┐")
    print(f"  │  Total params:     {stats['total_params_M']:>8.2f}M                │")
    print(f"  │  Trainable params: {stats['trainable_params_M']:>8.2f}M                │")
    if stats['gflops'] is not None:
        print(f"  │  GFLOPs:           {stats['gflops']:>8.2f}                 │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)    │")
    print(f"  │  Inference:        {stats['ms_per_query']:>8.1f} ms/query         │")
    print(f"  │  Image size:       {stats['img_size']:>4d}×{stats['img_size']:<4d}               │")
    print(f"  └─────────────────────────────────────────────────┘")


def generate_latex_table(all_results, normal_results):
    """Generate LaTeX table for paper."""
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Weather robustness evaluation on SUES-200 (Drone$\to$Satellite).}")
    lines.append(r"\label{tab:weather_robustness}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{l|cccc|cc}")
    lines.append(r"\toprule")
    lines.append(r"Weather & R@1 & R@5 & R@10 & mAP & $\Delta$R@1 & $\Delta$mAP \\")
    lines.append(r"\midrule")
    for w in CFG.WEATHER_NAMES:
        r = all_results[w]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        s_dr1 = f"+{dr1:.2f}" if dr1 >= 0 else f"{dr1:.2f}"
        s_dmap = f"+{dmap:.2f}" if dmap >= 0 else f"{dmap:.2f}"
        label = w.replace('_', r'\_')
        if w == 'normal':
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & --- & --- \\\\")
        else:
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & {s_dr1} & {s_dmap} \\\\")
    lines.append(r"\midrule")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1 = np.mean([all_results[w]['overall']['R@1'] for w in adv])
    avg_r5 = np.mean([all_results[w]['overall']['R@5'] for w in adv])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in adv])
    avg_map = np.mean([all_results[w]['overall']['mAP'] for w in adv])
    lines.append(f"Avg(adverse) & {avg_r1:.2f} & {avg_r5:.2f} & {avg_r10:.2f} & "
                 f"{avg_map:.2f} & {avg_r1-nr1:+.2f} & {avg_map-nmap:+.2f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    bench_start = datetime.datetime.now()

    print("=" * 75)
    print("  EXP49: AcrossWeather Robustness Benchmark — SUES-200")
    print(f"  Model checkpoint: {CFG.CHECKPOINT}")
    print(f"  Weather data:     {CFG.WEATHER_ROOT}")
    print(f"  SUES-200 root:    {CFG.SUES_ROOT}")
    print(f"  Device:           {DEVICE}")
    print(f"  Weathers:         {len(CFG.WEATHER_NAMES)} conditions")
    print("=" * 75)

    # ---- 1. Build Model & Load Checkpoint ----
    print("\n[1/4] Loading model …")
    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)

    ckpt_path = CFG.CHECKPOINT
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Checkpoint loaded (epoch={ckpt.get('epoch', '?')})")
    if 'metrics' in ckpt:
        m = ckpt['metrics']
        print(f"  Training best: R@1={m.get('R@1', '?'):.2f}%  mAP={m.get('mAP', '?'):.2f}%")

    # ---- 2. Model Complexity ----
    print("\n[2/4] Measuring model complexity …")
    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # ---- 3. Build Satellite Gallery (once) ----
    print("\n[3/4] Building satellite gallery …")
    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    sat_feats, sat_labels = build_satellite_gallery(model, CFG.SUES_ROOT, DEVICE, test_tf)

    # ---- 4. Evaluate per weather condition ----
    print("\n[4/4] Evaluating weather robustness …")
    all_results = OrderedDict()

    for w_name in CFG.WEATHER_NAMES:
        print(f"\n  --- {w_name} ---")
        query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, CFG.SUES_ROOT, w_name, transform=test_tf)
        if len(query_ds) == 0:
            print(f"    [SKIP] No images found for weather={w_name}")
            all_results[w_name] = {'overall': {'R@1': 0, 'R@5': 0, 'R@10': 0, 'mAP': 0, 'num_queries': 0},
                                   'per_alt': {}}
            continue

        print(f"    Queries: {len(query_ds)}")
        overall, per_alt = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
        all_results[w_name] = {'overall': overall, 'per_alt': per_alt}

        print(f"    R@1={overall['R@1']:.2f}%  R@5={overall['R@5']:.2f}%  "
              f"R@10={overall['R@10']:.2f}%  mAP={overall['mAP']:.2f}%")

    # ---- Print Results ----
    normal_results = all_results['normal']['overall']
    print_weather_table(all_results, normal_results)
    print_cross_table(all_results)

    # Per-altitude detail for each weather
    for w in CFG.WEATHER_NAMES:
        print_altitude_breakdown(all_results, w)

    # ---- S2D Evaluation (Satellite→Drone) ----
    print(f"\n{'='*75}")
    print(f"  [S2D] Satellite→Drone Evaluation …")
    all_s2d_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        drone_feats_s2d, drone_lbls_s2d = build_drone_gallery_s2d(
            model, CFG.WEATHER_ROOT, CFG.SUES_ROOT, w_name, DEVICE, test_tf)
        s2d_res = evaluate_s2d(sat_feats, sat_labels, drone_feats_s2d, drone_lbls_s2d)
        all_s2d_results[w_name] = s2d_res
        print(f"    {w_name:>12s}  R@1={s2d_res['R@1']:.2f}%  R@5={s2d_res['R@5']:.2f}%  "
              f"R@10={s2d_res['R@10']:.2f}%  mAP={s2d_res['mAP']:.2f}%")
        del drone_feats_s2d, drone_lbls_s2d
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    normal_s2d = all_s2d_results['normal']
    print_s2d_weather_table(all_s2d_results, normal_s2d)

    # ---- LaTeX Table ----
    latex = generate_latex_table(all_results, normal_results)
    print(f"\n{'='*75}")
    print("  LaTeX Table (copy-paste into paper):")
    print(f"{'='*75}")
    print(latex)

    # ---- Save JSON ----
    bench_end = datetime.datetime.now()
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    summary = {
        'experiment': 'EXP49_AcrossWeather_Robustness',
        'model': 'SPDGeo-DPEA-MAR (no weather training)',
        'dataset': 'SUES-200',
        'direction': 'drone → satellite (D2S) + satellite → drone (S2D)',
        'checkpoint': CFG.CHECKPOINT,
        'timestamp': {
            'start': bench_start.isoformat(),
            'end': bench_end.isoformat(),
            'duration_s': (bench_end - bench_start).total_seconds(),
        },
        'model_complexity': complexity,
        'config': {
            'img_size': CFG.IMG_SIZE,
            'embed_dim': CFG.EMBED_DIM,
            'n_parts': CFG.N_PARTS,
            'unfreeze_blocks': CFG.UNFREEZE_BLOCKS,
            'backbone': 'DINOv2 ViT-S/14',
        },
        'satellite_gallery_size': len(sat_feats),
        'weather_results': {
            w: {
                'overall': all_results[w]['overall'],
                'per_altitude': {str(k): v for k, v in all_results[w]['per_alt'].items()},
            }
            for w in CFG.WEATHER_NAMES
        },
        'summary_metrics': {
            'normal_R@1': normal_results['R@1'],
            'normal_mAP': normal_results['mAP'],
            'avg_all_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])),
            'drop_adverse_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers]) - normal_results['R@1']),
            'drop_adverse_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers]) - normal_results['mAP']),
            'worst_weather': min(adv_weathers, key=lambda w: all_results[w]['overall']['R@1']),
            'worst_R@1': float(min(all_results[w]['overall']['R@1'] for w in adv_weathers)),
        },
        'latex_table': latex,
        's2d_weather_results': {w: all_s2d_results[w] for w in CFG.WEATHER_NAMES},
        's2d_summary_metrics': {
            'normal_R@1': normal_s2d['R@1'],
            'normal_mAP': normal_s2d['mAP'],
            'avg_all_R@1':     float(np.mean([all_s2d_results[w]['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_s2d_results[w]['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_s2d_results[w]['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_s2d_results[w]['mAP'] for w in adv_weathers])),
            'drop_adverse_R@1': float(np.mean([all_s2d_results[w]['R@1'] for w in adv_weathers]) - normal_s2d['R@1']),
            'drop_adverse_mAP': float(np.mean([all_s2d_results[w]['mAP'] for w in adv_weathers]) - normal_s2d['mAP']),
        },
    }

    json_path = os.path.join(CFG.OUTPUT_DIR, 'exp49_weather_robustness.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}")

    # ---- Final Summary ----
    print(f"\n{'='*75}")
    print(f"  EXP49 COMPLETE — AcrossWeather Robustness Benchmark")
    print(f"{'='*75}")
    print(f"  Normal baseline:    R@1={normal_results['R@1']:.2f}%  mAP={normal_results['mAP']:.2f}%")
    sm = summary['summary_metrics']
    print(f"  Avg (all weather):  R@1={sm['avg_all_R@1']:.2f}%  mAP={sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse only): R@1={sm['avg_adverse_R@1']:.2f}%  mAP={sm['avg_adverse_mAP']:.2f}%")
    print(f"  Drop (adverse):     ΔR@1={sm['drop_adverse_R@1']:+.2f}%  ΔmAP={sm['drop_adverse_mAP']:+.2f}%")
    print(f"  Worst weather:      {sm['worst_weather']} → R@1={sm['worst_R@1']:.2f}%")
    print(f"  Duration:           {(bench_end - bench_start).total_seconds():.1f}s")
    s2d_sm = summary['s2d_summary_metrics']
    print(f"\n  S2D — Satellite→Drone:")
    print(f"  Normal:             R@1={normal_s2d['R@1']:.2f}%  mAP={normal_s2d['mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={s2d_sm['avg_all_R@1']:.2f}%  mAP={s2d_sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse only): R@1={s2d_sm['avg_adverse_R@1']:.2f}%  mAP={s2d_sm['avg_adverse_mAP']:.2f}%")
    print(f"  Drop (adverse):     ΔR@1={s2d_sm['drop_adverse_R@1']:+.2f}%  ΔmAP={s2d_sm['drop_adverse_mAP']:+.2f}%")
    print(f"{'='*75}")


main()
