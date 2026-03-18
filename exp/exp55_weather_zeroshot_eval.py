# =============================================================================
# EXP55: Zero-Shot Weather Robustness Evaluation  (New GeoPartLoss Pipeline)
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Pure inference — NO training.
# Loads the NEW best checkpoint (geopartloss_best.pth, 98.4% R@1 no-weather)
# built with the FULL GeoPartLoss pipeline (IMG_SIZE=448, 4-group adaptive loss)
# and evaluates on all 10 weather conditions.
#
# Designed for:
#   1. New best checkpoint (GeoPartLoss, EXP≥35 series, IMG=448, UNF=6)
#      → "EXP55_ZeroShot_NewBest"   ← DEFAULT
#   2. EXP56 best checkpoint (finetune + online aug, GeoPartLoss backbone)
#      → Uncomment EXP56 block in Config
#   3. EXP57 best checkpoint (scratch + online aug, GeoPartLoss backbone)
#      → Uncomment EXP57 block in Config
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone test
#   2. Weather synthetic:  weather-augmented drone TEST images (for eval)
#   3. Model checkpoint:   new geopartloss_best.pth
#      Path: /kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, json, time, datetime, gc
import numpy as np
from collections import OrderedDict

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
    # --- Which checkpoint to evaluate ---
    # NEW best checkpoint (GeoPartLoss pipeline, 98.4% no-weather R@1)
    EXPERIMENT_NAME = "EXP55_ZeroShot_NewBest"
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth"

    # For EXP56 best (finetune from new best):
    # EXPERIMENT_NAME = "EXP55_ZeroShot_EXP56"
    # CHECKPOINT      = "/kaggle/working/exp56_weather_finetune_online_best.pth"

    # For EXP57 best (scratch with new pipeline):
    # EXPERIMENT_NAME = "EXP55_ZeroShot_EXP57"
    # CHECKPOINT      = "/kaggle/working/exp57_weather_scratch_online_best.pth"

    # --- Paths ---
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT    = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    OUTPUT_DIR      = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120

    # --- Model architecture (MUST match the new GeoPartLoss checkpoint) ---
    IMG_SIZE        = 448   # NEW: 448 (was 336 in EXP52/53/54)
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6
    MASK_RATIO      = 0.30

    # --- Weather conditions ---
    WEATHER_NAMES   = ["normal", "fog", "rain", "snow", "dark", "light",
                       "fog_rain", "fog_snow", "rain_snow", "wind"]

    # --- Inference ---
    USE_AMP         = True
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    SEED            = 42


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL DEFINITION  (identical to New_Best_No_Weather98.4.py / GeoPartLoss)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=6):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

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
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(),
                                  nn.Linear(part_dim // 2, 1))
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

    def forward(self, projected_patches, part_features, assignment):
        B, N, D = projected_patches.shape
        num_mask = int(N * self.mask_ratio)
        noise = torch.rand(B, N, device=projected_patches.device)
        ids_shuffle = noise.argsort(dim=1)
        mask_indices = ids_shuffle[:, :num_mask]
        target = projected_patches.detach()
        mask_expand = mask_indices.unsqueeze(-1).expand(-1, -1, D)
        masked_targets = torch.gather(target, 1, mask_expand)
        K = part_features.shape[1]
        mask_expand_k = mask_indices.unsqueeze(-1).expand(-1, -1, K)
        masked_assign = torch.gather(assignment, 1, mask_expand_k)
        recon = torch.bmm(masked_assign, part_features)
        recon = self.decoder(recon)
        recon_norm = F.normalize(recon, dim=-1)
        target_norm = F.normalize(masked_targets, dim=-1)
        return (1 - (recon_norm * target_norm).sum(dim=-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.smooth_l1_loss(pred, alt_target)


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
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPEA-MAR (GeoPartLoss): {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)


# =============================================================================
# DATASET — weather test queries (pre-generated)
# =============================================================================
class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name
        self.samples = []

        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        if not os.path.isdir(weather_test_dir):
            return

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
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
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w == weather_name:
                    self.samples.append((os.path.join(class_path, fname), label_idx, alt))

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
# HELPERS
# =============================================================================
def get_test_transform():
    sz = CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def build_satellite_gallery(model, device, transform):
    sat_dir = os.path.join(CFG.SUES_ROOT, CFG.SAT_DIR)
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

    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch, alt_idx=None)
            sat_feats.append(feats.cpu())

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)
    print(f"  Satellite gallery: {len(sat_feats)} images ({distractor_cnt} distractors)")
    return sat_feats, sat_labels


@torch.no_grad()
def evaluate_retrieval(model, query_ds, sat_feats, sat_labels, device):
    model.eval()
    loader = DataLoader(query_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
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

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0)

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
        'num_queries': N,
    }

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
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops,
        'ms_per_query': ms_query,
        'img_size': sz,
    }


def print_complexity(stats):
    print(f"\n  ┌─ Model Complexity (IMG={stats['img_size']}) ─────────────────┐")
    print(f"  │  Total params:     {stats['total_params_M']:>8.2f}M                │")
    print(f"  │  Trainable params: {stats['trainable_params_M']:>8.2f}M                │")
    if stats['gflops'] is not None:
        print(f"  │  GFLOPs:           {stats['gflops']:>8.2f}                 │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)    │")
    print(f"  │  Inference:        {stats['ms_per_query']:>8.1f} ms/query         │")
    print(f"  └─────────────────────────────────────────────────┘")


def print_weather_table(all_results, normal_results):
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test, IMG={CFG.IMG_SIZE})")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'ΔR@1':>7s}  {'ΔmAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_results[w_name]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        sign_r1 = '+' if dr1 >= 0 else ''; sign_map = '+' if dmap >= 0 else ''
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {sign_r1}{dr1:5.2f}%  {sign_map}{dmap:5.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in CFG.WEATHER_NAMES])
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def print_cross_table(all_results):
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
            row += f"  {v:6.2f}%"; vals.append(v)
        row += f"  {np.mean(vals):6.2f}%"
        print(row)
    print(f"{'='*75}")


# =============================================================================
# KNOWN RESULTS FROM PREVIOUS EXPERIMENTS (for comparison table)
# =============================================================================
EXP49_RESULTS = {   # zero-shot (EXP35 checkpoint, old eval)
    'normal':    {'R@1': 95.94, 'mAP': 97.64},
    'fog':       {'R@1': 86.88, 'mAP': 92.68},
    'rain':      {'R@1': 43.75, 'mAP': 56.99},
    'snow':      {'R@1': 40.62, 'mAP': 54.48},
    'dark':      {'R@1': 66.88, 'mAP': 76.74},
    'light':     {'R@1': 83.44, 'mAP': 89.97},
    'fog_rain':  {'R@1': 42.19, 'mAP': 54.92},
    'fog_snow':  {'R@1': 22.19, 'mAP': 35.01},
    'rain_snow': {'R@1': 34.69, 'mAP': 46.85},
    'wind':      {'R@1': 81.25, 'mAP': 88.76},
}
EXP53_RESULTS = {   # scratch + online aug (previous best weather model)
    'normal':    {'R@1': 92.81, 'mAP': 95.81},
    'fog':       {'R@1': 91.88, 'mAP': 94.94},
    'rain':      {'R@1': 90.31, 'mAP': 94.18},
    'snow':      {'R@1': 87.19, 'mAP': 92.19},
    'dark':      {'R@1': 82.50, 'mAP': 87.90},
    'light':     {'R@1': 86.25, 'mAP': 91.25},
    'fog_rain':  {'R@1': 85.62, 'mAP': 91.19},
    'fog_snow':  {'R@1': 79.38, 'mAP': 86.89},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.76},
    'wind':      {'R@1': 87.50, 'mAP': 92.33},
}


def print_comparison(all_results):
    """Compare EXP55 (new best) against EXP49 (old zero-shot) and EXP53 (best weather model)."""
    exps = [
        ("EXP49", EXP49_RESULTS, "zero-shot-old"),
        ("EXP53", EXP53_RESULTS, "scratch-online"),
    ]
    exp_name_short = CFG.EXPERIMENT_NAME

    print(f"\n{'='*110}")
    print(f"  COMPARISON R@1(%): EXP49(zero-shot-old) vs EXP53(scratch-online,best) vs {exp_name_short}")
    print(f"{'='*110}")

    row = f"  {'Weather':>12s}"
    for exp_id, _, _ in exps:
        row += f"  {exp_id:>10s}"
    row += f"  {'EXP55':>10s}  {'Δ vs EXP53':>10s}"
    print(row)
    print(f"  {'-'*80}")

    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        for _, res, _ in exps:
            row += f"  {res[w]['R@1']:9.2f}%"
        e55_r1 = all_results[w]['overall']['R@1']
        delta = e55_r1 - EXP53_RESULTS[w]['R@1']
        row += f"  {e55_r1:9.2f}%  {delta:+9.2f}%"
        print(row)

    print(f"  {'-'*80}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    row = f"  {'Avg(adverse)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in adv]):9.2f}%"
    e55_adv = np.mean([all_results[w]['overall']['R@1'] for w in adv])
    row += f"  {e55_adv:9.2f}%  {e55_adv-np.mean([EXP53_RESULTS[w]['R@1'] for w in adv]):+9.2f}%"
    print(row)

    row = f"  {'Avg(all)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in CFG.WEATHER_NAMES]):9.2f}%"
    e55_all = np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])
    row += f"  {e55_all:9.2f}%  {e55_all-np.mean([EXP53_RESULTS[w]['R@1'] for w in CFG.WEATHER_NAMES]):+9.2f}%"
    print(row)
    print(f"{'='*110}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    import random
    random.seed(CFG.SEED); np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 75)
    print(f"  EXP55: Zero-Shot Weather Eval — GeoPartLoss Pipeline")
    print(f"  Experiment: {CFG.EXPERIMENT_NAME}")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  IMG_SIZE: {CFG.IMG_SIZE}  |  Device: {DEVICE}")
    print("=" * 75)

    # --- Build Model ---
    print("\nBuilding model…")
    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES, CFG).to(DEVICE)

    # --- Load Checkpoint ---
    print(f"\nLoading checkpoint: {CFG.CHECKPOINT}")
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval()

    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_r1    = ckpt.get('metrics', {}).get('R@1', '?')
    print(f"  Loaded epoch={ckpt_epoch}, reported R@1={ckpt_r1}")

    # --- Model Complexity ---
    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # --- Transforms ---
    transform = get_test_transform()

    # --- Build Satellite Gallery ---
    print("\nBuilding satellite gallery…")
    sat_feats, sat_labels = build_satellite_gallery(model, DEVICE, transform)

    # --- Evaluate all weather conditions ---
    print(f"\nEvaluating {len(CFG.WEATHER_NAMES)} weather conditions…")
    all_results = {}
    t_total = time.time()

    for w_name in CFG.WEATHER_NAMES:
        query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, w_name, transform)
        if len(query_ds) == 0:
            print(f"  [{w_name}] — NO queries found, skipping")
            all_results[w_name] = {
                'overall': {'R@1': 0, 'R@5': 0, 'R@10': 0, 'mAP': 0, 'num_queries': 0},
                'per_alt': {}
            }
            continue
        t0 = time.time()
        overall, per_alt = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
        elapsed = time.time() - t0
        all_results[w_name] = {'overall': overall, 'per_alt': per_alt}
        print(f"  [{w_name:>10s}] R@1={overall['R@1']:6.2f}%  mAP={overall['mAP']:6.2f}%  "
              f"({overall['num_queries']} queries, {elapsed:.1f}s)")

    total_time = time.time() - t_total

    # --- Print Tables ---
    normal_results = all_results['normal']['overall']
    print_weather_table(all_results, normal_results)
    print_cross_table(all_results)
    print_comparison(all_results)

    # --- Summary ---
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_all = np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])
    avg_adv = np.mean([all_results[w]['overall']['R@1'] for w in adv])
    print(f"\n  ★ EXP55 Summary — {CFG.EXPERIMENT_NAME}")
    print(f"    Normal R@1:       {normal_results['R@1']:6.2f}%")
    print(f"    Avg(all) R@1:     {avg_all:6.2f}%")
    print(f"    Avg(adverse) R@1: {avg_adv:6.2f}%")
    print(f"    Worst R@1:        {min(all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES):6.2f}%  "
          f"({min(CFG.WEATHER_NAMES, key=lambda w: all_results[w]['overall']['R@1'])})")
    print(f"    Total eval time:  {total_time:.1f}s")

    # --- Save JSON ---
    ts = datetime.datetime.now().isoformat()
    result_data = {
        'experiment': CFG.EXPERIMENT_NAME,
        'checkpoint': CFG.CHECKPOINT,
        'timestamp': ts,
        'img_size': CFG.IMG_SIZE,
        'model_complexity': complexity,
        'results': {w: all_results[w]['overall'] for w in CFG.WEATHER_NAMES},
        'per_alt': {w: all_results[w]['per_alt'] for w in CFG.WEATHER_NAMES},
        'summary': {
            'normal_r1': normal_results['R@1'],
            'avg_all_r1': avg_all,
            'avg_adverse_r1': avg_adv,
        }
    }
    out_path = os.path.join(CFG.OUTPUT_DIR, f"exp55_{CFG.EXPERIMENT_NAME}_results.json")
    with open(out_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == '__main__':
    main()
