# =============================================================================
# EXP58: Zero-Shot Weather Robustness Evaluation — University-1652
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Pure inference — NO training.
# Loads the NEW best checkpoint (geopartloss_best.pth, 98.4% R@1 on SUES-200)
# and evaluates on University-1652 test set under all 10 weather conditions.
#
# Architecture: SPDGeo-DPE-MAR (NO altitude — same as gpl_cross_uni2sues.py)
# Loss used for training (excluded here): 3-group GeoPartLoss (No Altitude group)
#
# WeatherPrompt (NeurIPS'25) baseline on Uni-1652:
#   D→S: 77.14% mean R@1 across 10 weathers
#   S→D: 87.72% mean R@1 across 10 weathers
#
# Kaggle Data Sources:
#   1. University-1652:        /kaggle/input/datasets/chinguyeen/university-1652/University-1652
#   2. New best checkpoint:    /kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth
#
# Evaluation Protocol (same as WeatherPrompt):
#   - Gallery: satellite images (constant, NO weather)
#   - Query:   drone images (weather-augmented on-the-fly via imgaug)
#   - 10 weather conditions: normal, fog, rain, snow, dark, light,
#                            fog_rain, fog_snow, rain_snow, wind
#   - Metrics: R@1, R@5, R@10, mAP (Drone→Satellite and Satellite→Drone)
#
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "imgaug"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, random, copy, time, json
import numpy as np

# NumPy 2.0 compatibility patch for imgaug
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import imgaug.augmenters as iaa


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # ---- University-1652 ----
    UNI_ROOT    = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"

    # ---- Checkpoint ----
    CHECKPOINT  = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth"

    OUTPUT_DIR  = "/kaggle/working"
    EXPERIMENT_NAME = "EXP58_ZeroShot_Uni1652"

    IMG_SIZE        = 448
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6

    USE_AMP         = True
    SEED            = 42
    BATCH_SIZE      = 32

    # 10 weather conditions (same as WeatherPrompt / SUES-200 experiments)
    WEATHER_CONDITIONS = [
        "normal", "fog", "rain", "snow", "dark", "light",
        "fog_rain", "fog_snow", "rain_snow", "wind"
    ]

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# WEATHER AUGMENTATION (same imgaug pipeline as WeatherPrompt)
# =============================================================================
def get_weather_augmenter(condition: str) -> iaa.Augmenter:
    """
    Exact same imgaug augmenters as WeatherPrompt source code.
    Returns identity for 'normal'.
    """
    if condition == "normal":
        return iaa.Identity()
    elif condition == "fog":
        return iaa.Fog()
    elif condition == "rain":
        return iaa.Rain(speed=(0.1, 0.3))
    elif condition == "snow":
        return iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))
    elif condition == "dark":
        return iaa.Multiply((0.3, 0.5))
    elif condition == "light":
        return iaa.Multiply((1.5, 2.0))
    elif condition == "fog_rain":
        return iaa.Sequential([iaa.Fog(), iaa.Rain(speed=(0.1, 0.3))])
    elif condition == "fog_snow":
        return iaa.Sequential([iaa.Fog(), iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))])
    elif condition == "rain_snow":
        return iaa.Sequential([iaa.Rain(speed=(0.1, 0.3)),
                               iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))])
    elif condition == "wind":
        return iaa.MotionBlur(k=15, angle=[-45, 45])
    else:
        return iaa.Identity()


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_base_transform(img_size=None):
    sz = img_size or CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


# =============================================================================
# DATASET — University-1652 Weather Eval
# =============================================================================
class Uni1652WeatherQueryDataset(Dataset):
    """
    Drone images from University-1652 test set (query_drone),
    with on-the-fly weather augmentation applied.
    """
    def __init__(self, root, weather_condition="normal", transform=None):
        self.transform = transform or get_base_transform()
        self.weather_aug = get_weather_augmenter(weather_condition)
        self.condition = weather_condition

        query_dir = os.path.join(root, "test", "query_drone")
        self.samples, self.labels = [], []
        if os.path.isdir(query_dir):
            for bid in sorted(os.listdir(query_dir)):
                bp = os.path.join(query_dir, bid)
                if not os.path.isdir(bp): continue
                for f in sorted(os.listdir(bp)):
                    if f.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(bp, f))
                        self.labels.append(int(bid))
        print(f"  [Uni-1652 Query '{weather_condition}'] {len(self.samples)} images")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        try:
            img = np.array(Image.open(path).convert('RGB'))
            if self.condition != "normal":
                img = self.weather_aug.augment_image(img)
            img = Image.fromarray(img.astype(np.uint8))
        except Exception:
            img = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))
        return self.transform(img), label


class Uni1652GalleryDataset(Dataset):
    """
    Satellite images from University-1652 test set (gallery_satellite).
    NO weather augmentation applied (satellite gallery stays clean).
    """
    def __init__(self, root, transform=None):
        self.transform = transform or get_base_transform()
        gallery_dir = os.path.join(root, "test", "gallery_satellite")
        self.samples, self.labels = [], []
        if os.path.isdir(gallery_dir):
            for bid in sorted(os.listdir(gallery_dir)):
                bp = os.path.join(gallery_dir, bid)
                if not os.path.isdir(bp): continue
                for f in sorted(os.listdir(bp)):
                    if f.endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append(os.path.join(bp, f))
                        self.labels.append(int(bid))
        print(f"  [Uni-1652 Gallery] {len(self.samples)} satellite images")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))
        return self.transform(img), self.labels[idx]


# =============================================================================
# MODEL — SPDGeo-DPE-MAR (no altitude, same as gpl_cross_uni2sues.py)
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

    def forward(self, x):
        features = self.model.forward_features(x)
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return features['x_norm_patchtokens'], features['x_norm_clstoken'], (H, W)


class SemanticPartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim*2),
                                    nn.GELU(), nn.Linear(part_dim*2, part_dim))
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat_norm = F.normalize(feat, dim=-1); proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1); assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H-1, 1)
        gx = torch.arange(W, device=device).float() / max(W-1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience, 'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim//2), nn.Tanh(), nn.Linear(part_dim//2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim*3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        return F.normalize(self.proj(torch.cat([attn_pool, mean_pool, max_pool], dim=-1)), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


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
        recon_norm = F.normalize(recon, dim=-1); target_norm = F.normalize(masked_targets, dim=-1)
        return (1 - (recon_norm * target_norm).sum(dim=-1)).mean()


class SPDGeoDPEMARModel(nn.Module):
    """
    Same as gpl_cross_uni2sues.py — NO altitude conditioning.
    num_classes is used only during training; for eval we use extract_embedding().
    """
    def __init__(self, num_classes=701, cfg=CFG):
        super().__init__()
        self.backbone    = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc   = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool        = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck     = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, 0.30)

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def forward(self, x, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'part_emb': emb, 'cls_emb': cls_emb_norm}
        if return_parts: out['parts'] = parts
        return out


# =============================================================================
# EVALUATION — University-1652 with Weather
# =============================================================================
@torch.no_grad()
def extract_features(model, paths, labels, weather_condition="normal", desc=""):
    """Extract features from a list of image paths with optional weather augmentation."""
    model.eval()
    tf = get_base_transform()
    aug = get_weather_augmenter(weather_condition) if weather_condition != "normal" else None
    feats, lbs = [], []
    batch_imgs, batch_labels = [], []

    for i, (path, label) in enumerate(tqdm(zip(paths, labels), total=len(paths), desc=desc, leave=False)):
        try:
            img = np.array(Image.open(path).convert('RGB'))
            if aug is not None:
                img = aug.augment_image(img)
            img = tf(Image.fromarray(img.astype(np.uint8)))
        except Exception:
            img = tf(Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128)))
        batch_imgs.append(img)
        batch_labels.append(label)

        if len(batch_imgs) == CFG.BATCH_SIZE or i == len(paths) - 1:
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                f = model.extract_embedding(torch.stack(batch_imgs).to(DEVICE))
            feats.append(f.cpu())
            lbs.extend(batch_labels)
            batch_imgs, batch_labels = [], []

    return torch.cat(feats), torch.tensor(lbs)


def compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels):
    """Compute R@1, R@5, R@10, mAP."""
    sim = query_feats @ gallery_feats.T
    _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0); r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        f = matches[0].item()
        if f < 1: r1 += 1
        if f < 5: r5 += 1
        if f < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100, 'N': N}


@torch.no_grad()
def evaluate_weather(model, uni_root):
    """
    Evaluate model on University-1652 under all 10 weather conditions.
    Protocol:
      - Gallery: satellite images (NO weather, constant)
      - Query D→S: drone images (weather augmented)
      - Query S→D: satellite images as query, drone as gallery (for consistency)
    """
    print("\n[Loading Gallery (satellite — no weather)]")
    # Load satellite gallery (constant — no weather)
    gallery_sat_paths, gallery_sat_labels = [], []
    gallery_dir = os.path.join(uni_root, "test", "gallery_satellite")
    for bid in sorted(os.listdir(gallery_dir)):
        bp = os.path.join(gallery_dir, bid)
        if not os.path.isdir(bp): continue
        for f in sorted(os.listdir(bp)):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                gallery_sat_paths.append(os.path.join(bp, f))
                gallery_sat_labels.append(int(bid))

    # Load drone query paths (will have weather applied per condition)
    query_drone_paths, query_drone_labels = [], []
    query_dir = os.path.join(uni_root, "test", "query_drone")
    for bid in sorted(os.listdir(query_dir)):
        bp = os.path.join(query_dir, bid)
        if not os.path.isdir(bp): continue
        for f in sorted(os.listdir(bp)):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                query_drone_paths.append(os.path.join(bp, f))
                query_drone_labels.append(int(bid))

    print(f"  Gallery: {len(gallery_sat_paths)} sat | Query: {len(query_drone_paths)} drone")

    # Extract satellite gallery features once (no weather)
    print("\n  Extracting satellite gallery features (no weather)...")
    gallery_feats, gallery_labels = extract_features(
        model, gallery_sat_paths, gallery_sat_labels, "normal", "Gallery")

    results = {}
    for condition in CFG.WEATHER_CONDITIONS:
        t0 = time.time()
        print(f"\n  [{condition.upper()}] Extracting features...")

        drone_feats, drone_labels = extract_features(
            model, query_drone_paths, query_drone_labels, condition, f"Drone[{condition}]")

        d2s = compute_metrics(drone_feats, drone_labels, gallery_feats, gallery_labels)
        elapsed = time.time() - t0

        results[condition] = {
            'D2S': d2s,
            'elapsed': elapsed
        }
        print(f"  ► [{condition}] D→S: R@1={d2s['R@1']:.2f}%  R@5={d2s['R@5']:.2f}%  "
              f"R@10={d2s['R@10']:.2f}%  mAP={d2s['mAP']:.2f}%  ({elapsed:.1f}s)")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 75)
    print(f"  {CFG.EXPERIMENT_NAME}")
    print(f"  Zero-Shot Weather Robustness Evaluation — University-1652")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  IMG_SIZE: {CFG.IMG_SIZE} | N_PARTS: {CFG.N_PARTS} | UNFREEZE: {CFG.UNFREEZE_BLOCKS}")
    print("=" * 75)

    # ---- Build Model ----
    print("\n[MODEL] Building SPDGeo-DPE-MAR (no altitude)...")
    # Use 701 as num_classes (Uni-1652 train size) — doesn't matter for inference
    model = SPDGeoDPEMARModel(num_classes=701, cfg=CFG).to(DEVICE)

    if os.path.exists(CFG.CHECKPOINT):
        ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
        # Handle various checkpoint formats
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {CFG.CHECKPOINT}")
        print(f"  Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"  Missing (first 5): {missing[:5]}")
    else:
        print(f"  WARNING: Checkpoint not found at {CFG.CHECKPOINT} — using random weights!")

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params/1e6:.2f}M")

    # ---- Evaluate ----
    t_total = time.time()
    results = evaluate_weather(model, CFG.UNI_ROOT)
    t_total = time.time() - t_total

    # ---- Summary ----
    print(f"\n{'=' * 75}")
    print(f"  RESULTS SUMMARY — {CFG.EXPERIMENT_NAME}")
    print(f"  University-1652 | Drone → Satellite (D2S) Retrieval")
    print(f"{'=' * 75}")
    print(f"  {'Condition':<12}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'mAP':>7}")
    print(f"  {'-'*50}")

    adverse_r1 = []
    for cond in CFG.WEATHER_CONDITIONS:
        m = results[cond]['D2S']
        print(f"  {cond:<12}  {m['R@1']:>6.2f}%  {m['R@5']:>6.2f}%  {m['R@10']:>6.2f}%  {m['mAP']:>6.2f}%")
        if cond != "normal":
            adverse_r1.append(m['R@1'])

    normal_r1 = results['normal']['D2S']['R@1']
    avg_adverse = sum(adverse_r1) / len(adverse_r1)
    avg_all = (normal_r1 + sum(adverse_r1)) / len(CFG.WEATHER_CONDITIONS)

    print(f"  {'-'*50}")
    print(f"  {'Avg(adverse)':<12}  {avg_adverse:>6.2f}%")
    print(f"  {'Avg(all)':<12}  {avg_all:>6.2f}%")
    print(f"\n  Normal R@1: {normal_r1:.2f}%")
    print(f"  Avg Adverse R@1: {avg_adverse:.2f}%")
    print(f"  Avg All R@1: {avg_all:.2f}%")
    print(f"  Total eval time: {t_total:.1f}s")
    print(f"{'=' * 75}")

    # WeatherPrompt comparison
    print(f"\n  === vs WeatherPrompt (NeurIPS'25 SOTA on Uni-1652 D→S) ===")
    wp_normal = 76.72; wp_avg = 62.52
    wp_per_cond = {
        'normal': 76.72, 'fog': 68.49, 'rain': 71.77, 'snow': 59.95,
        'dark': 40.42, 'light': 61.57, 'fog_rain': 58.24, 'fog_snow': 64.36,
        'rain_snow': 58.49, 'wind': 65.19
    }
    # Note: WeatherPrompt reports SUES-200 per-weather; uni-1652 summary is mean 77.14%
    print(f"  WeatherPrompt mean R@1 (D→S): 77.14% | Ours: {avg_all:.2f}%")
    print(f"  WeatherPrompt normal  R@1    : {wp_normal:.2f}% | Ours: {normal_r1:.2f}%")
    print(f"  Δ Normal: {normal_r1 - wp_normal:+.2f}%  |  Δ Mean: {avg_all - 77.14:+.2f}%")

    # ---- Save Results ----
    run_summary = {
        'experiment': CFG.EXPERIMENT_NAME,
        'dataset': 'University-1652',
        'task': 'Drone → Satellite under 10 weather conditions (zero-shot)',
        'checkpoint': CFG.CHECKPOINT,
        'img_size': CFG.IMG_SIZE,
        'normal_r1_d2s': normal_r1,
        'avg_adverse_r1_d2s': avg_adverse,
        'avg_all_r1_d2s': avg_all,
        'weather_results': {
            cond: {
                'R@1': results[cond]['D2S']['R@1'],
                'R@5': results[cond]['D2S']['R@5'],
                'R@10': results[cond]['D2S']['R@10'],
                'mAP': results[cond]['D2S']['mAP'],
            }
            for cond in CFG.WEATHER_CONDITIONS
        },
        'total_eval_time': t_total,
    }
    out_path = os.path.join(CFG.OUTPUT_DIR, f"{CFG.EXPERIMENT_NAME}_results.json")
    with open(out_path, 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
