# =============================================================================
# EXP56: Weather Fine-Tune with ONLINE Augmentation (GeoPartLoss Pipeline)
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Fine-tunes from the NEW best checkpoint (geopartloss_best.pth, 98.4% R@1)
# using WeatherPrompt's exact data strategy for FAIR COMPARISON:
#   - Online weather augmentation via imgaug (not pre-generated)
#   - 1 random drone image per location per epoch (like WeatherPrompt)
#   - Exact same imgaug weather augmenters from WeatherPrompt source code
#
# KEY DIFFERENCES from EXP52:
#   - EXP52: fine-tuned from EXP35 checkpoint (IMG=336, 12 separate losses)
#   - EXP56: fine-tunes from NEW BEST checkpoint (IMG=448, GeoPartLoss 4-group
#             adaptive uncertainty weighting — Kendall et al. 2018)
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone (train+test)
#   2. Weather synthetic:  weather-augmented drone TEST ONLY (for eval)
#   3. New best checkpoint:
#      /kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop", "imgaug"]:
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

# Monkey-patch for imgaug compatibility with NumPy 2.0
if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'float'):   np.float   = float
if not hasattr(np, 'int'):     np.int     = int
if not hasattr(np, 'bool'):    np.bool    = bool

import imgaug.augmenters as iaa


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Paths ---
    SUES_ROOT    = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    # NEW checkpoint — GeoPartLoss best (98.4% no-weather R@1)
    CHECKPOINT   = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth"
    OUTPUT_DIR   = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TRAIN_LOCS      = list(range(1, 121))
    TEST_LOCS       = list(range(121, 201))
    NUM_CLASSES     = 120

    # --- Model architecture (MUST match new geopartloss_best.pth) ---
    IMG_SIZE        = 448   # NEW vs EXP52 (was 336)
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

    # --- GeoPartLoss config (4-group adaptive) ---
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    EMA_DECAY           = 0.996
    RECON_WARMUP        = 0      # Already pretrained — no warmup needed
    DISTILL_TEMP        = 4.0
    NUM_LOSS_GROUPS     = 4

    # --- Fine-tune training ---
    NUM_EPOCHS      = 60
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 1e-4       # Fine-tune LR
    BACKBONE_LR     = 1e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 3
    USE_AMP         = True
    SEED            = 42

    # --- Eval ---
    EVAL_INTERVAL   = 5
    BATCH_SIZE      = 32         # Reduced: 448px images are larger
    NUM_WORKERS     = 2


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


# =============================================================================
# IMGAUG WEATHER AUGMENTERS (exact WeatherPrompt parameters)
# =============================================================================
def build_weather_augmenters():
    return [
        None,   # [0] normal
        iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
            intensity_coarse_scale=2, alpha_min=1.0, alpha_multiplier=0.9,
            alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            density_multiplier=0.5, seed=35)]),  # [1] fog
        iaa.Sequential([iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95)]),  # [2] rain
        iaa.Sequential([iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96)]),  # [3] snow
        iaa.Sequential([iaa.BlendAlpha(0.5, foreground=iaa.Add(100),
            background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991)]),  # [4] dark
        iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)]),  # [5] light
        iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
            intensity_coarse_scale=2, alpha_min=1.0, alpha_multiplier=0.9,
            alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            density_multiplier=0.5, seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36)]),  # [6] fog_rain
        iaa.Sequential([iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
            intensity_coarse_scale=2, alpha_min=1.0, alpha_multiplier=0.9,
            alpha_size_px_max=10, alpha_freq_exponent=-2, sparsity=0.9,
            density_multiplier=0.5, seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36)]),  # [7] fog_snow
        iaa.Sequential([iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74)]),  # [8] rain_snow
        iaa.Sequential([iaa.MotionBlur(15, seed=17)]),  # [9] wind
    ]


# =============================================================================
# MODEL (identical to New_Best_No_Weather98.4.py — GeoPartLoss pipeline)
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


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.output_dim = 768
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)['x_norm_clstoken']


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            return feat * self.gamma.mean(0, keepdim=True).unsqueeze(0) + self.beta.mean(0, keepdim=True).unsqueeze(0)
        return feat * self.gamma[alt_idx].unsqueeze(1) + self.beta[alt_idx].unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07, num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim*2), nn.GELU(), nn.Linear(part_dim*2, part_dim))
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.altitude_film(self.feat_proj(patch_features), alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass + 0  # refined below
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
        self.proj = nn.Sequential(nn.Linear(part_dim*3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        combined = torch.cat([(aw * part_features).sum(1), part_features.mean(1), part_features.max(1)[0]], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim//2), nn.ReLU(inplace=True), nn.Linear(embed_dim//2, 1))
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(nn.Linear(part_dim, part_dim*2), nn.GELU(), nn.Linear(part_dim*2, part_dim), nn.LayerNorm(part_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)

    def forward(self, projected_patches, part_features, assignment):
        B, N, D = projected_patches.shape
        num_mask = int(N * self.mask_ratio)
        ids_shuffle = torch.rand(B, N, device=projected_patches.device).argsort(dim=1)
        mask_indices = ids_shuffle[:, :num_mask]
        target = projected_patches.detach()
        mask_expand = mask_indices.unsqueeze(-1).expand(-1, -1, D)
        masked_targets = torch.gather(target, 1, mask_expand)
        K = part_features.shape[1]
        masked_assign = torch.gather(assignment, 1, mask_indices.unsqueeze(-1).expand(-1, -1, K))
        recon = self.decoder(torch.bmm(masked_assign, part_features))
        return (1 - (F.normalize(recon, dim=-1) * F.normalize(masked_targets, dim=-1)).sum(dim=-1)).mean()


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, embedding, alt_target):
        return F.smooth_l1_loss(self.head(embedding).squeeze(-1), alt_target)


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone   = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc  = AltitudeAwarePartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        self.pool       = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM), nn.LayerNorm(cfg.TEACHER_DIM))
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

    def forward(self, x, alt_idx=None, return_parts=False):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        bn = self.bottleneck(emb); logits = self.classifier(bn)
        cls_emb_raw = self.cls_proj(cls_tok)
        cls_logits = self.cls_classifier(cls_emb_raw)
        cls_emb_norm = F.normalize(cls_emb_raw, dim=-1)
        fused = self.fusion_gate(emb, cls_emb_norm)
        projected_feat = self.teacher_proj(emb)
        out = {'embedding': fused, 'logits': logits, 'cls_logits': cls_logits,
               'projected_feat': projected_feat, 'part_emb': emb, 'cls_emb': cls_emb_norm}
        if return_parts: out['parts'] = parts
        return out


class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.model.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# GEOPARTLOSS — 4-group adaptive uncertainty weighting
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__(); self.log_t = nn.Parameter(torch.tensor(temp).log())
    def forward(self, q_emb, r_emb, labels):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.t() / t
        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()

class CrossDistillationLoss(nn.Module):
    def forward(self, s, t):
        s = F.normalize(s, dim=-1); t = F.normalize(t, dim=-1)
        return F.mse_loss(s, t) + (1.0 - F.cosine_similarity(s, t).mean())

class EMADistillationLoss(nn.Module):
    def forward(self, s, e): return (1 - F.cosine_similarity(s, e)).mean()

class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    @staticmethod
    def _entropy(logits):
        p = F.softmax(logits, dim=1); return -(p * (p + 1e-8).log()).sum(1).mean()
    def forward(self, drone_logits, sat_logits):
        T = self.T0 * (1 + torch.sigmoid(self._entropy(drone_logits) - self._entropy(sat_logits)))
        return (T**2) * F.kl_div(F.log_softmax(drone_logits / T, 1), F.softmax(sat_logits / T, 1).detach(), reduction='batchmean')

class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin; self.alpha = alpha; self.num_classes = num_classes
    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=-1); sim = embeddings @ P.T
        one_hot = F.one_hot(labels, self.num_classes).float()
        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        has_pos = one_hot.sum(0) > 0
        pos_loss = torch.log(1 + pos_exp.sum(0))[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0., device=embeddings.device)
        neg_mask = 1 - one_hot
        neg_loss = torch.log(1 + torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask).sum(0).mean()
        return pos_loss + neg_loss

class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1); sim = P @ P.T; K = sim.size(0)
        return (sim * (1 - torch.eye(K, device=sim.device))).abs().sum() / (K * (K - 1))


class GeoPartLoss(nn.Module):
    """4-group adaptive uncertainty weighting (Kendall et al. 2018).
    G1: Alignment  G2: Distillation  G3: Part Quality  G4: Altitude"""
    def __init__(self, num_classes, embed_dim, num_groups=4, cfg=None):
        super().__init__()
        cfg = cfg or CFG
        self.log_vars = nn.Parameter(torch.zeros(num_groups))
        self.infonce = SupInfoNCELoss(temp=0.05)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.proxy_anchor = ProxyAnchorLoss(num_classes, embed_dim, cfg.PROXY_MARGIN, cfg.PROXY_ALPHA)
        self.uapa = UAPALoss(base_temperature=cfg.DISTILL_TEMP)
        self.cross_dist = CrossDistillationLoss()
        self.ema_dist = EMADistillationLoss()
        self.diversity = PrototypeDiversityLoss()
        self.recon_warmup = cfg.RECON_WARMUP
        print(f"  GeoPartLoss: {num_groups} groups, adaptive uncertainty weighting")

    def forward(self, d_out, s_out, labels, model, teacher, ema, drone, sat, alt_idx, alt_norm, epoch, device):
        # G1: Alignment
        l_ce = (self.ce(d_out['logits'], labels) + self.ce(s_out['logits'], labels))
        l_ce += 0.3 * (self.ce(d_out['cls_logits'], labels) + self.ce(s_out['cls_logits'], labels))
        l_nce = self.infonce(d_out['embedding'], s_out['embedding'], labels)
        l_proxy = 0.5 * (self.proxy_anchor(d_out['embedding'], labels) + self.proxy_anchor(s_out['embedding'], labels))
        l_uapa  = self.uapa(d_out['logits'], s_out['logits'])
        L_align = l_ce + l_nce + l_proxy + l_uapa
        ld = {'ce': l_ce.item(), 'nce': l_nce.item(), 'proxy': l_proxy.item(), 'uapa': l_uapa.item()}

        # G2: Distillation
        if teacher is not None:
            with torch.no_grad(): t_drone = teacher(drone); t_sat = teacher(sat)
            l_cross = self.cross_dist(d_out['projected_feat'], t_drone) + self.cross_dist(s_out['projected_feat'], t_sat)
        else:
            l_cross = torch.tensor(0., device=device)
        with torch.no_grad():
            ema_d = ema.forward(drone, alt_idx=alt_idx); ema_s = ema.forward(sat, alt_idx=None)
        l_ema = 0.5 * (self.ema_dist(d_out['embedding'], ema_d) + self.ema_dist(s_out['embedding'], ema_s))
        L_distill = l_cross + l_ema
        ld.update({'cross': l_cross.item() if torch.is_tensor(l_cross) else l_cross, 'ema': l_ema.item()})

        # G3: Part Quality
        if epoch >= self.recon_warmup:
            l_recon = 0.5 * (model.mask_recon(d_out['parts']['projected_patches'], d_out['parts']['part_features'], d_out['parts']['assignment']) +
                             model.mask_recon(s_out['parts']['projected_patches'], s_out['parts']['part_features'], s_out['parts']['assignment']))
        else:
            l_recon = torch.tensor(0., device=device)
        l_div = self.diversity(model.part_disc.prototypes)
        L_part = l_recon + l_div
        ld.update({'recon': l_recon.item() if torch.is_tensor(l_recon) else l_recon, 'div': l_div.item()})

        # G4: Altitude
        l_alt = model.alt_pred(d_out['embedding'].detach(), alt_norm)
        L_alt = l_alt
        ld['alt_p'] = l_alt.item()

        # Adaptive weighting
        groups = [L_align, L_distill, L_part, L_alt]
        group_names = ['align', 'distill', 'part', 'alt']
        total_loss = torch.tensor(0., device=device)
        gw = {}
        for i, (L, name) in enumerate(zip(groups, group_names)):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * L + self.log_vars[i]
            gw[name] = precision.item()
            ld[f'G_{name}'] = L.item()
        ld['total'] = total_loss.item()
        return total_loss, ld, gw


# =============================================================================
# DATASETS
# =============================================================================
class OnlineWeatherTrainDataset(Dataset):
    def __init__(self, sues_root, drone_transform=None, sat_transform=None, weather_augmenters=None):
        self.drone_transform = drone_transform
        self.sat_transform = sat_transform
        self.weather_augmenters = weather_augmenters or []
        self.iaa_spatial = iaa.Sequential([
            iaa.Resize({"height": CFG.IMG_SIZE, "width": CFG.IMG_SIZE}, interpolation=3),
            iaa.Pad(px=10, pad_mode="edge", keep_size=False),
            iaa.CropToFixedSize(width=CFG.IMG_SIZE, height=CFG.IMG_SIZE),
            iaa.Fliplr(0.5),
        ])
        self.sat_dir = os.path.join(sues_root, CFG.SAT_DIR)
        self.loc_buckets = defaultdict(list)
        train_locs = [f"{l:04d}" for l in CFG.TRAIN_LOCS]
        self.loc_to_idx = {l: i for i, l in enumerate(train_locs)}
        self.loc_list = train_locs
        drone_dir = os.path.join(sues_root, CFG.DRONE_DIR)
        for loc_name in train_locs:
            loc_idx = self.loc_to_idx[loc_name]
            for alt in CFG.ALTITUDES:
                alt_dir = os.path.join(drone_dir, loc_name, alt)
                if not os.path.isdir(alt_dir): continue
                for fname in sorted(os.listdir(alt_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.loc_buckets[loc_idx].append((os.path.join(alt_dir, fname), alt))
        self.drone_by_location = defaultdict(list)
        for loc_idx in self.loc_buckets: self.drone_by_location[loc_idx].append(loc_idx)
        print(f"  [online_weather_train] {len(self.loc_buckets)} locations, "
              f"{sum(len(v) for v in self.loc_buckets.values())} total drone images")

    def __len__(self): return len(self.loc_buckets)

    def __getitem__(self, idx):
        drone_path, alt = random.choice(self.loc_buckets[idx])
        try: drone_img = Image.open(drone_path).convert('RGB')
        except: drone_img = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))
        sat_path = os.path.join(self.sat_dir, self.loc_list[idx], "0.png")
        try: sat_img = Image.open(sat_path).convert('RGB')
        except: sat_img = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128))
        drone_np = np.array(drone_img)
        if self.weather_augmenters:
            aug = self.weather_augmenters[random.randint(0, len(self.weather_augmenters)-1)]
            if aug is not None: drone_np = aug(image=drone_np)
        drone_np = self.iaa_spatial(image=drone_np)
        drone_img = Image.fromarray(drone_np)
        if self.drone_transform: drone_img = self.drone_transform(drone_img)
        if self.sat_transform:   sat_img   = self.sat_transform(sat_img)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0
        return {'drone': drone_img, 'satellite': sat_img, 'label': idx,
                'altitude': int(alt), 'alt_idx': alt_idx, 'alt_norm': alt_norm}


class PKSamplerOnline:
    def __init__(self, ds, p, k): self.ds = ds; self.p = p; self.k = k; self.locs = list(ds.loc_buckets.keys())
    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            batch.extend([l]*self.k)
            if len(batch) >= self.p*self.k: yield batch[:self.p*self.k]; batch = batch[self.p*self.k:]
    def __len__(self): return len(self.locs) // self.p


class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, sues_root, weather_name, transform=None):
        self.transform = transform; self.weather_name = weather_name; self.samples = []
        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}
        if not os.path.isdir(weather_test_dir): return
        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path): continue
            parts = class_dir_name.split("_")
            if len(parts) != 2: continue
            loc_id, alt = parts[0], parts[1]
            if loc_id not in loc_to_idx: continue
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem: continue
                if stem.rsplit('-',1)[-1] == weather_name:
                    self.samples.append((os.path.join(class_path, fname), loc_to_idx[loc_id], alt))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label, alt = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform: img = self.transform(img)
        return {'drone': img, 'label': label, 'altitude': int(alt), 'alt_idx': CFG.ALT_TO_IDX.get(alt, 0)}


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transforms(mode="train", img_size=None):
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(),
                                   transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    return transforms.Compose([transforms.Resize((sz, sz)), transforms.ToTensor(),
                                transforms.Normalize([.485,.456,.406],[.229,.224,.225])])


# =============================================================================
# EVAL UTILITIES
# =============================================================================
def build_satellite_gallery(model, sues_root, device, transform):
    sat_dir = os.path.join(sues_root, CFG.SAT_DIR)
    test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]; loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    all_locs = [f"{l:04d}" for l in range(1, 201)]
    sat_imgs, sat_labels = [], []; distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp): continue
        sat_imgs.append(transform(Image.open(sp).convert('RGB')))
        if loc in loc_to_idx: sat_labels.append(loc_to_idx[loc])
        else: sat_labels.append(-1000 - distractor_cnt); distractor_cnt += 1
    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i+CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                sat_feats.append(model.extract_embedding(batch, alt_idx=None).cpu())
    return torch.cat(sat_feats), torch.tensor(sat_labels)


@torch.no_grad()
def evaluate_weather(model, sues_root, weather_root, device, transform):
    """Evaluate on all 10 weather conditions; return dict of R@1 per weather."""
    sat_feats, sat_labels = build_satellite_gallery(model, sues_root, device, transform)
    results = {}
    for w_name in CFG.WEATHER_NAMES:
        ds = WeatherQueryDataset(weather_root, sues_root, w_name, transform)
        if len(ds) == 0: results[w_name] = 0.0; continue
        loader = DataLoader(ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)
        drone_feats, drone_labels = [], []
        for batch in loader:
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch['drone'].to(device), alt_idx=batch['alt_idx'].to(device))
            drone_feats.append(feats.cpu()); drone_labels.append(batch['label'])
        drone_feats = torch.cat(drone_feats); drone_labels = torch.cat(drone_labels)
        sim = drone_feats @ sat_feats.T; _, rank = sim.sort(1, descending=True)
        N = drone_feats.size(0); r1 = 0
        for i in range(N):
            matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
            if len(matches) > 0 and matches[0].item() < 1: r1 += 1
        results[w_name] = r1 / N * 100
    return results


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, geo_loss, optimizer, scaler, device, epoch):
    model.train(); teacher.eval() if teacher else None
    total_sum = 0; n = 0
    loss_sums = defaultdict(float); all_weights = defaultdict(float)
    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone = batch['drone'].to(device); sat = batch['satellite'].to(device)
        labels = batch['label'].to(device); alt_idx = batch['alt_idx'].to(device)
        alt_norm = batch['alt_norm'].to(device).float()
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat,   alt_idx=None,    return_parts=True)
            loss, ld, gw = geo_loss(d_out, s_out, labels, model, teacher, ema,
                                    drone, sat, alt_idx, alt_norm, epoch, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update(); ema.update(model)
        total_sum += loss.item(); n += 1
        for k, v in ld.items(): loss_sums[k] += v
        for k, v in gw.items(): all_weights[k] += v
    return total_sum/max(n,1), {k: v/max(n,1) for k,v in loss_sums.items()}, {k: v/max(n,1) for k,v in all_weights.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  EXP56: Weather Fine-Tune — GeoPartLoss Pipeline")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  IMG_SIZE: {CFG.IMG_SIZE} | Epochs: {CFG.NUM_EPOCHS} | Device: {DEVICE}")
    print("=" * 65)

    # Build model and load checkpoint
    print("\nBuilding model…")
    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES, CFG).to(DEVICE)
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded checkpoint (epoch={ckpt.get('epoch','?')}, R@1={ckpt.get('metrics',{}).get('R@1','?')})")
    if missing:    print(f"  [WARN] Missing: {len(missing)} keys")
    if unexpected: print(f"  [WARN] Unexpected: {len(unexpected)} keys")

    # Teacher and EMA
    teacher = None
    try: teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e: print(f"  [WARN] Teacher not loaded: {e}")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)

    # GeoPartLoss
    geo_loss = GeoPartLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM, CFG.NUM_LOSS_GROUPS, CFG).to(DEVICE)

    # Weather augmenters and datasets
    weather_augs = build_weather_augmenters()
    normalize_tf = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = get_transforms("test")

    train_ds = OnlineWeatherTrainDataset(CFG.SUES_ROOT, drone_transform=normalize_tf,
                                         sat_transform=eval_tf, weather_augmenters=weather_augs)
    train_loader = DataLoader(train_ds, batch_sampler=PKSamplerOnline(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Optimizer
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,       'lr': CFG.BACKBONE_LR},
        {'params': head_params,           'lr': CFG.LR},
        {'params': geo_loss.parameters(), 'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_avg_r1 = 0.0; train_start = datetime.datetime.now().isoformat()

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS: lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = max(0.5 * (1 + math.cos(math.pi * progress)), 0.01)
        for i, pg in enumerate(optimizer.param_groups):
            base = [CFG.BACKBONE_LR, CFG.LR, CFG.LR][i]
            pg['lr'] = base * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld, gw = train_one_epoch(model, teacher, ema, train_loader, geo_loss, optimizer, scaler, DEVICE, epoch)
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"G_align {ld.get('G_align',0):.3f}(w={gw.get('align',0):.2f})  "
              f"G_dist {ld.get('G_distill',0):.3f}(w={gw.get('distill',0):.2f})  "
              f"G_part {ld.get('G_part',0):.3f}(w={gw.get('part',0):.2f})  "
              f"G_alt {ld.get('G_alt',0):.3f}(w={gw.get('alt',0):.2f}) | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            weather_results = evaluate_weather(model, CFG.SUES_ROOT, CFG.WEATHER_ROOT, DEVICE, eval_tf)
            adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
            avg_adv_r1 = np.mean([weather_results[w] for w in adv_weathers])
            avg_all_r1 = np.mean([weather_results[w] for w in CFG.WEATHER_NAMES])
            print(f"  ► Weather eval: normal={weather_results['normal']:.2f}%  "
                  f"avg_adv={avg_adv_r1:.2f}%  avg_all={avg_all_r1:.2f}%")
            for w, r1 in weather_results.items():
                print(f"    {w:>12s}: {r1:.2f}%")
            if avg_adv_r1 > best_avg_r1:
                best_avg_r1 = avg_adv_r1
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'weather_results': weather_results, 'avg_adv_r1': avg_adv_r1},
                           os.path.join(CFG.OUTPUT_DIR, 'exp56_weather_finetune_online_best.pth'))
                print(f"  ★ New best avg_adv_r1: {best_avg_r1:.2f}%! Saved.")

            # Also check EMA
            ema_results = evaluate_weather(ema.model, CFG.SUES_ROOT, CFG.WEATHER_ROOT, DEVICE, eval_tf)
            ema_adv = np.mean([ema_results[w] for w in adv_weathers])
            print(f"  ► EMA avg_adv_r1: {ema_adv:.2f}%")
            if ema_adv > best_avg_r1:
                best_avg_r1 = ema_adv
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'weather_results': ema_results, 'avg_adv_r1': ema_adv, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, 'exp56_weather_finetune_online_best.pth'))
                print(f"  ★ New best avg_adv_r1 (EMA): {best_avg_r1:.2f}%! Saved.")

    print(f"\n{'='*65}")
    print(f"  EXP56 Complete — Best avg_adv_r1: {best_avg_r1:.2f}%")
    print(f"  Checkpoint: {CFG.OUTPUT_DIR}/exp56_weather_finetune_online_best.pth")
    print(f"  → Run EXP55 with EXPERIMENT_NAME='EXP55_ZeroShot_EXP56' to evaluate")
    print(f"{'='*65}")

    # Save run summary
    run_summary = {
        'experiment': 'EXP56_WeatherFinetune_GeoPartLoss',
        'source_checkpoint': CFG.CHECKPOINT,
        'img_size': CFG.IMG_SIZE,
        'timestamp': {'start': train_start, 'end': datetime.datetime.now().isoformat()},
        'best_avg_adv_r1': best_avg_r1,
        'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')},
    }
    with open(os.path.join(CFG.OUTPUT_DIR, 'exp56_summary.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)


if __name__ == '__main__':
    main()
