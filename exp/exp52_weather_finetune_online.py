# =============================================================================
# EXP52: Weather Fine-Tune with ONLINE Augmentation (WeatherPrompt-style)
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Fine-tunes SPDGeo-DPEA-MAR from EXP35 checkpoint using WeatherPrompt's
# exact data strategy for FAIR COMPARISON:
#   - Online weather augmentation via imgaug (not pre-generated)
#   - 1 random drone image per location per epoch (like WeatherPrompt)
#   - Exact same imgaug weather augmenters from WeatherPrompt source code
#
# Key difference from EXP50:
#   - EXP50: used 4800 pre-generated weather images (120 loc × 4 alt × 10 weathers)
#   - EXP52: uses original drone images, weather applied on-the-fly
#            1 random image per location per epoch (matching WeatherPrompt)
#
# Evaluation: still uses pre-generated weather test images for consistency
#             with EXP49/50/51
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone (train+test)
#   2. Weather synthetic:  weather-augmented drone TEST ONLY (for eval)
#   3. EXP35 checkpoint:   pre-trained model weights
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
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'bool'):
    np.bool = bool

import imgaug.augmenters as iaa


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Paths ---
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
    TRAIN_LOCS      = list(range(1, 121))     # 120 train locations
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120                      # train classes

    # --- Model architecture (must match EXP35) ---
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

    # --- Training (fine-tune from EXP35 + online augmentation) ---
    NUM_EPOCHS      = 60          # Fine-tune epochs
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 1e-4        # Fine-tune LR
    BACKBONE_LR     = 1e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 3
    USE_AMP         = True
    SEED            = 42

    # --- Loss weights (same as EXP35) ---
    LAMBDA_CE           = 1.0
    LAMBDA_INFONCE      = 1.0
    LAMBDA_CONSISTENCY  = 0.1
    LAMBDA_CROSS_DIST   = 0.3
    LAMBDA_SELF_DIST    = 0.3
    LAMBDA_UAPA         = 0.2
    LAMBDA_PROXY        = 0.5
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    LAMBDA_EMA_DIST     = 0.2
    EMA_DECAY           = 0.996
    LAMBDA_ALT_CONSIST  = 0.2
    MASK_RATIO          = 0.30
    LAMBDA_MASK_RECON   = 0.3
    RECON_WARMUP        = 0       # No warmup (already pretrained)
    LAMBDA_ALT_PRED     = 0.15
    LAMBDA_DIVERSITY    = 0.05
    DISTILL_TEMP        = 4.0

    # --- Eval ---
    EVAL_INTERVAL   = 5
    BATCH_SIZE      = 64
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
    """
    Exact imgaug weather augmenters from WeatherPrompt source code (train.py).
    Index 0 = None (normal/clean), indices 1-9 = weather conditions.
    """
    return [
        # [0] normal — no augmentation
        None,
        # [1] fog
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2,
                intensity_coarse_scale=2, alpha_min=1.0,
                alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9,
                density_multiplier=0.5, seed=35
            )
        ]),
        # [2] rain
        iaa.Sequential([
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
        ]),
        # [3] snow
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
        ]),
        # [4] dark
        iaa.Sequential([
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100), background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),
        ]),
        # [5] light
        iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992)
        ]),
        # [6] fog_rain
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2,
                intensity_coarse_scale=2, alpha_min=1.0,
                alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9,
                density_multiplier=0.5, seed=35
            ),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),
        ]),
        # [7] fog_snow
        iaa.Sequential([
            iaa.CloudLayer(
                intensity_mean=225, intensity_freq_exponent=-2,
                intensity_coarse_scale=2, alpha_min=1.0,
                alpha_multiplier=0.9, alpha_size_px_max=10,
                alpha_freq_exponent=-2, sparsity=0.9,
                density_multiplier=0.5, seed=35
            ),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),
        ]),
        # [8] rain_snow
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2), speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        ]),
        # [9] wind
        iaa.Sequential([
            iaa.MotionBlur(15, seed=17)
        ]),
    ]


# =============================================================================
# MODEL DEFINITION (identical to EXP35)
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
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DINOv2Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 teacher …")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=True)
        self.feature_dim = 768
        for p in self.model.parameters(): p.requires_grad = False
        print("  DINOv2 ViT-B/14 teacher loaded (all frozen).")

    def forward(self, x):
        features = self.model.forward_features(x)
        return F.normalize(features['x_norm_clstoken'], dim=-1)


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

    def forward(self, projected_patches, part_features, assignment):
        B, K, D = part_features.shape
        n_mask = max(1, int(K * self.mask_ratio))
        mask_idx = torch.randperm(K, device=part_features.device)[:n_mask]
        masked = part_features.clone()
        masked[:, mask_idx] = self.mask_token.expand(B, n_mask, -1)
        reconstructed = self.decoder(masked)
        loss = F.mse_loss(reconstructed[:, mask_idx], part_features[:, mask_idx].detach())
        return loss


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.mse_loss(pred, alt_target)


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1)
        sim = P @ P.T
        K = sim.size(0)
        mask = 1 - torch.eye(K, device=sim.device)
        return (sim * mask).abs().sum() / (K * (K - 1))


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
        print(f"  SPDGeo-DPEA-MAR student: {total/1e6:.1f}M trainable parameters")

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


# =============================================================================
# EMA Model
# =============================================================================
class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x, alt_idx=None):
        return self.model.extract_embedding(x, alt_idx=alt_idx)


# =============================================================================
# LOSSES (identical to EXP35)
# =============================================================================
class SupInfoNCELoss(nn.Module):
    def __init__(self, temp=0.05):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(temp).log())

    def forward(self, q_emb, r_emb, labels):
        t = self.log_t.exp().clamp(0.01, 1.0)
        sim = q_emb @ r_emb.t() / t
        labels = labels.view(-1, 1); pos_mask = labels.eq(labels.T).float()
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        return (-(log_prob * pos_mask).sum(1) / pos_mask.sum(1).clamp(min=1)).mean()


class PartConsistencyLoss(nn.Module):
    def forward(self, assign_q, assign_r):
        dist_q = assign_q.mean(dim=1); dist_r = assign_r.mean(dim=1)
        kl_qr = F.kl_div((dist_q + 1e-8).log(), dist_r, reduction='batchmean', log_target=False)
        kl_rq = F.kl_div((dist_r + 1e-8).log(), dist_q, reduction='batchmean', log_target=False)
        return 0.5 * (kl_qr + kl_rq)


class CrossDistillationLoss(nn.Module):
    def forward(self, student_feat, teacher_feat):
        s = F.normalize(student_feat, dim=-1); t = F.normalize(teacher_feat, dim=-1)
        return F.mse_loss(s, t) + (1.0 - F.cosine_similarity(s, t).mean())


class SelfDistillationLoss(nn.Module):
    def __init__(self, temperature=4.0): super().__init__(); self.T = temperature
    def forward(self, weak_logits, strong_logits):
        p_teacher = F.softmax(strong_logits / self.T, dim=1).detach()
        p_student = F.log_softmax(weak_logits / self.T, dim=1)
        return (self.T ** 2) * F.kl_div(p_student, p_teacher, reduction='batchmean')


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    @staticmethod
    def _entropy(logits):
        probs = F.softmax(logits, dim=1)
        return -(probs * (probs + 1e-8).log()).sum(dim=1).mean()
    def forward(self, drone_logits, sat_logits):
        delta_U = self._entropy(drone_logits) - self._entropy(sat_logits)
        T = self.T0 * (1 + torch.sigmoid(delta_U))
        p_sat = F.softmax(sat_logits / T, dim=1).detach()
        return (T ** 2) * F.kl_div(F.log_softmax(drone_logits / T, dim=1), p_sat, reduction='batchmean')


class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embed_dim, margin=0.1, alpha=32):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.01)
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        self.margin = margin; self.alpha = alpha; self.num_classes = num_classes

    def forward(self, embeddings, labels):
        P = F.normalize(self.proxies, dim=-1)
        sim = embeddings @ P.T
        one_hot = F.one_hot(labels, self.num_classes).float()
        pos_exp = torch.exp(-self.alpha * (sim * one_hot - self.margin)) * one_hot
        P_plus = one_hot.sum(0); has_pos = P_plus > 0
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = pos_term[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        neg_loss = torch.log(1 + neg_exp.sum(0)).mean()
        return pos_loss + neg_loss


class EMADistillationLoss(nn.Module):
    def forward(self, student_emb, ema_emb):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()


class AltitudeConsistencyLoss(nn.Module):
    def forward(self, embeddings, labels, altitudes):
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device)
        lbl = labels.view(-1, 1); alt = altitudes.view(-1, 1)
        same_loc = lbl.eq(lbl.T); diff_alt = alt.ne(alt.T)
        mask = same_loc & diff_alt
        if mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        emb_norm = F.normalize(embeddings, dim=-1)
        cos_sim = emb_norm @ emb_norm.T
        cos_dist = 1 - cos_sim
        loss = (cos_dist * mask.float()).sum() / mask.float().sum().clamp(min=1)
        return loss


# =============================================================================
# ONLINE WEATHER TRAINING DATASET (WeatherPrompt-style)
# =============================================================================
class OnlineWeatherTrainDataset(Dataset):
    """
    WeatherPrompt-style training dataset:
      - Loads ORIGINAL drone images from SUES-200 (no pre-generated weathers)
      - Applies imgaug weather augmentation ONLINE in __getitem__
      - 1 random image per location per epoch (select=True behavior)
      - Drone pipeline IDENTICAL to WeatherPrompt:
          weather(imgaug) → spatial(Resize+Pad+Crop+Flip via imgaug) → ToTensor+Normalize

    Each location bucket contains all drone images across altitudes.
    __len__ = number of locations (120), NOT total images.
    """
    def __init__(self, sues_root, drone_transform=None, sat_transform=None, weather_augmenters=None):
        self.drone_transform = drone_transform  # ToTensor+Normalize only (WeatherPrompt-style)
        self.sat_transform = sat_transform
        self.weather_augmenters = weather_augmenters or []
        # WeatherPrompt-identical spatial augment (iaa_drone_transform in train.py)
        self.iaa_spatial = iaa.Sequential([
            iaa.Resize({"height": CFG.IMG_SIZE, "width": CFG.IMG_SIZE}, interpolation=3),
            iaa.Pad(px=10, pad_mode="edge", keep_size=False),
            iaa.CropToFixedSize(width=CFG.IMG_SIZE, height=CFG.IMG_SIZE),
            iaa.Fliplr(0.5),
        ])
        self.sat_dir = os.path.join(sues_root, CFG.SAT_DIR)

        # Build per-location buckets: {loc_idx: [(drone_path, alt), ...]}
        self.loc_buckets = defaultdict(list)
        train_locs = [f"{l:04d}" for l in CFG.TRAIN_LOCS]
        self.loc_to_idx = {l: i for i, l in enumerate(train_locs)}
        self.loc_list = train_locs  # ordered list

        drone_dir = os.path.join(sues_root, CFG.DRONE_DIR)
        for loc_name in train_locs:
            loc_idx = self.loc_to_idx[loc_name]
            for alt in CFG.ALTITUDES:
                alt_dir = os.path.join(drone_dir, loc_name, alt)
                if not os.path.isdir(alt_dir):
                    continue
                for fname in sorted(os.listdir(alt_dir)):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.loc_buckets[loc_idx].append(
                            (os.path.join(alt_dir, fname), alt)
                        )

        # drone_by_location for PKSampler compatibility
        self.drone_by_location = {k: list(range(len(v))) for k, v in self.loc_buckets.items()}
        # But PKSampler uses indices into samples list — we need a different approach
        # Since __len__ = num_locations, indices ARE location indices
        self.drone_by_location = defaultdict(list)
        for loc_idx in self.loc_buckets:
            self.drone_by_location[loc_idx].append(loc_idx)

        total_imgs = sum(len(v) for v in self.loc_buckets.values())
        print(f"  [online_weather_train] {len(self.loc_buckets)} locations, "
              f"{total_imgs} total drone images")
        print(f"  [online_weather_train] {len(self.weather_augmenters)} weather augmenters "
              f"(applied online)")
        print(f"  [online_weather_train] 1 random image per location per access "
              f"(WeatherPrompt-style)")

    def __len__(self):
        return len(self.loc_buckets)

    def __getitem__(self, idx):
        # idx = location index → pick 1 random drone image from this location
        bucket = self.loc_buckets[idx]
        drone_path, alt = random.choice(bucket)

        # Load images
        try:
            drone_img = Image.open(drone_path).convert('RGB')
        except Exception:
            sz = CFG.IMG_SIZE
            drone_img = Image.new('RGB', (sz, sz), (128, 128, 128))

        sat_path = os.path.join(self.sat_dir, self.loc_list[idx], "0.png")
        try:
            sat_img = Image.open(sat_path).convert('RGB')
        except Exception:
            sz = CFG.IMG_SIZE
            sat_img = Image.new('RGB', (sz, sz), (128, 128, 128))

        # Drone pipeline — IDENTICAL to WeatherPrompt:
        #   1. weather augmentation (imgaug)
        #   2. spatial augmentation: Resize→Pad→CropToFixed→Fliplr (imgaug)
        #   3. ToTensor + Normalize (torchvision)
        drone_np = np.array(drone_img)
        if self.weather_augmenters:
            w_idx = random.randint(0, len(self.weather_augmenters) - 1)
            augmenter = self.weather_augmenters[w_idx]
            if augmenter is not None:
                drone_np = augmenter(image=drone_np)
        drone_np = self.iaa_spatial(image=drone_np)
        drone_img = Image.fromarray(drone_np)

        if self.drone_transform:
            drone_img = self.drone_transform(drone_img)
        if self.sat_transform:
            sat_img = self.sat_transform(sat_img)

        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        alt_norm = (int(alt) - 150) / 150.0
        return {'drone': drone_img, 'satellite': sat_img, 'label': idx,
                'altitude': int(alt), 'alt_idx': alt_idx, 'alt_norm': alt_norm}


class PKSamplerOnline:
    """
    PK-sampler adapted for OnlineWeatherTrainDataset where each index
    is a location. Yields batches of P locations × K repeated accesses.
    Each access picks a fresh random image + random weather.
    """
    def __init__(self, ds, p, k):
        self.ds = ds; self.p = p; self.k = k
        self.locs = list(ds.loc_buckets.keys())

    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            # Each access to same location picks a different random image + weather
            batch.extend([l] * self.k)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]

    def __len__(self):
        return len(self.locs) // self.p


# =============================================================================
# WEATHER QUERY DATASET (for evaluation — uses pre-generated test images)
# =============================================================================
class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, sues_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name

        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        self.samples = []

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        if not os.path.isdir(weather_test_dir):
            return

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
# TRANSFORMS
# =============================================================================
def get_transforms(mode="train", img_size=None):
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


# =============================================================================
# SATELLITE GALLERY BUILDER
# =============================================================================
def build_satellite_gallery(model, sues_root, device, transform):
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


# =============================================================================
# RETRIEVAL EVALUATION
# =============================================================================
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


# =============================================================================
# SATELLITE→DRONE EVALUATION (S2D)
# =============================================================================
def build_drone_gallery_s2d(model, weather_root, sues_root, weather_name, device, transform):
    """
    Builds drone gallery for Satellite→Drone (S2D) retrieval.
    Gallery = weather-augmented test drones (labeled 0..79) +
              clean train drones as distractors (labeled -1000...).
    """
    test_locs  = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    train_locs = [f"{l:04d}" for l in CFG.TRAIN_LOCS]

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
# TRAINING
# =============================================================================
def train_one_epoch(model, teacher, ema, loader, losses, new_losses, optimizer,
                    scaler, device, epoch):
    model.train()
    if teacher: teacher.eval()

    infonce, ce, consist, cross_dist, self_dist, uapa = losses
    proxy_anchor, ema_dist, alt_consist, diversity_loss = new_losses

    # Reconstruction warmup
    if epoch <= CFG.RECON_WARMUP:
        recon_weight = CFG.LAMBDA_MASK_RECON * (epoch / max(CFG.RECON_WARMUP, 1))
    else:
        recon_weight = CFG.LAMBDA_MASK_RECON

    total_sum = 0; n = 0; loss_sums = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone    = batch['drone'].to(device)
        sat      = batch['satellite'].to(device)
        labels   = batch['label'].to(device)
        alt_idx  = batch['alt_idx'].to(device)
        alts     = batch['altitude'].to(device)
        alt_norm = batch['alt_norm'].to(device).float()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, alt_idx=alt_idx, return_parts=True)
            s_out = model(sat, alt_idx=None, return_parts=True)

            l_ce = (ce(d_out['logits'], labels) + ce(s_out['logits'], labels))
            l_ce += 0.3 * (ce(d_out['cls_logits'], labels) + ce(s_out['cls_logits'], labels))
            l_nce = infonce(d_out['embedding'], s_out['embedding'], labels)
            l_con = consist(d_out['parts']['assignment'], s_out['parts']['assignment'])

            if teacher is not None:
                with torch.no_grad():
                    t_drone = teacher(drone); t_sat = teacher(sat)
                l_cross = cross_dist(d_out['projected_feat'], t_drone) + \
                          cross_dist(s_out['projected_feat'], t_sat)
            else:
                l_cross = torch.tensor(0.0, device=device)

            l_self = self_dist(d_out['cls_logits'], d_out['logits']) + \
                     self_dist(s_out['cls_logits'], s_out['logits'])
            l_uapa = uapa(d_out['logits'], s_out['logits'])

            l_proxy = 0.5 * (proxy_anchor(d_out['embedding'], labels) +
                             proxy_anchor(s_out['embedding'], labels))
            with torch.no_grad():
                ema_drone_emb = ema.forward(drone, alt_idx=alt_idx)
                ema_sat_emb   = ema.forward(sat, alt_idx=None)
            l_ema = 0.5 * (ema_dist(d_out['embedding'], ema_drone_emb) +
                           ema_dist(s_out['embedding'], ema_sat_emb))

            l_alt_con = alt_consist(d_out['embedding'], labels, alts)

            l_recon_d = model.mask_recon(d_out['parts']['projected_patches'],
                                         d_out['parts']['part_features'],
                                         d_out['parts']['assignment'])
            l_recon_s = model.mask_recon(s_out['parts']['projected_patches'],
                                         s_out['parts']['part_features'],
                                         s_out['parts']['assignment'])
            l_recon = 0.5 * (l_recon_d + l_recon_s)

            l_alt_pred = model.alt_pred(d_out['embedding'].detach(), alt_norm)

            l_div = diversity_loss(model.part_disc.prototypes)

            loss = (CFG.LAMBDA_CE          * l_ce +
                    CFG.LAMBDA_INFONCE     * l_nce +
                    CFG.LAMBDA_CONSISTENCY * l_con +
                    CFG.LAMBDA_CROSS_DIST  * l_cross +
                    CFG.LAMBDA_SELF_DIST   * l_self +
                    CFG.LAMBDA_UAPA        * l_uapa +
                    CFG.LAMBDA_PROXY       * l_proxy +
                    CFG.LAMBDA_EMA_DIST    * l_ema +
                    CFG.LAMBDA_ALT_CONSIST * l_alt_con +
                    recon_weight           * l_recon +
                    CFG.LAMBDA_ALT_PRED    * l_alt_pred +
                    CFG.LAMBDA_DIVERSITY   * l_div)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum += loss.item(); n += 1
        loss_sums['ce']    += l_ce.item()
        loss_sums['nce']   += l_nce.item()
        loss_sums['con']   += l_con.item()
        loss_sums['cross'] += l_cross.item() if torch.is_tensor(l_cross) else l_cross
        loss_sums['self']  += l_self.item()
        loss_sums['uapa']  += l_uapa.item()
        loss_sums['proxy'] += l_proxy.item()
        loss_sums['ema']   += l_ema.item()
        loss_sums['alt_c'] += l_alt_con.item()
        loss_sums['recon'] += l_recon.item()
        loss_sums['alt_p'] += l_alt_pred.item()
        loss_sums['div']   += l_div.item()

    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


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
        'total_params': total_params, 'total_params_M': total_params / 1e6,
        'trainable_params': trainable_params, 'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops, 'ms_per_query': ms_query, 'img_size': sz,
    }


# =============================================================================
# PRETTY PRINTERS
# =============================================================================
def print_weather_table(all_results, normal_results):
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test)")
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
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1']  for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP']  for w in adv_weathers])
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
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Fine-tune with online WeatherPrompt-style augmentation (Drone$\to$Satellite).}")
    lines.append(r"\label{tab:weather_finetune_online}")
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
# COMPARISON WITH EXP49 + EXP50 + EXP51
# =============================================================================
EXP49_RESULTS = {
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

EXP50_RESULTS = {
    'normal':    {'R@1': 94.38, 'mAP': 96.78},
    'fog':       {'R@1': 90.94, 'mAP': 95.00},
    'rain':      {'R@1': 85.31, 'mAP': 91.07},
    'snow':      {'R@1': 88.12, 'mAP': 92.86},
    'dark':      {'R@1': 80.00, 'mAP': 86.58},
    'light':     {'R@1': 86.25, 'mAP': 91.42},
    'fog_rain':  {'R@1': 78.12, 'mAP': 86.61},
    'fog_snow':  {'R@1': 73.44, 'mAP': 82.36},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.36},
    'wind':      {'R@1': 85.62, 'mAP': 90.78},
}

EXP51_RESULTS = {
    'normal':    {'R@1': 90.94, 'mAP': 95.18},
    'fog':       {'R@1': 89.69, 'mAP': 94.37},
    'rain':      {'R@1': 86.56, 'mAP': 92.09},
    'snow':      {'R@1': 89.69, 'mAP': 93.66},
    'dark':      {'R@1': 80.31, 'mAP': 87.74},
    'light':     {'R@1': 86.88, 'mAP': 91.98},
    'fog_rain':  {'R@1': 81.25, 'mAP': 88.60},
    'fog_snow':  {'R@1': 75.00, 'mAP': 84.63},
    'rain_snow': {'R@1': 86.25, 'mAP': 91.27},
    'wind':      {'R@1': 88.12, 'mAP': 92.98},
}


def print_comparison(all_results):
    print(f"\n{'='*115}")
    print(f"  COMPARISON: EXP49 (zero-shot) vs EXP50 (finetune-pregenerated) vs "
          f"EXP51 (scratch-pregenerated) vs EXP52 (finetune-online)")
    print(f"{'='*115}")
    print(f"  {'Weather':>12s}  {'EXP49':>8s}  {'EXP50':>8s}  {'EXP51':>8s}  {'EXP52':>8s}  "
          f"{'Δ50→52':>8s}  {'Δ51→52':>8s}")
    print(f"  {'-'*70}")
    for w in CFG.WEATHER_NAMES:
        e49 = EXP49_RESULTS[w]['R@1']
        e50 = EXP50_RESULTS[w]['R@1']
        e51 = EXP51_RESULTS[w]['R@1']
        e52 = all_results[w]['overall']['R@1']
        d50 = e52 - e50; d51 = e52 - e51
        print(f"  {w:>12s}  {e49:7.2f}%  {e50:7.2f}%  {e51:7.2f}%  {e52:7.2f}%  "
              f"{d50:+7.2f}%  {d51:+7.2f}%")
    print(f"  {'-'*70}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    e49_adv = np.mean([EXP49_RESULTS[w]['R@1'] for w in adv])
    e50_adv = np.mean([EXP50_RESULTS[w]['R@1'] for w in adv])
    e51_adv = np.mean([EXP51_RESULTS[w]['R@1'] for w in adv])
    e52_adv = np.mean([all_results[w]['overall']['R@1'] for w in adv])
    print(f"  {'Avg(adverse)':>12s}  {e49_adv:7.2f}%  {e50_adv:7.2f}%  {e51_adv:7.2f}%  {e52_adv:7.2f}%  "
          f"{e52_adv-e50_adv:+7.2f}%  {e52_adv-e51_adv:+7.2f}%")
    e49_all = np.mean([EXP49_RESULTS[w]['R@1'] for w in CFG.WEATHER_NAMES])
    e50_all = np.mean([EXP50_RESULTS[w]['R@1'] for w in CFG.WEATHER_NAMES])
    e51_all = np.mean([EXP51_RESULTS[w]['R@1'] for w in CFG.WEATHER_NAMES])
    e52_all = np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])
    print(f"  {'Avg(all)':>12s}  {e49_all:7.2f}%  {e50_all:7.2f}%  {e51_all:7.2f}%  {e52_all:7.2f}%  "
          f"{e52_all-e50_all:+7.2f}%  {e52_all-e51_all:+7.2f}%")
    print(f"{'='*115}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    train_start = datetime.datetime.now()

    print("=" * 75)
    print("  EXP52: Weather Fine-Tune with ONLINE Augmentation (WeatherPrompt-style)")
    print(f"  SUES-200:     {CFG.SUES_ROOT}")
    print(f"  Checkpoint:   {CFG.CHECKPOINT}")
    print(f"  Device:       {DEVICE}")
    print(f"  Epochs:       {CFG.NUM_EPOCHS} (fine-tune)")
    print(f"  LR:           {CFG.LR} (backbone: {CFG.BACKBONE_LR})")
    print(f"  Data strategy: ONLINE imgaug (1 random img/loc/epoch, WeatherPrompt-style)")
    print(f"  Weathers:     10 conditions (applied on-the-fly)")
    print("=" * 75)

    # ---- 1. Build Model & Load Checkpoint ----
    print("\n[1/5] Loading model + checkpoint …")
    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)

    ckpt_path = CFG.CHECKPOINT
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: Checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"  Checkpoint loaded (epoch={ckpt.get('epoch', '?')})")
    if 'metrics' in ckpt:
        m = ckpt['metrics']
        print(f"  Pretrained best: R@1={m.get('R@1', '?'):.2f}%  mAP={m.get('mAP', '?'):.2f}%")

    # Teacher
    teacher = None
    try:
        teacher = DINOv2Teacher().to(DEVICE); teacher.eval()
    except Exception as e:
        print(f"  [WARN] Could not load teacher: {e}")

    # EMA
    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA model initialized (decay={CFG.EMA_DECAY})")

    # Model complexity
    print("\n[2/5] Measuring model complexity …")
    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # ---- 2. Build Online Weather Training Dataset ----
    print("\n[3/5] Building online weather training dataset …")
    test_tf = get_transforms("test")

    # WeatherPrompt-identical drone transform: spatial done by imgaug, only ToTensor+Normalize here
    drone_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Satellite uses full train-time augmentation
    sat_tf = get_transforms("train")

    # Build imgaug weather augmenters (exact WeatherPrompt parameters)
    weather_augmenters = build_weather_augmenters()
    print(f"  Built {len(weather_augmenters)} weather augmenters (WeatherPrompt-exact)")

    weather_train_ds = OnlineWeatherTrainDataset(
        CFG.SUES_ROOT,
        drone_transform=drone_tf,
        sat_transform=sat_tf,
        weather_augmenters=weather_augmenters,
    )
    if len(weather_train_ds) == 0:
        print("  ERROR: No training data found!")
        return

    train_loader = DataLoader(
        weather_train_ds,
        batch_sampler=PKSamplerOnline(weather_train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
        num_workers=CFG.NUM_WORKERS, pin_memory=True
    )

    # ---- 3. Setup Losses & Optimizer ----
    infonce    = SupInfoNCELoss(temp=0.05).to(DEVICE)
    ce         = nn.CrossEntropyLoss(label_smoothing=0.1)
    consist    = PartConsistencyLoss()
    cross_dist = CrossDistillationLoss()
    self_dist  = SelfDistillationLoss(temperature=CFG.DISTILL_TEMP)
    uapa_loss  = UAPALoss(base_temperature=CFG.DISTILL_TEMP)
    base_losses = (infonce, ce, consist, cross_dist, self_dist, uapa_loss)

    proxy_anchor   = ProxyAnchorLoss(CFG.NUM_CLASSES, CFG.EMBED_DIM,
                                     margin=CFG.PROXY_MARGIN, alpha=CFG.PROXY_ALPHA).to(DEVICE)
    ema_dist       = EMADistillationLoss()
    alt_consist    = AltitudeConsistencyLoss()
    diversity_loss = PrototypeDiversityLoss()
    new_losses     = (proxy_anchor, ema_dist, alt_consist, diversity_loss)

    # Optimizer (fine-tune LR)
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,            'lr': CFG.BACKBONE_LR},
        {'params': head_params,                'lr': CFG.LR},
        {'params': infonce.parameters(),       'lr': CFG.LR},
        {'params': proxy_anchor.parameters(),  'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)

    # ---- 4. Training Loop ----
    print(f"\n[4/5] Fine-tuning with online weather augmentation ({CFG.NUM_EPOCHS} epochs) …")
    best_r1 = 0.0
    best_avg_weather_r1 = 0.0
    results_log = []
    loss_history = []

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        # Cosine LR schedule with warmup
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)
        optimizer.param_groups[0]['lr'] = CFG.BACKBONE_LR * lr_scale
        optimizer.param_groups[1]['lr'] = CFG.LR * lr_scale
        optimizer.param_groups[2]['lr'] = CFG.LR * lr_scale
        optimizer.param_groups[3]['lr'] = CFG.LR * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, teacher, ema, train_loader, base_losses,
                                       new_losses, optimizer, scaler, DEVICE, epoch)
        loss_history.append({'epoch': epoch, 'total': avg_loss, 'lr': cur_lr, **ld})

        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"CE {ld['ce']:.3f}  NCE {ld['nce']:.3f}  "
              f"Proxy {ld['proxy']:.3f}  EMA {ld['ema']:.3f} | "
              f"AltC {ld['alt_c']:.3f}  Rec {ld['recon']:.3f}  AltP {ld['alt_p']:.3f}  Div {ld['div']:.3f} | "
              f"LR {cur_lr:.2e}")

        # Evaluate periodically
        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            print(f"\n  --- Eval at epoch {epoch} ---")
            sat_feats, sat_labels = build_satellite_gallery(model, CFG.SUES_ROOT, DEVICE, test_tf)

            eval_weathers = ["normal", "fog_snow", "rain", "snow"]
            epoch_results = {}
            for w_name in eval_weathers:
                query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, CFG.SUES_ROOT, w_name, transform=test_tf)
                if len(query_ds) == 0:
                    epoch_results[w_name] = {'R@1': 0, 'mAP': 0}
                    continue
                overall, _ = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
                epoch_results[w_name] = overall
                print(f"    {w_name:>10s}: R@1={overall['R@1']:.2f}%  mAP={overall['mAP']:.2f}%")

            normal_r1 = epoch_results.get('normal', {}).get('R@1', 0)
            avg_eval_r1 = np.mean([epoch_results[w].get('R@1', 0) for w in eval_weathers])
            results_log.append({'epoch': epoch, 'normal_R@1': normal_r1,
                                'avg_eval_R@1': avg_eval_r1,
                                **{f'{w}_R@1': epoch_results[w].get('R@1', 0) for w in eval_weathers}})

            # Save best by average weather R@1
            if avg_eval_r1 > best_avg_weather_r1:
                best_avg_weather_r1 = avg_eval_r1
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'metrics': epoch_results, 'avg_weather_r1': avg_eval_r1,
                }, os.path.join(CFG.OUTPUT_DIR, 'exp52_weather_finetune_online_best.pth'))
                print(f"  ★ New best avg weather R@1: {avg_eval_r1:.2f}%!")

            if normal_r1 > best_r1:
                best_r1 = normal_r1

            # EMA eval
            ema_sat_feats, _ = build_satellite_gallery(ema.model, CFG.SUES_ROOT, DEVICE, test_tf)
            ema_normal_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, CFG.SUES_ROOT, "normal", transform=test_tf)
            if len(ema_normal_ds) > 0:
                ema_overall, _ = evaluate_retrieval(ema.model, ema_normal_ds, ema_sat_feats, sat_labels, DEVICE)
                print(f"    EMA normal: R@1={ema_overall['R@1']:.2f}%")
                if ema_overall['R@1'] > best_r1:
                    best_r1 = ema_overall['R@1']

            del sat_feats, sat_labels; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- 5. Full Weather Evaluation (best model) ----
    print(f"\n[5/5] Full weather evaluation with best model …")

    best_ckpt_path = os.path.join(CFG.OUTPUT_DIR, 'exp52_weather_finetune_online_best.pth')
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
        print(f"  Loaded best checkpoint (epoch={best_ckpt.get('epoch', '?')}, "
              f"avg_weather_R@1={best_ckpt.get('avg_weather_r1', '?'):.2f}%)")
    else:
        print(f"  Using final model (no best checkpoint found)")

    model.eval()
    sat_feats, sat_labels = build_satellite_gallery(model, CFG.SUES_ROOT, DEVICE, test_tf)

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
    print_comparison(all_results)

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

    # LaTeX
    latex = generate_latex_table(all_results, normal_results)
    print(f"\n{'='*75}")
    print("  LaTeX Table:")
    print(f"{'='*75}")
    print(latex)

    # ---- Save JSON ----
    train_end = datetime.datetime.now()
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    summary = {
        'experiment': 'EXP52_Weather_Finetune_Online',
        'model': 'SPDGeo-DPEA-MAR (fine-tuned with online weather augmentation)',
        'dataset': 'SUES-200',
        'direction': 'drone → satellite (D2S) + satellite → drone (S2D)',
        'pretrained_from': 'EXP35 checkpoint',
        'data_strategy': 'Online imgaug augmentation (WeatherPrompt-style, 1 img/loc/epoch)',
        'timestamp': {
            'start': train_start.isoformat(),
            'end': train_end.isoformat(),
            'duration_s': (train_end - train_start).total_seconds(),
        },
        'training_config': {
            'epochs': CFG.NUM_EPOCHS,
            'lr': CFG.LR,
            'backbone_lr': CFG.BACKBONE_LR,
            'warmup_epochs': CFG.WARMUP_EPOCHS,
            'recon_warmup': CFG.RECON_WARMUP,
            'weather_conditions': CFG.WEATHER_NAMES,
            'augmentation': 'online_imgaug_weatherprompt_exact',
            'samples_per_epoch': f'{len(weather_train_ds)} locations × 1 img × random weather',
        },
        'model_complexity': complexity,
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
            'vs_exp49': {
                w: float(all_results[w]['overall']['R@1'] - EXP49_RESULTS[w]['R@1'])
                for w in CFG.WEATHER_NAMES
            },
            'vs_exp50': {
                w: float(all_results[w]['overall']['R@1'] - EXP50_RESULTS[w]['R@1'])
                for w in CFG.WEATHER_NAMES
            },
            'vs_exp51': {
                w: float(all_results[w]['overall']['R@1'] - EXP51_RESULTS[w]['R@1'])
                for w in CFG.WEATHER_NAMES
            },
        },
        'training_history': {
            'eval_log': results_log,
            'loss_history': loss_history,
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
        },
    }

    json_path = os.path.join(CFG.OUTPUT_DIR, 'exp52_weather_finetune_online.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}")

    # ---- Final Summary ----
    sm = summary['summary_metrics']
    print(f"\n{'='*75}")
    print(f"  EXP52 COMPLETE — Fine-Tune with Online Weather Augmentation")
    print(f"{'='*75}")
    print(f"  Data strategy:      Online imgaug (WeatherPrompt-style)")
    print(f"  Training samples:   {len(weather_train_ds)} locs × 1 img × random weather / epoch")
    print(f"  Normal:             R@1={sm['normal_R@1']:.2f}%  mAP={sm['normal_mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={sm['avg_all_R@1']:.2f}%  mAP={sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={sm['avg_adverse_R@1']:.2f}%  mAP={sm['avg_adverse_mAP']:.2f}%")
    print(f"  Training duration:  {(train_end - train_start).total_seconds():.1f}s")
    s2d_sm = summary['s2d_summary_metrics']
    print(f"\n  S2D — Satellite→Drone:")
    print(f"  Normal:             R@1={normal_s2d['R@1']:.2f}%  mAP={normal_s2d['mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={s2d_sm['avg_all_R@1']:.2f}%  mAP={s2d_sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={s2d_sm['avg_adverse_R@1']:.2f}%  mAP={s2d_sm['avg_adverse_mAP']:.2f}%")
    print(f"\n  vs EXP50 (finetune + pre-generated):")
    e50_adv = np.mean([EXP50_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"    Avg adverse R@1:  {e50_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e50_adv:+.2f}%)")
    print(f"\n  vs EXP51 (scratch + pre-generated):")
    e51_adv = np.mean([EXP51_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"    Avg adverse R@1:  {e51_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e51_adv:+.2f}%)")
    print(f"{'='*75}")


main()
