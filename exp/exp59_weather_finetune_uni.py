# =============================================================================
# EXP59: Weather Fine-Tune with ONLINE Augmentation — University-1652
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Fine-tunes from the NEW best checkpoint (geopartloss_best.pth, 98.4% SUES-200)
# on University-1652 training set with online weather augmentation.
#
# Training strategy (same as WeatherPrompt for FAIR COMPARISON):
#   - Online weather augmentation via imgaug (not pre-generated)
#   - 1 random drone image per location per epoch (WeatherPrompt-style)
#   - Exact same imgaug weather augmenters from WeatherPrompt source code
#
# KEY DIFFERENCES from EXP56 (SUES-200 fine-tune):
#   - EXP56: fine-tuned on SUES-200 (200 locations, 4 altitudes)
#   - EXP59: fine-tuned on University-1652 (701 buildings, 54 drone views each)
#   - EXP59 uses NO altitude conditioning (no FiLM — same as gpl_cross_uni2sues.py)
#   - EXP59 evaluates on Uni-1652 test set (D→S and S→D)
#
# Architecture: SPDGeo-DPE-MAR with NO altitude conditioning
# Loss: GeoPartLoss (3-group adaptive: Alignment + EMA + Part Quality)
#
# WeatherPrompt D→S baseline on Uni-1652: 77.14% mean R@1
#
# Kaggle Data Sources:
#   1. University-1652:       /kaggle/input/datasets/chinguyeen/university-1652/University-1652
#   2. New best checkpoint:   /kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth
#
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "imgaug"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, math, json, random, copy, time
import numpy as np
from collections import defaultdict

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
    UNI_ROOT        = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"

    # ---- Checkpoint ----
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/2/geopartloss_best.pth"

    OUTPUT_DIR      = "/kaggle/working"
    EXPERIMENT_NAME = "EXP59_FineTune_Uni1652_OnlineAug"

    IMG_SIZE        = 448
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6

    # Training params (fine-tune — lower LR, fewer epochs)
    NUM_EPOCHS      = 60
    P_CLASSES       = 16       # building classes per batch
    K_SAMPLES       = 4        # drone images per building
    LR              = 1e-4
    BACKBONE_LR     = 1e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 3
    USE_AMP         = True
    SEED            = 42
    RECON_WARMUP    = 0        # already pretrained — no warmup needed

    # GeoPartLoss (3-group, no altitude — same as gpl_cross_uni2sues.py)
    EMA_DECAY       = 0.996
    PROXY_MARGIN    = 0.1
    PROXY_ALPHA     = 32
    MASK_RATIO      = 0.30
    DISTILL_TEMP    = 4.0
    NUM_LOSS_GROUPS = 3
    EVAL_INTERVAL   = 5
    NUM_WORKERS     = 2

    # WeatherPrompt-style training: 1 random drone per building per epoch
    WEATHER_CONDITIONS = [
        "normal", "fog", "rain", "snow", "dark", "light",
        "fog_rain", "fog_snow", "rain_snow", "wind"
    ]
    # Evaluation weather subset (fast eval during training)
    EVAL_WEATHER_SUBSET = ["normal", "fog", "rain", "dark", "fog_snow"]

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# WEATHER AUGMENTATION
# =============================================================================
def get_weather_augmenter(condition: str) -> iaa.Augmenter:
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
    return iaa.Identity()


# Pre-build all augmenters once
WEATHER_AUGMENTERS = {c: get_weather_augmenter(c) for c in Config.WEATHER_CONDITIONS}


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
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


# =============================================================================
# DATASET — University-1652 Online Weather Training
# =============================================================================
class Uni1652OnlineWeatherTrainDataset(Dataset):
    """
    University-1652 training dataset with online weather augmentation.
    WeatherPrompt strategy: 1 random drone image per building per epoch,
    random weather applied on-the-fly.

    Each item: (weather_drone_image, satellite_image, building_label)
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.drone_tf = transform or get_transforms("train")
        self.sat_tf = get_transforms("test")  # satellite stays clean

        drone_dir = os.path.join(root, "train", "drone")
        sat_dir   = os.path.join(root, "train", "satellite")

        self.building_ids = sorted([
            d for d in os.listdir(drone_dir)
            if os.path.isdir(os.path.join(drone_dir, d))
            and os.path.isdir(os.path.join(sat_dir, d))
        ])
        self.bid_to_idx = {b: i for i, b in enumerate(self.building_ids)}
        self.num_classes = len(self.building_ids)

        # For each building: list of drone image paths, satellite image paths
        self.drone_by_building = {}
        self.sat_by_building = {}
        for bid in self.building_ids:
            dp = os.path.join(drone_dir, bid)
            sp = os.path.join(sat_dir, bid)
            drones = sorted([os.path.join(dp, f) for f in os.listdir(dp)
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
            sats = sorted([os.path.join(sp, f) for f in os.listdir(sp)
                          if f.endswith(('.jpg', '.jpeg', '.png'))])
            if drones and sats:
                self.drone_by_building[bid] = drones
                self.sat_by_building[bid] = sats

        # Build index: one entry per building (WeatherPrompt-style: 1 random drone per building)
        self.valid_bids = [b for b in self.building_ids
                          if b in self.drone_by_building and b in self.sat_by_building]
        # drone_by_class for PK sampler
        self.drone_by_class = {self.bid_to_idx[b]: list(range(len(self.valid_bids)))
                               for b in self.valid_bids}
        # Flat list for sampling
        self.samples = [(b, self.bid_to_idx[b]) for b in self.valid_bids]

        print(f"  [Uni-1652 Train] {len(self.valid_bids)} buildings | "
              f"{self.num_classes} classes | Online weather aug")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bid, label = self.samples[idx]

        # Random drone image from this building
        drone_path = random.choice(self.drone_by_building[bid])
        # Random satellite image from this building
        sat_path = random.choice(self.sat_by_building[bid])
        # Random weather condition (including normal)
        condition = random.choice(CFG.WEATHER_CONDITIONS)

        try:
            drone_np = np.array(Image.open(drone_path).convert('RGB'))
            if condition != "normal":
                drone_np = WEATHER_AUGMENTERS[condition].augment_image(drone_np)
            drone_img = self.drone_tf(Image.fromarray(drone_np.astype(np.uint8)))
        except Exception:
            drone_img = self.drone_tf(Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128)))

        try:
            sat_img = self.sat_tf(Image.open(sat_path).convert('RGB'))
        except Exception:
            sat_img = self.sat_tf(Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128)))

        return {'drone': drone_img, 'satellite': sat_img, 'label': label}


# =============================================================================
# PK Sampler (P buildings × K samples)
# =============================================================================
class PKSampler:
    def __init__(self, dataset, p, k):
        self.dataset = dataset; self.p = p; self.k = k
        self.n = len(dataset.valid_bids)

    def __iter__(self):
        indices = list(range(self.n))
        random.shuffle(indices)
        batch = []
        for idx in indices:
            # Each building yields k samples (same building, different drone+weather each time)
            batch.extend([idx] * self.k)
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]
                batch = batch[self.p * self.k:]

    def __len__(self):
        return self.n // self.p


# =============================================================================
# MODEL (no altitude — same as gpl_cross_uni2sues.py)
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
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone    = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc   = SemanticPartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP)
        self.pool        = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck  = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                         nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier  = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj    = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                         nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.mask_recon  = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPE-MAR: {total/1e6:.1f}M trainable parameters")

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
# EMA
# =============================================================================
class EMAModel:
    def __init__(self, model, decay=0.996):
        self.decay = decay
        self.model = copy.deepcopy(model); self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def forward(self, x):
        return self.model.extract_embedding(x)


# =============================================================================
# LOSSES — GeoPartLoss 3-group (no altitude)
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


class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature

    @staticmethod
    def _entropy(logits):
        p = F.softmax(logits, dim=1); return -(p*(p+1e-8).log()).sum(dim=1).mean()

    def forward(self, drone_logits, sat_logits):
        T = self.T0 * (1 + torch.sigmoid(self._entropy(drone_logits) - self._entropy(sat_logits)))
        return (T**2) * F.kl_div(F.log_softmax(drone_logits/T, 1),
                                  F.softmax(sat_logits/T, 1).detach(), reduction='batchmean')


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
        pos_term = torch.log(1 + pos_exp.sum(0))
        pos_loss = pos_term[has_pos].mean() if has_pos.sum() > 0 else torch.tensor(0.0, device=embeddings.device)
        neg_mask = 1 - one_hot
        neg_exp = torch.exp(self.alpha * (sim * neg_mask + self.margin)) * neg_mask
        return pos_loss + torch.log(1 + neg_exp.sum(0)).mean()


class EMADistillationLoss(nn.Module):
    def forward(self, student_emb, ema_emb):
        return (1 - F.cosine_similarity(student_emb, ema_emb)).mean()


class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1); sim = P @ P.T; K = sim.size(0)
        mask = 1 - torch.eye(K, device=sim.device)
        return (sim * mask).abs().sum() / (K * (K - 1))


class GeoPartLoss(nn.Module):
    """3-group adaptive weighting (Alignment + EMA + Part Quality). No altitude group."""
    def __init__(self, num_classes, embed_dim, num_groups=3, cfg=None):
        super().__init__()
        cfg = cfg or CFG
        self.log_vars = nn.Parameter(torch.zeros(num_groups))
        self.infonce = SupInfoNCELoss(temp=0.05)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.proxy_anchor = ProxyAnchorLoss(num_classes, embed_dim,
                                            margin=cfg.PROXY_MARGIN, alpha=cfg.PROXY_ALPHA)
        self.uapa = UAPALoss(base_temperature=cfg.DISTILL_TEMP)
        self.ema_dist = EMADistillationLoss()
        self.diversity = PrototypeDiversityLoss()
        self.recon_warmup = cfg.RECON_WARMUP
        print(f"  GeoPartLoss: {num_groups} groups (align+ema+part) with adaptive weighting")

    def forward(self, d_out, s_out, labels, model, ema, drone, sat, epoch, device):
        loss_dict = {}
        # G1: Alignment
        l_ce = (self.ce(d_out['logits'], labels) + self.ce(s_out['logits'], labels))
        l_ce += 0.3 * (self.ce(d_out['cls_logits'], labels) + self.ce(s_out['cls_logits'], labels))
        l_nce = self.infonce(d_out['embedding'], s_out['embedding'], labels)
        l_proxy = 0.5 * (self.proxy_anchor(d_out['embedding'], labels) +
                         self.proxy_anchor(s_out['embedding'], labels))
        l_uapa = self.uapa(d_out['logits'], s_out['logits'])
        L_align = l_ce + l_nce + l_proxy + l_uapa
        loss_dict.update({'ce': l_ce.item(), 'nce': l_nce.item(),
                          'proxy': l_proxy.item(), 'uapa': l_uapa.item()})
        # G2: EMA
        with torch.no_grad():
            ema_d = ema.forward(drone); ema_s = ema.forward(sat)
        l_ema = 0.5 * (self.ema_dist(d_out['embedding'], ema_d) +
                       self.ema_dist(s_out['embedding'], ema_s))
        L_ema = l_ema; loss_dict['ema'] = l_ema.item()
        # G3: Part Quality
        if epoch >= self.recon_warmup:
            l_recon = 0.5 * (
                model.mask_recon(d_out['parts']['projected_patches'],
                                 d_out['parts']['part_features'], d_out['parts']['assignment']) +
                model.mask_recon(s_out['parts']['projected_patches'],
                                 s_out['parts']['part_features'], s_out['parts']['assignment']))
        else:
            l_recon = torch.tensor(0.0, device=device)
        l_div = self.diversity(model.part_disc.prototypes)
        L_part = l_recon + l_div
        loss_dict.update({'recon': l_recon.item() if torch.is_tensor(l_recon) else l_recon,
                          'div': l_div.item()})
        # Adaptive uncertainty weighting
        groups = [L_align, L_ema, L_part]
        group_names = ['align', 'ema', 'part']
        total_loss = torch.tensor(0.0, device=device); group_weights = {}
        for i, (L, name) in enumerate(zip(groups, group_names)):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * L + self.log_vars[i]
            group_weights[name] = precision.item()
            loss_dict[f'G_{name}'] = L.item()
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict, group_weights


# =============================================================================
# EVALUATION — University-1652 weather
# =============================================================================
@torch.no_grad()
def extract_features(model, paths, labels, weather_condition="normal", desc=""):
    model.eval()
    tf = get_transforms("test")
    aug = WEATHER_AUGMENTERS.get(weather_condition) if weather_condition != "normal" else None
    feats, lbs = [], []
    batch_imgs, batch_labels = [], []
    for i, (path, label) in enumerate(zip(paths, labels)):
        try:
            img_np = np.array(Image.open(path).convert('RGB'))
            if aug is not None:
                img_np = aug.augment_image(img_np)
            img = tf(Image.fromarray(img_np.astype(np.uint8)))
        except Exception:
            img = tf(Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128, 128, 128)))
        batch_imgs.append(img); batch_labels.append(label)
        if len(batch_imgs) == 32 or i == len(paths) - 1:
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                f = model.extract_embedding(torch.stack(batch_imgs).to(DEVICE))
            feats.append(f.cpu()); lbs.extend(batch_labels)
            batch_imgs, batch_labels = [], []
    return torch.cat(feats), torch.tensor(lbs)


def compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels):
    sim = query_feats @ gallery_feats.T; _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0); r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        f = matches[0].item()
        if f < 1: r1 += 1
        if f < 5: r5 += 1
        if f < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100}


@torch.no_grad()
def evaluate_weather(model, uni_root, weather_subset=None):
    """Evaluate on University-1652 under selected weather conditions."""
    conditions = weather_subset or CFG.WEATHER_CONDITIONS

    # Load paths
    gallery_paths, gallery_labels = [], []
    gallery_dir = os.path.join(uni_root, "test", "gallery_satellite")
    for bid in sorted(os.listdir(gallery_dir)):
        bp = os.path.join(gallery_dir, bid)
        if not os.path.isdir(bp): continue
        for f in sorted(os.listdir(bp)):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                gallery_paths.append(os.path.join(bp, f))
                gallery_labels.append(int(bid))

    query_paths, query_labels = [], []
    query_dir = os.path.join(uni_root, "test", "query_drone")
    for bid in sorted(os.listdir(query_dir)):
        bp = os.path.join(query_dir, bid)
        if not os.path.isdir(bp): continue
        for f in sorted(os.listdir(bp)):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                query_paths.append(os.path.join(bp, f))
                query_labels.append(int(bid))

    # Gallery features (no weather — constant)
    gallery_feats, g_labels = extract_features(model, gallery_paths, gallery_labels, "normal", "Gallery")

    results = {}
    r1_list = []
    for cond in conditions:
        drone_feats, d_labels = extract_features(model, query_paths, query_labels, cond, f"D[{cond}]")
        m = compute_metrics(drone_feats, d_labels, gallery_feats, g_labels)
        results[cond] = m
        r1_list.append(m['R@1'])

    avg_r1 = sum(r1_list) / len(r1_list)
    adverse_conds = [c for c in conditions if c != "normal"]
    adverse_r1_list = [results[c]['R@1'] for c in adverse_conds if c in results]
    avg_adv_r1 = sum(adverse_r1_list) / len(adverse_r1_list) if adverse_r1_list else 0.0

    return results, avg_r1, avg_adv_r1


# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, ema, geo_loss, optimizer, scaler, loader, device, epoch):
    model.train()
    total_sum = 0; n = 0; loss_sums = defaultdict(float)
    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, return_parts=True)
            s_out = model(sat, return_parts=True)
            loss, ld, gw = geo_loss(d_out, s_out, labels, model, ema, drone, sat, epoch, device)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)
        total_sum += loss.item(); n += 1
        for k, v in ld.items(): loss_sums[k] += v
    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 75)
    print(f"  {CFG.EXPERIMENT_NAME}")
    print(f"  Fine-Tune from GeoPartLoss checkpoint on University-1652")
    print(f"  WeatherPrompt-style online augmentation | {CFG.NUM_EPOCHS} epochs")
    print(f"  IMG: {CFG.IMG_SIZE} | UNFREEZE: {CFG.UNFREEZE_BLOCKS} | GeoPartLoss 3-group")
    print("=" * 75)

    # ---- Data ----
    print("\n[DATA] Building University-1652 training dataset...")
    train_ds = Uni1652OnlineWeatherTrainDataset(CFG.UNI_ROOT)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )
    print(f"  {len(train_ds)} buildings | P={CFG.P_CLASSES} K={CFG.K_SAMPLES}")

    # ---- Model ----
    print("\n[MODEL] Building SPDGeo-DPE-MAR...")
    model = SPDGeoDPEMARModel(train_ds.num_classes, cfg=CFG).to(DEVICE)

    # Load checkpoint
    if os.path.exists(CFG.CHECKPOINT):
        ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded checkpoint: {CFG.CHECKPOINT}")
        print(f"  Missing keys: {len(missing)} (likely classifier head — expected for new dataset)")
    else:
        print(f"  WARNING: Checkpoint not found — starting from DINOv2 pretrained only!")

    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA initialized (decay={CFG.EMA_DECAY})")

    # ---- Loss ----
    geo_loss = GeoPartLoss(train_ds.num_classes, CFG.EMBED_DIM,
                           num_groups=CFG.NUM_LOSS_GROUPS, cfg=CFG).to(DEVICE)

    # ---- Optimizer ----
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': CFG.BACKBONE_LR},
        {'params': head_params,     'lr': CFG.LR},
        {'params': geo_loss.parameters(), 'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)

    # ---- Training Loop ----
    best_avg_adverse = 0.0; best_avg_all = 0.0; best_ema_adverse = 0.0
    results_log = []

    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        # LR schedule
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)
        for i in range(3):
            optimizer.param_groups[i]['lr'] = (CFG.BACKBONE_LR if i == 0 else CFG.LR) * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld = train_one_epoch(model, ema, geo_loss, optimizer, scaler,
                                       train_loader, DEVICE, epoch)
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"G_align {ld.get('G_align',0):.3f}  G_ema {ld.get('G_ema',0):.3f}  "
              f"G_part {ld.get('G_part',0):.3f} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            # Fast eval on subset
            eval_results, avg_all, avg_adv = evaluate_weather(
                model, CFG.UNI_ROOT, weather_subset=CFG.EVAL_WEATHER_SUBSET)
            normal_r1 = eval_results.get('normal', {}).get('R@1', 0)
            print(f"  ► Weather eval: normal={normal_r1:.2f}%  "
                  f"avg_adv={avg_adv:.2f}%  avg_all={avg_all:.2f}%")

            entry = {'epoch': epoch, 'normal_r1': normal_r1,
                     'avg_all_r1': avg_all, 'avg_adv_r1': avg_adv, 'results': {
                         c: eval_results[c]['R@1'] for c in eval_results}}
            results_log.append(entry)

            if avg_adv > best_avg_adverse:
                best_avg_adverse = avg_adv
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'avg_adv_r1': avg_adv, 'normal_r1': normal_r1},
                           os.path.join(CFG.OUTPUT_DIR, f'{CFG.EXPERIMENT_NAME}_best.pth'))
                print(f"  ★ New best avg_adv_r1: {best_avg_adverse:.2f}%! Saved.")

            # EMA eval
            ema_results, ema_avg_all, ema_avg_adv = evaluate_weather(
                ema.model, CFG.UNI_ROOT, weather_subset=CFG.EVAL_WEATHER_SUBSET)
            ema_normal = ema_results.get('normal', {}).get('R@1', 0)
            print(f"  ► EMA avg_adv_r1: {ema_avg_adv:.2f}%")
            if ema_avg_adv > best_ema_adverse:
                best_ema_adverse = ema_avg_adv
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'avg_adv_r1': ema_avg_adv, 'normal_r1': ema_normal, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, f'{CFG.EXPERIMENT_NAME}_ema_best.pth'))
                print(f"  ★ New best avg_adv_r1 (EMA): {best_ema_adverse:.2f}%! Saved.")

    print(f"\n{'='*75}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best avg_adv_r1 (direct): {best_avg_adverse:.2f}%")
    print(f"  Best avg_adv_r1 (EMA):    {best_ema_adverse:.2f}%")
    print(f"{'='*75}")

    # ---- Final Full Evaluation ----
    print("\n[FINAL EVAL] Loading best checkpoint for full 10-weather evaluation...")
    best_ckpt_path = os.path.join(CFG.OUTPUT_DIR, f'{CFG.EXPERIMENT_NAME}_ema_best.pth')
    if not os.path.exists(best_ckpt_path):
        best_ckpt_path = os.path.join(CFG.OUTPUT_DIR, f'{CFG.EXPERIMENT_NAME}_best.pth')

    ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    final_results, final_avg_all, final_avg_adv = evaluate_weather(
        model, CFG.UNI_ROOT, weather_subset=CFG.WEATHER_CONDITIONS)
    final_normal = final_results.get('normal', {}).get('R@1', 0)

    print(f"\n{'=' * 75}")
    print(f"  FINAL RESULTS — {CFG.EXPERIMENT_NAME}")
    print(f"  University-1652 | Drone → Satellite | All 10 Weather Conditions")
    print(f"{'=' * 75}")
    print(f"  {'Condition':<12}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  {'mAP':>7}")
    print(f"  {'-'*50}")
    for cond in CFG.WEATHER_CONDITIONS:
        if cond in final_results:
            m = final_results[cond]
            print(f"  {cond:<12}  {m['R@1']:>6.2f}%  {m['R@5']:>6.2f}%  {m['R@10']:>6.2f}%  {m['mAP']:>6.2f}%")
    print(f"  {'-'*50}")
    print(f"  {'Avg(adverse)':<12}  {final_avg_adv:>6.2f}%")
    print(f"  {'Avg(all)':<12}  {final_avg_all:>6.2f}%")
    print(f"\n  Normal R@1: {final_normal:.2f}%  |  Avg Adverse: {final_avg_adv:.2f}%")
    print(f"  WeatherPrompt mean R@1 (D→S): 77.14%  |  Ours: {final_avg_all:.2f}%")
    print(f"  Δ vs WeatherPrompt: {final_avg_all - 77.14:+.2f}%")
    print(f"{'=' * 75}")

    # Save
    run_summary = {
        'experiment': CFG.EXPERIMENT_NAME,
        'dataset': 'University-1652',
        'checkpoint_loaded': CFG.CHECKPOINT,
        'best_checkpoint': best_ckpt_path,
        'normal_r1': final_normal,
        'avg_adverse_r1': final_avg_adv,
        'avg_all_r1': final_avg_all,
        'weather_results': {c: final_results[c] for c in CFG.WEATHER_CONDITIONS if c in final_results},
        'training_log': results_log,
    }
    with open(os.path.join(CFG.OUTPUT_DIR, f'{CFG.EXPERIMENT_NAME}_results.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f"  Results saved to {CFG.OUTPUT_DIR}/{CFG.EXPERIMENT_NAME}_results.json")


NUM_WORKERS = CFG.NUM_WORKERS  # re-export for DataLoader

if __name__ == '__main__':
    main()
