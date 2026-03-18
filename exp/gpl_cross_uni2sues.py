# =============================================================================
# GeoPartLoss Cross-Dataset: Train University-1652 → Test SUES-200 + TTA
# =============================================================================
# Best config from SUES-200 (98.48% R@1) adapted for cross-dataset eval:
#   - GeoPartLoss: 3-group adaptive (Alignment + EMA + Part Quality)
#   - UNFREEZE_BLOCKS=6, IMG_SIZE=448, 120 epochs
#   - Phase 1: Train on Uni-1652, eval on both datasets each 5 epochs
#   - Phase 2: TTA post-processing on both test sets
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, math, json, gc, random, copy, time
import numpy as np
from collections import defaultdict

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
    # ---- University-1652 (training) ----
    UNI_ROOT        = "/kaggle/input/datasets/chinguyeen/university-1652/University-1652"

    # ---- SUES-200 (cross-domain test) ----
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    SUES_DRONE_DIR  = "drone-view"
    SUES_SAT_DIR    = "satellite-view"
    SUES_ALTITUDES  = ["150", "200", "250", "300"]
    SUES_TEST_LOCS  = list(range(1, 201))      # test on ALL 200 locs (cross-domain)

    OUTPUT_DIR      = "/kaggle/working"

    IMG_SIZE        = 448       # 14×32 patches for DINOv2-S/14
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6       # best config from SUES-200

    NUM_EPOCHS      = 80      # best config from SUES-200
    P_CLASSES       = 16
    K_SAMPLES       = 4
    LR              = 3e-4
    BACKBONE_LR     = 3e-5
    WEIGHT_DECAY    = 0.01
    WARMUP_EPOCHS   = 5
    USE_AMP         = True
    SEED            = 42

    # GeoPartLoss: Learnable uncertainty weighting (Kendall et al. 2018)
    PROXY_MARGIN        = 0.1
    PROXY_ALPHA         = 32
    EMA_DECAY           = 0.996
    MASK_RATIO          = 0.30
    RECON_WARMUP        = 10
    DISTILL_TEMP        = 4.0
    NUM_LOSS_GROUPS     = 3  # Alignment, EMA, Part Quality (no altitude)
    EVAL_INTERVAL       = 5
    NUM_WORKERS         = 2

    # TTA Config (Phase 2)
    TTA_SCALES          = [336, 448, 518]  # multiples of patch_size=14
    TTA_FLIP            = True

CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s); torch.backends.cudnn.benchmark = True


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transforms(mode="train", img_size=None):
    sz = img_size or CFG.IMG_SIZE
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((sz, sz)), transforms.RandomHorizontalFlip(0.5),
            transforms.RandomResizedCrop(sz, scale=(0.8, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.RandomGrayscale(p=0.02),
            transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
        ])
    return transforms.Compose([
        transforms.Resize((sz, sz)), transforms.ToTensor(),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225]),
    ])


# =============================================================================
# DATASET — University-1652 (Training + In-Domain Test)
# =============================================================================
class University1652TrainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root; self.transform = transform
        self.drone_dir = os.path.join(root, "train", "drone")
        self.sat_dir   = os.path.join(root, "train", "satellite")
        self.building_ids = sorted([
            d for d in os.listdir(self.drone_dir)
            if os.path.isdir(os.path.join(self.drone_dir, d))
            and os.path.isdir(os.path.join(self.sat_dir, d))
        ])
        self.bid_to_idx = {b: i for i, b in enumerate(self.building_ids)}
        self.num_classes = len(self.building_ids)
        self.samples = []; self.drone_by_class = defaultdict(list)
        for bid in self.building_ids:
            idx = self.bid_to_idx[bid]
            dp = os.path.join(self.drone_dir, bid)
            sp = os.path.join(self.sat_dir, bid)
            sat_imgs = sorted([os.path.join(sp, f) for f in os.listdir(sp) if f.endswith(('.jpg','.jpeg','.png'))])
            if not sat_imgs: continue
            for f in sorted(os.listdir(dp)):
                if f.endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(dp, f), random.choice(sat_imgs), idx))
                    self.drone_by_class[idx].append(len(self.samples) - 1)
        print(f"  [Uni-1652 train] {len(self.samples)} drone-sat pairs | {self.num_classes} classes")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        dp, sp, label = self.samples[idx]
        try:
            d = Image.open(dp).convert('RGB'); s = Image.open(sp).convert('RGB')
        except Exception:
            d = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128,128,128))
            s = Image.new('RGB', (CFG.IMG_SIZE, CFG.IMG_SIZE), (128,128,128))
        if self.transform: d = self.transform(d); s = self.transform(s)
        return {'drone': d, 'satellite': s, 'label': label}


class University1652TestDataset:
    def __init__(self, root):
        self.root = root
        self.query_dir   = os.path.join(root, "test", "query_drone")
        self.gallery_dir = os.path.join(root, "test", "gallery_satellite")
        self.query_samples, self.query_labels = [], []
        if os.path.isdir(self.query_dir):
            for bid in sorted(os.listdir(self.query_dir)):
                bp = os.path.join(self.query_dir, bid)
                if not os.path.isdir(bp): continue
                for f in sorted(os.listdir(bp)):
                    if f.endswith(('.jpg','.jpeg','.png')):
                        self.query_samples.append(os.path.join(bp, f))
                        self.query_labels.append(int(bid))
        self.gallery_samples, self.gallery_labels = [], []
        if os.path.isdir(self.gallery_dir):
            for bid in sorted(os.listdir(self.gallery_dir)):
                bp = os.path.join(self.gallery_dir, bid)
                if not os.path.isdir(bp): continue
                for f in sorted(os.listdir(bp)):
                    if f.endswith(('.jpg','.jpeg','.png')):
                        self.gallery_samples.append(os.path.join(bp, f))
                        self.gallery_labels.append(int(bid))
        print(f"  [Uni-1652 test] Query: {len(self.query_samples)} drone, "
              f"Gallery: {len(self.gallery_samples)} satellite")


# =============================================================================
# DATASET — SUES-200 (Cross-Domain Test Only)
# =============================================================================
class SUES200CrossDomainTest:
    """
    SUES-200 test set loader for cross-domain evaluation.
    No altitude conditioning (model from University-1652 has no FiLM).
    """
    def __init__(self, root, altitudes=None):
        self.root = root
        self.altitudes = altitudes or CFG.SUES_ALTITUDES
        self.drone_dir = os.path.join(root, CFG.SUES_DRONE_DIR)
        self.sat_dir   = os.path.join(root, CFG.SUES_SAT_DIR)
        loc_ids = CFG.SUES_TEST_LOCS
        self.locations = [f"{l:04d}" for l in loc_ids]
        self.location_to_idx = {l: i for i, l in enumerate(self.locations)}

        # Drone images (all altitudes)
        self.drone_samples, self.drone_labels, self.drone_alts = [], [], []
        for loc in self.locations:
            li = self.location_to_idx[loc]
            for alt in self.altitudes:
                ad = os.path.join(self.drone_dir, loc, alt)
                if not os.path.isdir(ad): continue
                for img in sorted(os.listdir(ad)):
                    if img.endswith(('.png', '.jpg', '.jpeg')):
                        self.drone_samples.append(os.path.join(ad, img))
                        self.drone_labels.append(li)
                        self.drone_alts.append(int(alt))

        # Satellite images (1 per location)
        self.sat_samples, self.sat_labels = [], []
        for loc in self.locations:
            sp = os.path.join(self.sat_dir, loc, "0.png")
            if not os.path.exists(sp): continue
            self.sat_samples.append(sp)
            self.sat_labels.append(self.location_to_idx[loc])

        print(f"  [SUES-200 cross-test] {len(self.drone_samples)} drone imgs, "
              f"{len(self.sat_samples)} sat imgs, {len(self.locations)} locations")


class PKSampler:
    def __init__(self, ds, p, k):
        self.ds = ds; self.p = p; self.k = k
        self.locs = list(ds.drone_by_class.keys())
    def __iter__(self):
        locs = self.locs.copy(); random.shuffle(locs); batch = []
        for l in locs:
            idx = self.ds.drone_by_class[l]
            if len(idx) < self.k: idx = idx * (self.k // len(idx) + 1)
            batch.extend(random.sample(idx, self.k))
            if len(batch) >= self.p * self.k:
                yield batch[:self.p * self.k]; batch = batch[self.p * self.k:]
    def __len__(self): return len(self.locs) // self.p


# =============================================================================
# BACKBONE
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
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return features['x_norm_patchtokens'], features['x_norm_clstoken'], (H, W)


# =============================================================================
# PART DISCOVERY + POOLING + FUSION (no altitude — same as University-1652)
# =============================================================================
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
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
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
        recon_norm = F.normalize(recon, dim=-1)
        target_norm = F.normalize(masked_targets, dim=-1)
        return (1 - (recon_norm * target_norm).sum(dim=-1)).mean()


# =============================================================================
# MODEL (no altitude)
# =============================================================================
class SPDGeoDPEMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
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
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPE-MAR student: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)

    def extract_with_assignment(self, x):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb), parts['assignment']

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
# LOSSES
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
        kl_qr = F.kl_div((dist_q+1e-8).log(), dist_r, reduction='batchmean', log_target=False)
        kl_rq = F.kl_div((dist_r+1e-8).log(), dist_q, reduction='batchmean', log_target=False)
        return 0.5*(kl_qr+kl_rq)

class UAPALoss(nn.Module):
    def __init__(self, base_temperature=4.0): super().__init__(); self.T0 = base_temperature
    @staticmethod
    def _entropy(logits):
        p = F.softmax(logits, dim=1); return -(p*(p+1e-8).log()).sum(dim=1).mean()
    def forward(self, drone_logits, sat_logits):
        T = self.T0 * (1 + torch.sigmoid(self._entropy(drone_logits) - self._entropy(sat_logits)))
        return (T**2) * F.kl_div(F.log_softmax(drone_logits/T, 1), F.softmax(sat_logits/T, 1).detach(), reduction='batchmean')

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


# =============================================================================
# MODEL COMPLEXITY
# =============================================================================
def print_model_complexity(model, device, img_size=None):
    """Print #params, GFLOPs, and ms/query."""
    sz = img_size or CFG.IMG_SIZE
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
    except Exception: pass
    model.eval()
    with torch.no_grad():
        for _ in range(10): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000
    print(f"\n  ┌─ Model Complexity ─────────────────────────┐")
    print(f"  │  Total params:     {total_params/1e6:>8.2f}M              │")
    print(f"  │  Trainable params: {trainable_params/1e6:>8.2f}M              │")
    if gflops is not None:
        print(f"  │  GFLOPs:           {gflops:>8.2f}               │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)  │")
    print(f"  │  Inference:        {ms_query:>8.1f} ms/query       │")
    print(f"  └─────────────────────────────────────────────┘")
    return {'total_params': total_params, 'trainable_params': trainable_params,
            'gflops': gflops, 'ms_query': ms_query}




# =============================================================================
# Prototype Diversity Loss + GeoPartLoss (3-Group, No Altitude)
# =============================================================================
class PrototypeDiversityLoss(nn.Module):
    def forward(self, prototypes):
        P = F.normalize(prototypes, dim=-1)
        sim = P @ P.T; K = sim.size(0)
        mask = 1 - torch.eye(K, device=sim.device)
        return (sim * mask).abs().sum() / (K * (K - 1))


class GeoPartLoss(nn.Module):
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
        print(f"  GeoPartLoss: {num_groups} groups (no altitude) with adaptive weighting")

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
        loss_dict.update({'recon': l_recon.item() if torch.is_tensor(l_recon) else l_recon, 'div': l_div.item()})
        # Adaptive weighting
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
# EVALUATION — University-1652 (in-domain, drone→sat)
# =============================================================================
@torch.no_grad()
def evaluate_uni1652(model, test_data, device):
    model.eval()
    test_tf = get_transforms("test")
    t_start = time.time()

    query_feats, batch_imgs = [], []
    for i, img_path in enumerate(tqdm(test_data.query_samples, desc='Uni Query', leave=False)):
        batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
        if len(batch_imgs) == 64 or i == len(test_data.query_samples) - 1:
            query_feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
            batch_imgs = []
    query_feats = torch.cat(query_feats)
    query_labels = torch.tensor(test_data.query_labels)

    gallery_feats, batch_imgs = [], []
    for i, img_path in enumerate(tqdm(test_data.gallery_samples, desc='Uni Gallery', leave=False)):
        batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
        if len(batch_imgs) == 64 or i == len(test_data.gallery_samples) - 1:
            gallery_feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
            batch_imgs = []
    gallery_feats = torch.cat(gallery_feats)
    gallery_labels = torch.tensor(test_data.gallery_labels)

    sim = query_feats @ gallery_feats.T; _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0); r1=r5=r10=ap=0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        f = matches[0].item()
        if f < 1: r1 += 1
        if f < 5: r5 += 1
        if f < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)
    t_elapsed = time.time() - t_start
    return {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100,
            'inference_time': t_elapsed}


# =============================================================================
# EVALUATION — SUES-200 (cross-domain)
# =============================================================================
@torch.no_grad()
def evaluate_sues200_cross(model, sues_test, device, direction="drone2sat"):
    """
    Cross-domain evaluation on SUES-200 using model trained on University-1652.
    No altitude conditioning (model has no FiLM).

    direction: "drone2sat" or "sat2drone"
    """
    model.eval()
    test_tf = get_transforms("test")
    t_start = time.time()

    # Extract drone features
    drone_feats, batch_imgs = [], []
    for i, img_path in enumerate(tqdm(sues_test.drone_samples, desc=f'SUES Drone', leave=False)):
        try: batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
        except: batch_imgs.append(torch.zeros(3, CFG.IMG_SIZE, CFG.IMG_SIZE))
        if len(batch_imgs) == 64 or i == len(sues_test.drone_samples) - 1:
            drone_feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
            batch_imgs = []
    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.tensor(sues_test.drone_labels)
    drone_alts = torch.tensor(sues_test.drone_alts)

    # Extract satellite features
    sat_feats, batch_imgs = [], []
    for i, img_path in enumerate(tqdm(sues_test.sat_samples, desc=f'SUES Sat', leave=False)):
        try: batch_imgs.append(test_tf(Image.open(img_path).convert('RGB')))
        except: batch_imgs.append(torch.zeros(3, CFG.IMG_SIZE, CFG.IMG_SIZE))
        if len(batch_imgs) == 64 or i == len(sues_test.sat_samples) - 1:
            sat_feats.append(model.extract_embedding(torch.stack(batch_imgs).to(device)).cpu())
            batch_imgs = []
    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sues_test.sat_labels)

    # Retrieval
    if direction == "drone2sat":
        query_feats, query_labels = drone_feats, drone_labels
        gallery_feats, gallery_labels = sat_feats, sat_labels
        tag = "DRONE → SAT"
    else:
        query_feats, query_labels = sat_feats, sat_labels
        gallery_feats, gallery_labels = drone_feats, drone_labels
        tag = "SAT → DRONE"

    sim = query_feats @ gallery_feats.T; _, rank = sim.sort(1, descending=True)
    N = query_feats.size(0); r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
        if len(matches) == 0: continue
        first = matches[0].item()
        if first < 1: r1 += 1
        if first < 5: r5 += 1
        if first < 10: r10 += 1
        ap += sum((j+1)/(p.item()+1) for j, p in enumerate(matches)) / len(matches)

    t_elapsed = time.time() - t_start
    overall = {'R@1': r1/N*100, 'R@5': r5/N*100, 'R@10': r10/N*100, 'mAP': ap/N*100,
               'inference_time': t_elapsed}

    # Per-altitude breakdown (only for drone2sat — drone as query)
    per_alt = {}
    if direction == "drone2sat":
        altitudes = sorted(drone_alts.unique().tolist())
        for alt in altitudes:
            mask = drone_alts == alt
            if mask.sum() == 0: continue
            af = drone_feats[mask]; al = drone_labels[mask]
            s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
            n = af.size(0); a1 = a5 = a10 = aap = 0
            for i in range(n):
                m = torch.where(sat_labels[rk[i]] == al[i])[0]
                if len(m) == 0: continue
                f = m[0].item()
                if f < 1: a1 += 1
                if f < 5: a5 += 1
                if f < 10: a10 += 1
                aap += sum((j+1)/(p.item()+1) for j, p in enumerate(m)) / len(m)
            per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100, 'R@10': a10/n*100,
                                 'mAP': aap/n*100, 'n': n}

    # Print
    print(f"\n{'='*75}")
    print(f"  SUES-200 Cross-Domain ({tag}) — Model trained on University-1652")
    print(f"  Queries: {N} | Gallery: {len(gallery_feats)}")
    print(f"{'='*75}")
    if per_alt:
        print(f"  {'Altitude':>8s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'#Query':>6s}")
        print(f"  {'-'*50}")
        for alt in sorted(per_alt.keys()):
            a = per_alt[alt]
            print(f"  {alt:>6d}m  {a['R@1']:6.2f}%  {a['R@5']:6.2f}%  {a['R@10']:6.2f}%  {a['mAP']:6.2f}%  {a['n']:>6d}")
        print(f"  {'-'*50}")

    print(f"  {'Overall':>8s}  {overall['R@1']:6.2f}%  {overall['R@5']:6.2f}%  {overall['R@10']:6.2f}%  {overall['mAP']:6.2f}%  {N:>6d}")

    print(f"  Inference time: {t_elapsed:.2f}s ({N/t_elapsed:.1f} queries/s)")

    print(f"{'='*75}\n")

    return overall, per_alt



# =============================================================================
# TRAINING
# =============================================================================
def train_one_epoch(model, ema, geo_loss, optimizer, scaler, loader, device, epoch):
    model.train()
    total_sum = 0; n = 0; loss_sums = defaultdict(float)
    all_weights = defaultdict(float)

    for batch in tqdm(loader, desc=f'Ep{epoch:3d}', leave=False):
        drone  = batch['drone'].to(device)
        sat    = batch['satellite'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            d_out = model(drone, return_parts=True)
            s_out = model(sat, return_parts=True)
            loss, ld, gw = geo_loss(d_out, s_out, labels, model, ema,
                                     drone, sat, epoch, device)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update()
        ema.update(model)

        total_sum += loss.item(); n += 1
        for k, v in ld.items(): loss_sums[k] += v
        for k, v in gw.items(): all_weights[k] += v

    avg_weights = {k: v / max(n, 1) for k, v in all_weights.items()}
    return total_sum / max(n, 1), {k: v / max(n, 1) for k, v in loss_sums.items()}, avg_weights


# =============================================================================
# MAIN
# =============================================================================
def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    print("=" * 75)
    print("  GeoPartLoss CROSS-DATASET: Train Uni-1652 → Test SUES-200 + TTA")
    print(f"  3-group adaptive (Alignment + EMA + Part Quality)")
    print(f"  Epochs: {CFG.NUM_EPOCHS} | Img: {CFG.IMG_SIZE} | UNF: {CFG.UNFREEZE_BLOCKS}")
    print(f"  TTA: {CFG.TTA_SCALES} x flip={CFG.TTA_FLIP}")
    print("=" * 75)

    print('\n[DATA] Loading University-1652 …')
    train_ds  = University1652TrainDataset(CFG.UNI_ROOT, transform=get_transforms("train"))
    uni_test  = University1652TestDataset(CFG.UNI_ROOT)
    train_loader = DataLoader(train_ds, batch_sampler=PKSampler(train_ds, CFG.P_CLASSES, CFG.K_SAMPLES),
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    print('\n[DATA] Loading SUES-200 …')
    sues_test = SUES200CrossDomainTest(CFG.SUES_ROOT)

    print('\n[MODEL] Building SPDGeo-DPE-MAR …')
    model = SPDGeoDPEMARModel(train_ds.num_classes).to(DEVICE)
    ema = EMAModel(model, decay=CFG.EMA_DECAY)
    print(f"  EMA initialized (decay={CFG.EMA_DECAY})")
    model_stats = print_model_complexity(model, DEVICE)

    geo_loss = GeoPartLoss(train_ds.num_classes, CFG.EMBED_DIM,
                           num_groups=CFG.NUM_LOSS_GROUPS, cfg=CFG).to(DEVICE)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('backbone')]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params,              'lr': CFG.BACKBONE_LR},
        {'params': head_params,                  'lr': CFG.LR},
        {'params': geo_loss.parameters(),        'lr': CFG.LR},
    ], weight_decay=CFG.WEIGHT_DECAY)

    scaler = torch.amp.GradScaler('cuda', enabled=CFG.USE_AMP)
    best_r1_uni = 0.0; best_r1_sues = 0.0
    results_log = []; loss_history = []
    import datetime
    train_start = datetime.datetime.now().isoformat()

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: TRAINING
    # ════════════════════════════════════════════════════════════════
    for epoch in range(1, CFG.NUM_EPOCHS + 1):
        if epoch <= CFG.WARMUP_EPOCHS:
            lr_scale = epoch / CFG.WARMUP_EPOCHS
        else:
            progress = (epoch - CFG.WARMUP_EPOCHS) / (CFG.NUM_EPOCHS - CFG.WARMUP_EPOCHS)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        lr_scale = max(lr_scale, 0.01)
        for i in range(3):
            optimizer.param_groups[i]['lr'] = (CFG.BACKBONE_LR if i == 0 else CFG.LR) * lr_scale
        cur_lr = optimizer.param_groups[1]['lr']

        avg_loss, ld, gw = train_one_epoch(model, ema, geo_loss, optimizer, scaler,
                                            train_loader, DEVICE, epoch)
        loss_history.append({'epoch': epoch, 'total': avg_loss, 'lr': cur_lr, **ld, 'group_weights': gw})
        print(f"Ep {epoch:3d}/{CFG.NUM_EPOCHS} | Loss {avg_loss:.4f} | "
              f"G_align {ld.get('G_align',0):.3f}  G_ema {ld.get('G_ema',0):.3f}  "
              f"G_part {ld.get('G_part',0):.3f} | LR {cur_lr:.2e}")

        if epoch % CFG.EVAL_INTERVAL == 0 or epoch == CFG.NUM_EPOCHS:
            uni_metrics = evaluate_uni1652(model, uni_test, DEVICE)
            print(f"  ► [Uni-1652] R@1: {uni_metrics['R@1']:.2f}%  mAP: {uni_metrics['mAP']:.2f}%")

            sues_d2s, sues_d2s_alt = evaluate_sues200_cross(model, sues_test, DEVICE, "drone2sat")
            print(f"  ► [SUES D→S] R@1: {sues_d2s['R@1']:.2f}%  mAP: {sues_d2s['mAP']:.2f}%")

            sues_s2d, _ = evaluate_sues200_cross(model, sues_test, DEVICE, "sat2drone")
            print(f"  ► [SUES S→D] R@1: {sues_s2d['R@1']:.2f}%  mAP: {sues_s2d['mAP']:.2f}%")

            entry = {'epoch': epoch, 'uni1652': uni_metrics,
                     'sues_d2s': sues_d2s, 'sues_d2s_alt': sues_d2s_alt,
                     'sues_s2d': sues_s2d}
            results_log.append(entry)

            if uni_metrics['R@1'] > best_r1_uni:
                best_r1_uni = uni_metrics['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'uni_metrics': uni_metrics, 'sues_d2s': sues_d2s, 'sues_s2d': sues_s2d},
                           os.path.join(CFG.OUTPUT_DIR, 'gpl_cross_uni2sues_best.pth'))
                print(f"  ★ New best Uni-1652 R@1: {best_r1_uni:.2f}%!")
            if sues_d2s['R@1'] > best_r1_sues:
                best_r1_sues = sues_d2s['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'uni_metrics': uni_metrics, 'sues_d2s': sues_d2s, 'sues_s2d': sues_s2d},
                           os.path.join(CFG.OUTPUT_DIR, 'gpl_cross_uni2sues_best_sues.pth'))
                print(f"  ★ New best SUES D→S R@1: {best_r1_sues:.2f}%!")

            ema_uni = evaluate_uni1652(ema.model, uni_test, DEVICE)
            print(f"  ► [EMA Uni-1652] R@1: {ema_uni['R@1']:.2f}%")
            if ema_uni['R@1'] > best_r1_uni:
                best_r1_uni = ema_uni['R@1']
                torch.save({'epoch': epoch, 'model_state_dict': ema.model.state_dict(),
                            'uni_metrics': ema_uni, 'is_ema': True},
                           os.path.join(CFG.OUTPUT_DIR, 'gpl_cross_uni2sues_best.pth'))
                print(f"  ★ New best Uni-1652 R@1 (EMA): {best_r1_uni:.2f}%!")

    print(f'\n{"="*75}')
    print(f'  TRAINING COMPLETE — Best Uni R@1: {best_r1_uni:.2f}%  Best SUES D→S: {best_r1_sues:.2f}%')
    print(f'{"="*75}')

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: TTA POST-PROCESSING
    # ════════════════════════════════════════════════════════════════
    import time as _time
    print(f'\n{"="*70}')
    print(f'  PHASE 2: TTA Post-Processing')
    print(f'  Scales: {CFG.TTA_SCALES} | Flip: {CFG.TTA_FLIP}')
    print(f'{"="*70}')

    # Load best in-domain model
    ckpt = torch.load(os.path.join(CFG.OUTPUT_DIR, 'gpl_cross_uni2sues_best.pth'),
                       map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  Loaded best model (epoch {ckpt['epoch']})")
    model.eval()

    test_tf = get_transforms("test")

    def extract_feats_tta(img_paths, model, device):
        BS = 8; all_feats = []
        for i in tqdm(range(0, len(img_paths), BS), desc="TTA", leave=False):
            batch_paths = img_paths[i:i+BS]
            imgs_raw = [Image.open(p).convert('RGB') for p in batch_paths]
            aug_feats = []
            for scale in CFG.TTA_SCALES:
                tf = transforms.Compose([transforms.Resize((scale, scale)), transforms.ToTensor(),
                                          transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
                imgs = torch.stack([tf(im) for im in imgs_raw]).to(device)
                with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                    aug_feats.append(model.extract_embedding(imgs))
                    if CFG.TTA_FLIP:
                        aug_feats.append(model.extract_embedding(torch.flip(imgs, [3])))
            all_feats.append(F.normalize(torch.stack(aug_feats).mean(0), dim=-1).cpu())
        return torch.cat(all_feats)

    def compute_metrics(query_feats, query_labels, gallery_feats, gallery_labels):
        sim = query_feats @ gallery_feats.T; _, rank = sim.sort(1, descending=True)
        N = query_feats.size(0); r1=r5=r10=ap=0
        for i in range(N):
            m = torch.where(gallery_labels[rank[i]] == query_labels[i])[0]
            if len(m)==0: continue
            f = m[0].item()
            if f < 1: r1 += 1
            if f < 5: r5 += 1
            if f < 10: r10 += 1
            ap += sum((j+1)/(p.item()+1) for j,p in enumerate(m))/len(m)
        return {'R@1':r1/N*100, 'R@5':r5/N*100, 'R@10':r10/N*100, 'mAP':ap/N*100}

    # --- TTA on University-1652 (both directions) ---
    print(f"\n  [1/3] TTA University-1652 (D2S + S2D)...")
    t0 = _time.time()
    tta_q = extract_feats_tta(uni_test.query_samples, model, DEVICE)
    tta_g = extract_feats_tta(uni_test.gallery_samples, model, DEVICE)
    q_labels = torch.tensor(uni_test.query_labels)
    g_labels = torch.tensor(uni_test.gallery_labels)
    tta_uni_d2s = compute_metrics(tta_q, q_labels, tta_g, g_labels)
    tta_uni_s2d = compute_metrics(tta_g, g_labels, tta_q, q_labels)
    print(f"    D2S: R@1={tta_uni_d2s['R@1']:.2f}%  mAP={tta_uni_d2s['mAP']:.2f}%")
    print(f"    S2D: R@1={tta_uni_s2d['R@1']:.2f}%  mAP={tta_uni_s2d['mAP']:.2f}%  ({_time.time()-t0:.0f}s)")

    # --- TTA on SUES-200 (both directions) ---
    print(f"\n  [2/3] TTA SUES-200 D2S...")
    t0 = _time.time()
    tta_d = extract_feats_tta(sues_test.drone_samples, model, DEVICE)
    tta_s = extract_feats_tta(sues_test.sat_samples, model, DEVICE)
    d_labels = torch.tensor(sues_test.drone_labels)
    s_labels = torch.tensor(sues_test.sat_labels)
    tta_sues_d2s = compute_metrics(tta_d, d_labels, tta_s, s_labels)
    print(f"    R@1={tta_sues_d2s['R@1']:.2f}%  mAP={tta_sues_d2s['mAP']:.2f}%  ({_time.time()-t0:.0f}s)")

    print(f"\n  [3/3] TTA SUES-200 S2D...")
    t0 = _time.time()
    tta_sues_s2d = compute_metrics(tta_s, s_labels, tta_d, d_labels)
    print(f"    R@1={tta_sues_s2d['R@1']:.2f}%  mAP={tta_sues_s2d['mAP']:.2f}%  ({_time.time()-t0:.0f}s)")

    # Summary
    print(f"\n  {'='*65}")
    print(f"  {'Dataset':<25s}  {'Direction':<8s}  {'R@1':>8s}  {'mAP':>8s}")
    print(f"  {'-'*55}")
    print(f"  {'Uni-1652 (TTA)':<25s}  {'D2S':<8s}  {tta_uni_d2s['R@1']:7.2f}%  {tta_uni_d2s['mAP']:7.2f}%")
    print(f"  {'Uni-1652 (TTA)':<25s}  {'S2D':<8s}  {tta_uni_s2d['R@1']:7.2f}%  {tta_uni_s2d['mAP']:7.2f}%")
    print(f"  {'SUES-200 Cross (TTA)':<25s}  {'D2S':<8s}  {tta_sues_d2s['R@1']:7.2f}%  {tta_sues_d2s['mAP']:7.2f}%")
    print(f"  {'SUES-200 Cross (TTA)':<25s}  {'S2D':<8s}  {tta_sues_s2d['R@1']:7.2f}%  {tta_sues_s2d['mAP']:7.2f}%")
    print(f"  {'='*65}")

    train_end = datetime.datetime.now().isoformat()
    run_summary = {
        'experiment': 'GeoPartLoss Cross-Dataset: Uni1652 to SUES200',
        'train_dataset': 'University-1652',
        'test_datasets': ['University-1652', 'SUES-200'],
        'direction': 'both (drone2sat + sat2drone)',
        'timestamp': {'start': train_start, 'end': train_end},
        'model_complexity': model_stats,
        'config': {k: v for k, v in vars(CFG).items() if not k.startswith('_')},
        'loss_type': 'GeoPartLoss_adaptive_3groups',
        'best_r1_uni': best_r1_uni,
        'best_r1_sues': best_r1_sues,
        'eval_history': results_log,
        'loss_history': loss_history,
        'tta_config': {'scales': CFG.TTA_SCALES, 'flip': CFG.TTA_FLIP},
        'tta_uni1652_d2s': tta_uni_d2s,
        'tta_uni1652_s2d': tta_uni_s2d,
        'tta_sues_d2s': tta_sues_d2s,
        'tta_sues_s2d': tta_sues_s2d,
    }
    with open(os.path.join(CFG.OUTPUT_DIR, 'gpl_cross_uni2sues_results.json'), 'w') as f:
        json.dump(run_summary, f, indent=2)
    print(f'Results saved to {CFG.OUTPUT_DIR}/gpl_cross_uni2sues_results.json')


if __name__ == '__main__':
    main()
