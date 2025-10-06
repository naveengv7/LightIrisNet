#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iris segmentation (VIS/NIR) — DeepLabv3+-style with multi-task heads.
Combined iris + pupil upgrades with opt-in flags.

New (opt-in) features:
  --uncertainty_weighting            # homoscedastic loss weighting (learned log-sigmas)
  --iris_floor 1.1                   # minimum multiplier on iris loss even with uncertainty weighting
  --extra_decoder_conv               # add a 3x3 conv block pre-head for more local context
  --iris_head_3x3                    # iris head uses 3x3->1x1 (vs 1x1) for local context
  --pupil_refine_depth 1             # depth (1-2) of pupil boundary refinement residual DW convs
  --iris_tversky                     # add Tversky to iris loss
  --pupil_tversky                    # add Tversky to pupil loss
  --iris_tversky_alpha 0.6 --iris_tversky_beta 0.4 --iris_tversky_gamma 1.0
  --pupil_tversky_alpha 0.7 --pupil_tversky_beta 0.3 --pupil_tversky_gamma 1.3333
  --iris_boundary_boost 1.2          # scale contour/consistency terms for iris
  --pupil_boundary_boost 1.2         # scale contour/consistency terms for pupil
  --iris_sdt_r 20 --pupil_sdt_r 16   # SDT radius per class (default 12)
  --aug_limbus                       # vignette, eyelash occluders, mild perspective, per-channel gamma
  --warmup_epochs 5                  # LR warmup steps before cosine anneal
  --grad_clip 1.0                    # gradient clip (L2)
  --dataset_loss_scale               # enable per-dataset multipliers via split suffixes
                                     # mapping: mich1→MICHE, ubv1→UBIRISv1, ubv2→UBIRISv2, cor/visJ/visR→CUVIRIS

Everything else preserves your original defaults and behavior.
"""

import os, argparse, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
cv2.setNumThreads(16)

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def imread_rgb(path, size=None, keep_aspect=True):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None: raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2: img = np.stack([img, img, img], -1)
    if img.shape[2] == 4: img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    if size is not None and size > 0:
        if keep_aspect:
            h, w = img.shape[:2]; ar = w / h
            new_h = size; new_w = int(round(size * ar))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

def imread_mask(path, size=None, keep_aspect=True):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None: raise FileNotFoundError(f"Could not read mask: {path}")
    if size is not None and size > 0:
        if keep_aspect:
            h, w = m.shape[:2]; ar = w / h
            m = cv2.resize(m, (int(round(size*ar)), size), interpolation=cv2.INTER_NEAREST)
        else:
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.float32)

def imread_edge(path, size=None, keep_aspect=True):
    e = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if e is None: raise FileNotFoundError(f"Could not read edge map: {path}")
    if size is not None and size > 0:
        if keep_aspect:
            h, w = e.shape[:2]; ar = w / h
            e = cv2.resize(e, (int(round(size*ar)), size), interpolation=cv2.INTER_NEAREST)
        else:
            e = cv2.resize(e, (size, size), interpolation=cv2.INTER_NEAREST)
    return (e > 127).astype(np.float32)

def gaussian_blur_soft_boundary(edge_mask, sigma=1.0):
    src = edge_mask.astype(np.float32)
    k = max(3, int(2 * round(3 * sigma) + 1))
    soft = cv2.GaussianBlur(src, (k, k), sigma, sigma, borderType=cv2.BORDER_REFLECT101)
    if soft.max() > 0: soft = soft / soft.max()
    return soft

def distance_transform_loss_weight(edge_mask, eps=1e-6):
    non_edge = (edge_mask < 0.5).astype(np.uint8) * 255
    dist = cv2.distanceTransform(non_edge, cv2.DIST_L2, 3).astype(np.float32)
    dist = dist / (dist.max() + eps)  # 0 at edge, 1 far
    return 1.0 - dist                  # 1 near edge, 0 far

def signed_dt_from_mask(mask, r=12):
    pos = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    neg = cv2.distanceTransform((mask <= 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    sdt = pos - neg
    sdt = np.clip(sdt, -r, r) / float(r)
    return sdt.astype(np.float32)

# ---------- Ellipse GT helpers ----------
def _fit_ellipse_params(bin_mask):
    """
    Fit an ellipse to a binary mask. Returns (cx, cy, rx, ry, sinθ, cosθ) + valid flag.
    """
    cnts, _ = cv2.findContours((bin_mask>0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, False
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 20: return None, False
    (cx, cy), (ma, mi), ang_deg = cv2.fitEllipse(cnt)
    rx = 0.5 * max(ma, mi); ry = 0.5 * min(ma, mi)
    ang = np.deg2rad(ang_deg)
    return (cx, cy, rx, ry, np.sin(ang), np.cos(ang)), True

def _norm_ellipse_params(params, H, W):
    cx, cy, rx, ry, s, c = params
    # FIX retained: ry normalized by H
    return np.array([cx/W, cy/H, rx/W, ry/H, s, c], dtype=np.float32)

# -----------------------------
# Dataset
# -----------------------------

DATASET_SUFFIX_MAP = {
    'mich1': 'MICHE',
    'ubv1' : 'UBIRISv1',
    'ubv2' : 'UBIRISv2',
    'cor'  : 'CUVIRIS',
    'visJ' : 'CUVIRIS',
    'visR' : 'CUVIRIS',
}

def infer_dataset_from_id(stem: str):
    for suf, name in DATASET_SUFFIX_MAP.items():
        if stem.endswith('_' + suf):
            return name
    return 'UNKNOWN'

class IrisSegDataset(Dataset):
    def __init__(self, root, split_file, img_size=-1,
                 images_dir='images', iris_dir='labels_iris', pupil_dir='labels_pupil',
                 edges_iris_dir='edges_iris', edges_pupil_dir='edges_pupil',
                 augment=False, paper_aug=False, aug_limbus=False,
                 iris_sdt_r=12, pupil_sdt_r=12):
        self.root = Path(root)
        self.img_size = img_size
        self.images_dir = self.root / images_dir
        self.iris_dir = self.root / iris_dir
        self.pupil_dir = self.root / pupil_dir
        self.edges_iris_dir = self.root / edges_iris_dir if edges_iris_dir else None
        self.edges_pupil_dir = self.root / edges_pupil_dir if edges_pupil_dir else None
        self.augment = augment
        self.paper_aug = paper_aug
        self.aug_limbus = aug_limbus
        self.iris_sdt_r = iris_sdt_r
        self.pupil_sdt_r = pupil_sdt_r

        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]
        if not self.ids: raise ValueError(f"No IDs in {split_file}")

    def __len__(self): return len(self.ids)

    # ---- Paper-style augmentations (IrisParseNet): scale, blur, translate, flip, rotate, crop 321×321 ----
    def _augment_paper(self, img, iris, pupil, e_iris, e_pupil):
        # (a) isotropic scale
        s = random.choice([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        interp_img = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        img   = cv2.resize(img,   None, fx=s, fy=s, interpolation=interp_img)
        iris  = cv2.resize(iris,  None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        pupil = cv2.resize(pupil, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        e_iris  = cv2.resize(e_iris,  None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        e_pupil = cv2.resize(e_pupil, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

        Hs, Ws = img.shape[:2]

        # (b) blur: one of mean/gauss/median/bilateral/box
        blur_mode = random.choice(['mean','gauss','median','bilateral','box'])
        if blur_mode == 'mean':
            k = random.choice([3,5]); img = cv2.blur(img, (k,k))
        elif blur_mode == 'gauss':
            k = random.choice([3,5]); img = cv2.GaussianBlur(img, (k,k), 0)
        elif blur_mode == 'median':
            k = random.choice([3,5]); img = cv2.medianBlur((img*255).astype(np.uint8), k).astype(np.float32)/255.0
        elif blur_mode == 'bilateral':
            img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
        else: # box
            k = random.choice([3,5]); img = cv2.boxFilter(img, -1, (k,k))

        # (c) translation in [-30, +30] px
        tx = random.randint(-30, 30); ty = random.randint(-30, 30)
        M_t = np.float32([[1,0,tx],[0,1,ty]])
        def warp(m, mode_img):
            interp = cv2.INTER_LINEAR if mode_img else cv2.INTER_NEAREST
            border = cv2.BORDER_CONSTANT
            val = (0,0,0) if mode_img else 0
            return cv2.warpAffine(m, M_t, (Ws, Hs), flags=interp, borderMode=border, borderValue=val)
        img   = warp(img, True)
        iris  = warp(iris, False)
        pupil = warp(pupil, False)
        e_iris  = warp(e_iris, False)
        e_pupil = warp(e_pupil, False)

        # (d) horizontal flip 0.5
        if random.random() < 0.5:
            img   = np.fliplr(img).copy()
            iris  = np.fliplr(iris).copy()
            pupil = np.fliplr(pupil).copy()
            e_iris  = np.fliplr(e_iris).copy()
            e_pupil = np.fliplr(e_pupil).copy()

        # (e) rotation [-60, 60] deg
        Hs, Ws = img.shape[:2]
        M_r = cv2.getRotationMatrix2D((Ws/2, Hs/2), random.uniform(-60,60), 1.0)
        img   = cv2.warpAffine(img,   M_r, (Ws, Hs), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        iris  = cv2.warpAffine(iris,  M_r, (Ws, Hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        pupil = cv2.warpAffine(pupil, M_r, (Ws, Hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        e_iris  = cv2.warpAffine(e_iris,  M_r, (Ws, Hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        e_pupil = cv2.warpAffine(e_pupil, M_r, (Ws, Hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # (f) final random crop to 321×321
        CH, CW = 321, 321
        if Hs < CH or Ws < CW:
            img    = _pad_to_min_hw(img,   CH, CW)
            iris   = _pad_to_min_hw(iris,  CH, CW)
            pupil  = _pad_to_min_hw(pupil, CH, CW)
            e_iris = _pad_to_min_hw(e_iris, CH, CW)
            e_pupil= _pad_to_min_hw(e_pupil, CH, CW)
            Hs, Ws = img.shape[:2]
        y0 = random.randint(0, Hs - CH)
        x0 = random.randint(0, Ws - CW)
        img    = img[y0:y0+CH, x0:x0+CW]
        iris   = iris[y0:y0+CH, x0:x0+CW]
        pupil  = pupil[y0:y0+CH, x0:x0+CW]
        e_iris = e_iris[y0:y0+CH, x0:x0+CW]
        e_pupil= e_pupil[y0:y0+CH, x0:x0+CW]
        return img, iris, pupil, e_iris, e_pupil

    def _augment_limbus(self, img, iris, pupil):
        H, W = img.shape[:2]
        # v1: per-channel gamma
        if random.random() < 0.7:
            gammas = np.clip(np.random.normal(loc=1.0, scale=0.15, size=3), 0.6, 1.6)
            img = np.clip(img ** gammas.reshape(1,1,3), 0.0, 1.0)
        # v2: vignette (low-light rim)
        if random.random() < 0.5:
            yy, xx = np.mgrid[0:H, 0:W]
            cx = W*0.5 + np.random.uniform(-0.1*W, 0.1*W)
            cy = H*0.5 + np.random.uniform(-0.1*H, 0.1*H)
            r = max(H, W)*np.random.uniform(0.6, 1.0)
            dist = np.sqrt((xx-cx)**2 + (yy-cy)**2) / r
            vig = np.clip(1.0 - 0.5*(dist**2), 0.4, 1.0)[...,None]
            img = np.clip(img*vig, 0.0, 1.0)
        # v3: thin occluders (eyelashes)
        if random.random() < 0.35:
            k = random.randint(1,3)
            for _ in range(k):
                x1 = random.randint(0, W-1); x2 = random.randint(0, W-1)
                y  = random.randint(int(0.05*H), int(0.45*H))
                cv2.line(img, (x1,y), (x2,y+random.randint(-5,5)), (0,0,0), thickness=random.randint(1,2))
        # v4: mild perspective
        if random.random() < 0.25:
            dx = int(0.03*W)
            src = np.float32([[0,0],[W-1,0],[0,H-1],[W-1,H-1]])
            dst = src + np.float32([[random.randint(-dx,dx),0],
                                    [random.randint(-dx,dx),0],
                                    [random.randint(-dx,dx),0],
                                    [random.randint(-dx,dx),0]])
            M = cv2.getPerspectiveTransform(src, dst)
            img = cv2.warpPerspective(img, M, (W,H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
            iris  = cv2.warpPerspective(iris, M, (W,H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            pupil = cv2.warpPerspective(pupil, M, (W,H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img, iris, pupil

    def _augment(self, img, iris, pupil, e_iris, e_pupil):
        # Prefer paper aug if requested
        if self.paper_aug:
            img, iris, pupil, e_iris, e_pupil = self._augment_paper(img, iris, pupil, e_iris, e_pupil)
        else:
            # Original VIS aug subset
            H, W = img.shape[:2]
            if random.random() < 0.5:
                gamma = random.uniform(0.6, 1.6)
                img = np.clip(img ** gamma, 0.0, 1.0)
            if random.random() < 0.3:
                lab = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2LAB)
                L, A, B = cv2.split(lab)
                L = cv2.createCLAHE(2.0, (8,8)).apply(L)
                img = cv2.cvtColor(cv2.merge([L,A,B]), cv2.COLOR_LAB2RGB).astype(np.float32)/255.0
            if random.random() < 0.25:
                k = random.choice([3,5,7]); kernel = np.zeros((k,k), np.float32); kernel[k//2,:] = 1.0/k
                img = cv2.filter2D(img, -1, kernel)
            if random.random() < 0.5:
                img = np.fliplr(img).copy(); iris = np.fliplr(iris).copy(); pupil = np.fliplr(pupil).copy()
                e_iris = np.fliplr(e_iris).copy(); e_pupil = np.fliplr(e_pupil).copy()
            if random.random() < 0.5:
                angle = random.uniform(-20, 20); scale = random.uniform(0.9, 1.1)
                M = cv2.getRotationMatrix2D((W/2, H/2), angle, scale)
                img   = cv2.warpAffine(img,   M, (W, H), flags=cv2.INTER_LINEAR,  borderMode=cv2.BORDER_REFLECT101)
                iris  = cv2.warpAffine(iris,  M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                pupil = cv2.warpAffine(pupil, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                e_iris  = cv2.warpAffine(e_iris,  M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                e_pupil = cv2.warpAffine(e_pupil, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # synthetic glints
            if random.random() < 0.3:
                for _ in range(random.randint(1,3)):
                    r = random.randint(2,5)
                    cx = random.randint(r, W-r-1); cy = random.randint(r, H-r-1)
                    cv2.circle(img, (cx,cy), r, (1.0,1.0,1.0), -1)
                    img = np.clip(img, 0, 1)

        if self.aug_limbus:
            img, iris, pupil = self._augment_limbus(img, iris, pupil)

        return img, iris, pupil, e_iris, e_pupil

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        stem = id_
        def find_file(folder, stem, exts):
            for e in exts:
                p = folder / f"{stem}{e}"
                if p.exists(): return p
            for e in exts:
                cs = list(folder.glob(f"{stem}*{e}"))
                if cs: return cs[0]
            raise FileNotFoundError(f"Could not find {stem} in {folder}")

        exts_img = ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']
        exts_msk = ['.png','.bmp','.jpg','.jpeg','.tif','.tiff']
        img_path   = find_file(self.images_dir, stem, exts_img)
        iris_path  = find_file(self.iris_dir,  stem, exts_msk)
        pupil_path = find_file(self.pupil_dir, stem, exts_msk)

        img = imread_rgb(img_path, size=self.img_size)
        iris = imread_mask(iris_path, size=self.img_size)
        pupil = imread_mask(pupil_path, size=self.img_size)

        if self.edges_iris_dir and self.edges_iris_dir.exists():
            e_iris = imread_edge(find_file(self.edges_iris_dir, stem, exts_msk), size=self.img_size)
        else:
            e_iris = (cv2.Canny((iris*255).astype(np.uint8), 50, 100) > 0).astype(np.float32)
        if self.edges_pupil_dir and self.edges_pupil_dir.exists():
            e_pupil = imread_edge(find_file(self.edges_pupil_dir, stem, exts_msk), size=self.img_size)
        else:
            e_pupil = (cv2.Canny((pupil*255).astype(np.uint8), 50, 100) > 0).astype(np.float32)

        if self.augment:
            img, iris, pupil, e_iris, e_pupil = self._augment(img, iris, pupil, e_iris, e_pupil)

        H, W = img.shape[:2]
        bnd_iris = gaussian_blur_soft_boundary(e_iris, sigma=1.2)
        bnd_pupil = gaussian_blur_soft_boundary(e_pupil, sigma=1.0)
        dtw_iris = distance_transform_loss_weight(e_iris)
        dtw_pupil = distance_transform_loss_weight(e_pupil)
        sdt_i = signed_dt_from_mask(iris, r=self.iris_sdt_r)
        sdt_p = signed_dt_from_mask(pupil, r=self.pupil_sdt_r)

        # Ellipse GTs (normalized) + valid flags
        ell_i_params, ell_i_valid = _fit_ellipse_params(iris)
        ell_p_params, ell_p_valid = _fit_ellipse_params(pupil)
        ell_i = _norm_ellipse_params(ell_i_params, H, W) if ell_i_valid else np.zeros((6,), np.float32)
        ell_p = _norm_ellipse_params(ell_p_params, H, W) if ell_p_valid else np.zeros((6,), np.float32)

        img_t = torch.from_numpy(img.transpose(2,0,1))
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        img_t = (img_t - mean) / std

        dataset_name = infer_dataset_from_id(stem)

        return {
            'image': img_t.float(),
            'iris': torch.from_numpy(iris).unsqueeze(0).float(),
            'pupil': torch.from_numpy(pupil).unsqueeze(0).float(),
            'bnd_i': torch.from_numpy(bnd_iris).unsqueeze(0).float(),
            'bnd_p': torch.from_numpy(bnd_pupil).unsqueeze(0).float(),
            'dtw_i': torch.from_numpy(dtw_iris).unsqueeze(0).float(),
            'dtw_p': torch.from_numpy(dtw_pupil).unsqueeze(0).float(),
            'sdt_i': torch.from_numpy(sdt_i).unsqueeze(0).float(),
            'sdt_p': torch.from_numpy(sdt_p).unsqueeze(0).float(),
            's_ell_i': torch.from_numpy(ell_i).float(),
            's_ell_p': torch.from_numpy(ell_p).float(),
            'ell_i_valid': torch.tensor(float(ell_i_valid)).float(),
            'ell_p_valid': torch.tensor(float(ell_p_valid)).float(),
            'id': stem,
            'size': (H, W),
            'dataset': dataset_name,
        }

# -----------------------------
# Variable-size padding collate
# -----------------------------

def _pad2d(x, Ht, Wt, val=0):
    if x.dim()==3:  # C,H,W
        _,H,W = x.shape
        return F.pad(x, (0, Wt-W, 0, Ht-H), value=val)
    elif x.dim()==2:  # H,W
        H,W = x.shape
        return F.pad(x, (0, Wt-W, 0, Ht-H), value=val)
    return x

def collate_pad(batch, multiple=32):
    Hm = max(b['image'].shape[-2] for b in batch)
    Wm = max(b['image'].shape[-1] for b in batch)
    Ht = ((Hm + multiple - 1)//multiple)*multiple
    Wt = ((Wm + multiple - 1)//multiple)*multiple

    keys_1ch = ['iris','pupil','bnd_i','bnd_p','dtw_i','dtw_p','sdt_i','sdt_p']
    out = {}
    out['image'] = torch.stack([_pad2d(b['image'], Ht, Wt) for b in batch], 0)
    for k in keys_1ch:
        out[k] = torch.stack([_pad2d(b[k].squeeze(0), Ht, Wt).unsqueeze(0) for b in batch], 0)
    for k in ['s_ell_i','s_ell_p','ell_i_valid','ell_p_valid','id','size','dataset']:
        out[k] = [b[k] for b in batch]
    return out

# -----------------------------
# Models
# -----------------------------

class ResNet50Encoder(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        bb = resnet50(weights=weights)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1  # /4  ch 256
        self.layer2 = bb.layer2  # /8  ch 512
        self.layer3 = bb.layer3  # /16 ch 1024
        self.layer4 = bb.layer4  # /32 ch 2048
        self.low_ch, self.high_ch = 256, 2048
    def forward(self, x):
        x = self.stem(x); s4 = self.layer1(x)
        s8 = self.layer2(s4); s16 = self.layer3(s8); s32 = self.layer4(s16)
        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

class MNetV3Encoder(nn.Module):
    def __init__(self, weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1):
        super().__init__()
        m = mobilenet_v3_large(weights=weights).features
        self.s4  = nn.Sequential(*m[:6])    # ~40
        self.s8  = nn.Sequential(*m[6:10])  # ~80
        self.s16 = nn.Sequential(*m[10:13]) # ~112
        self.s32 = nn.Sequential(*m[13:])   # 960
        self.low_ch, self.high_ch = 40, 960
    def forward(self, x):
        x = self.s4(x); s4 = x
        x = self.s8(x); s8 = x
        x = self.s16(x); s16 = x
        x = self.s32(x); s32 = x
        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

class ASPP(nn.Module):
    def __init__(self, c_in, c_out=256, rates=(1,6,12,18)):
        super().__init__()
        self.b0 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.b1 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=rates[1], dilation=rates[1], bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.b2 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=rates[2], dilation=rates[2], bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=rates[3], dilation=rates[3], bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.proj = nn.Sequential(nn.Conv2d(c_out*5, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True), nn.Dropout2d(0.1))
    def forward(self, x):
        h,w = x.shape[-2:]
        ys = [self.b0(x), self.b1(x), self.b2(x), self.b3(x)]
        yp = self.pool(x); yp = F.interpolate(yp, size=(h,w), mode='bilinear', align_corners=False)
        ys.append(yp)
        return self.proj(torch.cat(ys,1))

class ASPP_dw(nn.Module):
    def __init__(self, c_in, c_out=128, rates=(1,6,12,18)):
        super().__init__()
        def dw(inC,outC,ks,pad,d):
            return nn.Sequential(
                nn.Conv2d(inC, inC, ks, padding=pad, dilation=d, groups=inC, bias=False),
                nn.BatchNorm2d(inC), nn.ReLU(True),
                nn.Conv2d(inC, outC, 1, bias=False),
                nn.BatchNorm2d(outC), nn.ReLU(True))
        self.b0 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.b1 = dw(c_in, c_out, 3, rates[1], rates[1])
        self.b2 = dw(c_in, c_out, 3, rates[2], rates[2])
        self.b3 = dw(c_in, c_out, 3, rates[3], rates[3])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True))
        self.proj = nn.Sequential(nn.Conv2d(c_out*5, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True), nn.Dropout2d(0.1))
    def forward(self, x):
        h,w = x.shape[-2:]
        ys = [self.b0(x), self.b1(x), self.b2(x), self.b3(x)]
        yp = self.pool(x); yp = F.interpolate(yp, (h,w), mode='bilinear', align_corners=False)
        ys.append(yp)
        return self.proj(torch.cat(ys,1))

class DecoderDeepLabV3Plus(nn.Module):
    def __init__(self, low_ch, aspp_ch, out_ch, extra_decoder_conv=False):
        super().__init__()
        self.low_proj = nn.Sequential(nn.Conv2d(low_ch, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(True))
        body = [
            nn.Conv2d(48+aspp_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        ]
        if extra_decoder_conv:
            body += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)]
        self.fuse = nn.Sequential(*body)
    def forward(self, aspp, low):
        aspp_up = F.interpolate(aspp, size=low.shape[-2:], mode='bilinear', align_corners=False)
        low = self.low_proj(low)
        return self.fuse(torch.cat([aspp_up, low], 1))

class PupilRefine(nn.Module):
    """Light boundary refinement block for pupil head."""
    def __init__(self, ch, depth=1):
        super().__init__()
        blocks = []
        for _ in range(max(1, depth)):
            blocks += [
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(True),
                nn.Conv2d(ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(True)
            ]
        self.block = nn.Sequential(*blocks)
    def forward(self, x):
        return x + self.block(x)

class IrisNetDeepLab(nn.Module):
    def __init__(self, backbone='resnet50', use_ellipse=False, extra_decoder_conv=False,
                 iris_head_3x3=False, pupil_refine_depth=0):
        super().__init__()
        self.backbone = backbone
        self.use_ellipse = use_ellipse
        if backbone == 'mobilenetv3':
            self.enc = MNetV3Encoder()
            self.aspp = ASPP_dw(self.enc.high_ch, 128)
            self.dec  = DecoderDeepLabV3Plus(self.enc.low_ch, 128, 128, extra_decoder_conv=extra_decoder_conv)
            heads_ch = 128
        else:
            self.enc = ResNet50Encoder(ResNet50_Weights.IMAGENET1K_V2)
            self.aspp = ASPP(self.enc.high_ch, 256)
            self.dec  = DecoderDeepLabV3Plus(self.enc.low_ch, 256, 256, extra_decoder_conv=extra_decoder_conv)
            heads_ch = 256

        # Heads
        if iris_head_3x3:
            self.head_iris = nn.Sequential(nn.Conv2d(heads_ch, heads_ch, 3, padding=1, bias=False),
                                           nn.BatchNorm2d(heads_ch), nn.ReLU(True),
                                           nn.Conv2d(heads_ch, 1, 1))
        else:
            self.head_iris  = nn.Conv2d(heads_ch, 1, 1)
        # Pupil with optional refinement
        self.pre_pupil = nn.Identity()
        if pupil_refine_depth and pupil_refine_depth > 0:
            self.pre_pupil = PupilRefine(heads_ch, depth=int(pupil_refine_depth))
        self.head_pupil = nn.Conv2d(heads_ch, 1, 1)

        # auxiliary heads kept (training losses)
        self.head_bnd_i = nn.Conv2d(heads_ch, 1, 1)
        self.head_bnd_p = nn.Conv2d(heads_ch, 1, 1)
        self.head_sdt_i = nn.Conv2d(heads_ch, 1, 1)
        self.head_sdt_p = nn.Conv2d(heads_ch, 1, 1)

        if self.use_ellipse:
            self.ell_conv = nn.Sequential(nn.Conv2d(heads_ch, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
            self.ell_fc_i = nn.Linear(64, 6)
            self.ell_fc_p = nn.Linear(64, 6)

    def forward(self, x):
        f = self.enc(x)
        z = self.dec(self.aspp(f['s32']), f['s4'])
        z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pupil_feat = self.pre_pupil(z)
        out = {
            'iris':  self.head_iris(z),
            'pupil': self.head_pupil(pupil_feat),
            'bnd_i': self.head_bnd_i(z),
            'bnd_p': self.head_bnd_p(z),
            'sdt_i': self.head_sdt_i(z),
            'sdt_p': self.head_sdt_p(z),
        }
        if self.use_ellipse:
            g = F.adaptive_avg_pool2d(self.ell_conv(z), 1).flatten(1)
            out['ell_i'] = self.ell_fc_i(g)
            out['ell_p'] = self.ell_fc_p(g)
        return out

# -----------------------------
# Losses & metrics
# -----------------------------

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6): super().__init__(); self.eps=eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        num = 2.0*(p*targets).sum((2,3)) + self.eps
        den = (p**2).sum((2,3)) + (targets**2).sum((2,3)) + self.eps
        return (1.0 - num/den).mean()

class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4/3, eps=1e-6):
        super().__init__(); self.a=alpha; self.b=beta; self.g=gamma; self.eps=eps
    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        tp = (p*targets).sum((2,3))
        fp = (p*(1-targets)).sum((2,3))
        fn = ((1-p)*targets).sum((2,3))
        t = (tp + self.eps) / (tp + self.a*fp + self.b*fn + self.eps)
        return ((1 - t) ** self.g).mean()

def bce_with_pos_weight(logits, targets, maxw=100.0):
    with torch.no_grad():
        pos = targets.sum() + 1.0
        neg = (1.0 - targets).sum() + 1.0
        w = float(torch.clamp(neg / pos, 1.0, maxw))
    pw = torch.tensor([w], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

def contour_loss(logits, soft_target, dt_weight):
    probs = torch.sigmoid(logits)
    loss = (dt_weight * (1.0 - probs)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, soft_target)
    return 0.5*loss + 0.5*bce

def iou_score(logits, targets, thr=0.5):
    preds = (torch.sigmoid(logits) > thr).float()
    inter = (preds*targets).sum((2,3))
    union = (preds + targets - preds*targets).sum((2,3)) + 1e-6
    return (inter/union).mean().item()

def dice_score_bin(logits, targets, thr=0.5):
    preds = (torch.sigmoid(logits) > thr).float()
    num = 2.0*(preds*targets).sum((2,3)) + 1e-6
    den = (preds**2).sum((2,3)) + (targets**2).sum((2,3)) + 1e-6
    return (num/den).mean().item()

def e1_error(logits, targets):
    probs = torch.sigmoid(logits)
    return (probs - targets).abs().mean().item()

# Priors
def pupil_inside_iris_loss(logits_pupil, logits_iris):
    p_pupil = torch.sigmoid(logits_pupil)
    p_iris  = torch.sigmoid(logits_iris).detach()
    return F.relu(p_pupil - p_iris).mean()

def sobel_edges(x):
    gx = F.conv2d(x, x.new_tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]), padding=1)
    gy = F.conv2d(x, x.new_tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]), padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-6)

def edge_mask_consistency(logits_mask, edge_target):
    e = sobel_edges(logits_mask)
    e = (e / (e.amax((2,3), keepdim=True) + 1e-6))
    dice = (2*(e*edge_target).sum((2,3)) / (e.sum((2,3)) + edge_target.sum((2,3)) + 1e-6)).mean()
    return 1.0 - dice

def ellipse_regression_loss(pred, target, valid):
    if valid.ndim == 1: valid = valid.view(-1,1)
    w = (valid > 0.5).float()
    if w.sum() == 0: return pred.new_tensor(0.0)
    return F.smooth_l1_loss(pred*w, target*w, reduction='sum') / (w.sum()*pred.shape[1])

# Uncertainty weighting (homoscedastic)
class UncertaintyWeights(nn.Module):
    def __init__(self, n_terms):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(n_terms))  # initialized equal
    def forward(self, losses: list):
        # Sum_i (exp(-s_i)*L_i + s_i)
        s = torch.exp(-self.log_sigmas)
        total = 0.0
        terms = []
        for i, L in enumerate(losses):
            Li = s[i]*L + self.log_sigmas[i]
            terms.append(Li)
            total = total + Li
        return total, terms

# -----------------------------
# Train & eval
# -----------------------------

def build_dataset(args, split_file, augment, paper_aug):
    ds = IrisSegDataset(
        root=args.root, split_file=split_file, img_size=args.img_size,
        images_dir=args.images_dir, iris_dir=args.iris_dir, pupil_dir=args.pupil_dir,
        edges_iris_dir=args.edges_iris_dir, edges_pupil_dir=args.edges_pupil_dir,
        augment=augment, paper_aug=paper_aug, aug_limbus=args.aug_limbus,
        iris_sdt_r=args.iris_sdt_r, pupil_sdt_r=args.pupil_sdt_r
    )
    return ds

def dataset_multipliers(batch_datasets, default=1.0):
    # Optionally scale losses by dataset (stabilize across domains)
    mapping = {
        'MICHE':    1.00,
        'UBIRISv1': 1.00,
        'UBIRISv2': 1.00,
        'CUVIRIS':  1.00,
        'UNKNOWN':  1.00,
    }
    # Tweak here if you want slight boosts, e.g., smaller pupils in UBIRISv2
    scales = []
    for name in batch_datasets:
        scales.append(mapping.get(name, default))
    return torch.tensor(scales, dtype=torch.float32)

def lr_lambda_warmup_cosine(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return max(1e-3, (epoch+1)/(warmup_epochs+1e-9))
    # cosine on [warmup_epochs, total_epochs]
    t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return 0.5*(1 + np.cos(np.pi * t))

def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args,
                    uw: UncertaintyWeights = None):
    model.train(); total=0.0
    dice = DiceLoss()
    tv_iris  = TverskyFocalLoss(args.iris_tversky_alpha, args.iris_tversky_beta, args.iris_tversky_gamma)
    tv_pupil = TverskyFocalLoss(args.pupil_tversky_alpha, args.pupil_tversky_beta, args.pupil_tversky_gamma)

    pbar = tqdm(loader, total=len(loader), desc=f"Train {epoch:03d}", dynamic_ncols=True)
    for batch in pbar:
        img = batch['image'].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        iris = batch['iris'].to(device); pupil = batch['pupil'].to(device)
        bnd_i = batch['bnd_i'].to(device); bnd_p = batch['bnd_p'].to(device)
        dtw_i = batch['dtw_i'].to(device); dtw_p = batch['dtw_p'].to(device)
        sdt_i = batch['sdt_i'].to(device); sdt_p = batch['sdt_p'].to(device)

        if args.use_ellipse:
            s_ell_i = torch.stack(batch['s_ell_i']).to(device) if isinstance(batch['s_ell_i'], list) else batch['s_ell_i'].to(device)
            s_ell_p = torch.stack(batch['s_ell_p']).to(device) if isinstance(batch['s_ell_p'], list) else batch['s_ell_p'].to(device)
            v_ell_i = torch.tensor(batch['ell_i_valid'], device=device).view(-1,1).float() if isinstance(batch['ell_i_valid'], list) else batch['ell_i_valid'].to(device).view(-1,1)
            v_ell_p = torch.tensor(batch['ell_p_valid'], device=device).view(-1,1).float() if isinstance(batch['ell_p_valid'], list) else batch['ell_p_valid'].to(device).view(-1,1)

        # per-batch dataset scaling
        ds_scale = dataset_multipliers(batch['dataset']) if args.dataset_loss_scale else None
        if ds_scale is not None:
            ds_scale = ds_scale.to(device).view(-1,1,1,1)

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        ctx = torch.amp.autocast('cuda', enabled=use_amp and device.type=='cuda')
        with ctx:
            out = model(img)
            # Base mask losses
            l_iris_bce  = bce_with_pos_weight(out['iris'], iris)
            l_iris_dice = dice(out['iris'], iris)
            l_pupil_bce  = bce_with_pos_weight(out['pupil'], pupil)
            l_pupil_dice = dice(out['pupil'], pupil)
            if args.iris_tversky:
                l_iris_dice = 0.5*l_iris_dice + 0.5*tv_iris(out['iris'], iris)
            if args.pupil_tversky:
                l_pupil_dice = 0.5*l_pupil_dice + 0.5*tv_pupil(out['pupil'], pupil)

            # Edges (contours) + consistency
            l_bi = contour_loss(out['bnd_i'], bnd_i, dtw_i)
            l_bp = contour_loss(out['bnd_p'], bnd_p, dtw_p)
            l_em_i = edge_mask_consistency(out['iris'],  bnd_i)
            l_em_p = edge_mask_consistency(out['pupil'], bnd_p)

            # Priors
            l_contain = pupil_inside_iris_loss(out['pupil'], out['iris'])

            # SDT
            w_sdt = args.sdt_weight
            l_sdt = w_sdt * (F.smooth_l1_loss(out['sdt_i'], sdt_i) + F.smooth_l1_loss(out['sdt_p'], sdt_p))

            # Optional per-dataset scaling (applied to primary heads)
            if ds_scale is not None:
                l_iris_bce  = (l_iris_bce  * ds_scale.mean())  # scalar terms
                l_pupil_bce = (l_pupil_bce * ds_scale.mean())
                # keep auxiliary unscaled, or scale uniformly by mean

            # Boundary boosts
            l_bi   = args.iris_boundary_boost  * l_bi
            l_em_i = args.iris_boundary_boost  * l_em_i
            l_bp   = args.pupil_boundary_boost * l_bp
            l_em_p = args.pupil_boundary_boost * l_em_p

            # Compose
            L_iris  = l_iris_bce + 0.7*l_iris_dice
            L_pupil = 1.2*l_pupil_bce + 1.0*l_pupil_dice
            L_edges = 0.60*l_bi + 0.70*l_bp
            L_pr    = 0.08*l_contain + 0.05*l_em_i + 0.07*l_em_p
            L_aux   = l_sdt

            # Ellipse (optional)
            if args.use_ellipse and 'ell_i' in out and 'ell_p' in out:
                l_ell = 0.1 * (ellipse_regression_loss(out['ell_i'], s_ell_i, v_ell_i) +
                               ellipse_regression_loss(out['ell_p'], s_ell_p, v_ell_p))
            else:
                l_ell = out['iris'].mean()*0.0  # zero

            if uw is not None:
                # order: iris, pupil, edges, priors, sdt, ell
                parts = [L_iris, L_pupil, L_edges, L_pr, L_aux, l_ell]
                loss_u, _ = uw(parts)
                loss = loss_u
                # Iris floor multiplier to keep iris strong even if sigma grows
                loss = loss + (args.iris_floor - 1.0) * L_iris
            else:
                loss = L_iris + L_pupil + L_edges + L_pr + L_aux + l_ell

        if use_amp:
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, device, writer=None, epoch=0, log_images_every=0):
    model.eval()
    iou_i=dice_i=e1_i=iou_p=dice_p=e1_p=0.0; count=0
    for i, batch in enumerate(loader):
        img = batch['image'].to(device, non_blocking=True).to(memory_format=torch.channels_last)
        iris = batch['iris'].to(device); pupil = batch['pupil'].to(device)
        out = model(img)
        iou_i  += iou_score(out['iris'], iris);  dice_i  += dice_score_bin(out['iris'], iris);  e1_i += e1_error(out['iris'], iris)
        iou_p  += iou_score(out['pupil'], pupil); dice_p += dice_score_bin(out['pupil'], pupil); e1_p += e1_error(out['pupil'], pupil)
        count += 1

        if writer is not None and log_images_every>0 and epoch % log_images_every == 0 and i==0:
            img_vis = img[:4].detach().cpu().clone()
            mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            img_vis = (img_vis * std + mean).clamp(0,1)
            iris_gt = batch['iris'][:4].repeat(1,3,1,1).cpu()
            iris_pr = (torch.sigmoid(out['iris'][:4]).cpu()>0.5).float().repeat(1,3,1,1)
            pupil_gt = batch['pupil'][:4].repeat(1,3,1,1).cpu()
            pupil_pr = (torch.sigmoid(out['pupil'][:4]).cpu()>0.5).float().repeat(1,3,1,1)
            grid = torch.cat([img_vis, iris_gt, iris_pr, pupil_gt, pupil_pr], dim=0)
            grid = vutils.make_grid(grid, nrow=4, pad_value=1.0)
            writer.add_image('val/preview', grid, global_step=epoch)

    return {
        'iou_iris': iou_i/count, 'dice_iris': dice_i/count, 'e1_iris': e1_i/count,
        'iou_pupil': iou_p/count, 'dice_pupil': dice_p/count, 'e1_pupil': e1_p/count
    }

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def _pad_to_min_hw(arr, CH, CW):
    Hs, Ws = arr.shape[:2]
    dh = max(0, CH - Hs)
    dw = max(0, CW - Ws)
    if dh == 0 and dw == 0:
        return arr
    top = dh // 2; bottom = dh - top
    left = dw // 2; right = dw - left
    if arr.ndim == 3:
        pad_width = ((top, bottom), (left, right), (0, 0))
    else:
        pad_width = ((top, bottom), (left, right))
    return np.pad(arr, pad_width, mode='constant', constant_values=0)

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description='Iris segmentation with multi-task + iris/pupil upgrades')

# --- Default arguments injected (can be overridden on CLI) ---
ap.set_defaults(
    root=r".\data",
    train_split=r".\data\train.txt",
    val_split=r".\data\val.txt",
    img_size=-1,
    batch_size=8,
    epochs=120,
    lr=3e-4,
    seed=42,
    workers=8,
    outdir=r"runs\train",
    paper_aug=True,
    amp=True,
    freeze_enc_bn=True,
    use_ellipse=True,
    backbone="mobilenetv3",
    aug_limbus=True,
    extra_decoder_conv=True,
    pupil_refine_depth=1,
    iris_tversky=True,
    pupil_tversky=True,
    iris_boundary_boost=1.2,
    pupil_boundary_boost=1.2,
    iris_sdt_r=20,
    pupil_sdt_r=16,
    warmup_epochs=5,
    grad_clip=1.0,
    uncertainty_weighting=True,
    iris_floor=1.1,
    dataset_loss_scale=True
)
    # data
    ap.add_argument('--root', type=str, required=False)
    ap.add_argument('--train_split', type=str, required=False)
    ap.add_argument('--val_split', type=str, required=False)
    ap.add_argument('--images_dir', type=str, default='images')
    ap.add_argument('--iris_dir', type=str, default='labels_iris')
    ap.add_argument('--pupil_dir', type=str, default='labels_pupil')
    ap.add_argument('--edges_iris_dir', type=str, default='edges_iris')
    ap.add_argument('--edges_pupil_dir', type=str, default='edges_pupil')
    ap.add_argument('--img_size', type=int, default=-1, help='-1=native; else height (keeps aspect)')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--outdir', type=str, default='checkpoints_iris_mobilenet')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--log_images_every', type=int, default=10)
    ap.add_argument('--backbone', type=str, default='mobilenetv3', choices=['resnet50','mobilenetv3'])
    ap.add_argument('--use_ellipse', action='store_true', help='enable ellipse regression prior')
    ap.add_argument('--freeze_enc_bn', action='store_true', help='freeze encoder BN (helpful for small batches)')
    ap.add_argument('--paper_aug', action='store_true')
    ap.add_argument('--aug_limbus', action='store_true')
    ap.add_argument('--extra_decoder_conv', action='store_true')
    ap.add_argument('--iris_head_3x3', action='store_true')
    ap.add_argument('--pupil_refine_depth', type=int, default=0)

    # new loss knobs
    ap.add_argument('--iris_tversky', action='store_true')
    ap.add_argument('--iris_tversky_alpha', type=float, default=0.6)
    ap.add_argument('--iris_tversky_beta', type=float, default=0.4)
    ap.add_argument('--iris_tversky_gamma', type=float, default=1.0)

    ap.add_argument('--pupil_tversky', action='store_true')
    ap.add_argument('--pupil_tversky_alpha', type=float, default=0.7)
    ap.add_argument('--pupil_tversky_beta', type=float, default=0.3)
    ap.add_argument('--pupil_tversky_gamma', type=float, default=4/3)

    ap.add_argument('--iris_boundary_boost', type=float, default=1.0)
    ap.add_argument('--pupil_boundary_boost', type=float, default=1.0)
    ap.add_argument('--sdt_weight', type=float, default=0.2)
    ap.add_argument('--iris_sdt_r', type=int, default=12)
    ap.add_argument('--pupil_sdt_r', type=int, default=12)

    # auto weighting
    ap.add_argument('--uncertainty_weighting', action='store_true')
    ap.add_argument('--iris_floor', type=float, default=1.0)

    # sched / opt niceties
    ap.add_argument('--warmup_epochs', type=int, default=0)
    ap.add_argument('--grad_clip', type=float, default=0.0)

    # dataset balancing
    ap.add_argument('--dataset_loss_scale', action='store_true')

    # quick inference parity (unchanged)
    ap.add_argument('--predict_image', type=str, default=None)
    ap.add_argument('--checkpoint', type=str, default=None)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inference only
    if args.predict_image and args.checkpoint:
        model = IrisNetDeepLab(backbone=args.backbone, use_ellipse=args.use_ellipse,
                               extra_decoder_conv=args.extra_decoder_conv,
                               iris_head_3x3=args.iris_head_3x3,
                               pupil_refine_depth=args.pupil_refine_depth).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        model.eval()
        img = imread_rgb(args.predict_image, size=args.img_size)
        from math import exp
        def predict_once(img_rgb_np):
            x = torch.from_numpy(img_rgb_np.transpose(2,0,1)).unsqueeze(0).float()
            mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            x = ((x - mean)/std).to(device)
            out = model(x)
            iris  = (torch.sigmoid(out['iris'])  > 0.5).float().cpu().numpy()[0, 0]
            pupil = (torch.sigmoid(out['pupil']) > 0.5).float().cpu().numpy()[0, 0]
            return {'iris': iris, 'pupil': pupil}
        pred = predict_once(img)
        base = (img*255).astype(np.uint8); base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        vis = base.copy()
        for mm, col in [(pred['pupil'],(0,255,255)), (pred['iris'],(0,255,0))]:
            contours,_ = cv2.findContours((mm>0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, col, 2)
        out_path = str(Path(args.outdir) / 'predict_vis.png')
        os.makedirs(args.outdir, exist_ok=True); cv2.imwrite(out_path, vis)
        print(f"Saved visualization to {out_path}")
        return

    # Datasets / loaders
    train_ds = build_dataset(args, args.train_split, augment=True,  paper_aug=args.paper_aug)
    val_ds   = build_dataset(args, args.val_split,   augment=False, paper_aug=False)

    nw = args.workers
    if args.img_size < 0:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True,
                                  persistent_workers=(nw > 0), prefetch_factor=4,
                                  collate_fn=collate_pad)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=nw, pin_memory=True,
                                  persistent_workers=(nw > 0), prefetch_factor=4,
                                  collate_fn=collate_pad)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=nw, pin_memory=True, drop_last=True,
                                  persistent_workers=(nw > 0), prefetch_factor=4)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                  num_workers=nw, pin_memory=True,
                                  persistent_workers=(nw > 0), prefetch_factor=4)

    writer = SummaryWriter(log_dir=args.outdir)

    model = IrisNetDeepLab(backbone=args.backbone, use_ellipse=args.use_ellipse,
                           extra_decoder_conv=args.extra_decoder_conv,
                           iris_head_3x3=args.iris_head_3x3,
                           pupil_refine_depth=args.pupil_refine_depth).to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.freeze_enc_bn:
        for m in model.enc.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval(); m.weight.requires_grad_(False); m.bias.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type=='cuda')

    # LR schedule: warmup + cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: lr_lambda_warmup_cosine(ep, args.warmup_epochs, args.epochs)
    )

    uw = UncertaintyWeights(6).to(device) if args.uncertainty_weighting else None
    if uw is not None:
        optimizer.add_param_group({'params': uw.parameters(), 'lr': args.lr})

    best = -1.0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args, uw=uw)
        val = evaluate(model, val_loader, device, writer=writer, epoch=epoch, log_images_every=args.log_images_every)

        # Composite score emphasizing iris+pupil; small penalty on e1
        score = 0.5*val['dice_iris'] + 0.5*val['dice_pupil'] - 0.05*(val['e1_iris'] + val['e1_pupil'])

        print(f"Epoch {epoch:03d} | loss {tr_loss:.4f} | "
              f"IoU(iris) {val['iou_iris']:.4f} Dice(iris) {val['dice_iris']:.4f} E1(iris) {val['e1_iris']:.4f} | "
              f"IoU(pupil) {val['iou_pupil']:.4f} Dice(pupil) {val['dice_pupil']:.4f} E1(pupil) {val['e1_pupil']:.4f}",
              flush=True)

        writer.add_scalar('Loss/train', tr_loss, epoch)
        for k,v in val.items():
            writer.add_scalar(f'Val/{k}', v, epoch)

        if score > best:
            best = score
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'val': val},
                            os.path.join(args.outdir, f'best_{args.backbone}.pt'))
            print(f"  -> Saved best to {os.path.join(args.outdir, f'best_{args.backbone}.pt')}")

        if epoch % 10 == 0:
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'val': val},
                            os.path.join(args.outdir, f'epoch_{epoch}.pt'))

        scheduler.step()

    writer.close()

if __name__ == '__main__':
    main()
