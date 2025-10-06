#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Test-time iris segmentation (native resolution) with diagnostics.

What this script does:
- Loads a trained checkpoint (MobileNetV3 or ResNet50 backbone) with optional
  architecture knobs: --extra_decoder_conv, --pupil_refine_depth, --iris_head_3x3, --use_ellipse
- Reads IDs from --test_split
- Saves predicted masks (+ optional overlays)
- If GT masks exist, computes IoU/Dice per image and macro means
- Reports pre-PP (raw threshold) vs post-PP (ellipse refit + containment) metrics
- Reports pupil/iris positive rates, ellipse refit usage/fallbacks
- Warns on checkpoint key mismatches to catch arch flag mismatches

Post-processing containment:
  --containment_mode off|soft|soft_erode|hard
     off        : no pupil-in-iris gating
     soft       : pupil := pupil ∧ iris (intersect)
     soft_erode : erode pupil (k iters) then intersect with iris
     hard       : always intersect (same as soft; kept for clarity)
  --inside_thresh    : if fraction_inside < inside_thresh we apply containment (for soft/soft_erode)
  --keep_area_frac   : if intersect area too small (< keep_area_frac * pupil_area), keep original pupil

Run (PowerShell example):
  python test_diagnostics.py `
    --root "D:\Smartphone_Data\analysis\data" `
    --test_split "D:\Smartphone_Data\analysis\data\splits\ubv1.txt" `
    --checkpoint "runs\irisparsenet_aug_all\best_mobilenetv3.pt" `
    --backbone mobilenetv3 `
    --use_ellipse --extra_decoder_conv --pupil_refine_depth 1 `
    --export_dir "runs\irisparsenet_aug_all\ubv1_diag" `
    --thr_iris 0.5 --thr_pupil 0.5 `
    --containment_mode soft --inside_thresh 0.85 --keep_area_frac 0.90 `
    --save_overlay --amp
"""

import os
from pathlib import Path
import argparse
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights

# -------- speed niceties --------
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
cv2.setNumThreads(2)

# -----------------------------
# I/O helpers
# -----------------------------
def imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    if img.ndim == 2: img = np.stack([img, img, img], -1)
    if img.shape[2] == 4: img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)
    return img

def _find_file(folder: Path, stem: str, exts):
    for e in exts:
        p = folder / f"{stem}{e}"
        if p.exists(): return p
    for e in exts:
        cs = list(folder.glob(f"{stem}*{e}"))
        if cs: return cs[0]
    return None

def _save_mask_png(mask01, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.clip(mask01, 0, 1) * 255).astype(np.uint8))

def _bin_iou(pred01, gt01, eps=1e-6):
    inter = (pred01 & gt01).sum()
    union = (pred01 | gt01).sum() + eps
    return float(inter / union)

def _bin_dice(pred01, gt01, eps=1e-6):
    inter = (pred01 & gt01).sum()
    return float((2.0 * inter) / (pred01.sum() + gt01.sum() + eps))

# -----------------------------
# Post-processing
# -----------------------------
def largest_cc(m):
    n, cc = cv2.connectedComponents(m.astype(np.uint8))
    if n <= 2: return m
    areas = [(cc==i).sum() for i in range(1,n)]
    k = 1 + int(np.argmax(areas))
    return (cc==k).astype(np.uint8)

def refit_ellipse(binary_mask, min_pts=20):
    cnts,_ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: return None, None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < min_pts: return None, None
    ellipse = cv2.fitEllipse(cnt)  # (cx,cy),(ma,mi),angle
    mask = np.zeros_like(binary_mask, np.uint8)
    cv2.ellipse(mask, ellipse, 1, -1)
    return ellipse, mask

def rasterize_ellipse_from_params(params, H, W):
    cx = params[0]*W; cy = params[1]*H
    rx = max(params[2]*W, 1.0); ry = max(params[3]*H, 1.0)
    ang = np.degrees(np.arctan2(params[4], params[5]))
    mask = np.zeros((H,W), np.uint8)
    try:
        cv2.ellipse(mask, ((cx,cy),(2*rx,2*ry), ang), 1, -1)
    except:
        pass
    return mask

def apply_containment(pupil_mask, iris_mask, mode='soft', erode_iters=1,
                      inside_thresh=0.70, keep_area_frac=0.70):
    """Return adjusted pupil given iris, plus diagnostics dict."""
    d = {'applied': False, 'inside_ratio': 1.0, 'p_area': int(pupil_mask.sum()), 'p_area_new': int(pupil_mask.sum())}
    p = pupil_mask.astype(np.uint8)
    i = iris_mask.astype(np.uint8)
    if p.sum() == 0:
        d['inside_ratio'] = 1.0
        return p, d
    inside = (p & i).sum()
    frac_inside = inside / float(p.sum())
    d['inside_ratio'] = float(frac_inside)
    if mode == 'off':
        return p, d
    # decide if we need to gate
    need = (frac_inside < inside_thresh) or (mode == 'hard')
    if not need and mode in ('soft','soft_erode'):
        return p, d
    p2 = p.copy()
    if mode == 'soft_erode':
        if erode_iters > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            p2 = cv2.erode(p2, k, iterations=int(erode_iters))
    p2 = p2 & i
    # keep if not catastrophically small
    if p2.sum() >= keep_area_frac * max(1, p.sum()):
        d['applied'] = True
        d['p_area_new'] = int(p2.sum())
        return p2, d
    else:
        # fall back to original
        d['applied'] = True
        d['p_area_new'] = int(p.sum())
        return p, d

# -----------------------------
# Model (must mirror training flags)
# -----------------------------
class ResNet50Encoder(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        super().__init__()
        bb = resnet50(weights=weights)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.low_ch, self.high_ch = 256, 2048
    def forward(self, x):
        x = self.stem(x); s4 = self.layer1(x)
        s8 = self.layer2(s4); s16 = self.layer3(s8); s32 = self.layer4(s16)
        return {'s4': s4, 's8': s8, 's16': s16, 's32': s32}

class MNetV3Encoder(nn.Module):
    def __init__(self, weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1):
        super().__init__()
        m = mobilenet_v3_large(weights=weights).features
        self.s4  = nn.Sequential(*m[:6])
        self.s8  = nn.Sequential(*m[6:10])
        self.s16 = nn.Sequential(*m[10:13])
        self.s32 = nn.Sequential(*m[13:])
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
        yp = self.pool(x); yp = F.interpolate(yp, (h,w), mode='bilinear', align_corners=False)
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

        if iris_head_3x3:
            self.head_iris = nn.Sequential(nn.Conv2d(heads_ch, heads_ch, 3, padding=1, bias=False),
                                           nn.BatchNorm2d(heads_ch), nn.ReLU(True),
                                           nn.Conv2d(heads_ch, 1, 1))
        else:
            self.head_iris  = nn.Conv2d(heads_ch, 1, 1)

        self.pre_pupil = nn.Identity()
        if pupil_refine_depth and pupil_refine_depth > 0:
            self.pre_pupil = PupilRefine(heads_ch, depth=int(pupil_refine_depth))
        self.head_pupil = nn.Conv2d(heads_ch, 1, 1)

        # aux heads (present in ckpt)
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
        }
        if hasattr(self, 'head_bnd_i'):
            out['bnd_i'] = self.head_bnd_i(z)
            out['bnd_p'] = self.head_bnd_p(z)
            out['sdt_i'] = self.head_sdt_i(z)
            out['sdt_p'] = self.head_sdt_p(z)
        if self.use_ellipse:
            g = F.adaptive_avg_pool2d(self.ell_conv(z), 1).flatten(1)
            out['ell_i'] = self.ell_fc_i(g)
            out['ell_p'] = self.ell_fc_p(g)
        return out

# -----------------------------
# Inference utils
# -----------------------------
@torch.no_grad()
def predict(model, img_rgb_np, device='cpu', thr_iris=0.5, thr_pupil=0.5, do_post=True,
            use_ellipse=False, amp=False, min_pts=20, containment_mode='soft',
            erode_iters=1, inside_thresh=0.70, keep_area_frac=0.70):
    x = torch.from_numpy(img_rgb_np.transpose(2,0,1)).unsqueeze(0).float()
    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
    x = ((x - mean)/std).to(device).to(memory_format=torch.channels_last)

    ctx = torch.autocast(device_type='cuda', dtype=torch.float16, enabled=amp and device.type=='cuda')
    with ctx:
        out = model(x)

    # use torch.sigmoid for numerical stability then numpy
    iris_prob  = torch.sigmoid(out['iris']).float().cpu().numpy()[0,0]
    pupil_prob = torch.sigmoid(out['pupil']).float().cpu().numpy()[0,0]
    iris_bin_pre  = (iris_prob  > thr_iris ).astype(np.uint8)
    pupil_bin_pre = (pupil_prob > thr_pupil).astype(np.uint8)

    diag = {'iris': {}, 'pupil': {}}
    diag['iris']['pos_rate_pre']  = float(iris_bin_pre.mean())
    diag['pupil']['pos_rate_pre'] = float(pupil_bin_pre.mean())

    if not do_post:
        return {'iris': iris_bin_pre.astype(np.float32), 'pupil': pupil_bin_pre.astype(np.float32)}, diag

    # ellipse refit with fallback to predicted ellipse params
    H, W = iris_prob.shape
    used = {'iris_refit': False, 'iris_fallback': False, 'pupil_refit': False, 'pupil_fallback': False}

    _, iris_ell = refit_ellipse(iris_bin_pre, min_pts=min_pts)
    if iris_ell is None and use_ellipse and 'ell_i' in out:
        iris_ell = rasterize_ellipse_from_params(out['ell_i'].cpu().numpy()[0], H, W)
        used['iris_fallback'] = True
    elif iris_ell is not None:
        used['iris_refit'] = True

    _, pup_ell  = refit_ellipse(pupil_bin_pre, min_pts=min_pts)
    if pup_ell is None and use_ellipse and 'ell_p' in out:
        pup_ell = rasterize_ellipse_from_params(out['ell_p'].cpu().numpy()[0], H, W)
        used['pupil_fallback'] = True
    elif pup_ell is not None:
        used['pupil_refit'] = True

    iris_post = iris_bin_pre.copy()
    pupil_post = pupil_bin_pre.copy()
    if iris_ell is not None:
        iris_post = iris_post & (iris_ell>0)
    if pup_ell is not None:
        pupil_post = pupil_post & (pup_ell>0)

    # containment
    pupil_adj, gate_d = apply_containment(
        pupil_post, iris_post,
        mode=containment_mode, erode_iters=erode_iters,
        inside_thresh=inside_thresh, keep_area_frac=keep_area_frac
    )

    diag['iris'].update(used)
    diag['pupil'].update(used)
    diag['pupil'].update(gate_d)

    iris_post = largest_cc(iris_post)
    pupil_post = largest_cc(pupil_adj)

    return {'iris': iris_post.astype(np.float32), 'pupil': pupil_post.astype(np.float32)}, diag

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Test iris segmentation (diagnostics)")

# --- Default arguments injected (can be overridden on CLI) ---
ap.set_defaults(
    root=r".\data",
    test_split=r".\data\test.txt",
    checkpoint=r"runs\train\best_mobilenetv3.pt",
    backbone="mobilenetv3",
    use_ellipse=True,
    extra_decoder_conv=True,
    pupil_refine_depth=1,
    amp=True,
    save_overlay=True,
    export_dir=r"runs\irisparsenet_aug_all\miche_diag",
    thr_iris=0.5,
    thr_pupil=0.5,
    containment_mode="soft",
    inside_thresh=0.85,
    keep_area_frac=0.90
)
    ap.add_argument('--root', type=str, required=False)
    ap.add_argument('--test_split', type=str, required=False)
    ap.add_argument('--images_dir', type=str, default='images')
    ap.add_argument('--iris_dir', type=str, default='labels_iris')   # optional GT
    ap.add_argument('--pupil_dir', type=str, default='labels_pupil') # optional GT
    ap.add_argument('--checkpoint', type=str, required=False)
    ap.add_argument('--backbone', type=str, default='mobilenetv3', choices=['mobilenetv3','resnet50'])

    # arch flags (MUST match training)
    ap.add_argument('--use_ellipse', action='store_true')
    ap.add_argument('--extra_decoder_conv', action='store_true')
    ap.add_argument('--iris_head_3x3', action='store_true')
    ap.add_argument('--pupil_refine_depth', type=int, default=0)

    # io / viz
    ap.add_argument('--export_dir', type=str, default='preds_test')
    ap.add_argument('--thr_iris', type=float, default=0.5)
    ap.add_argument('--thr_pupil', type=float, default=0.5)
    ap.add_argument('--save_overlay', action='store_true')

    # PP controls
    ap.add_argument('--no_post', action='store_true', help='disable ellipse refit & containment PP')
    ap.add_argument('--min_pts', type=int, default=20)
    ap.add_argument('--containment_mode', type=str, default='soft', choices=['off','soft','soft_erode','hard'])
    ap.add_argument('--erode_iters', type=int, default=1)
    ap.add_argument('--inside_thresh', type=float, default=0.70)
    ap.add_argument('--keep_area_frac', type=float, default=0.70)

    ap.add_argument('--amp', action='store_true', help='mixed-precision inference')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = IrisNetDeepLab(backbone=args.backbone, use_ellipse=args.use_ellipse,
                           extra_decoder_conv=args.extra_decoder_conv,
                           iris_head_3x3=args.iris_head_3x3,
                           pupil_refine_depth=args.pupil_refine_depth).to(device)
    model = model.to(memory_format=torch.channels_last)

    # safer load (PyTorch versions < 2.4 won't have weights_only)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    state = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] load_state_dict mismatches: "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print("  Missing (first 10):", missing[:10])
        if unexpected:
            print("  Unexpected (first 10):", unexpected[:10])
        print("  -> If many head/decoder keys are missing, check your test flags match training flags.")

    model.eval()

    root = Path(args.root)
    images_dir = root / args.images_dir
    iris_dir   = root / args.iris_dir
    pupil_dir  = root / args.pupil_dir
    out_dir    = Path(args.export_dir); out_dir.mkdir(parents=True, exist_ok=True)

    exts_img = ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']
    exts_msk = ['.png','.bmp','.jpg','.jpeg','.tif','.tiff']

    with open(args.test_split, 'r') as f:
        ids = [ln.strip() for ln in f if ln.strip()]

    have_gt_any = False
    rows = []
    # diagnostics accumulators
    iris_pos_rates = []
    pupil_pos_rates = []
    used_counts = {'iris_refit':0, 'iris_fallback':0, 'pupil_refit':0, 'pupil_fallback':0}

    pbar = tqdm(ids, desc="Testing", dynamic_ncols=True)
    for sid in pbar:
        img_path = _find_file(images_dir, sid, exts_img)
        if img_path is None:
            print(f"[WARN] Missing image for ID '{sid}'"); continue

        img = imread_rgb(img_path)
        pred_prepost, diag = predict(
            model, img, device=device,
            thr_iris=args.thr_iris, thr_pupil=args.thr_pupil,
            do_post=not args.no_post, use_ellipse=args.use_ellipse, amp=args.amp,
            min_pts=args.min_pts, containment_mode=args.containment_mode,
            erode_iters=args.erode_iters, inside_thresh=args.inside_thresh,
            keep_area_frac=args.keep_area_frac
        )

        iris_post = pred_prepost['iris']; pupil_post = pred_prepost['pupil']
        # We also want raw pre-PP for metrics; recompute quickly with do_post=False
        pred_pre, _ = predict(
            model, img, device=device,
            thr_iris=args.thr_iris, thr_pupil=args.thr_pupil,
            do_post=False, use_ellipse=args.use_ellipse, amp=args.amp
        )
        iris_pre = pred_pre['iris']; pupil_pre = pred_pre['pupil']

        _save_mask_png(iris_post,  out_dir / f"{sid}_iris.png")
        _save_mask_png(pupil_post, out_dir / f"{sid}_pupil.png")

        if args.save_overlay:
            base = (img*255).astype(np.uint8)
            vis  = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
            for mm, col in [(pupil_post,(0,255,255)), (iris_post,(0,255,0))]:
                cnts,_ = cv2.findContours((mm>0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cnts, -1, col, 2)
            cv2.imwrite(str(out_dir / f"{sid}_overlay.png"), vis)

        iris_gt_path  = _find_file(iris_dir,  sid, exts_msk)
        pupil_gt_path = _find_file(pupil_dir, sid, exts_msk)

        # accumulate diagnostics
        iris_pos_rates.append(diag['iris']['pos_rate_pre'])
        pupil_pos_rates.append(diag['pupil']['pos_rate_pre'])
        for k in ['iris_refit','iris_fallback','pupil_refit','pupil_fallback']:
            if diag['iris'].get(k, False) or diag['pupil'].get(k, False):
                used_counts[k] += 1

        if iris_gt_path is not None and pupil_gt_path is not None:
            have_gt_any = True
            iris_gt  = (cv2.imread(str(iris_gt_path), 0)  > 127).astype(np.uint8)
            pupil_gt = (cv2.imread(str(pupil_gt_path), 0) > 127).astype(np.uint8)
            if iris_gt.shape != iris_post.shape:
                iris_gt  = cv2.resize(iris_gt,  (iris_post.shape[1], iris_post.shape[0]), interpolation=cv2.INTER_NEAREST)
            if pupil_gt.shape != pupil_post.shape:
                pupil_gt = cv2.resize(pupil_gt, (pupil_post.shape[1], pupil_post.shape[0]), interpolation=cv2.INTER_NEAREST)

            rows.append({
                'id': sid,
                # pre-PP
                'iou_iris_pre':  _bin_iou((iris_pre>0.5),  (iris_gt>0)),
                'dice_iris_pre': _bin_dice((iris_pre>0.5), (iris_gt>0)),
                'iou_pupil_pre':  _bin_iou((pupil_pre>0.5),  (pupil_gt>0)),
                'dice_pupil_pre': _bin_dice((pupil_pre>0.5), (pupil_gt>0)),
                # post-PP
                'iou_iris_post':  _bin_iou((iris_post>0.5),  (iris_gt>0)),
                'dice_iris_post': _bin_dice((iris_post>0.5), (iris_gt>0)),
                'iou_pupil_post':  _bin_iou((pupil_post>0.5),  (pupil_gt>0)),
                'dice_pupil_post': _bin_dice((pupil_post>0.5), (pupil_gt>0)),
                # containment diag
                'pupil_inside_ratio': diag['pupil'].get('inside_ratio', 1.0),
            })

    if have_gt_any and rows:
        csv_path = out_dir / "metrics_diag.csv"
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

        # macro means (pre)
        mi_pre  = sum(r['iou_iris_pre']  for r in rows)/len(rows)
        mdi_pre = sum(r['dice_iris_pre'] for r in rows)/len(rows)
        mp_pre  = sum(r['iou_pupil_pre'] for r in rows)/len(rows)
        mdp_pre = sum(r['dice_pupil_pre']for r in rows)/len(rows)
        print(f"[TEST] === PRE-PP (raw thresholds) ===")
        print(f"mean IoU iris={mi_pre:.4f}, dice iris={mdi_pre:.4f} | mean IoU pupil={mp_pre:.4f}, dice pupil={mdp_pre:.4f}")

        # macro means (post)
        mi  = sum(r['iou_iris_post']  for r in rows)/len(rows)
        mdi = sum(r['dice_iris_post'] for r in rows)/len(rows)
        mp  = sum(r['iou_pupil_post'] for r in rows)/len(rows)
        mdp = sum(r['dice_pupil_post']for r in rows)/len(rows)
        print(f"[TEST] === POST-PP (after ellipse & containment) ===")
        print(f"mean IoU iris={mi:.4f}, dice iris={mdi:.4f} | mean IoU pupil={mp:.4f}, dice pupil={mdp:.4f}")

        # rates & fallbacks
        print(f"[TEST] === RATES & FALLBACKS ===")
        print(f"pupil_pos_rate (mean pre-PP): {np.mean(pupil_pos_rates):.4f}")
        print(f"iris_pos_rate  (mean pre-PP): {np.mean(iris_pos_rates):.4f}")
        n = len(ids) 
        print(f"iris refit used     : {used_counts['iris_refit']} / {n}")
        print(f"iris fallback used  : {used_counts['iris_fallback']} / {n}")
        print(f"iris none available : {n - used_counts['iris_refit'] - used_counts['iris_fallback']} / {n}")
        print(f"pupil refit used    : {used_counts['pupil_refit']} / {n}")
        print(f"pupil fallback used : {used_counts['pupil_fallback']} / {n}")
        print(f"pupil none available: {n - used_counts['pupil_refit'] - used_counts['pupil_fallback']} / {n}")
        print(f"[TEST] Wrote per-image metrics to: {csv_path}")
    else:
        print("[TEST] No GT masks found — saved predictions only.")

    print(f"[TEST] Settings: thr_iris={args.thr_iris}, thr_pupil={args.thr_pupil}, "
          f"containment_mode={args.containment_mode}, erode_iters={args.erode_iters}, "
          f"min_pts={args.min_pts}, use_ellipse={args.use_ellipse}, amp={args.amp}, "
          f"inside_thresh={args.inside_thresh}, keep_area_frac={args.keep_area_frac}")

if __name__ == "__main__":
    main()
