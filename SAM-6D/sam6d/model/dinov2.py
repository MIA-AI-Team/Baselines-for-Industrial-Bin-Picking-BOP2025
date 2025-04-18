import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
import logging
import numpy as np
from utils.bbox_utils import CropResizePad, CustomResizeLongestSide
from torchvision.utils import make_grid, save_image
from model.utils import BatchedData
from copy import deepcopy
import os.path as osp

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

descriptor_map = {
    "dinov2_vits14": "vit_small",
    "dinov2_vitb14": "vit_base",
    "dinov2_vitl14": "vit_large",
    "dinov2_vitg14": "vit_giant2",
}

from enum import Enum
from typing import Union


class Weights(Enum):
    LVD142M = "LVD142M"

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Utils
# ─────────────────────────────────────────────────────────────────────────────
from contextlib import contextmanager
import torch
from torch import Tensor
from torch.amp import autocast


@contextmanager
def maybe_autocast(enabled: bool = True):
    """
    Context‑manager that activates torch.autocast *only* when CUDA is available
    and `enabled` is True.  Saves ~40 % VRAM in forward passes that are
    dominated by matrix‑mult / convolution (DINOv2 falls in that bucket).
    """
    if enabled and torch.cuda.is_available():
        with autocast(device_type="cuda"):
            yield
    else:
        yield


def _chunk_indices(n: int, chunk: int):
    """Generator that yields (start, end) tuples to slice `n` items in steps of `chunk`."""
    for s in range(0, n, chunk):
        yield s, min(s + chunk, n)


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from . import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model




class CustomDINOv2(pl.LightningModule):
    def __init__(
        self,
        model_name,
        token_name,
        image_size,
        chunk_size,
        descriptor_width_size,
        checkpoint_dir,
        patch_size=14,
        validpatch_thresh=0.5,
        use_amp: bool = True
    ):
        super().__init__()
        self.model_name = model_name
        self.model = _make_dinov2_model(arch_name=descriptor_map[model_name], pretrained=False)
        self.model.load_state_dict(torch.load(osp.join(checkpoint_dir, f"{model_name}_pretrain.pth")))
        self.validpatch_thresh = validpatch_thresh
        self.token_name = token_name
        self.chunk_size = chunk_size
        self.patch_size = patch_size
        self.proposal_size = image_size
        self.descriptor_width_size = descriptor_width_size
        logging.info(f"Init CustomDINOv2 done!")
        self.rgb_normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.458971, 0.458971, 0.458971], std=[0.225609, 0.225609, 0.225609]),
            ]
        )
        # use for global feature
        self.rgb_proposal_processor = CropResizePad(self.proposal_size)
        self.rgb_resize = CustomResizeLongestSide(
            descriptor_width_size, dividable_size=self.patch_size
        )
        self.patch_kernel = torch.nn.AvgPool2d(kernel_size=self.patch_size, stride=self.patch_size)


        self._use_amp = bool(use_amp)

        logging.info(
            f"Init CustomDINOv2 with full size={descriptor_width_size} and proposal size={self.proposal_size} done!"
        )

    @torch.no_grad()
    def process_rgb_proposals(self, image_np: np.ndarray, masks: Tensor, boxes: Tensor) -> Tensor:
        """
        Previous implementation duplicated the *whole* (C,H,W) tensor on the GPU
        `num_proposals` times.  That explodes as soon as there are many masks.

        New strategy
        ------------
        • Keep every heavy op on the **CPU** until the very last moment.
        • Stream proposals to the GPU in `self.chunk_size` mini‑batches.
        • Optionally process in FP16 (controlled by self.use_amp).
        """
        device = masks.device               # "remember" where the caller lives
        cpu_rgb = self.rgb_normalize(image_np).float().cpu()        # (3,H,W) on *CPU*
        cpu_masks = masks.cpu()                                   # (N, H, W) on CPU
        outputs = []

        for start, end in _chunk_indices(len(cpu_masks), self.chunk_size):
            # Slice the current mini‑batch  -----------------------------------
            m = cpu_masks[start:end]                                # (B,H,W)
            rgbs = cpu_rgb.unsqueeze(0).expand(len(m), -1, -1, -1)  # (B,3,H,W)
            masked = rgbs * m.unsqueeze(1)                          # (B,3,H,W)

            # Crop / resize / pad – *still* on CPU ----------------------------
            out = self.rgb_proposal_processor(masked, boxes[start:end])
            outputs.append(out)                                     # (B,3,S,S)

        # Concatenate, but keep on CPU; we will push to GPU per‑chunk later
        return torch.cat(outputs, 0)                                # (N,3,S,S)

    # ≡≡≡ 2‑b  Mask proposals  ≡≡≡
    @torch.no_grad()
    def process_masks_proposals(self, masks: Tensor, boxes: Tensor) -> Tensor:
        cpu_masks = masks.cpu().unsqueeze(1)                        # (N,1,H,W)
        outs = []
        for start, end in _chunk_indices(len(cpu_masks), self.chunk_size):
            outs.append(
                self.rgb_proposal_processor(cpu_masks[start:end], boxes[start:end])
            )
        return torch.cat(outs, 0).squeeze(1)                        # (N,S,S)

    # ────────────────────────────────────────────────────────────────────────
    # 3.  Chunked inference + mixed precision
    # ────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward_by_chunk(self, processed_rgbs: Tensor) -> Tensor:
        """Same public behaviour – but the *internals* now push one mini‑batch
        at a time to the GPU and immediately free it afterwards."""
        feats = []
        for start, end in _chunk_indices(len(processed_rgbs), self.chunk_size):
            with maybe_autocast(self._use_amp):
                x = processed_rgbs[start:end].to(next(self.model.parameters()).device, non_blocking=True)
                feats.append(self.model(x))            # returns (B,D)
            del x                                      # free ASAP
            torch.cuda.empty_cache()
        return torch.cat(feats, 0)                     # (N,D)


    @torch.no_grad()
    def compute_features(self, images, token_name):
        if token_name == "x_norm_clstoken":
            if images.shape[0] > self.chunk_size:
                features = self.forward_by_chunk(images)
            else:
                features = self.model(images)
        else:  # get both features
            raise NotImplementedError
        return features

    # @torch.no_grad()
    # def forward_by_chunk(self, processed_rgbs):
    #     batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
    #     del processed_rgbs  # free memory
    #     features = BatchedData(batch_size=self.chunk_size)
    #     for idx_batch in range(len(batch_rgbs)):
    #         feats = self.compute_features(
    #             batch_rgbs[idx_batch], token_name="x_norm_clstoken"
    #         )
    #         features.cat(feats)
    #     return features.data


    @torch.no_grad()
    def forward_cls_token(self, image_np, proposals):
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        return self.forward_by_chunk(processed_rgbs)


    # def process_masks_proposals(self, masks, boxes):
    #     """
    #     1. Normalize image with DINOv2 transfom
    #     2. Mask and crop each proposals
    #     3. Resize each proposals to predefined longest image size
    #     """
    #     num_proposals = len(masks)
    #     masks.unsqueeze_(1) # [N_proposal, 1, ImgH, ImgW]
    #     processed_masks = self.rgb_proposal_processor(
    #         masks, boxes
    #     ).squeeze_()  # [N, 1, target_size, target_size]
    #     return processed_masks

    @torch.no_grad()
    def forward_patch_tokens(self, image_np, proposals):
        # with preprocess
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        processed_masks = self.process_masks_proposals(proposals.masks, proposals.boxes)
        return self.forward_by_chunk_v2(processed_rgbs, processed_masks)

    @torch.no_grad()
    def forward_by_chunk_v2(self, processed_rgbs, masks):
        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=masks)
        del processed_rgbs  # free memory
        del masks
        features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            feats = self.compute_masked_patch_feature(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            features.cat(feats)
        return features.data

    @torch.no_grad()
    def compute_masked_patch_feature(self, images, masks):
        # without preprocess
        if images.shape[0] > self.chunk_size:
            features = self.forward_by_chunk_v2(images, masks)
        else:
            features = self.model(images, is_training=True)["x_norm_patchtokens"]
            features_mask = self.patch_kernel(masks).flatten(-2) > self.validpatch_thresh
            features_mask = features_mask.unsqueeze(-1).repeat(1, 1, features.shape[-1])
            features = F.normalize(features * features_mask, dim=-1)
        return features


    @torch.no_grad()
    def forward(self, image_np, proposals):
        # with preprocess
        processed_rgbs = self.process_rgb_proposals(
            image_np, proposals.masks, proposals.boxes
        )
        processed_masks = self.process_masks_proposals(proposals.masks, proposals.boxes)

        batch_rgbs = BatchedData(batch_size=self.chunk_size, data=processed_rgbs)
        batch_masks = BatchedData(batch_size=self.chunk_size, data=processed_masks)
        del processed_rgbs  # free memory
        del processed_masks
        cls_features = BatchedData(batch_size=self.chunk_size)
        patch_features = BatchedData(batch_size=self.chunk_size)
        for idx_batch in range(len(batch_rgbs)):
            cls_feats, patch_feats = self.compute_cls_and_patch_features(
                batch_rgbs[idx_batch], batch_masks[idx_batch]
            )
            cls_features.cat(cls_feats)
            patch_features.cat(patch_feats)
        
        return cls_features.data, patch_features.data

    def compute_cls_and_patch_features(self, images, masks):
        device  = next(self.model.parameters()).device       # GPU (or CPU)

        images  = images.to(device,  non_blocking=True)
        masks   = masks.to(device,   non_blocking=True)
        with maybe_autocast(self._use_amp):                  # mixed‑precision opt‑in
            features = self.model(images, is_training=True)
        patch_features = features["x_norm_patchtokens"]
        cls_features = features["x_norm_clstoken"]
        features_mask = self.patch_kernel(masks).flatten(-2) > self.validpatch_thresh
        features_mask = features_mask.unsqueeze(-1).repeat(1, 1, patch_features.shape[-1])
        patch_features = F.normalize(patch_features * features_mask, dim=-1)

        return cls_features, patch_features
        
