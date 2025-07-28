import torch
import numpy as np
import cv2
from tqdm import tqdm
from omegaconf import OmegaConf

import sys
sys.path.append('/workspace-SR008.fs2/philurame/VMODEL/VBench')

from vbench.third_party.amt.utils.utils import check_dim_and_resize
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder

class FrameProcess:
    """Handle in‑memory video tensors of shape (T, C, H, W), dtype float64, range [-1, 1]."""

    def __init__(self):
        pass

    @staticmethod
    def get_frames(video_tensor: torch.Tensor):
        """Split a video tensor into a list of *batched* frame tensors.

        Each output tensor has shape **(1, 3, H, W)**, dtype float32, range [0, 1].
        This exactly mirrors the behaviour of the original `img2tensor()` helper:

        ```python
        def img2tensor(img):
            if img.shape[-1] > 3:
                img = img[:,:,:3]
            return torch.tensor(img).permute(2, 0, 1).unsqueeze(0) / 255.0
        ```
        """
        if not isinstance(video_tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(video_tensor)}")
        if video_tensor.ndim != 4:
            raise ValueError(f"Video must have shape (T, C, H, W); got {video_tensor.shape}")
        
        # assert -1e-6 > video_tensor.min() > -1.5
        # assert 1.5 > video_tensor.max() > 1e-6

        # Convert to float32 in [0, 1]
        video_tensor = video_tensor.to(torch.float32)
        video_tensor = (video_tensor + 1.0) / 2.0           # [-1, 1] → [0, 1]
        video_tensor = video_tensor.clamp_(0.0, 1.0)

        # Split into frames and add batch‑dim exactly like img2tensor
        frame_list = [frame.unsqueeze(0).contiguous() for frame in video_tensor]
        if not frame_list:
            raise ValueError("Input video contains no frames")
        return frame_list

    @staticmethod
    def extract_frame(frame_list, start_from: int = 0):
        """Return every second frame starting at *start_from*."""
        return [frame_list[i] for i in range(start_from, len(frame_list), 2)]


class MotionSmoothness:
    """Compute motion‑smoothness score for in‑memory video tensors."""

    def __init__(self, config, ckpt, device):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self._init_resources()
        self._load_model()
        self.fp = FrameProcess()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        print(f"Loading [{network_name}] from [{ckpt_path}]…")
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"], strict=True)
        self.model = self.model.to(self.device).eval()

    def _init_resources(self):
        if self.device == "cuda":
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024 ** 2
            self.anchor_memory_bias = 2500 * 1024 ** 2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print(f"VRAM available: {self.vram_avail / 1024 ** 2:.1f} MB")
        else:
            # No resizing in CPU mode
            self.anchor_resolution = 8192 * 8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(0.5, dtype=torch.float32, device=self.device).view(1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def motion_score(self, video_tensor: torch.Tensor):
        """Compute the normalised VFI score (1 → perfect, 0 → worst)."""
        # --------------------------------------------------------------
        # 1) Prepare inputs
        # --------------------------------------------------------------
        frames = self.fp.get_frames(video_tensor)
        frame_list = self.fp.extract_frame(frames, start_from=0)
        inputs = [frame.to(self.device) for frame in frame_list]
        if len(inputs) < 2:
            raise ValueError("Need at least two frames to compute motion score")

        # Dynamically scale depending on VRAM
        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = (
            self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        )
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            print(f"Due to limited VRAM, the video will be scaled by {scale:.2f}")

        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        # --------------------------------------------------------------
        # 2) Frame interpolation (identical to original pipeline)
        # --------------------------------------------------------------
        for _ in range(self.niters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                with torch.no_grad():
                    imgt_pred = self.model(
                        in_0, in_1, self.embt, scale_factor=scale, eval=True
                    )["imgt_pred"]
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        # --------------------------------------------------------------
        # 3) Compute VFI score
        # --------------------------------------------------------------
        outputs = padder.unpad(*outputs)
        vfi_score = self._vfi_score(frames, outputs)
        norm = (255.0 - vfi_score) / 255.0
        return float(norm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _vfi_score(self, original_frames, interpolated_frames):
        ori = self.fp.extract_frame(original_frames, start_from=1)
        inter = self.fp.extract_frame(interpolated_frames, start_from=1)
        scores = [self._frame_diff(o, p) for o, p in zip(ori, inter)]
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def _tensor_to_uint8(img: torch.Tensor):
        """Convert (1, 3, H, W) or (3, H, W) → H×W×3 uint8 ndarray."""
        if img.ndim == 4:
            img = img[0]
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        return img_np

    def _frame_diff(self, img1: torch.Tensor, img2: torch.Tensor):
        diff = cv2.absdiff(self._tensor_to_uint8(img1), self._tensor_to_uint8(img2))
        return float(diff.mean())


# =====================================================================
# Convenience wrappers (API mirrors original script)
# =====================================================================


def ms_loss(motion: MotionSmoothness, video_list):
    video_results = []
    for video_tensor in video_list:
        # (b) (T H W C) ->  (b) (T C H W)
        video_tensor = video_tensor.permute(0, 3, 1, 2).contiguous()

        score = motion.motion_score(video_tensor)
        video_results.append(1 - score)

    return torch.tensor(video_results)
