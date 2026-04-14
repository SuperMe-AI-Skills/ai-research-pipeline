"""
interpret_gradcam.py
====================
Gradient-weighted Class Activation Mapping (Grad-CAM) and ViT attention
visualisation for image classification models.

Provides:
  - :class:`GradCAM` — generic Grad-CAM implementation via hooks.
  - :func:`get_target_layer` — auto-selects the right conv layer per model type.
  - :func:`overlay_heatmap` — blends a heatmap onto the original image.
  - :func:`generate_gradcam_grid` — multi-sample Grad-CAM grid saved as PNG.
  - :func:`generate_attention_map` — ViT-specific attention visualisation.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .data_prep import IMAGENET_MEAN, IMAGENET_STD
from .dl_train_utils import get_device


# ===================================================================== #
#  GradCAM                                                               #
# ===================================================================== #
class GradCAM:
    """Grad-CAM: Gradient-weighted Class Activation Mapping.

    Registers forward and backward hooks on a target convolutional layer
    to compute gradient-weighted activation maps.

    Parameters
    ----------
    model : nn.Module
        Trained classification model.
    target_layer : nn.Module
        Convolutional layer to visualise (e.g. ``model.layer4[-1]`` for
        ResNet).
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(
            self._bwd_hook
        )

    # ------------------------------------------------------------------ #
    def _fwd_hook(
        self,
        module: nn.Module,
        input: tuple,
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _bwd_hook(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        self._gradients = grad_output[0].detach()

    # ------------------------------------------------------------------ #
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate a Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Image tensor of shape ``(1, 3, H, W)``.
        target_class : int or None
            Class index to visualise.  If ``None`` the predicted class
            is used.

        Returns
        -------
        heatmap : np.ndarray
            Normalised heatmap of shape ``(H, W)`` in ``[0, 1]``.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        if self._gradients is None or self._activations is None:
            raise RuntimeError(
                "Hooks did not capture activations/gradients. "
                "Check that the target layer is correct."
            )

        # Global average pool of gradients -> channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input spatial size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    # ------------------------------------------------------------------ #
    def remove_hooks(self) -> None:
        """Remove the forward and backward hooks."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# ===================================================================== #
#  Auto target-layer selection                                           #
# ===================================================================== #
def get_target_layer(model: nn.Module, model_type: str) -> nn.Module:
    """Return the recommended convolutional layer for Grad-CAM.

    Parameters
    ----------
    model : nn.Module
    model_type : str
        One of ``'resnet'``, ``'efficientnet'``, ``'vgg'``,
        ``'densenet'``, ``'simple_cnn'``.

    Returns
    -------
    nn.Module
    """
    model_type = model_type.lower()

    if model_type == "resnet":
        return model.layer4[-1]
    elif model_type == "efficientnet":
        return model.features[-1]
    elif model_type == "vgg":
        return model.features[-1]
    elif model_type == "densenet":
        return model.features.denseblock4
    elif model_type == "simple_cnn":
        # Last conv block before adaptive pool (index 6 = Conv2d(64,128))
        return model.features[6]
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Choose from: resnet, efficientnet, vgg, densenet, simple_cnn."
        )


# ===================================================================== #
#  Overlay heatmap on image                                              #
# ===================================================================== #
def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Overlay a coloured heatmap on the original image.

    Parameters
    ----------
    image : PIL.Image.Image
        Original RGB image.
    heatmap : np.ndarray
        Normalised heatmap ``[0, 1]`` of shape ``(H, W)``.
    alpha : float
        Blending weight for the heatmap overlay.

    Returns
    -------
    PIL.Image.Image
        Blended image.
    """
    img_np = np.array(image.resize((heatmap.shape[1], heatmap.shape[0])))
    img_np = img_np.astype(np.float32)

    # Apply colour map (jet)
    heatmap_uint8 = np.uint8(255 * heatmap)
    coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.float32)

    blended = (1.0 - alpha) * img_np + alpha * coloured
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(blended)


# ===================================================================== #
#  De-normalise tensor to PIL                                            #
# ===================================================================== #
def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised image tensor back to a PIL Image.

    Parameters
    ----------
    tensor : torch.Tensor  (3, H, W) or (1, 3, H, W)
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    tensor = tensor.clamp(0, 1)
    np_img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


# ===================================================================== #
#  Grad-CAM grid                                                         #
# ===================================================================== #
def generate_gradcam_grid(
    model: nn.Module,
    model_type: str,
    data_loader: DataLoader,
    class_names: Sequence[str],
    n_samples: int = 3,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
) -> Optional[str]:
    """Generate a grid of Grad-CAM overlays (n_samples per class).

    Parameters
    ----------
    model : nn.Module
    model_type : str
        Passed to :func:`get_target_layer`.
    data_loader : DataLoader
    class_names : list of str
    n_samples : int
        Number of images per class to include.
    device : optional torch.device
    save_path : str or None
        Path for the output PNG.  If ``None`` the plot is displayed
        interactively instead.

    Returns
    -------
    str or None
        Path to the saved file, or ``None`` if not saved.
    """
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    target_layer = get_target_layer(model, model_type)
    cam = GradCAM(model, target_layer)

    n_classes = len(class_names)

    # Collect samples per class
    class_images: dict[int, list[torch.Tensor]] = {c: [] for c in range(n_classes)}
    class_labels_collected: dict[int, int] = {c: 0 for c in range(n_classes)}

    for images, labels in data_loader:
        for img, lbl in zip(images, labels):
            c = int(lbl.item())
            if class_labels_collected[c] < n_samples:
                class_images[c].append(img)
                class_labels_collected[c] += 1
        if all(v >= n_samples for v in class_labels_collected.values()):
            break

    # Build grid
    fig, axes = plt.subplots(
        n_classes,
        n_samples * 2,
        figsize=(n_samples * 4, n_classes * 2.2),
    )
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for c in range(n_classes):
        for s in range(n_samples):
            if s >= len(class_images[c]):
                continue
            img_tensor = class_images[c][s].unsqueeze(0).to(device)
            heatmap = cam.generate(img_tensor, target_class=c)
            pil_img = _tensor_to_pil(img_tensor)
            overlay = overlay_heatmap(pil_img, heatmap)

            col_orig = s * 2
            col_cam = s * 2 + 1

            axes[c, col_orig].imshow(np.array(pil_img))
            axes[c, col_orig].set_title(class_names[c], fontsize=8)
            axes[c, col_orig].axis("off")

            axes[c, col_cam].imshow(np.array(overlay))
            axes[c, col_cam].set_title("GradCAM", fontsize=8)
            axes[c, col_cam].axis("off")

    plt.suptitle(f"Grad-CAM Visualisation ({model_type})", fontsize=12)
    plt.tight_layout()

    cam.remove_hooks()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    plt.close(fig)
    return None


# ===================================================================== #
#  ViT attention map                                                     #
# ===================================================================== #
def generate_attention_map(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract and visualise attention from a Vision Transformer.

    Computes the mean attention from [CLS] token to all patch tokens
    in the last encoder block and reshapes it to a 2-D spatial heatmap.

    Parameters
    ----------
    model : nn.Module
        A torchvision ViT-B/16 model.
    image_tensor : torch.Tensor
        ``(1, 3, 224, 224)`` or ``(3, 224, 224)``.
    device : optional torch.device

    Returns
    -------
    attention_heatmap : np.ndarray
        Normalised heatmap of shape ``(patch_grid, patch_grid)`` —
        for ViT-B/16 with 224x224 input this is ``(14, 14)``.
    """
    if device is None:
        device = get_device()

    # Re-use the extraction helper from dl_vit (import here to avoid
    # circular import at module level).
    from .dl_vit import extract_attention_weights

    attn = extract_attention_weights(model, image_tensor, device=device)
    # attn shape: (num_heads, seq_len, seq_len)  seq_len = 1 + num_patches

    # Average over heads
    attn_mean = attn.mean(axis=0)  # (seq_len, seq_len)

    # CLS token attention to patch tokens (row 0, skip CLS column 0)
    cls_attn = attn_mean[0, 1:]  # (num_patches,)

    # Reshape to spatial grid
    num_patches = cls_attn.shape[0]
    grid_size = int(math.sqrt(num_patches))
    assert grid_size * grid_size == num_patches, (
        f"Cannot reshape {num_patches} patches into a square grid."
    )
    heatmap = cls_attn.reshape(grid_size, grid_size)

    # Normalise to [0, 1]
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap
