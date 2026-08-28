"""Factory de modelos para U-Net, DeepLabV3+ y SegFormer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerBinaryWrapper(nn.Module):
    """Wrapper para SegFormer binario con salida al tamano de entrada."""

    def __init__(self, image_size: int = 512) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig(
            num_channels=3,
            num_labels=1,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
            semantic_loss_ignore_index=-100,
        )
        self.model = SegformerForSemanticSegmentation(config)
        self.image_size = image_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predice logits binarios."""
        output = self.model(pixel_values=x)
        logits = output.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


def create_model(architecture: str, image_size: int = 512) -> nn.Module:
    """Crea modelo binario sin descargar pesos externos por defecto."""
    architecture = architecture.lower()

    if architecture in {"unet", "u-net"}:
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    if architecture in {"deeplabv3", "deeplabv3+", "deeplabv3plus"}:
        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

    if architecture == "segformer":
        return SegFormerBinaryWrapper(image_size=image_size)

    raise ValueError(f"Arquitectura no soportada: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Cuenta parametros entrenables."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
