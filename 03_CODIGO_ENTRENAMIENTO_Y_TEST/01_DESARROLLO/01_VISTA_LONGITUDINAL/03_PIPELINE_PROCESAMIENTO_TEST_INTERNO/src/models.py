"""Modelos binarios y metadatos arquitectonicos."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegFormerBinary(nn.Module):
    """SegFormer-B0 binario con salida interpolada al tamano de entrada."""

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=1,
                id2label={0: "foreground"},
                label2id={"foreground": 0},
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_channels=3,
                num_labels=1,
                depths=[2, 2, 2, 2],
                hidden_sizes=[32, 64, 160, 256],
                decoder_hidden_size=256,
                semantic_loss_ignore_index=-100,
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=images).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def create_model(architecture: str, pretrained: bool) -> tuple[nn.Module, dict]:
    """Crea la arquitectura solicitada y devuelve su configuracion real."""
    architecture = architecture.lower()
    if architecture in {"unet", "deeplabv3"}:
        import segmentation_models_pytorch as smp

        encoder_weights = "imagenet" if pretrained else None
        if architecture == "unet":
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
                activation=None,
            )
            name = "U-Net"
        else:
            model = smp.DeepLabV3Plus(
                encoder_name="resnet34",
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1,
                activation=None,
            )
            name = "DeepLabV3+"
        return model, {
            "architecture_display": name,
            "implementation": "segmentation_models_pytorch",
            "encoder": "resnet34",
            "encoder_weights": encoder_weights or "none",
            "fine_tuning": "completo; encoder no congelado",
        }

    if architecture == "segformer":
        model = SegFormerBinary(pretrained=pretrained)
        return model, {
            "architecture_display": "SegFormer",
            "implementation": "transformers",
            "encoder": "MiT-B0",
            "encoder_weights": "nvidia/mit-b0" if pretrained else "none (configuracion B0)",
            "fine_tuning": "completo; encoder no congelado",
            "initialization_notes": (
                "encoder MiT-B0 preentrenado; decoder y cabezal binario inicializados de nuevo"
                if pretrained
                else "encoder, decoder y cabezal inicializados desde cero"
            ),
        }
    raise ValueError(f"Arquitectura no soportada: {architecture}")


def count_parameters(model: nn.Module) -> int:
    """Cuenta parametros entrenables."""
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
