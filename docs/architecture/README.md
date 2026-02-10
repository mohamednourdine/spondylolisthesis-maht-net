# MAHT-Net Architecture Components

This folder contains detailed explanations of each component in the MAHT-Net architecture.

## Overview

```
MAHT-Net Architecture
├── 1. CNN Backbone (EfficientNetV2-S)
├── 2. Transformer Bridge
├── 3. Vertebral Attention Module (VAM)
├── 4. Multi-scale Decoder
├── 5. Coordinate Extraction (DARK)
└── 6. Loss Functions
```

## Component Files

| File | Component | Description |
|------|-----------|-------------|
| [01_cnn_backbone.md](01_cnn_backbone.md) | CNN Backbone | EfficientNetV2-S feature extraction |
| [02_transformer_bridge.md](02_transformer_bridge.md) | Transformer Bridge | Global context modeling |
| [03_vertebral_attention_module.md](03_vertebral_attention_module.md) | VAM | Anatomically-aware keypoint attention |
| [04_multiscale_decoder.md](04_multiscale_decoder.md) | Multi-scale Decoder | Heatmap generation with skip connections |
| [05_dark_decoding.md](05_dark_decoding.md) | DARK Decoding | Sub-pixel coordinate extraction |
| [06_loss_functions.md](06_loss_functions.md) | Loss Functions | Multi-component training loss |

## Architecture Diagram

```
Input Image (512×512×3)
        │
        ▼
┌───────────────────┐
│   CNN Backbone    │  ← 01_cnn_backbone.md
│  (EfficientNetV2) │
│   F1, F2, F3, F4  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Transformer Bridge│  ← 02_transformer_bridge.md
│   (4 layers)      │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│       VAM         │  ← 03_vertebral_attention_module.md
│ (K query tokens)  │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Multi-scale       │  ← 04_multiscale_decoder.md
│ Decoder           │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ DARK Decoding     │  ← 05_dark_decoding.md
└───────────────────┘
        │
        ▼
Output: K keypoints (x, y, σ)
```

## Quick Links

- [Main Architecture Document](../MAHT_NET_ARCHITECTURE.md)
- [Related Works](../related-works/)
