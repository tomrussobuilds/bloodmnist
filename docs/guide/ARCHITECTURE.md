← [Back to Main README](../../README.md)

# Supported Models

## ResNet-18 Adapted (28x28)

Standard ResNet-18 is optimized for 224x224 ImageNet inputs. Direct application to 28x28 domains causes catastrophic information loss. Our adaptation strategy:

| Layer | Standard ResNet-18 | VisionForge Adapted | Rationale |
|-------|-------------------|---------------------|-----------|
| **Input Conv** | 7x7, stride=2, pad=3 | **3x3, stride=1, pad=1** | Preserve spatial resolution |
| **Max Pooling** | 3x3, stride=2 | **Identity (bypassed)** | Prevent 75% feature loss |
| **Stage 1 Input** | 56x56 (from 224) | **28x28 (from 28)** | Native resolution entry |

**Key Modifications:**
1. **Stem Redesign**: Replacing large-receptive-field convolution avoids immediate downsampling
2. **Pooling Removal**: MaxPool bypass maintains full spatial fidelity into residual stages
3. **Bicubic Weight Transfer**: Pretrained 7x7 weights are spatially interpolated to 3x3 geometry

---

## MiniCNN (28x28)

A compact, custom architecture designed specifically for low-resolution medical imaging:

| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Architecture** | 3 conv blocks + global pooling | Fast convergence with minimal parameters |
| **Parameters** | ~94K | 220x fewer than ResNet-18-Adapted |
| **Input Processing** | 28x28 → 14x14 → 7x7 → 1x1 | Progressive spatial compression |
| **Regularization** | Configurable dropout before FC | Overfitting prevention |

**Advantages:**
- **Speed**: 2-3 minutes for full 60-epoch training on GPU
- **Efficiency**: Ideal for rapid prototyping and ablation studies
- **Interpretability**: Simple architecture for educational purposes

---

## EfficientNet-B0 (224x224)

Implements compound scaling (depth, width, resolution) for optimal parameter efficiency:

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | Mobile Inverted Bottleneck Convolution (MBConv) | Memory-efficient feature extraction |
| **Parameters** | ~4.0M | 50% fewer than ResNet-50 |
| **Pretrained Weights** | ImageNet-1k | Strong initialization for transfer learning |
| **Input Adaptation** | Dynamic first-layer modification for grayscale | Preserves pretrained knowledge via weight morphing |

---

## Vision Transformer Tiny (ViT-Tiny) (224x224)

Patch-based attention architecture with multiple pretrained weight variants:

| Feature | Specification | Benefit |
|---------|--------------|---------|
| **Architecture** | 12-layer transformer encoder | Global context modeling via self-attention |
| **Parameters** | ~5.5M | Comparable to EfficientNet-B0 |
| **Patch Size** | 16x16 (196 patches from 224x224) | Efficient sequence length for transformers |
| **Weight Variants** | ImageNet-1k, ImageNet-21k, ImageNet-21k→1k fine-tuned | Optuna-searchable pretraining strategies |

**Supported Weight Variants:**
1. `vit_tiny_patch16_224.augreg_in21k_ft_in1k`: ImageNet-21k pretrained, fine-tuned on 1k (recommended)
2. `vit_tiny_patch16_224.augreg_in21k`: ImageNet-21k pretrained (requires custom head tuning)
3. `vit_tiny_patch16_224`: ImageNet-1k baseline

---

## Weight Transfer

To retain ImageNet-learned feature detectors when adapting to grayscale inputs, we apply bicubic interpolation for CNNs and channel averaging for ViT:

**CNN Weight Morphing (ResNet, EfficientNet):**

**Source Tensor:**
```math
W_{\text{src}} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}
```

**Transformation:**
```math
W_{\text{dest}} = \mathcal{I}_{\text{bicubic}}(W_{\text{src}}, \text{size}=(K', K'))
```

For grayscale adaptation:
```math
W_{\text{gray}} = \frac{1}{3} \sum_{c=1}^{3} W_{\text{src}}[:, c, :, :]
```

**ViT Patch Embedding Adaptation:**
```math
W_{\text{gray}} = \text{mean}(W_{\text{src}}, \text{dim}=1) \quad \text{where} \quad W_{\text{src}} \in \mathbb{R}^{192 \times 3 \times 16 \times 16}
```

**Result:** Preserves learned edge detectors and texture patterns while adapting to custom input geometry.

---

## Training Regularization

**MixUp Augmentation** synthesizes training samples via convex combinations:

```math
\tilde{x} = \lambda x_i + (1 - \lambda) x_j \quad \text{where} \quad \lambda \sim \text{Beta}(\alpha, \alpha)
```

```math
\tilde{y} = \lambda y_i + (1 - \lambda) y_j
```

This prevents overfitting on small-scale textures and improves generalization.

---
