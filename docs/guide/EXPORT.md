← [Back to Main README](../../README.md)

<h1 align="center">Model Export Guide</h1>

Convert trained Orchard ML models to ONNX or TorchScript for production deployment.

<h2>How It Works</h2>

Add an `export:` section to any YAML recipe. The pipeline will **train, evaluate, and export** in a single run:

```bash
orchard run recipes/config_efficientnet_b0.yaml
```

After training completes, the export phase automatically:
1. Loads the best checkpoint from `models/`
2. Traces the model and converts to the target format
3. Validates PyTorch vs exported output (optional)
4. Saves the exported file alongside the checkpoint

```
outputs/20260219_galaxy10_efficientnetb0_abc123/
  models/
    best_efficientnetb0.pth     # PyTorch checkpoint
    best_efficientnetb0.onnx    # Exported ONNX model
```

<h2>Configuration</h2>

All export behavior is controlled via the `export:` section of your YAML recipe.

**Minimal (defaults are sensible):**

```yaml
export:
  format: onnx
```

**Full reference:**

```yaml
export:
  # Format
  format: onnx                    # onnx | torchscript | both
  output_path: null               # auto-generated if omitted

  # ONNX settings
  opset_version: 18               # 18 = latest, no conversion warnings
  dynamic_axes: true              # dynamic batch size for flexible inference
  do_constant_folding: true       # fold constants at export time

  # TorchScript settings
  torchscript_method: trace       # trace | script

  # Quantization
  quantize: false                 # apply INT8 post-training quantization
  quantization_backend: qnnpack   # qnnpack (mobile/ARM) | fbgemm (x86)

  # Validation
  validate_export: true           # compare PyTorch vs exported output
  validation_samples: 10          # number of samples for validation
  max_deviation: 1.0e-05          # max allowed numerical deviation
```

| Field | Default | Description |
|-------|---------|-------------|
| `format` | `onnx` | Export format: `onnx`, `torchscript`, or `both` |
| `output_path` | `null` | Custom output path (auto-generated if omitted) |
| `opset_version` | `18` | ONNX opset version (18 recommended) |
| `dynamic_axes` | `true` | Enable dynamic batch size for inference flexibility |
| `do_constant_folding` | `true` | Optimize constant operations during export |
| `torchscript_method` | `trace` | TorchScript method: `trace` (recommended) or `script` |
| `quantize` | `false` | Apply INT8 post-training quantization |
| `quantization_backend` | `qnnpack` | Quantization backend: `qnnpack` (mobile/ARM), `fbgemm` (x86) |
| `validate_export` | `true` | Run numerical validation after export |
| `validation_samples` | `10` | Number of random samples for validation |
| `max_deviation` | `1e-5` | Maximum allowed output deviation (PyTorch vs export) |

<h2>Quantization</h2>

INT8 quantization reduces model size ~4x and speeds up CPU inference, with minimal accuracy loss:

```yaml
export:
  format: onnx
  quantize: true
  quantization_backend: fbgemm    # x86 server deployment
  validate_export: true           # verify accuracy after quantization
  max_deviation: 1.0e-03          # relax tolerance for quantized models
```

> **Note:** Use `qnnpack` for mobile/ARM targets, `fbgemm` for x86 servers.

<h2>Troubleshooting</h2>

<h3>Validation failed</h3>

If quantization causes deviations beyond `max_deviation`, either relax the tolerance or set `validate_export: false`.

<h3>Missing onnxscript</h3>

```bash
pip install onnx onnxruntime onnxscript
```

<h3>Export warnings</h3>

`opset_version: 18` produces clean output. Lower versions may emit harmless conversion warnings.

<h2>Next Steps</h2>

- Deploy with [ONNX Runtime](https://onnxruntime.ai/)
- Optimize for edge devices with [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/)
- Convert to TensorRT for NVIDIA GPUs
