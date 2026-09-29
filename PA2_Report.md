# Pruning Report

This report summarizes results recorded in the executed notebook. All values below match the notebook’s result tables and Final Report section.

---

## Main comparison

| Model | Test Top-1 | Test Top-5 | Macro-F1 | Sparsity / reduction |
|---|---:|---:|---:|---:|
| Dense FP32 | 93.07% | 99.74% | 93.07% | 0% |
| Task 1 Local magnitude | 92.99% | 99.77% | 92.99% | 80% |
| Task 1 Global magnitude | 92.82% | 99.75% | 92.81% | 80% |
| Task 2 Iterative magnitude | 91.42% | 99.71% | 91.43% | 80% |
| Task 2 Iterative GraSP | 88.65% | 99.42% | 88.64% | 80% |
| Task 3 Structured regression | 92.61% | 99.72% | 92.61% | 79.99% |

Final target sparsity for Tasks 1 and 2: **80%**. Task 3 parameter reduction: **79.99%**.

---

## Task 0: Load and profile the base model

The official CIFAR-10 ResNet-18 checkpoint loaded with no missing or unexpected keys.

### Data and preprocessing

| Setting | Value |
|---|---|
| Batch size | 256 |
| Input size | 3 × 32 × 32 |
| Normalization mean | [0.4914, 0.4822, 0.4465] |
| Normalization std | [0.2471, 0.2435, 0.2616] |
| Training augmentation | RandomCrop(32, padding=4), RandomHorizontalFlip |

### Dense baseline metrics

| Metric | Result |
|---|---:|
| Train Top-1 | 99.866% |
| Test Top-1 | 93.070% |
| Train Top-5 | 99.996% |
| Test Top-5 | 99.740% |
| Test Macro-F1 | 93.070% |
| Parameters | 11,173,962 |
| Serialized size | 44.77 MB |
| MACs | 140.19 M |
| FLOPs | 280.37 M |
| Mean inference latency | 12.00 ms / batch (0.0469 ms / image) |
| Average GPU memory | 57.48 MB |
| Peak GPU memory | 191.68 MB |
| CPU energy | 0.002245 J / image |
| GPU energy | 0.003683 J / image |

Test Top-1 differs from the published baseline by about **0.001 percentage points**.

---

## Task 1: Unstructured post-training magnitude pruning

Both methods used **80% overall sparsity**.

| Metric | Local | Global |
|---|---:|---:|
| Train Top-1 | 99.86% | 99.89% |
| Test Top-1 before fine-tuning | 92.91% | 93.11% |
| Test Top-1 after fine-tuning | 92.99% | 92.82% |
| Test Top-5 | 99.77% | 99.75% |
| Macro-F1 | 92.99% | 92.81% |
| Accuracy change from fine-tuning | +0.08 pp | −0.29 pp |
| Dense weight size | 44.77 MB | 44.77 MB |
| COO package size | 91.61 MB | 91.59 MB |
| Mean dense-masked latency | 12.17 ms | 12.29 ms |
| Average GPU memory | 343.20 MB | 343.20 MB |
| Peak GPU memory | 477.41 MB | 477.41 MB |
| CPU energy | 0.001287 J/image | 0.001262 J/image |
| GPU energy | 0.003748 J/image | 0.003775 J/image |

Mask verification passed. The classifier can use `torch.sparse.mm`; convolutions still use reconstructed dense weights.

### Required discussion (Task 1)

**Which layers were the most sensitive to pruning?**  
`conv1.weight` was the most sensitive (**22.15%** test Top-1 at **90%** layer sparsity). `layer2.0.conv1.weight` was also sensitive at high sparsity.

**Did global pruning produce a reasonable layer-wise sparsity distribution?**  
**Yes**—strongly non-uniform sparsities; **no layer was completely pruned**; some late layers exceeded **95%** sparsity.

**How much accuracy was recovered through fine-tuning?**  
Local: **+0.08 pp**. Global: **−0.29 pp** vs pre–fine-tuning test Top-1.

**Did COO storage reduce the actual model size?**  
**No** (local COO **91.61 MB** vs dense **44.77 MB**; also larger than dense weights plus masks at **55.94 MB**).

**Why can many zeros still give ~dense latency?**  
Tensor shapes are unchanged; dense convolution still processes zero-valued positions (~**12.17–12.29 ms** vs dense **~12.00 ms**).

---

## Task 2: Saliency-based iterative pruning (GraSP)

ResNet-18 was trained from **random initialization** (not the Task 0 checkpoint). Iterative magnitude and GraSP shared initialization hash `1f7b0761da53c6f0654eaa0d5bfde4ab84537d26890ffe3e9904662a17e2f758`, a **100-epoch** budget, and **80%** final sparsity.

### Calibration (GraSP)

| Setting | Value |
|---|---:|
| Calibration samples | 1024 |
| Calibration batch size | 128 |
| Calibration batches | 8 |

Warm-up reached **48.92%** validation Top-1 on the first epoch (above the ~20% trigger before the first epoch-level checkpoint).

### Final results

| Method | Final train Top-1 | Final test Top-1 | Test Top-5 | Macro-F1 | Total criterion time |
|---|---:|---:|---:|---:|---:|
| Iterative magnitude | 99.166% | 91.42% | 99.71% | 91.43% | 0.666 s |
| Iterative GraSP | 98.282% | 88.65% | 99.42% | 88.64% | 4.086 s |

### Pruning stages (test Top-1 before → immediately after)

| Method | Stage sparsity | Before | After |
|---|---:|---:|---:|
| Magnitude | 40% | 47.91% | 46.70% |
| Magnitude | 60% | 61.75% | 57.34% |
| Magnitude | 80% | 65.13% | 43.88% |
| GraSP | 40% | 47.91% | 10.00% |
| GraSP | 60% | 46.88% | 11.87% |
| GraSP | 80% | 61.80% | 10.41% |

Training and validation curves and layer-wise remaining-weight ratios are in the notebook (Task 2.5 and Show Task 2).

### Required discussion (Task 2)

**Did iterative pruning outperform magnitude-based pruning (Task 1)?**  
**No** for the reference comparison to Task 1 **global** magnitude (**92.82%** test Top-1 after fine-tuning vs iterative magnitude **91.42%**). Task 1 uses a pretrained checkpoint; Task 2 trains from random init, so this is not a controlled match of initialization.

**Did GraSP perform better than iterative magnitude pruning?**  
**No.** Final test Top-1: iterative magnitude **91.42%**, iterative GraSP **88.65%**.

**Did any layer collapse at high sparsity?**  
**No.** All layers retained some weights at **80%** final sparsity.

---

## Task 3: Structured (channel) pruning

Structured pruning targeted **80%** parameter reduction. The physical model was fine-tuned for **20 epochs**.

| Metric | Dense baseline | Structured model |
|---|---:|---:|
| Test Top-1 | 93.07% | 92.61% |
| Test Top-5 | 99.74% | 99.72% |
| Macro-F1 | 93.07% | 92.61% |
| Parameters | 11,173,962 | 2,241,290 |
| Model size | 44.77 MB | 9.03 MB |
| MACs | 140.19 M | 94.84 M |
| FLOPs | 280.37 M | 189.69 M |
| Mean latency | 12.00 ms | 9.90 ms |
| Average GPU memory | 57.48 MB | 786.13 MB |
| Peak GPU memory | 191.68 MB | 920.34 MB |
| CPU energy | 0.002245 J/image | 0.001022 J/image |
| GPU energy | 0.003683 J/image | 0.003030 J/image |

Test Top-1 drop vs dense: **0.46 percentage points**.

### Required discussion (Task 3)

**Did you see an improvement in performance (memory, latency) while meeting the accuracy constraint? Why or why not?**  
**Latency improved; peak GPU memory did not.** Mean batch latency **12.00 ms → 9.90 ms** (~**17.5%**). Model size decreased ~**79.8%** and MACs ~**32.3%**, with **0.46 pp** test accuracy loss. **Peak GPU memory increased** (191.68 MB → 920.34 MB) despite fewer parameters, likely due to activations, allocator/caching behavior, and profiling overhead—not parameter count alone. Physical channel removal reduces parameters and MACs, but measured peak memory can still rise on this profiling setup.

Channel removal updated matching BatchNorm and `conv2` input channels; residual block output widths stayed compatible with shortcuts.

---
