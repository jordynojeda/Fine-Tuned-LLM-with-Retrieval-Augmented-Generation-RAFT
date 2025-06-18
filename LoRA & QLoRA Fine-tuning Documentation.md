## 🔧 Fine-Tuning Strategies: Full, LoRA, and QLoRA

Large Language Models (LLMs) are powerful but computationally expensive to fine-tune. This section outlines different strategies for adapting LLMs to specific tasks or domains, including full-parameter fine-tuning, Low-Rank Adaptation (LoRA), and Quantized LoRA (QLoRA).

---

### 🧠 Full-Parameter Fine-Tuning

**Full fine-tuning** involves updating *all* of a model’s parameters across multiple training epochs. While effective in adapting models to new tasks or domains, it requires massive compute and memory.

#### Key Characteristics:
- Updates **all weight matrices** in the model.
- Very high memory requirements — not practical on consumer GPUs.
- Scales with model size:  
  - A 7B model has ~7 billion parameters  
  - A 13B model has ~13 billion parameters, and so on.
- Suitable for large GPU clusters with high memory capacity (e.g., A100s or H100s).

---

### 🛠️ LoRA (Low-Rank Adaptation)

**LoRA** is a parameter-efficient fine-tuning method that injects small, trainable "adapter" matrices into the model’s architecture, allowing updates without modifying the original weights directly.

#### How it Works:
1. LoRA freezes the original model weights.
2. It adds two low-rank matrices (A and B) to each targeted weight matrix.
3. During training, only the A and B matrices are updated.
4. At inference, the updates are merged into the original weights or applied on-the-fly.

This greatly reduces the number of trainable parameters and memory requirements, enabling fine-tuning even on modest hardware.

#### Number of Trainable Parameters (Approximate)

| Rank | 7B     | 13B    | 70B    | 180B   |
|------|--------|--------|--------|--------|
| 1    | 167k   | 288k   | 529k   | 849k   |
| 2    | 334k   | 456k   | 1M     | 2M     |
| 8    | 1M     | 2M     | 4M     | 7M     |
| 16   | 3M     | 4M     | 8M     | 14M    |
| 512  | 86M    | 117M   | 270M   | 434M   |
| 1024 | 171M   | 233M   | 542M   | 869M   |

> ✅ **Note:** Even at low ranks (e.g. 8 or 16), LoRA performs competitively on many tasks.

#### Does Rank Matter?

The core idea is that **many downstream tasks are intrinsically low-rank**, meaning only a small subset of updates is needed to capture task-specific knowledge. Rank selection involves a tradeoff between performance and efficiency.

---

### 💡 QLoRA (Quantized LoRA)

**QLoRA** builds on LoRA by incorporating quantization — reducing the model's precision to save memory while retaining accuracy.

#### Key Concepts:
- Applies **4-bit quantization** to model weights (using techniques like NF4).
- Stores LoRA adapter weights in full precision.
- Uses **paged optimizers** and **offloading** to enable training on consumer GPUs.
- Enables **full model training** (not just adapters) by keeping the quantized model in memory-efficient formats.

#### Benefits:
- Allows fine-tuning of 65B+ models on a single A100 GPU.
- Maintains similar performance to full fine-tuning at a fraction of the cost.
- Particularly effective when used with ranks between **8 and 256**, beyond which accuracy gains diminish.

---

### 🧪 Summary Comparison

| Method             | Updates All Weights | Memory Usage | Trainable Params (7B) | Suitable For               |
|--------------------|---------------------|---------------|------------------------|----------------------------|
| Full Fine-Tuning   | ✅                  | 🔴 Very High  | ~7B                    | Large GPU clusters         |
| LoRA               | ❌ (Adapters only)  | 🟡 Low         | ~1M (Rank 8)           | Laptops / Single GPU       |
| QLoRA              | ✅ (via quantization) | 🟢 Very Low  | ~1M + quantized weights| Consumer GPUs (e.g. RTX 4090) |

---

### 🔗 References

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [BitsAndBytes Quantization](https://github.com/TimDettmers/bitsandbytes)


  
