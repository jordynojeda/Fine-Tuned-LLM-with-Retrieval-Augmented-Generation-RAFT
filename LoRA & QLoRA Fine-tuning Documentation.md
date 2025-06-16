Full-parameter fine-tuning 

- Updates all the model weights
- The weight matrices are very large
  - 7B model has billion weights, 13B has 13 billion, and so on
- All weights get updated repeatedly for multiple "epochs"
- Storing and updating weights requires a lot of memory, which limits fine-tuning to large GPUs or GPU clusters

How is LoRA different?

1. Instead of updating weights directly, we track changes.
2. These weight changes are tracked in two seperate, smaller matrices that get multipled together to form a matrix the same size as the model's weight matrix.

Number of Trainable Parameters

| Rank |  7B  |  13B |  70B | 180B |
| ---- | ---- | ---- | ---- | ---- |
|   1  | 167k | 288k | 529K | 849k |
|   2  | 334k | 456k |  1M  |  2M  |
|   8  | 1M   |  2M  |  4M  |  7M  |
|   16 | 3M   |  4M  |  8M  | 14M  |
|  512 | 86M  | 117M | 270M | 434M |
| 1,024| 171M | 233M | 542M | 869M |

Does rank matter?

The theory is that downstream tasks are intrinsically low-rank. 

QLoRA

- Uses even less memory with "recoverable" quantization
- Training all layers of the network is necessary to match performance of full-parameter fine-tuning
- Rank may not matter from 8 to 256
  
