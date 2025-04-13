# IF_GPT
Inverse Folding with Multimodal Protein Language Model



This code integrates a pretrained backbone encoder (from a VQ‑VAE pipeline: https://github.com/SATANtrainsAI/PROTOKEN) with a pretrained GPT-based protein language model(https://github.com/SATANtrainsAI/tinyPLM) to train a multimodal inverse folding system. In essence, the model generates protein sequences conditioned on structural images of the protein backbone.

---

## Key Components

- **LayerNorm & MLP Block**  
  - *LayerNorm:* A straightforward normalization layer (without bias by default) used before attention and MLP sublayers.
  - *MLP:* A two-layer network with GELU activation that processes token embeddings in each transformer block.

- **Multi-Head Self-Attention with Rotary Embedding (MLA)**  
  - Projects input embeddings into queries, keys, and values.
  - Applies rotary embeddings to encode positional information by rotating the queries and keys.
  - Performs scaled dot-product attention via PyTorch’s built-in function.
  
- **Transformer Block**  
  - Each block comprises residual connections around a self-attention sublayer and an MLP sublayer, with LayerNorm applied before each module.

- **Vision Transformer Encoder**  
  - The `VitEncoder` downsamples protein backbone images via a stack of small-stride convolutional layers (the DownsampleStack) and processes the resulting feature map with transformer blocks.
  - Only the encoder part is used (i.e. no quantizer or decoder) since the structural tokens are pre-extracted.

- **Cross-Attention Mechanism (CMLA and cBlock)**  
  - *CMLA:* Implements cross-attention by projecting sequence representations as queries and the flattened image latents as keys/values. Rotary embeddings are applied for relative positional encoding.
  - *cBlock:* A transformer sublayer that wraps a cross-attention branch with residual connections and further normalization.

- **Interleaved Transformer Stack (IFTransformer / IFTransformer2)**  
  - Pretrained GPT transformer blocks are interleaved with new cross-attention (cBlock) layers at specific indices to inject structural conditioning from image latents.
  - This enables the fused model to condition sequence generation on structural features.

- **Inverse Folding Model (IFModel / IFModel2)**  
  - Combines the pretrained backbone encoder (which produces latent tokens from the structure image) with the pretrained GPT model.
  - The model selects certain encoder levels as conditioning signals and passes them to an interleaved transformer stack.
  - The final output is a sequence of logits over the amino acid vocabulary for protein sequence generation.
  
- **Conditional Generation and Training**  
  - The model’s generation routine uses repetition penalty and n-gram blocking to avoid redundant outputs.
  - The training loss is computed as cross-entropy over the target sequence, with the pretrained components fine-tuned only via the new cross-attention layers.
  - Distributed training, gradient accumulation, and mixed precision are used to improve efficiency.

---

## Summary

After pretraining the VQ-VAE encoder for protein backbone representation and the GPT language model for sequence learning separately, the code fuses both modalities into a unified model for the inverse folding task. It does so by conditioning the GPT transformer with latent structural features extracted by the encoder and integrating cross-attention layers at strategic positions. This design allows the model to generate protein sequences that are structurally consistent with a given backbone, thereby solving the inverse folding problem in a multimodal framework.
