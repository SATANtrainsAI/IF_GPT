import torch
import torch.nn as nn
import torch.nn.functional as F



from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange


from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import math
import inspect


def exists(val):
    return val is not None

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


# =============================================================================
# 1. LayerNorm
# =============================================================================
class LayerNorm(nn.Module):
    """
    A simple LayerNorm without bias by default, as used in many transformer blocks.
    """
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, eps=1e-5)

# =============================================================================
# 2. Multi-Head Self-Attention with Rotary Embedding (MLA)
# =============================================================================
class MLA(nn.Module):
    """
    Multi-head attention block for 2D patches flattened into sequences.
    Uses scaled_dot_product_attention (PyTorch 2.0).
    """
    def __init__(self, config):
        super().__init__()
        # Ensure latent_dim is divisible by n_head
        assert config.latent_dim % config.n_head == 0, "latent_dim must be divisible by n_head"

        # Project from (n_embd) -> (latent_dim * 3) to get Q, K, V
        self.c_attn = nn.Linear(config.n_embd, config.latent_dim * 3, bias=False)
        # Final projection back from latent_dim -> n_embd
        self.c_proj = nn.Linear(config.latent_dim, config.n_embd, bias=False)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(dim=config.rope_dim)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.latent_dim = config.latent_dim

        # For VQ‐VAE, we typically do not want causal masking
        self.is_causal = config.is_causal  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (B, T, n_embd), T is the flattened 2D spatial dimension
        """
        B, T, _ = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*latent_dim)
        q, k, v = qkv.split(self.latent_dim, dim=2)

        # Reshape into (B, heads, T, dim_per_head)
        q = q.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)

        # Apply rotary embeddings to q and k
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Scaled dot product attention (PyTorch 2.0+)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=self.is_causal
        )
        # Reshape back
        y = y.transpose(1, 2).contiguous().view(B, T, self.latent_dim)
        # Final linear back to n_embd
        y = self.c_proj(y)
        return y

# =============================================================================
# 3. MLP Block
# =============================================================================
class MLP(nn.Module):
    """
    Simple 2-layer MLP for feedforward portion of a transformer block.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.mlp_hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# =============================================================================
# 4. Transformer Block
# =============================================================================
class Block(nn.Module):
    """
    A single Transformer block, consisting of:
      - LayerNorm
      - Multi-head Self-Attention
      - LayerNorm
      - MLP
    Each is a residual connection around LN+Attention or LN+MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=False)
        self.mla = MLA(config)
        self.ln2 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mla(self.ln1(x))   # Self-attention
        x = x + self.mlp(self.ln2(x))  # Feedforward
        return x

# =============================================================================
# 5. Vision Transformer Encoder
# =============================================================================

class DownsampleStack(nn.Module):
    """
    Replaces the single patch_emb Conv2d with multiple smaller-stride conv layers.
    Overlaps by using kernel_size=3, stride=2, repeated log2(downscale_factor) times.
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        layers = []
        current_ch = in_channels
        f = factor
        while f > 1:
            layers.append(nn.Conv2d(current_ch, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            current_ch = out_channels
            f //= 2
        self.down = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpsampleStack(nn.Module):
    """
    Replaces the single patch_unemb ConvTranspose2d with multiple smaller-stride transposed conv layers.
    Each step doubles the spatial resolution (stride=2).
    """
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        layers = []
        current_ch = in_channels
        f = factor
        while f > 1:
            layers.append(nn.ConvTranspose2d(current_ch, out_channels, kernel_size=2, stride=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.GELU())
            current_ch = out_channels
            f //= 2
        self.up = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class VitEncoder(nn.Module):
    """
    Takes an image, downsamples it by 'downscale_factor' via a Conv2d,
    flattens the resulting feature map, then applies several Transformer blocks.
    Finally, reshapes back to (B, C, H, W).
    """
    def __init__(self, config, downscale_factor: int, in_channels: Optional[int] = None):
        super().__init__()
        if in_channels is None:
            in_channels = config.n_embd  # default

        # Patch embedding by conv with kernel_size = stride = downscale_factor
        self.patch_emb = DownsampleStack(
            in_channels=in_channels,
            out_channels=config.n_embd,
            factor=downscale_factor
        )
        # Stacked transformer blocks
        self.encoder_blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer_encoder)
        ])
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, in_channels, H, W)
        returns: (B, n_embd, H//downscale, W//downscale)
        """
        x = self.patch_emb(img)  # shape: (B, n_embd, H/down, W/down)
        B, C, H, W = x.shape
        # Flatten (spatial) => (B, H*W, C)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # Apply the Transformer
        x = self.encoder_blocks(x)
        # Reshape back
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x

# =============================================================================
# 6. Vision Transformer Decoder
# =============================================================================
class VitDecoder(nn.Module):
    """
    Takes a multi-code concatenation as input: shape (B, n_embd*(level+1), H, W).
    1) Projects down to n_embd
    2) Applies stacked transformer blocks
    3) ConvTranspose back up by 'upscale_factor'.
    """
    def __init__(self, config, upscale_factor: int, level: int, out_channels: Optional[int] = None):
        """
        :param level: the number of codes being concatenated minus 1.
                      For example, if we're concatenating 2 codes, level=1 => input channels = n_embd * 2.
        """
        super().__init__()
        if out_channels is None:
            out_channels = config.n_embd  # default to embedding dimension

        self.config = config
        self.level = level  # (level+1) codes are concatenated
        in_ch = config.n_embd * (level + 1)

        # Project from in_ch -> n_embd via 1x1 conv
        self.in_proj = nn.Conv2d(in_ch, config.n_embd, kernel_size=1)

        # Stacked transformer blocks
        self.decoder_blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer_decoder)
        ])

        # Transpose conv to "unpatch" with kernel_size = stride = upscale_factor
        self.patch_unemb = UpsampleStack(
            in_channels=config.n_embd,
            out_channels=out_channels,
            factor=upscale_factor
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_embd*(level+1), H, W)
        returns: (B, out_channels, H*upscale, W*upscale)
        """
        # 1) Project input to n_embd
        x = self.in_proj(x)  # shape: (B, n_embd, H, W)
        B, C, H, W = x.shape

        # 2) Flatten and run Transformer
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.decoder_blocks(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        # 3) Transpose conv up to original resolution
        x = self.patch_unemb(x)
        return x

# =============================================================================
# 7. Quantize (VQ-VAE Codebook with EMA) in NCHW
# =============================================================================
class Quantize(nn.Module):
    """
    Learns a discrete codebook with an exponential moving average update,
    as in the original VQ-VAE2 by Oord et al. or Rosinality's code.

    NOTE: We keep everything in NCHW for the commitment loss to avoid shape mismatches.
    """
    def __init__(self, config, in_channels: Optional[int] = None):
        super().__init__()
        self.codebook_dim = config.codebook_dim  # embedding dimension
        if in_channels is None:
            self.in_channels = config.q_channels
        else:
            self.in_channels = in_channels

        self.codebook_size = config.codebook_size  # number of entries
        self.decay = 0.99
        self.eps = 1e-5

        # 1x1 conv to go from in_channels -> codebook_dim
        self.conv_in = nn.Conv2d(self.in_channels, self.codebook_dim, 1)

        # Initialize the codebook: shape (codebook_dim, codebook_size)
        codebook = torch.randn(self.codebook_dim, self.codebook_size, dtype=torch.float32)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(self.codebook_size, dtype=torch.float32))
        self.register_buffer("codebook_avg", codebook.clone())

    @torch.autocast("cuda", enabled=False)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: (B, in_channels, H, W)
        :return: quantized, diff, codebook_indices
           quantized: (B, codebook_dim, H, W)
           diff: scalar codebook commitment loss
           codebook_indices: (B, H, W), the argmax index per spatial location
        """
        # 1) Project to codebook_dim, staying in NCHW
        x_projected = self.conv_in(x.float())  # => (B, codebook_dim, H, W)

        B, D, H, W = x_projected.shape

        # 2) Flatten spatial dims => (B*H*W, D)
        x_reshaped = rearrange(x_projected, 'b d h w -> (b h w) d')

        # 3) Compute distances to each codebook entry, shape => (B*H*W, codebook_size)
        #    dist(i, j) = || x_i - e_j ||^2
        codebook_t = self.codebook  # shape (codebook_dim, codebook_size)
        dist = (
            x_reshaped.pow(2).sum(dim=1, keepdim=True)
            - 2 * (x_reshaped @ codebook_t)
            + codebook_t.pow(2).sum(dim=0, keepdim=True)
        )  # (B*H*W, codebook_size)

        # 4) Find nearest codebook entry
        _, indices = (-dist).max(dim=1)          # (B*H*W,)
        indices = indices.view(B, H, W)          # (B, H, W)
        codebook_onehot = F.one_hot(indices, self.codebook_size).type(x_reshaped.dtype)
        codebook_onehot = rearrange(codebook_onehot, 'b h w c -> (b h w) c')  # (B*H*W, codebook_size)

        # 5) Lookup embedding from the codebook => shape (B, H, W, codebook_dim)
        quantized_hw_d = self.embed_code(indices)

        # 6) If training, do EMA updates
        if self.training:
            # sum of one-hot per codebook entry => cluster size
            codebook_onehot_sum = codebook_onehot.sum(dim=0)  # (codebook_size,)

            # sum of x_reshaped for each code => (codebook_dim, codebook_size)
            codebook_sum = x_reshaped.transpose(0, 1) @ codebook_onehot  # (D, codebook_size)

            # Update cluster size
            self.cluster_size.data.mul_(self.decay).add_(codebook_onehot_sum, alpha=1 - self.decay)
            # Update codebook averages
            self.codebook_avg.data.mul_(self.decay).add_(codebook_sum, alpha=1 - self.decay)

            # Normalize so each codebook vector is the average of all assigned vectors
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            codebook_normalized = self.codebook_avg / cluster_size.unsqueeze(0)
            self.codebook.data.copy_(codebook_normalized)

        # 7) Compute commitment loss in NCHW
        #    quantized_hw_d is (B, H, W, D)
        #    let's transpose that to (B, D, H, W)
        quantized_nchw = rearrange(quantized_hw_d, 'b h w d -> b d h w')
        diff = (quantized_nchw.detach() - x_projected).pow(2).mean()

        # 8) Straight-through estimator
        #    final quantized = x_projected + (quantized - x_projected).detach()
        quantized_nchw = x_projected + (quantized_nchw - x_projected).detach()

        # 9) Return NCHW quantized
        return quantized_nchw, diff, indices

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        """
        embed_id: (B, H, W) of codebook indices in [0, codebook_size)
        returns: (B, H, W, codebook_dim)
        """
        # codebook: (codebook_dim, codebook_size)
        # F.embedding => shape (B*H*W, codebook_dim), we reshape to (B,H,W, D)
        out_flat = F.embedding(embed_id.view(-1), self.codebook.transpose(0, 1))
        out = rearrange(out_flat, '(b h w) d -> b h w d', b=embed_id.shape[0], h=embed_id.shape[1], w=embed_id.shape[2])
        return out



class RefinementHead(nn.Module):
    """
    A small transformer-based refinement step for the final (B,3,H,W) output,
    to help sharpen or denoise.
    """
    def __init__(self, config):
        super().__init__()
        # Project 3 -> n_embd
        self.in_proj = nn.Conv2d(3, config.n_embd, kernel_size=1)
        # A small stack of Transformer blocks (e.g. 2)
        self.refine_blocks = nn.Sequential(*[
            Block(config) for _ in range(2)
        ])
        # Project back n_embd -> 3
        self.out_proj = nn.Conv2d(config.n_embd, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        returns: (B, 3, H, W) refined
        """
        B, C, H, W = x.shape
        r = self.in_proj(x)  # (B, n_embd, H, W)
        r = rearrange(r, 'b c h w -> b (h w) c')
        r = self.refine_blocks(r)  # Transform
        r = rearrange(r, 'b (h w) c -> b c h w', h=H, w=W)
        r = self.out_proj(r)  # back to 3 channels
        return r

# =============================================================================
# 8. Model Configuration
# =============================================================================
@dataclass
class ModelArgs:
    """
    Hyperparameters for the model.
    Adjust to your preference. Typically n_embd ~ 256..768,
    n_head ~ 4..16, and so on, depending on memory and resolution.
    """
    # Basic
    n_embd: int = 1024
    mlp_hidden_dim: int = int(1024 * 2)
    n_head: int = 16
    latent_dim: int = 256   # dimension for Q/K/V
    rope_dim: int = 8
    is_causal: bool = False  # Usually no causal mask for VQ-VAE

    # Codebook
    q_channels: int = 1024
    codebook_dim: int = 1024
    codebook_size: int = 2048

    # Depth
    n_layer_encoder: int = 4
    n_layer_decoder: int = 4

    beta: float = 0.25

# =============================================================================
# 9. Hierarchical VQ-VAE (Transformer-based)
# =============================================================================
class VQVAE(nn.Module):
    """
    A multi-level VQ-VAE-2 style model with Transformers at each stage.
    - Each stage: 
      1) Encode with VitEncoder
      2) Next stage uses the output of the previous stage
    - Decoding top-down in reverse order, each code conditioned on
      the corresponding encoder output + upsampled next-lower-level decoder output
    - Then quantize each stage's conditioning, feed into a VitDecoder
    """
    def __init__(self, config: ModelArgs, scaling_rates=[2, 2, 2]):
        super().__init__()
        self.config = config
        self.scaling_rates = scaling_rates
        self.num_levels = len(scaling_rates)

        # ---------------------
        # Build encoders
        # ---------------------
        self.encoders = nn.ModuleList()
        for level in range(self.num_levels):
            # The first encoder sees 3-channel images
            in_ch = 3 if level == 0 else None
            self.encoders.append(
                VitEncoder(config, downscale_factor=scaling_rates[level], in_channels=in_ch)
            )

        # ---------------------
        # Build codebooks
        # ---------------------
        self.codebooks = nn.ModuleList()
        for level in range(self.num_levels):
            # The topmost level (level == num_levels - 1) has in_ch = n_embd
            # otherwise in_ch = 2 * n_embd (concatenate encoder_out + lower-level dec_out)
            if level == self.num_levels - 1:
                in_ch = config.n_embd
            else:
                in_ch = config.n_embd * 2
            self.codebooks.append(Quantize(config, in_channels=in_ch))

        # ---------------------
        # Build decoders
        # Decoding in top-down order: level=2 => coarsest => outputs n_embd
        #                             level=1 => outputs n_embd
        #                             level=0 => outputs 3
        # but we store them in normal order, so index=0 => factor=8 => final upsample
        # We'll just carefully pick out_channels = 3 if level==0, else n_embd
        # We'll feed (num_levels - level) codes into the decoder.
        # ---------------------
        self.decoders = nn.ModuleList()
        for level in range(self.num_levels):
            out_ch = 3 if level == 0 else config.n_embd
            codes_to_cat = (self.num_levels - level)
            self.decoders.append(
                VitDecoder(
                    config,
                    upscale_factor=scaling_rates[level],
                    level=(codes_to_cat - 1),
                    out_channels=out_ch
                )
            )

        self.refiner = RefinementHead(config)
        
        self.apply(self._init_weights)

        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, x: torch.Tensor):
        """
        :param x: (B, 3, H, W) input image
        :return: (recon, code_diffs, encoder_outputs, decoder_outputs, code_indices)
        """
        # ----------------------------------
        # 1) Bottom-up encoding
        # ----------------------------------
        encoder_outputs = []
        for level, encoder in enumerate(self.encoders):
            if level == 0:
                out = encoder(x)
            else:
                out = encoder(encoder_outputs[-1])
            encoder_outputs.append(out)  # shape: (B, n_embd, H/scale, W/scale)

        # ----------------------------------
        # 2) Top-down decoding
        # ----------------------------------
        code_diffs = []
        code_indices = []
        decoder_outputs = []  # store from coarse to fine
        code_outputs = []     # store the quantized codes from coarse to fine

        # We go in reversed order: self.num_levels-1 (coarsest) down to 0 (finest)
        for l in reversed(range(self.num_levels)):
            # Combine the encoder output with upsampled next-lower-level decoder output (if any)
            if len(decoder_outputs) == 0:
                # no finer-level decoder yet
                cond = encoder_outputs[l]
            else:
                # upsample the last decoder output to match this scale
                prev_dec = decoder_outputs[-1]
                target_size = encoder_outputs[l].shape[2:]  # (H, W) at current scale
                prev_dec_upsampled = F.interpolate(prev_dec, size=target_size, mode='bilinear', align_corners=False)
                # Cat => (B, 2*n_embd, H, W)
                cond = torch.cat([encoder_outputs[l], prev_dec_upsampled], dim=1)

            # Quantize the conditioning
            q, diff, inds = self.codebooks[l](cond)
            code_diffs.append(diff)
            code_indices.append(inds)

            # Also upsample all previously computed codes so they match this resolution
            upsampled_lower_codes = []
            for code_map in code_outputs:
                c_up = F.interpolate(code_map, size=cond.shape[2:], mode='bilinear', align_corners=False)
                upsampled_lower_codes.append(c_up)

            # Concatenate the new code q with all upsampled codes from even finer levels
            if len(upsampled_lower_codes) > 0:
                dec_input = torch.cat([q] + upsampled_lower_codes, dim=1)
            else:
                dec_input = q

            # Decode
            dec_out = self.decoders[l](dec_input)
            decoder_outputs.append(dec_out)
            code_outputs.append(q)

        # The final reconstruction is decoder_outputs[-1] (the last appended)
        reconstruction = decoder_outputs[-1]

        # Reverse lists if you prefer them from level=0..N-1
        # but it's optional. Right now they are in [coarsest -> finest] order
        reconstruction_refined = self.refiner(reconstruction)

        return reconstruction_refined, code_diffs, encoder_outputs, decoder_outputs, code_indices
    
    
    

class Encoder(nn.Module):
    def __init__(self, config: ModelArgs, scaling_rates=[2, 2, 2, 2, 2, 2]):
        """
        This encoder-only model consists solely of the bottom-up VitEncoder modules.
        :param config: Model configuration (ModelArgs)
        :param scaling_rates: List of downscale factors for each encoder level.
        """
        super().__init__()
        self.num_levels = len(scaling_rates)
        # Build encoder modules only.
        self.encoders = nn.ModuleList()
        for level in range(self.num_levels):
            # The first encoder sees 3-channel images
            in_ch = 3 if level == 0 else None
            self.encoders.append(
                VitEncoder(config, downscale_factor=scaling_rates[level], in_channels=in_ch)
            )


    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through all encoder stages.
        :param x: Input image tensor of shape (B, 3, H, W)
        :return: List of encoder outputs at each level.
                 Each output has shape (B, n_embd, H/scale, W/scale)
        """
        encoder_outputs = []
        for level, encoder in enumerate(self.encoders):
            if level == 0:
                out = encoder(x)
            else:
                # Feed the previous encoder's output into the next level.
                out = encoder(encoder_outputs[-1])
            encoder_outputs.append(out)
        return encoder_outputs


@dataclass
class GPT_Config:
    n_embd: int = 1024
    latent_dim: int = 256
    max_seq_len: int = 1024
    n_head: int = 16
    n_layer: int = 32
    vocab_size: int = 32            
    ignore_index: int = -100
    block_size: int = max_seq_len
    seq_padding_idx: int = 0
    is_causal: bool = True
    rope_dim: int = 8
    mlp_hidden_dim: int = n_embd * 2


# Our token dictionary (including our special separator token "1")
token_dict = {
    "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
    "A": 4, "B": 5, "C": 6, "D": 7, "E": 8, "F": 9,
    "G": 10, "H": 11, "I": 12, "J": 13, "K": 14, "L": 15,
    "M": 16, "N": 17, "O": 18, "P": 19, "Q": 20, "R": 21,
    "S": 22, "T": 23, "U": 24, "V": 25, "W": 26, "X": 27,
    "Y": 28, "Z": 29, "1": 30, "2": 31
}
# Inverse dictionary:
token_dict_inv = {v: k for k, v in token_dict.items()}

class ProteinTokenizer:
    def __init__(self, token_dict):
        self.token_dict = token_dict
        self.inv_token_dict = {v: k for k, v in token_dict.items()}
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.pad_id = token_dict[self.pad_token]
        self.bos_id = token_dict[self.bos_token]
        self.eos_id = token_dict[self.eos_token]
    def tokenize(self, sequence):
        return list(sequence)
    def convert_tokens_to_ids(self, tokens):
        return [self.token_dict.get(t, self.token_dict[self.unk_token]) for t in tokens]
    def encode(self, sequence, add_special_tokens=True):
        tokens = self.tokenize(sequence)
        if add_special_tokens:
            # For single‑protein examples we add both bos and eos.
            return self.convert_tokens_to_ids([self.bos_token] + tokens + [self.eos_token])
        else:
            return self.convert_tokens_to_ids(tokens)
    def decode(self, token_ids):
        return "".join(self.inv_token_dict.get(tid, self.unk_token) for tid in token_ids)
    def pad_sequences(self, sequences, padding_value=None, block_size=None):
        if block_size is None:
            block_size = max(len(seq) for seq in sequences)
        if padding_value is None:
            padding_value = self.token_dict[self.pad_token]
        return [seq[:block_size] + [padding_value] * max(0, block_size - len(seq)) for seq in sequences]

tokenizer = ProteinTokenizer(token_dict)



class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = ProteinTokenizer(token_dict)
        
        self.seq_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=0)

        self.ln = LayerNorm(config.n_embd, bias=False)
        
        # Initialize transformer layers
        self.transformer = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])

        self.project = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        
        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, seq, targets=None):
      
        x = self.seq_embedding(seq)
    
        for layer in self.transformer:
            x = layer(x)
    
        x = self.ln(x)

    
        return x
    
    


class CMLA(nn.Module):
    """
    Cross-attention block: x is the sequence representation (B, T, n_embd) and 
    y is the image representation (B, n_embd, H, W). It flattens y and applies scaled
    dot-product attention.
    """
    def __init__(self, config):
        super().__init__()
        assert config.latent_dim % config.n_head == 0, "latent_dim must be divisible by n_head"
        self.c_attn_q = nn.Linear(config.n_embd, config.latent_dim, bias=False)
        self.c_attn_k = nn.Linear(config.n_embd, config.latent_dim, bias=False)
        self.c_attn_v = nn.Linear(config.n_embd, config.latent_dim, bias=False)
        self.c_proj = nn.Linear(config.latent_dim, config.n_embd, bias=False)
        self.rotary_emb = RotaryEmbedding(dim=config.rope_dim)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.latent_dim = config.latent_dim
        self.is_causal = config.is_causal  

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (B, n_embd, H, W) -> flatten spatial dims: (B, H*W, n_embd)
        y = y.contiguous()
        B1, T1, _ = y.size()
        B, T, _ = x.size()
        q = self.c_attn_q(x)
        k = self.c_attn_k(y)
        v = self.c_attn_v(y)
        q = q.view(B, T, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        k = k.view(B1, T1, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        v = v.view(B1, T1, self.n_head, self.latent_dim // self.n_head).transpose(1, 2)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.latent_dim)
        out = self.c_proj(out)
        return out
    

class cBlock(nn.Module):
    """
    A single Transformer block, consisting of:
      - LayerNorm
      - Multi-head Self-Attention
      - LayerNorm
      - MLP
    Each is a residual connection around LN+Attention or LN+MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=False)
        self.ln2 = LayerNorm(config.n_embd, bias=False)
        self.cmla = CMLA(config)
        self.ln3 = LayerNorm(config.n_embd, bias=False)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = rearrange(y, 'b c h w -> b (h w) c')
        x = x + self.cmla(self.ln1(x), self.ln2(y))   # Self-attention
        x = x + self.mlp(self.ln3(x))  # Feedforward
        return x

# ----------------------------
# Modified Transformer Stack for IFModel
# ----------------------------
class IFTransformer(nn.Module):
    def __init__(self, gpt_model: 'GPT', config):
        """
        Interleaves the pretrained GPT blocks with new cross-attention blocks.
        Assumes the pretrained GPT model has 32 transformer blocks.
        The new transformer stack will have 32 + 4 = 36 layers.
        The cross-attention blocks will be inserted at positions:
            after block 8, 16, 24, and 32.
        """
        super().__init__()
        gpt_blocks = gpt_model.transformer  # Pretrained GPT blocks (ModuleList of length 32)
        assert len(gpt_blocks) == 32, "Expected 32 GPT blocks"
        self.layers = nn.ModuleList()
        # Add blocks 0-7
        for i in range(8):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention for first image latent (encoder level 2)
        self.layers.append(cBlock(config))
        # Blocks 8-15
        for i in range(8, 16):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention for second image latent (encoder level 3)
        self.layers.append(cBlock(config))
        # Blocks 16-23
        for i in range(16, 24):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention for third image latent (encoder level 4)
        self.layers.append(cBlock(config))
        # Blocks 24-31
        for i in range(24, 32):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention for fourth image latent (encoder level 5)
        self.layers.append(cBlock(config))
        
        # Record indices of inserted cross-attention layers in the new stack.
        self.cross_attn_positions = [8, 17, 26, 35]
        self.ln = gpt_model.ln  # Final layer norm from GPT.

    def forward(self, x: torch.Tensor, img_latents: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: Sequence representation (B, T, n_embd)
        :param img_latents: List of 4 image representations (conditioning tokens),
                            each of shape (B, n_embd, H, W).
        """
        cross_idx = 0
        for i, layer in enumerate(self.layers):
            if i in self.cross_attn_positions:
                # Insert cross-attention: add output of cross-attn (conditioned on the corresponding image latent)
                y = img_latents[cross_idx]
                x = layer(x, y)
                cross_idx += 1
            else:
                x = layer(x)
        x = self.ln(x)
        return x

# ----------------------------
# Complete IFModel Definition
# ----------------------------
class IFModel(nn.Module):
    def __init__(self, gpt_model: 'GPT', encodr_model: 'Encoder', config: GPT_Config):
        """
        Inverse Folding Model that integrates:
          - A pretrained VQ-VAE encoder (to convert protein structure backbone into latent tokens).
          - A pretrained GPT model (for protein sequence generation).
          - A modified transformer stack that interleaves GPT blocks with cross-attention blocks.
        
        We use the VQ-VAE encoder outputs from levels 2, 3, 4, 5 (ignoring the first two).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.gpt = gpt_model       # Pretrained GPT (with its embedding, transformer, etc.)
        self.encoder = encodr_model   # Pretrained Encoder model; we use only its encoder.
        # We assume vqvae.encode(x) returns a list of encoder outputs.
        self.encoder_levels_to_use = [2, 3, 4, 5]  # Use these 4 levels as conditioning.
        
        # Build the new transformer stack that interleaves GPT blocks with cross-attention blocks.
        self.if_transformer = IFTransformer(gpt_model, config)

        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
        
    
    def forward(self, seq: torch.Tensor, img: torch.Tensor, targets = None) -> torch.Tensor:
        """
        :param seq: Protein sequence tokens (B, T)
        :param img: Protein structure backbone image (B, 3, H, W)
        :return: Logits over vocabulary (B, T, vocab_size)
        """
        # Get image latent representations from VQ-VAE encoder.
        encoder_outputs = self.encoder(img)
        # Select only levels 2,3,4,5.
        img_latents = [encoder_outputs[i] for i in self.encoder_levels_to_use]
        # Get sequence embeddings.
        x = self.gpt.seq_embedding(seq)  # (B, T, n_embd)
        # Run through interleaved transformer.

        x = self.if_transformer(x, img_latents)
        if exists(targets):
            logits = self.gpt.project(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )
        else:
            logits = self.gpt.project(x[:, [-1], :])
            loss = None
    
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print0(f"using fused AdamW: {use_fused}")

        return optimizer
    

    @torch.inference_mode()     
    def generate_IF(self, prefix, max_size, img, temperature=1.0, top_k=7, 
                rep_penalty=5, ngram_block=4):
        generated = prefix.clone()
        tokens_to_generate = max_size - prefix.size(1)
        if tokens_to_generate.item() < 0:
            tokens_to_generate = 2
            #raise ValueError(f"Desired size {max_size} <= prefix length {prefix.size(1)}.")
        
        for _ in range(tokens_to_generate):
            logits, _ = self.forward(seq = generated, img = img)
            # Take the last-step logits
            next_token_logits = logits[:, -1, :]  # shape [B, vocab_size]
            
            # 1) Apply repetition penalty
            next_token_logits = adjust_logits_for_repetition(next_token_logits, generated, 
                                                            rep_penalty=rep_penalty)
            # 2) Apply n-gram blocking
            next_token_logits = adjust_logits_for_ngram_blocking(next_token_logits, generated, 
                                                                n=ngram_block)
            
            # Now sample from the adjusted logits
            next_token = self._sample_next_token(next_token_logits, temperature, top_k)
            generated = torch.cat([generated, next_token], dim=1)
            
            # If we hit the EOS token, break
            if next_token.item() == self.tokenizer.eos_id:
                break
        
        return generated
    
    def _sample_next_token(self, logits, temperature=1.0, top_k=7):
        logits = logits / temperature
        if top_k > 0:
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=1)
            next_tokens = top_indices.gather(-1, indices)
        else:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
    
        return next_tokens
    






def adjust_logits_for_repetition(logits, generated_seq, rep_penalty=4):
    """
    For each example in the batch, if the last token is repeated consecutively
    `rep_penalty` times or more in the generated sequence, set its logit to -∞.
    """
    logits = logits.clone()
    B = logits.size(0)
    for i in range(B):
        seq = generated_seq[i]
        if seq.numel() == 0:
            continue
        last_token = seq[-1].item()
        count = 1
        j = seq.size(0) - 2
        while j >= 0 and seq[j].item() == last_token:
            count += 1
            j -= 1
        if count >= rep_penalty:
            logits[i, last_token] = float('-inf')  # block
    return logits

def adjust_logits_for_ngram_blocking(logits, generated_seq, n=3):
    """
    For each example in the batch, if appending a candidate token would form an n-gram
    that has already appeared in the generated sequence, then set its logit to -∞.
    
    For each batch element, we take the last (n-1) tokens as context. Then we search
    over the previously generated tokens for any occurrence of that (n-1)-gram. For every
    occurrence, we add the token that followed that occurrence to a banned set.
    """
    logits = logits.clone()
    B = logits.size(0)
    for i in range(B):
        seq = generated_seq[i]
        # Not enough tokens to form an n-gram
        if seq.size(0) < n - 1:
            continue
        # The last (n-1) tokens
        context = tuple(seq[-(n - 1):].tolist())
        banned_tokens = set()
        # Search the entire sequence for the same (n-1)-gram
        for start_idx in range(seq.size(0) - (n - 1)):
            window = seq[start_idx:start_idx + (n - 1)]
            if tuple(window.tolist()) == context:
                # The token that followed this (n-1)-gram is at start_idx + (n-1)
                # but be sure we don't run off the end
                if start_idx + (n - 1) < seq.size(0):
                    banned_tokens.add(seq[start_idx + (n - 1)].item())
        # Block all banned tokens
        for token in banned_tokens:
            logits[i, token] = float('-inf')
    return logits







class IFTransformer2(nn.Module):
    def __init__(self, gpt_model: 'GPT', config):
        """
        Interleaves the pretrained GPT blocks with new cross-attention blocks.
        Assumes the pretrained GPT model has 32 transformer blocks.
        The new transformer stack will have 32 + 4 = 36 layers.
        The cross-attention blocks will be inserted at positions:
            after block 8, 16, 24, and 32.
        """
        super().__init__()
        gpt_blocks = gpt_model.transformer  # Pretrained GPT blocks (ModuleList of length 32)
        assert len(gpt_blocks) == 32, "Expected 32 GPT blocks"
        self.layers = nn.ModuleList()
        # Group 1: GPT blocks 0 to 4 (5 blocks)
        for i in range(5):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention block #1
        self.layers.append(cBlock(config))
        
        # Group 2: GPT blocks 5 to 9 (5 blocks)
        for i in range(5, 10):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention block #2
        self.layers.append(cBlock(config))
        
        # Group 3: GPT blocks 10 to 14 (5 blocks)
        for i in range(10, 15):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention block #3
        self.layers.append(cBlock(config))
        
        # Group 4: GPT blocks 15 to 19 (5 blocks)
        for i in range(15, 20):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention block #4
        self.layers.append(cBlock(config))
        
        # Group 5: GPT blocks 20 to 24 (5 blocks)
        for i in range(20, 25):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention block #5
        self.layers.append(cBlock(config))
        
        # Group 6: GPT blocks 25 to 29 (5 blocks)
        for i in range(25, 30):
            self.layers.append(gpt_blocks[i])
        # Insert cross-attention block #6
        self.layers.append(cBlock(config))
        
        # Group 7: Remaining GPT blocks 30 to 31 (2 blocks)
        for i in range(30, 32):
            self.layers.append(gpt_blocks[i])
        
        # Record the positions where we inserted cross-attention blocks
        self.cross_attn_positions = [5, 11, 17, 23, 29, 35]
        self.ln = gpt_model.ln  # Use final layer norm from GPT

    def forward(self, x: torch.Tensor, img_latents: List[torch.Tensor]) -> torch.Tensor:
        """
        :param x: Sequence representation (B, T, n_embd)
        :param img_latents: List of 4 image representations (conditioning tokens),
                            each of shape (B, n_embd, H, W).
        """
        cross_idx = 0
        for i, layer in enumerate(self.layers):
            if i in self.cross_attn_positions:
                # Insert cross-attention: add output of cross-attn (conditioned on the corresponding image latent)
                y = img_latents[cross_idx]
                x = layer(x, y)
                cross_idx += 1
            else:
                x = layer(x)
        x = self.ln(x)
        return x

# ----------------------------
# Complete IFModel Definition
# ----------------------------
class IFModel2(nn.Module):
    def __init__(self, gpt_model: 'GPT', encodr_model: 'Encoder', config: GPT_Config):
        """
        Inverse Folding Model that integrates:
          - A pretrained VQ-VAE encoder (to convert protein structure backbone into latent tokens).
          - A pretrained GPT model (for protein sequence generation).
          - A modified transformer stack that interleaves GPT blocks with cross-attention blocks.
        
        We use the VQ-VAE encoder outputs from levels 2, 3, 4, 5 (ignoring the first two).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.gpt = gpt_model       # Pretrained GPT (with its embedding, transformer, etc.)
        self.encoder = encodr_model   # Pretrained Encoder model; we use only its encoder.
        # We assume vqvae.encode(x) returns a list of encoder outputs.
        self.encoder_levels_to_use = [0, 1, 2, 3, 4, 5]  # Use these 4 levels as conditioning.
        
        # Build the new transformer stack that interleaves GPT blocks with cross-attention blocks.
        self.if_transformer = IFTransformer2(gpt_model, config)

        print0("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
        
    
    def forward(self, seq: torch.Tensor, img: torch.Tensor, targets = None) -> torch.Tensor:
        """
        :param seq: Protein sequence tokens (B, T)
        :param img: Protein structure backbone image (B, 3, H, W)
        :return: Logits over vocabulary (B, T, vocab_size)
        """
        # Get image latent representations from VQ-VAE encoder.
        encoder_outputs = self.encoder(img)
        # Select only levels 2,3,4,5.
        img_latents = [encoder_outputs[i] for i in self.encoder_levels_to_use]
        # Get sequence embeddings.
        x = self.gpt.seq_embedding(seq)  # (B, T, n_embd)
        # Run through interleaved transformer.

        x = self.if_transformer(x, img_latents)
        if exists(targets):
            logits = self.gpt.project(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),   
                ignore_index=0
            )
        else:
            logits = self.gpt.project(x[:, [-1], :])
            loss = None
    
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print0(f"using fused AdamW: {use_fused}")

        return optimizer
    

    @torch.inference_mode()     
    def generate_IF(self, prefix, max_size, img, temperature=1.0, top_k=7, 
                rep_penalty=5, ngram_block=4):
        generated = prefix.clone()
        tokens_to_generate = max_size - prefix.size(1)
        if tokens_to_generate.item() < 0:
            tokens_to_generate = 2
            #raise ValueError(f"Desired size {max_size} <= prefix length {prefix.size(1)}.")
        
        for _ in range(tokens_to_generate):
            logits, _ = self.forward(seq = generated, img = img)
            # Take the last-step logits
            next_token_logits = logits[:, -1, :]  # shape [B, vocab_size]
            
            # 1) Apply repetition penalty
            next_token_logits = adjust_logits_for_repetition(next_token_logits, generated, 
                                                            rep_penalty=rep_penalty)
            # 2) Apply n-gram blocking
            next_token_logits = adjust_logits_for_ngram_blocking(next_token_logits, generated, 
                                                                n=ngram_block)
            
            # Now sample from the adjusted logits
            next_token = self._sample_next_token(next_token_logits, temperature, top_k)
            generated = torch.cat([generated, next_token], dim=1)
            
            # If we hit the EOS token, break
            if next_token.item() == self.tokenizer.eos_id:
                break
        
        return generated
    
    def _sample_next_token(self, logits, temperature=1.0, top_k=7):
        logits = logits / temperature
        if top_k > 0:
            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)
            indices = torch.multinomial(probs, num_samples=1)
            next_tokens = top_indices.gather(-1, indices)
        else:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
    
        return next_tokens
