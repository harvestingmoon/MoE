import torch
import torch.nn as nn 

class FeedForward(nn.Module):
    """
                  x (input from attention)
                  |
          +-------+-------+
          |               |
      Linear_gate       Linear_up
          |               |
     gate_proj         up_proj
          |               |
      Swish(gate_proj)    |
          |               |
          +-------*-------+
                  |
             gated_output
                  |
             Linear_down
                  |
            final_output (to next layer)

    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self,x):
        # SwiGLU function
        # SwiGLU(x) = Swish(x) * GLU(x)
        swiglu =  nn.functional.silu(self.fc1(x)) * self.fc2(x)
        return self.fc3(swiglu)
    


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None, dtype=torch.float32):
    """
    Pre-computes cosine and sine components for RoPE.

    Generates rotational frequencies and applies Llama 3's specific
    frequency scaling (if `freq_config` is provided) for context extension.

    Args:
        head_dim (int): Dimension of each attention head.
        theta_base (float): Base frequency for RoPE (e.g., 10,000 for Llama).
        context_length (int): Maximum sequence length.
        freq_config (dict, optional): Llama 3's RoPE scaling config for context extension.
        dtype (torch.dtype): Data type for computations.

    Returns:
        tuple: (cos, sin) tensors, shape (context_length, head_dim).
    
        

    Workflow:
    1.  Inverse Frequencies (`inv_freq`): Creates diverse frequencies for different head dimensions.
    2.  Frequency Adjustments (Llama 3 Scaling): Adjusts frequencies for extended context,
        applying global scaling and smooth transitions.
    3.  Angles: Computes angles based on positions and frequencies.
    4.  Sine/Cosine: Pre-calculates `sin` and `cos` for rotational transformation.


    """

    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Frequency adjustments
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def apply_rope(x, cos, sin):

    """
    Applies Rotary Positional Embeddings to a tensor (Query or Key).

    Performs the core RoPE rotation using pre-computed `cos` and `sin` components.

    Args:
        x (torch.Tensor): Input tensor (Query or Key). Shape: (B, H, S, D).
        cos (torch.Tensor): Pre-computed cosine components. Shape: (max_S, D).
        sin (torch.Tensor): Pre-computed sine components. Shape: (max_S, D).

    Returns:
        torch.Tensor: Tensor `x` with RoPE applied. Same shape as `x`.
    
    Workflow:
    1.  Split `x`: Divides `x` into two halves (`x1`, `x2`) along `head_dim` for 2D rotation.
    2.  Adjust `cos`/`sin`: Slices and expands `cos`/`sin` to match `x`'s shape for broadcasting.
    3.  Rotation: Applies the RoPE formula: `(x * cos) + (rotate_half(x) * sin)`.
    4.  Precision: Ensures output dtype matches input.
    """


    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)


def rescale_theta(theta_old, context_length_old, context_length_new):
    """
    Rescales the RoPE `theta_base` for context length extension.

    Linearly scales `theta_base` to adapt RoPE for new, longer context lengths.

    Args:
        theta_old (float): Original `theta_base`.
        context_length_old (int): Original max context length.
        context_length_new (int): New desired max context length.

    Returns:
        float: Rescaled `theta_base`.
    """

    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new


""" Attention Architecture (GQA with RoPe)

 Key:
(b, S, D) = (batch_size, num_tokens, dimension)
  H         = num_heads (for Queries)
   KV_G      = num_kv_groups (for Keys/Values in GQA)
   h_dim     = head_dim (D / H)
  Group_Size= H / KV_G

                          x (Input)
                (b, num_tokens, d_in)
                              |
                              V
      +-------------------------------------------------------------+
      |                  Linear Projections (Q, K, V)               |
      |   queries = W_query(x)                                      |
      |   keys    = W_key(x)                                        |
      |   values  = W_value(x)                                      |
      +-------------------------------------------------------------+
               |                     |                     |
               V                     V                     V
      +-------------------------------------------------------------+
      |            Reshape & Transpose for Heads                    |
      |                                                             |
      |   queries.view(b, S, H, h_dim).transpose(1,2)               |
      |   keys.view(b, S, KV_G, h_dim).transpose(1,2)               |
      |   values.view(b, S, KV_G, h_dim).transpose(1,2)             |
      +-------------------------------------------------------------+
               |                     |                     |
               V                     V                     V
        queries_raw             keys_raw             values_raw
       (b, H, S, h_dim)      (b, KV_G, S, h_dim)   (b, KV_G, S, h_dim)
               |                     |                     |
               |       +-----------------------------------+
               |       |                                   |
               |       V                                   V
               |    +---------------------------------------------+
               |    |  Apply RoPE (Rotary Positional Embeddings)  |
               |    |  (using pre-computed cos and sin)           |
               |    +---------------------------------------------+
               |         |                             |
               V         V                             |
        queries_rotated  keys_rotated                  |
       (b, H, S, h_dim) (b, KV_G, S, h_dim)            |
               |         |                             |
               |         +-----------------------------+
               |         |                             |
               |         V                             V
               |    +---------------------------------------+
               |    |    Repeat K/V Heads for GQA           |
               |    |    (keys_rotated.repeat_interleave)   |
               |    |    (values_raw.repeat_interleave)     |
               |    +---------------------------------------+
               |             |                             |
               V             V                             V
        queries_rotated   keys_repeated            values_repeated
       (b, H, S, h_dim)  (b, H, S, h_dim)        (b, H, S, h_dim)
               |             |                             |
               +-------------+-----------------------------+
                             |
                             V
      +-------------------------------------------------------------+
      |                Attention Score Calculation                  |
      |                                                             |
      |  attn_scores = queries_rotated @ keys_repeated.transpose(2,3)|
      |  attn_scores.masked_fill(mask, -inf)                        |
      |  attn_weights = softmax(attn_scores / sqrt(h_dim))          |
      +-------------------------------------------------------------+
                             |
                             V
                        attn_weights
                     (b, H, S, S)
                             |
                             V
      +-------------------------------------------------------------+
      |              Weighted Sum of Values                         |
      |                                                             |
      |  context_vec_per_head = attn_weights @ values_repeated      |
      +-------------------------------------------------------------+
                             |
                             V
                      context_vec_per_head
                     (b, H, S, h_dim)
                             |
                             V
      +-------------------------------------------------------------+
      |            Combine Heads & Output Projection                |
      |                                                             |
      |  context_vec.transpose(1, 2).reshape(b, S, d_out)           |
      |  context_vec = self.out_proj(context_vec)                   |
      +-------------------------------------------------------------+
                             |
                             V
                           Output
                     (b, num_tokens, d_out)
"""


class GroupQueryAttention(nn.Module):
    def __init__(
            self, d_in, d_out, num_heads,
            num_kv_groups,
            dtype = None
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num of heads must be divisible by num of kv groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias = False, dtype= dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias = False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias= False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias = False, dtype = dtype)
    
    def forward(self, x, mask, cos, sin):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim)

        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        keys = apply_rope(keys, cos, sin)
        queries = apply_rope(queries, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim = 1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2,3)
        attn_scores = attn_scores.masked_fill(mask[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        assert keys.shape[-1] == self.head_dim
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

"""
Our actual transformer block, after combining all of the features together


Data Flow:

                    x (Input to Transformer Block)
                           |
                           | (Initial shortcut for 1st residual)
                           V
              +---------------------------+
              |        self.norm1         |  (RMSNorm before Attention)
              +---------------------------+
                           |
                           V
              +---------------------------+
              |        self.attn          |  (Attention mechanism w/ mask, cos, sin)
              +---------------------------+
                           |
        +------------------+------------------+
        |                                     |
        V                                     V
   (Output of attn)                  (Original input 'x')
        +-------------------------------------+
        |          x + shortcut_attn          |  (1st Residual Connection) (the routing portion)
        +-------------------------------------+
                           |
                           | (Shortcut for 2nd residual)
                           V
              +---------------------------+
              |        self.norm2         |  (RMSNorm before FFN)
              +---------------------------+
                           |
                           V
              +---------------------------+
              |        self.ffn           |  (FeedForward Network)
              +---------------------------+
                           |
        +------------------+------------------+
        |                                     |
        V                                     V
   (Output of ffn)                  (Input after 1st residual)
        +-------------------------------------+
        |          x + shortcut_ffn           |    (2nd Residual Connection)
        +-------------------------------------+
                           |
                           V
                Output of Transformer Block

"""



class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GroupQueryAttention( 
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            num_heads= cfg["n_heads"],
            num_kv_groups= cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ffn = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, mask, cos, sin)
        x = x + shortcut

        return x



""" Actual Llama Model 

Key:
  in_idx      = Input tensor (token IDs)
  tok_emb     = Token Embedding Layer
  x           = Intermediate tensor representing activations
  trf_blocks  = List of Transformer Blocks (n_layers times)
  final_norm  = Final RMSNorm layer
  out_head    = Linear Output Head
  mask, cos, sin = Pre-computed buffers for attention/RoPE (from self.register_buffer)
  ->          = Data flow / Transformation


Data Flow:

                     in_idx (Input Token IDs)
                           |
                           V
              +---------------------------+
              |       self.tok_emb        |  (Token Embedding Layer)
              +---------------------------+
                           |
                           V
                           x
              (batch_size, num_tokens, emb_dim)
                           |
                           V
              +---------------------------+
              |    Loop through           |
              | self.trf_blocks (n_layers)|
              |          [Start Loop]     |
              +---------------------------+
                           |
                           V
          +--------------------------------------------+
          |  [Inside each Transformer Block iteration] |
          |                                            |
          |       x (Input to current block)           |
          |               |                            |
          |               V                            |
          |     +---------------------------+          |
          |     |  block(x, self.mask,      |          |
          |     |          self.cos,        |          |
          |     |          self.sin)        |          |
          |     |(Calls Transformer.forward)|          |
          |     +---------------------------+          |
          |               |                            |
          |               V                            |
          |       Output of current block              |
          +--------------------------------------------+
                           |
                           V
              +---------------------------+
              |          [End Loop]       |
              |                           |  (x is now output of last block)
              +---------------------------+
                           |
                           V
              +---------------------------+
              |      self.final_norm      |  (Final RMSNorm)
              +---------------------------+
                           |
                           V
              +---------------------------+
              |        self.out_head      |  (Linear Output Head)
              +---------------------------+
                           |
                           V
                        logits
              (batch_size, num_tokens, vocab_size)

"""

class Llama3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype= cfg["dtype"])

        self.trf_blocks = nn.ModuleList(
            [Transformer(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps= 1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]),diagonal=1).bool(),
            persistent=False
        )
        cfg["rope_base"] = rescale_theta(
            cfg["rope_base"],
            cfg["orig_context_length"],
            cfg["context_length"]
        )

        # this is for RoPe
        cos,sin = compute_rope_params(
            head_dim = cfg["emb_dim"] // cfg["n_heads"],
            theta_base= cfg["rope_base"],
            context_length = cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.cfg = cfg 
    
    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)
        
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits