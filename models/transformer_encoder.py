import math
import torch
import torch.nn as nn 
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, encoding_type="sinusoidal"):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type

        if encoding_type == "sinusoidal":
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        elif encoding_type == "learned":
            # learned positional embeddings
            self.pe = nn.Parameter(torch.randn(max_len, 1, d_model))
            nn.init.xavier_uniform_(self.pe)

        else:
            raise ValueError(f"Unknown positional encoding type: {encoding_type}")

    def forward(self, x):
        if self.encoding_type == "sinusoidal":
            return x + self.pe[:x.size(0), :]
        elif self.encoding_type == "learned":
            # for batched input with shape [seq_len, batch_size, d_model]
            return x + self.pe[:x.size(0), :].expand(-1, x.size(1), -1)
        else:
            return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        
        # For attention visualization
        self.attn = None

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear transformation
        query = self.q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        # Output of attention
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.fc(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Self-attention layer
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=mask)
        src = src + self.dropout(attn_output)  # Use only the attention output
        src = self.layernorm1(src)

        # Feedforward layer
        src2 = self.ffn(src)
        src = src + self.dropout(src2)
        src = self.layernorm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


class VpSatNetTransformer(nn.Module):
    def __init__(self, C):
        super(VpSatNetTransformer, self).__init__()

        self.positional_encoding = PositionalEncoding(C.d_model, max_len=5000)
        self.transformer_encoder = TransformerEncoder(
            num_layers=C.num_layers,
            d_model=C.d_model,
            num_heads=C.num_heads,
            dim_feedforward=C.dim_feedforward,
        )
    
    def forward(self, patches, line_segments=None):
        # Apply positional encoding to both patches and line segments
        patches = self.positional_encoding(patches)

        if line_segments is not None:
            line_segments = self.positional_encoding(line_segments)

            # Concatenate patches and line segments
            src = torch.cat((patches, line_segments), dim=1)

            # Apply transformer encoder
            output = self.transformer_encoder(src)
        
        else:
            output = self.transformer_encoder(patches)

        return output


class SimpleVPHead(nn.Module):
    def __init__(self, d_model, num_vpts=3, output_dim=2):
        super(SimpleVPHead, self).__init__()
        self.num_vpts = num_vpts
        self.output_dim = output_dim

        # simple linear projection
        self.projection = nn.Linear(d_model, num_vpts * output_dim)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.mean(dim=1)

        output = self.projection(x)
        output = output.view(-1, self.num_vpts, self.output_dim)
        return output


class VanishingPointPredictionHead(nn.Module):
    def __init__(self, d_model, num_vpts=1, output_dim=2):
        super(VanishingPointPredictionHead, self).__init__()
        self.num_vpts = num_vpts
        self.output_dim = output_dim

        # Attention-based pooling layer
        self.attention = nn.Linear(d_model, 1)
        
        # MLP for richer feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_vpts * output_dim)
        )

    def forward(self, x):
        # Compute attention scores
        attn_weights = F.softmax(self.attention(x), dim=1)  # Shape: [batch_size, seq_len, 1]
        x = (x * attn_weights).sum(dim=1)  # Weighted sum (attention pooling)
        
        # Pass through MLP for final VP prediction
        vp_predictions = self.mlp(x)
        vp_predictions = vp_predictions.view(-1, self.num_vpts, self.output_dim)
        
        return vp_predictions