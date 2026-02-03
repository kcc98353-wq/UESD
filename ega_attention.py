
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    """Layer normalization module."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute scaled dot product attention.
    
    Args:
        query: [batch, heads, seq_len, d_k]
        key: [batch, heads, seq_len, d_k]
        value: [batch, heads, seq_len, d_k]
        mask: [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
        dropout: dropout layer
    
    Returns:
        output: [batch, heads, seq_len, d_k]
        attn_weights: [batch, heads, seq_len, seq_len]
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attn_weights = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadedAttention(nn.Module):
    """
    Standard Multi-Head Attention module.
    
    Args:
        num_heads: number of attention heads
        d_model: dimension of the model
        dropout: dropout rate
    """
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights = None
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch, seq_len, d_model] or [batch, 1, d_model]
            key: [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask: [batch, seq_len] or [batch, seq_len, seq_len]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)  # [batch, 1, 1, seq_len]
        
        batch_size = query.size(0)
        
        # Linear projections
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, self.attn_weights = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # Final linear projection
        return self.linear_out(x)


class EGAAttention(nn.Module):
    """
    Exponential Gain Attention (EGA) mechanism.
    
    This mechanism performs:
    1. Standard multi-head attention to get A_i
    2. Decompose into Content (C_i) and Confidence Gate (g)
    3. Apply exponential gain: G_i = (e^g + 1) * a
    4. Final output: A_i^ega = G_i ⊙ C_i
    
    Args:
        num_heads: number of attention heads
        d_model: dimension of the model
        dropout: dropout rate for attention weights
        suppress_factor:防抑制系数 a，用于防止特征过度增强
        
    Reference:
        Based on the EGA mechanism described in the provided document
    """
    def __init__(self, num_heads, d_model, dropout=0.1, suppress_factor=0.1):
        super(EGAAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.suppress_factor = suppress_factor  # 防抑制系数 a
        
        # Linear projections for Q, K, V (公式1)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        # Content representation C_i (公式4)
        # C_i = W_q^C * Q_i + W_A^C * A_i + b^c
        self.W_q_C = nn.Linear(self.d_k, self.d_k, bias=False)
        self.W_A_C = nn.Linear(self.d_k, self.d_k, bias=False)
        self.b_C = nn.Parameter(torch.zeros(self.d_k))
        
        # Confidence gate g (公式5)
        # g = σ(W_q^g * Q_i + W_A^g * A_i + b^g)
        self.W_q_g = nn.Linear(self.d_k, self.d_k, bias=False)
        self.W_A_g = nn.Linear(self.d_k, self.d_k, bias=False)
        self.b_g = nn.Parameter(torch.zeros(self.d_k))
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(d_model)
        self.attn_weights = None
        
    def forward(self, query, key, value, mask=None):

        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)  # [batch, 1, 1, seq_len]
        
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        
        # Normalize query
        query = self.norm(query)
        

        Q = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        A, self.attn_weights = attention(Q, K, V, mask=mask, dropout=self.dropout)

        A_ega_heads = []
        
        for i in range(self.num_heads):

            Q_i = Q[:, i, :, :]  # [batch, seq_len_q, d_k]
            A_i = A[:, i, :, :]  # [batch, seq_len_q, d_k]
            

            C_i = self.W_q_C(Q_i) + self.W_A_C(A_i) + self.b_C

            g = torch.sigmoid(self.W_q_g(Q_i) + self.W_A_g(A_i) + self.b_g)

            G_i = (torch.exp(g) + 1) * self.suppress_factor

            A_i_ega = G_i * C_i

            
            A_ega_heads.append(A_i_ega)
        

        A_ega = torch.stack(A_ega_heads, dim=1)  # [batch, heads, seq_len_q, d_k]
        A_ega = A_ega.transpose(1, 2).contiguous()  # [batch, seq_len_q, heads, d_k]
        A_ega = A_ega.view(batch_size, seq_len_q, self.d_model)  # [batch, seq_len_q, d_model]
        
        # 最终线性投影 W^O
        output = self.output_proj(A_ega)
        
        return output


class EGABlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1, suppress_factor=0.1):
        super(EGABlock, self).__init__()
        
        self.ega_attention = EGAAttention(num_heads, d_model, dropout, suppress_factor)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):

        residual = query
        x = self.ega_attention(query, key, value, mask)
        x = self.norm1(residual + x)
        
        # Feed-forward with residual connection
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)
        
        return x


# 保留原始的AoA实现作为对比
class AoAAttention(nn.Module):

    def __init__(self, num_heads, d_model, dropout=0.1, dropout_aoa=0.3):
        super(AoAAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        self.aoa_layer = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GLU(dim=-1)
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_aoa = nn.Dropout(p=dropout_aoa)
        
        self.norm = LayerNorm(d_model)
        self.attn_weights = None
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(-2)
            mask = mask.unsqueeze(1)
        
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        
        query_original = query
        query = self.norm(query)
        
        query_proj = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key_proj = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_proj = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        x, self.attn_weights = attention(query_proj, key_proj, value_proj, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        aoa_input = torch.cat([x, query_original], dim=-1)
        aoa_input = self.dropout_aoa(aoa_input)
        output = self.aoa_layer(aoa_input)
        
        return output


class AoABlock(nn.Module):
    """Original AoA block (for comparison)."""
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1, dropout_aoa=0.3):
        super(AoABlock, self).__init__()
        
        self.aoa_attention = AoAAttention(num_heads, d_model, dropout, dropout_aoa)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        residual = query
        x = self.aoa_attention(query, key, value, mask)
        x = self.norm1(residual + x)
        
        residual = x
        x = self.ffn(x)
        x = self.norm2(residual + x)
        
        return x



if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "EGA - Exponential Gain Attention Mechanism" + " " * 16 + "║")
    print("║" + " " * 15 + "指数增益注意力机制 - 独立模块" + " " * 18 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    example_usage()

