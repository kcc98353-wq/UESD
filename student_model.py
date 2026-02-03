import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import copy

from ega_attention import EGAAttention
from fairseq.models.wav2vec import ConvFeatureExtractionModel
from fairseq.modules import SamePad, TransposeLast


class EGAAltAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attn_drop: float = 0.0,
        suppress_factor: float = 0.1,
        init_from_alt_attn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim {self.dim} must be divisible by num_heads {self.num_heads}")
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.ega = EGAAttention(
            num_heads=self.num_heads,
            d_model=self.dim,
            dropout=float(attn_drop),
            suppress_factor=float(suppress_factor),
        )

        if init_from_alt_attn is not None and hasattr(init_from_alt_attn, "qkv") and hasattr(init_from_alt_attn, "proj"):
            with torch.no_grad():
                qkv_w = init_from_alt_attn.qkv.weight
                self.ega.linear_q.weight.copy_(qkv_w[: self.dim])
                self.ega.linear_k.weight.copy_(qkv_w[self.dim : 2 * self.dim])
                self.ega.linear_v.weight.copy_(qkv_w[2 * self.dim : 3 * self.dim])
                if init_from_alt_attn.qkv.bias is not None:
                    qkv_b = init_from_alt_attn.qkv.bias
                    self.ega.linear_q.bias.copy_(qkv_b[: self.dim])
                    self.ega.linear_k.bias.copy_(qkv_b[self.dim : 2 * self.dim])
                    self.ega.linear_v.bias.copy_(qkv_b[2 * self.dim : 3 * self.dim])
                self.ega.output_proj.weight.copy_(init_from_alt_attn.proj.weight)
                if init_from_alt_attn.proj.bias is not None:
                    self.ega.output_proj.bias.copy_(init_from_alt_attn.proj.bias)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        B, N, C = x.shape
        H = self.num_heads
        D = self.head_dim

        q = self.ega.linear_q(x).view(B, N, H, D).transpose(1, 2)
        k = self.ega.linear_k(x).view(B, N, H, D).transpose(1, 2)
        v = self.ega.linear_v(x).view(B, N, H, D).transpose(1, 2)

        dtype = q.dtype
        scores = (q * self.scale) @ k.transpose(-2, -1)

        if alibi_bias is not None:
            scores = scores.type_as(alibi_bias)
            scores[:, : alibi_bias.size(1)] = scores[:, : alibi_bias.size(1)] + alibi_bias

        if padding_mask is not None and padding_mask.any():
            scores = scores.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.ega.dropout(attn)
        A = attn @ v

        q2 = q.reshape(B * H, N, D)
        A2 = A.reshape(B * H, N, D)

        C2 = self.ega.W_q_C(q2) + self.ega.W_A_C(A2) + self.ega.b_C
        g = torch.sigmoid(self.ega.W_q_g(q2) + self.ega.W_A_g(A2) + self.ega.b_g)
        G = (torch.exp(g) + 1.0) * float(self.ega.suppress_factor)
        Aega = (G * C2).view(B, H, N, D)

        out = Aega.transpose(1, 2).contiguous().view(B, N, C)
        out = self.ega.output_proj(out)
        return out


class NoiseRobustConvEncoder(nn.Module):
    """噪声鲁棒的卷积编码器"""
    
    def __init__(self, input_dim=1, embed_dim=512):
        super().__init__()
        
        # 多尺度卷积特征提取 - 针对噪声环境优化
        self.conv_layers = nn.ModuleList([
            # 第一层：大kernel捕获长时依赖，对噪声更鲁棒
            nn.Sequential(
                nn.Conv1d(input_dim, 128, kernel_size=15, stride=5, padding=7),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            # 第二层：中等kernel
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=7, stride=3, padding=3),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            # 第三层：小kernel精细特征
            nn.Sequential(
                nn.Conv1d(256, 384, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(384),
                nn.GELU(),
                nn.Dropout(0.1)
            ),
            # 第四层：进一步压缩
            nn.Sequential(
                nn.Conv1d(384, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        ])
        
        # 噪声自适应模块
        self.noise_adaptation = NoiseAdaptationModule(embed_dim)
        
        # 特征投影
        self.feature_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T] 原始音频波形
        Returns:
            features: [B, T', embed_dim] 提取的特征
        """
        # 添加通道维度
        x = x.unsqueeze(1)  # [B, 1, T]
        
        # 逐层卷积特征提取
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # 转置为 [B, T', embed_dim]
        x = x.transpose(1, 2)
        
        # 噪声自适应
        x = self.noise_adaptation(x)
        
        # 特征投影
        x = self.feature_projection(x)
        
        return x


class NoiseAdaptationModule(nn.Module):
    """噪声自适应模块 - 学习噪声特征并进行补偿"""
    
    def __init__(self, embed_dim):
        super().__init__()
        
        # 噪声检测分支
        self.noise_detector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )
        
        # 特征增强分支
        self.feature_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """
        Args:
            x: [B, T, embed_dim]
        Returns:
            enhanced_x: [B, T, embed_dim]
        """
        # 噪声权重估计
        noise_weights = self.noise_detector(x)
        
        # 特征增强
        enhanced_features = self.feature_enhancer(x)
        
        # 自适应融合
        output = x * self.residual_weight + enhanced_features * noise_weights
        
        return output


class LightweightTransformerBlock(nn.Module):
    """轻量化Transformer块 - 使用EGA注意力机制"""
    
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=2.0, dropout=0.1, suppress_factor=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 使用EGA注意力替代标准的MultiheadAttention
        self.attn = EGAAttention(
            num_heads=num_heads,
            d_model=embed_dim,
            dropout=dropout,
            suppress_factor=suppress_factor
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 更小的MLP比例以减少参数
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask=None):
        """
        Args:
            x: [B, T, embed_dim]
            padding_mask: [B, T] True表示padding位置
        Returns:
            x: [B, T, embed_dim]
        """
        # EGA Self-attention
        residual = x
        x = self.norm1(x)
        
        # EGA注意力计算 (query=key=value for self-attention)
        # padding_mask需要转换: True->0 (padding), False->1 (valid)
        mask = None
        if padding_mask is not None:
            mask = ~padding_mask  # 转换为: 0=padding, 1=valid
        
        attn_output = self.attn(x, x, x, mask=mask)
        x = residual + self.dropout(attn_output)
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x


class TeacherSlimFrontend(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        feature_encoder_spec="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        extractor_mode="layer_norm",
        conv_pos_width=95,
        conv_pos_groups=16,
        conv_pos_depth=2,
        dropout=0.0,
    ):
        super().__init__()
        self.feature_enc_layers = eval(feature_encoder_spec)
        feature_embed_dim = self.feature_enc_layers[-1][0]

        self.local_encoder = ConvFeatureExtractionModel(
            conv_layers=self.feature_enc_layers,
            dropout=0.0,
            mode=extractor_mode,
            conv_bias=False,
        )

        self.project_features = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        )

        num_pos_layers = max(1, int(conv_pos_depth))
        k = max(3, int(conv_pos_width) // num_pos_layers)
        self.positional_encoder = nn.Sequential(
            TransposeLast(),
            *[
                nn.Sequential(
                    nn.Conv1d(
                        embed_dim,
                        embed_dim,
                        kernel_size=k,
                        padding=k // 2,
                        groups=int(conv_pos_groups),
                    ),
                    SamePad(k),
                    TransposeLast(),
                    nn.LayerNorm(embed_dim),
                    TransposeLast(),
                    nn.GELU(),
                )
                for _ in range(num_pos_layers)
            ],
            TransposeLast(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features = self.local_encoder(x)
        features = self.project_features(features)
        pos = self.positional_encoder(features)
        features = features + pos
        features = self.dropout(features)
        return features


class TeacherSlimStudentModel(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_transformer_layers=6,
        num_heads=12,
        num_classes=4,
        hidden_dim=256,
        dropout=0.1,
        suppress_factor=0.1,
        feature_encoder_spec="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        conv_pos_depth=2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.frontend = TeacherSlimFrontend(
            embed_dim=embed_dim,
            feature_encoder_spec=feature_encoder_spec,
            conv_pos_depth=conv_pos_depth,
            dropout=0.0,
        )

        self.transformer_layers = nn.ModuleList(
            [
                LightweightTransformerBlock(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=2.0,
                    dropout=dropout,
                    suppress_factor=suppress_factor,
                )
                for _ in range(num_transformer_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        self.pre_net = nn.Linear(embed_dim, hidden_dim)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.post_net = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, padding_mask=None, return_features=False):
        features = self.frontend(x)

        feature_padding_mask = None
        if padding_mask is not None:
            feature_padding_mask = self._compute_feature_padding_mask(
                padding_mask, features.size(1)
            )

        intermediate_features = []
        for layer in self.transformer_layers:
            features = layer(features, feature_padding_mask)
            if return_features:
                intermediate_features.append(features)

        features = self.final_norm(features)

        if feature_padding_mask is not None:
            mask = (~feature_padding_mask.unsqueeze(-1)).float()
            pooled = (features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = features.mean(dim=1)

        pooled = self.activate(self.pre_net(pooled))
        pooled = self.dropout(pooled)
        logits = self.post_net(pooled)

        if return_features:
            return logits, features, intermediate_features
        return logits

    def _compute_feature_padding_mask(self, input_padding_mask, feature_length):
        batch_size = input_padding_mask.size(0)
        input_length = input_padding_mask.size(1)
        downsample_ratio = input_length / feature_length
        feature_padding_mask = torch.zeros(
            batch_size,
            feature_length,
            dtype=torch.bool,
            device=input_padding_mask.device,
        )
        for b in range(batch_size):
            valid_length = (~input_padding_mask[b]).sum().item()
            feature_valid_length = int(valid_length / downsample_ratio)
            if feature_valid_length < feature_length:
                feature_padding_mask[b, feature_valid_length:] = True
        return feature_padding_mask


class StudentModel(nn.Module):
    """学生模型 - 专门用于噪声环境下的语音情感识别"""
    
    def __init__(
        self, 
        embed_dim=512,
        num_transformer_layers=6,  # 比教师模型少
        num_heads=8,
        num_classes=4,
        dropout=0.1,
        suppress_factor=0.1  # EGA注意力的防抑制系数
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 噪声鲁棒的特征提取器
        self.feature_extractor = NoiseRobustConvEncoder(
            input_dim=1, embed_dim=embed_dim
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # 轻量化Transformer层 - 使用EGA注意力
        self.transformer_layers = nn.ModuleList([
            LightweightTransformerBlock(
                embed_dim, num_heads, mlp_ratio=2.0, dropout=dropout,
                suppress_factor=suppress_factor
            )
            for _ in range(num_transformer_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 用于蒸馏的特征对齐层
        self.feature_align = nn.Linear(embed_dim, 768)  # 对齐到教师模型的768维
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, padding_mask=None, return_features=False):
        """
        Args:
            x: [B, T] 噪声语音波形
            padding_mask: [B, T] padding mask
            return_features: 是否返回中间特征用于蒸馏
        Returns:
            logits: [B, num_classes] 分类结果
            features: [B, T', 768] 对齐后的特征 (如果return_features=True)
        """
        # 特征提取
        features = self.feature_extractor(x)  # [B, T', embed_dim]
        
        # 位置编码
        features = self.pos_encoder(features)
        
        # 计算padding mask for features
        if padding_mask is not None:
            # 根据卷积层的下采样计算新的padding mask
            feature_padding_mask = self._compute_feature_padding_mask(
                padding_mask, features.size(1)
            )
        else:
            feature_padding_mask = None
        
        # Transformer层
        intermediate_features = []
        for layer in self.transformer_layers:
            features = layer(features, feature_padding_mask)
            if return_features:
                intermediate_features.append(features)
        
        # 最终归一化
        features = self.final_norm(features)
        
        # 全局平均池化 (忽略padding部分)
        if feature_padding_mask is not None:
            mask = ~feature_padding_mask.unsqueeze(-1)  # [B, T', 1]
            pooled_features = (features * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled_features = features.mean(dim=1)  # [B, embed_dim]
        
        # 分类
        logits = self.classifier(pooled_features)
        
        if return_features:
            # 特征对齐到教师模型维度
            aligned_features = self.feature_align(features)  # [B, T', 768]
            return logits, aligned_features, intermediate_features
        
        return logits
    
    def _compute_feature_padding_mask(self, input_padding_mask, feature_length):
        """根据输入的padding mask计算特征的padding mask"""
        batch_size = input_padding_mask.size(0)
        input_length = input_padding_mask.size(1)
        
        # 计算下采样比例 (根据卷积层的stride)
        downsample_ratio = input_length / feature_length
        
        # 简单的下采样策略
        feature_padding_mask = torch.zeros(
            batch_size, feature_length, 
            dtype=torch.bool, device=input_padding_mask.device
        )
        
        for b in range(batch_size):
            # 找到有效长度
            valid_length = (~input_padding_mask[b]).sum().item()
            feature_valid_length = int(valid_length / downsample_ratio)
            if feature_valid_length < feature_length:
                feature_padding_mask[b, feature_valid_length:] = True
        
        return feature_padding_mask


class PaperDistilStudent(nn.Module):
    def __init__(
        self,
        teacher_base: nn.Module,
        num_transformer_layers: int,
        num_classes: int = 4,
        dropout: float = 0.0,
        use_ega_attention: bool = False,
        ega_suppress_factor: float = 0.1,
    ):
        super().__init__()
        if teacher_base is None:
            raise ValueError("PaperDistilStudent requires teacher_base")

        self.teacher_embed_dim = getattr(getattr(teacher_base, "cfg", None), "embed_dim", None)
        if self.teacher_embed_dim is None:
            raise ValueError("teacher_base.cfg.embed_dim not found")

        if not hasattr(teacher_base, "modality_encoders") or "AUDIO" not in teacher_base.modality_encoders:
            raise ValueError("teacher_base.modality_encoders['AUDIO'] not found")
        if not hasattr(teacher_base, "blocks"):
            raise ValueError("teacher_base.blocks not found")

        self.frontend = copy.deepcopy(teacher_base.modality_encoders["AUDIO"])
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        teacher_blocks = list(teacher_base.blocks)
        teacher_even_indices = [i for i in range(1, len(teacher_blocks), 2)]
        if not teacher_even_indices:
            raise ValueError("teacher transformer blocks are insufficient for even-layer init")

        num_transformer_layers = int(num_transformer_layers)
        num_layers = min(num_transformer_layers, len(teacher_even_indices))
        blocks = []
        for i in range(num_layers):
            blk = copy.deepcopy(teacher_blocks[teacher_even_indices[i]])
            if use_ega_attention and hasattr(blk, "attn"):
                original_attn = blk.attn
                num_heads = getattr(original_attn, "num_heads", None)
                if num_heads is None:
                    num_heads = int(getattr(getattr(teacher_base, "cfg", None), "num_heads", 12))
                attn_drop = 0.0
                if hasattr(original_attn, "attn_drop") and hasattr(original_attn.attn_drop, "p"):
                    attn_drop = float(original_attn.attn_drop.p)
                blk.attn = EGAAltAttention(
                    dim=self.teacher_embed_dim,
                    num_heads=num_heads,
                    attn_drop=attn_drop,
                    suppress_factor=float(ega_suppress_factor),
                    init_from_alt_attn=original_attn,
                )
            blocks.append(blk)
        self.transformer_layers = nn.ModuleList(blocks)

        self.norm = None
        if hasattr(teacher_base, "norm") and teacher_base.norm is not None:
            self.norm = copy.deepcopy(teacher_base.norm)

        self.classifier = nn.Linear(self.teacher_embed_dim, num_classes)

    def forward(self, x, padding_mask=None, return_features=False):
        extractor_out = self.frontend(
            x,
            padding_mask,
            mask=False,
            remove_masked=False,
            clone_batch=1,
            mask_seeds=None,
            precomputed_mask=None,
        )
        features = extractor_out["x"]
        feat_padding_mask = extractor_out.get("padding_mask", None)
        alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        intermediate_features = []
        for i, blk in enumerate(self.transformer_layers):
            ab = alibi_bias
            if ab is not None and alibi_scale is not None:
                scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                ab = ab * scale.type_as(ab)
            features, _ = blk(features, padding_mask=feat_padding_mask, alibi_bias=ab)
            if return_features:
                intermediate_features.append(features)

        if self.norm is not None:
            features = self.norm(features)

        num_extra = int(getattr(getattr(self.frontend, "modality_cfg", None), "num_extra_tokens", 0) or 0)
        if num_extra > 0:
            features = features[:, num_extra:, :]
            if feat_padding_mask is not None:
                feat_padding_mask = feat_padding_mask[:, num_extra:]

        if feat_padding_mask is not None:
            mask = (~feat_padding_mask).float()
            pooled = (features * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = features.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        if return_features:
            return logits, features, intermediate_features
        return logits


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def create_student_model(
    embed_dim=512,
    num_layers=4,
    num_heads=8,
    num_classes=4,
    dropout=0.1,
    suppress_factor=0.1,
    arch="noise_robust",
    teacher_base=None,
    use_ega_attention=False,
    ega_suppress_factor=0.1,
):
    """创建学生模型的工厂函数"""
    if arch == "paper_distil":
        return PaperDistilStudent(
            teacher_base=teacher_base,
            num_transformer_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            use_ega_attention=use_ega_attention,
            ega_suppress_factor=ega_suppress_factor,
        )
    if arch == "teacher_slim":
        return TeacherSlimStudentModel(
            embed_dim=embed_dim,
            num_transformer_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
            suppress_factor=suppress_factor,
        )
    return StudentModel(
        embed_dim=embed_dim,
        num_transformer_layers=num_layers,
        num_heads=num_heads,
        num_classes=num_classes,
        dropout=dropout,
        suppress_factor=suppress_factor
    )


if __name__ == "__main__":
    # 测试模型
    model = create_student_model()
    
    # 模拟输入
    batch_size = 2
    seq_length = 16000  # 1秒的16kHz音频
    x = torch.randn(batch_size, seq_length)
    
    # 前向传播
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # 带特征返回的前向传播
    logits, features, intermediate = model(x, return_features=True)
    print(f"Logits shape: {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Number of intermediate features: {len(intermediate)}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
