"""
Multimodal Emotion Recognition v4 - Model Definitions

Architecture:
  audio (wav)  -> SERModel  (WavLM-Base+ fine-tuned) -> 768d
  clip  (T,C,H,W) -> FERModel  (vit-face-expression + Temporal Transformer + Weighted Vote) -> 768d
                 -> GatedFusion (Deep MLP) -> 512d -> Classifier -> 4 classes
"""

import torch
import torch.nn as nn
from transformers import WavLMModel, WavLMConfig, AutoConfig, AutoModelForImageClassification

# =========================================================
# SER (Speech Emotion Recognition) backbone 
# =========================================================
class SpeakerNorm(nn.Module):
    """Per-utterance InstanceNorm over the time axis."""
    def __init__(self, dim: int = 768, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.InstanceNorm1d(dim, eps=eps, affine=True)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class WeightedLayerFusion(nn.Module):
    """Learnable softmax fusion over the top K transformer hidden layers."""
    def __init__(self, k: int = 12):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.zeros(k))

    def forward(self, hidden_states):
        chosen = hidden_states[-self.k:]
        w = torch.softmax(self.weight, dim=0)
        stacked = torch.stack(chosen, dim=0)
        return (stacked * w[:, None, None, None]).sum(0)


class AttentionPooling(nn.Module):
    """Single learned query attending over time frames."""
    def __init__(self, hidden_size: int = 768, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, batch_first=True, dropout=0.1
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

    def forward(self, x):
        q = self.query.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)


class SERModel(nn.Module):
    def __init__(self, n_class: int = 4,
                 backbone_name: str = "microsoft/wavlm-base-plus",
                 load_pretrained: bool = True):
        super().__init__()
        cfg = WavLMConfig.from_pretrained(backbone_name, output_hidden_states=True)

        if load_pretrained:
            self.backbone = WavLMModel.from_pretrained(
                backbone_name, config=cfg, use_safetensors=True
            )
        else:
            self.backbone = WavLMModel(cfg)

        self.fuse = WeightedLayerFusion(k=12)
        self.spk_norm = SpeakerNorm(dim=768)
        self.attn_pool = AttentionPooling(768, 4)
        self.layer_norm = nn.LayerNorm(768)

        self.emotion_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_class),
        )

    def _embed(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.backbone(wav, output_hidden_states=True)
        h = self.fuse(out.hidden_states)
        h = self.spk_norm(h)
        z = self.attn_pool(h)
        return self.layer_norm(z)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.emotion_head(self._embed(wav))

    def extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        return self._embed(wav)


# =========================================================
# FER Multi-view Aggregation Logic (v4)
# =========================================================
def aggregate_fer_multiview_logits(logits_3d: torch.Tensor, det_scores: torch.Tensor):
    probs = torch.softmax(logits_3d.float(), dim=-1)
    conf_scores = probs.max(dim=-1).values.clamp(min=1e-6)
    det_scores = det_scores.to(logits_3d.device, dtype=probs.dtype).clamp(min=1e-6)

    # Weights combine MediaPipe face score + ViT emotion confidence
    weights = (det_scores ** 1.0) * (conf_scores ** 1.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

    # Weighted vote aggregation
    bsz, _, n_classes = probs.shape
    hard_preds = probs.argmax(dim=-1)
    vote_scores = torch.zeros((bsz, n_classes), device=probs.device, dtype=probs.dtype)
    vote_scores.scatter_add_(1, hard_preds, weights)
    agg_logits = torch.log(vote_scores.clamp(min=1e-8))
    
    return agg_logits, weights


# =========================================================
# FER (Facial Emotion Recognition) backbone (v4)
# =========================================================
class FERModel(nn.Module):
    def __init__(self, n_class: int = 4, n_frames: int = 16,
                 embed_dim: int = 768, img_size: int = 224,
                 dropout: float = 0.4, load_pretrained: bool = True):
        super().__init__()
        self.output_dim = embed_dim
        self.n_frames = n_frames

        # ViT Backbone
        if load_pretrained:
            self.backbone = AutoModelForImageClassification.from_pretrained(
                "trpakov/vit-face-expression", output_hidden_states=True
            )
        else:
            cfg = AutoConfig.from_pretrained("trpakov/vit-face-expression", output_hidden_states=True)
            self.backbone = AutoModelForImageClassification.from_config(cfg)

        # Temporal Transformer Module
        self.temporal_proj = nn.Identity()
        self.temporal_norm = nn.LayerNorm(self.output_dim)
        
        # Positional Embeddings for frames
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max(16, self.n_frames), self.output_dim))
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,
            nhead=8, # FER_TEMPORAL_HEADS
            dim_feedforward=self.output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=1) # FER_TEMPORAL_NUM_LAYERS = 1
        
        # Classification Head
        self.frame_norm = nn.LayerNorm(self.output_dim)
        self.frame_classifier = nn.Linear(self.output_dim, n_class)

    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        feat = outputs.hidden_states[-1][:, 0] # CLS token
        return feat

    def _prepare_inputs(self, x: torch.Tensor, det_scores: torch.Tensor = None):
        if x.dim() == 4:
            x = x.unsqueeze(1)
            
        bsz, n_views, ch, h, w = x.shape
        if det_scores is None:
            det_scores = torch.ones((bsz, n_views), device=x.device, dtype=x.dtype)

        # Flatten frames for ViT
        flat_x = x.view(bsz * n_views, ch, h, w)
        flat_feat = self._extract_frame_features(flat_x)
        
        # Reshape for Temporal sequence
        seq = flat_feat.view(bsz, n_views, -1)
        seq = self.temporal_proj(seq)
        seq = self.temporal_norm(seq)
        seq = seq + self.temporal_pos_embed[:, :n_views, :]
        seq = self.temporal_encoder(seq)
        return seq, det_scores

    def _predict_from_sequence(self, seq: torch.Tensor, det_scores: torch.Tensor):
        frame_logits = self.frame_classifier(self.frame_norm(seq))
        agg_logits, view_weights = aggregate_fer_multiview_logits(frame_logits, det_scores)
        
        # Weight the temporal sequence features dynamically to form the final visual embedding
        pooled_feat = (view_weights.unsqueeze(-1) * seq).sum(dim=1)
        return agg_logits, pooled_feat, view_weights

    def forward(self, x: torch.Tensor, det_scores: torch.Tensor = None):
        seq, det_scores = self._prepare_inputs(x, det_scores=det_scores)
        agg_logits, _, _ = self._predict_from_sequence(seq, det_scores)
        return agg_logits

    def extract_features(self, x: torch.Tensor, det_scores: torch.Tensor = None):
        seq, det_scores = self._prepare_inputs(x, det_scores=det_scores)
        _, pooled_feat, _ = self._predict_from_sequence(seq, det_scores)
        return pooled_feat


# =========================================================
# Gated Fusion (v4 - Deep MLP Gate)
# =========================================================
class GatedFusion(nn.Module):
    def __init__(self, audio_dim: int = 768, visual_dim: int = 768,
                 fusion_dim: int = 512, dropout: float = 0.3, gate_hidden_mult: int = 2):
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU()
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU()
        )
        
        gate_hidden_dim = max(fusion_dim, fusion_dim * gate_hidden_mult)
        
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, gate_hidden_dim),
            nn.LayerNorm(gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(gate_hidden_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a, h_v, return_gate=False):
        a = self.audio_proj(h_a)
        v = self.visual_proj(h_v)
        z = self.gate(torch.cat([a, v], dim=-1))
        fused = self.dropout(z * a + (1.0 - z) * v)
        return (fused, z) if return_gate else fused


# =========================================================
# Full multimodal model (v4)
# =========================================================
class MultimodalEmotionModel(nn.Module):
    def __init__(self, ser: nn.Module, fer: nn.Module,
                 fusion_dim: int = 512, n_classes: int = 4,
                 visual_dim: int = 768, dropout: float = 0.3):
        super().__init__()
        self.ser = ser
        self.fer = fer
        
        # Initialize Fusion with gate_hidden_mult=2 as defined in v4.py
        self.fusion = GatedFusion(
            audio_dim=768, visual_dim=visual_dim,
            fusion_dim=fusion_dim, dropout=dropout, gate_hidden_mult=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, wav, clip, img_scores=None, return_gate=False):
        with torch.no_grad():
            h_a = self.ser.extract_features(wav)     
            h_v = self.fer.extract_features(clip, det_scores=img_scores)    
            
        if return_gate:
            fused, gate = self.fusion(h_a, h_v, return_gate=True)
            return self.classifier(fused), gate
            
        return self.classifier(self.fusion(h_a, h_v))