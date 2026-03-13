"""
Frequency-Aware Semantic Anchoring (FASA) Module

Prevents tail-class prototype drift by tethering prototypes to CLIP semantic
anchors with frequency-dependent strength.

Enhanced design:
1. Multi-layer anchor projection with bottleneck (preserves CLIP structure)
2. Cosine + L2 hybrid anchoring loss (scale-invariant + position-aware)
3. Continuous sigmoid frequency weighting
4. stop_gradient on anchors
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FASA(nn.Module):
    """
    Frequency-Aware Semantic Anchoring.
    
    Loss: L_anchor = Σ_c α_c · (λ_cos · (1 - cos(p_c, sg(a_c))) + λ_l2 · ||p_c - sg(a_c)||²_norm)
    where α_c = sigmoid(k · (log(N_median) - log(N_c)))
    """
    
    def __init__(self, clip_embed_dim=768, proto_dim=4096, num_rel_cls=51,
                 temperature_k=2.0, clip_embed_path=None, predicate_freq=None,
                 cos_weight=0.5, l2_weight=0.5):
        super(FASA, self).__init__()
        
        self.num_rel_cls = num_rel_cls
        self.temperature_k = temperature_k
        self.cos_weight = cos_weight
        self.l2_weight = l2_weight
        
        # Load CLIP predicate embeddings
        if clip_embed_path is not None and os.path.exists(clip_embed_path):
            clip_data = torch.load(clip_embed_path, map_location='cpu')
            clip_pred_embeds = clip_data['pred_embeddings']
            self.register_buffer('clip_pred_embeds', clip_pred_embeds)
            actual_dim = clip_pred_embeds.shape[1]
            if actual_dim != clip_embed_dim:
                clip_embed_dim = actual_dim
        else:
            self.register_buffer('clip_pred_embeds', torch.randn(num_rel_cls, clip_embed_dim))
            print(f"[FASA WARNING] No CLIP embeddings at {clip_embed_path}.")
        
        # Multi-layer anchor projection with bottleneck
        bottleneck_dim = min(clip_embed_dim, 512)
        self.anchor_proj = nn.Sequential(
            nn.Linear(clip_embed_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, proto_dim),
        )
        
        # Frequency-aware weights
        if predicate_freq is not None:
            self._compute_freq_weights(predicate_freq)
        else:
            self.register_buffer('freq_weights', torch.ones(num_rel_cls))
        
        # Initialize
        for m in self.anchor_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
    
    def _compute_freq_weights(self, predicate_freq):
        """α_c = sigmoid(k · (log(N_median) - log(N_c)))
        Background class (index 0) is excluded from median computation and gets weight=0.
        """
        freq = torch.FloatTensor(predicate_freq)
        freq = freq.clamp(min=1.0)
        log_freq = torch.log(freq)
        # Compute median over FOREGROUND classes only (exclude index 0)
        log_median = torch.median(log_freq[1:])
        weights = torch.sigmoid(self.temperature_k * (log_median - log_freq))
        weights[0] = 0.0  # background class should NOT be anchored
        self.register_buffer('freq_weights', weights)
    
    def set_predicate_freq(self, predicate_freq):
        self._compute_freq_weights(predicate_freq)
    
    def compute_anchors(self):
        return self.anchor_proj(self.clip_pred_embeds)  # [num_rel_cls, proto_dim]
    
    def forward(self, predicate_proto):
        """
        Compute FASA anchoring loss (cosine + normalized L2).
        
        Args:
            predicate_proto: [num_rel_cls, proto_dim]
        Returns:
            anchor_loss: scalar
        """
        anchors = self.compute_anchors().detach()  # stop gradient
        
        loss = torch.tensor(0.0, device=predicate_proto.device)
        
        # Cosine distance component (scale-invariant)
        if self.cos_weight > 0:
            proto_norm = F.normalize(predicate_proto, dim=-1)
            anchor_norm = F.normalize(anchors, dim=-1)
            cos_dist = 1.0 - (proto_norm * anchor_norm).sum(dim=-1)  # [num_rel_cls]
            loss = loss + self.cos_weight * (self.freq_weights * cos_dist).mean()
        
        # Normalized L2 component
        if self.l2_weight > 0:
            l2_dist = (predicate_proto - anchors).pow(2).sum(dim=-1)
            # Normalize by proto_dim to keep loss scale manageable
            l2_dist = l2_dist / predicate_proto.shape[-1]
            loss = loss + self.l2_weight * (self.freq_weights * l2_dist).mean()
        
        return loss
    
    def get_anchor_distances(self, predicate_proto):
        """Per-class analysis."""
        with torch.no_grad():
            anchors = self.compute_anchors()
            l2 = (predicate_proto - anchors).pow(2).sum(dim=-1).sqrt()
            cos = F.cosine_similarity(predicate_proto, anchors, dim=-1)
        return {'l2_distances': l2, 'cosine_sim': cos, 'freq_weights': self.freq_weights}
