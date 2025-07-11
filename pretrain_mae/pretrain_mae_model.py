import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import os
import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import datetime


def _build_corner_L_indices(grid: int, device) -> torch.Tensor:
    """
    Return a tensor with the 12 patch indices (0-based, *without* the CLS token)
    that form an L-shape around the four corners of a `grid × grid` ViT.

    Ordering:
        TL: [0, 1,  grid]                       (corner, right, down)
        TR: [g-1, g-2, 2g-1]                    (corner, left,  down)
        BL: [g(g-1), g(g-1)+1, g(g-2)]          (corner, right, up)
        BR: [g²-1,  g²-2,  g(g-1)-1]            (corner, left,  up)
    """
    g = grid
    idxs = [
        0, 1, g,
        g - 1, g - 2, 2*g - 1,
        g*(g - 1), g*(g - 1) + 1, g*(g - 2),
        g*g - 1, g*g - 2, g*(g - 1) - 1,
    ]
    return torch.tensor(idxs, device=device)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.0, attention_type='cls'):
        super().__init__()
        print(f"Using {attention_type} cross attention")
        # only one mhsa for t→i, one for i→t
        self.attn_t2i = nn.MultiheadAttention(embed_dim, num_heads, 
                                              dropout=dropout, batch_first=True)
        self.attn_i2t = nn.MultiheadAttention(embed_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.ln_tact  = nn.LayerNorm(embed_dim)
        self.ln_img   = nn.LayerNorm(embed_dim)
        self.attention_type = attention_type

    def forward(self, image_tokens, tactile_tokens):
        # image_tokens: (B, 1+P, D), tactile_tokens: (B, Q, D)
        cls_tok   = image_tokens[:, :1, :]   # (B,1,D)
        patch_tok = image_tokens[:, 1:, :]   # (B,P,D)

        if self.attention_type == "cls":
            # 1) tactile → CLS
            #    queries: tactile; keys/values: only the CLS slot
            tact_out, attn_weights = self.attn_t2i(
                query  = tactile_tokens,   # (B, Q, D)
                key    = cls_tok,          # (B, 1, D)
                value  = cls_tok,          # (B, 1, D)
                need_weights = True,
                average_attn_weights = False
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 2) CLS → tactile (optionally you could also only update CLS here,
            #    but this shows the symmetric step if you still want cls←tactile)
            img_out, _ = self.attn_i2t(
                query  = cls_tok,         # (B,1,D)
                key    = tactile_tokens,  # (B,Q,D)
                value  = tactile_tokens,  # (B,Q,D)
            )
            cls_tok = self.ln_img(cls_tok + img_out)

            # 3) put tokens back together
            image_tokens = torch.cat([cls_tok, patch_tok], dim=1)

        elif self.attention_type == "patch":
            B, N, D = image_tokens.shape          # N = 1( CLS ) + P
            P = N - 1
            grid = int(P ** 0.5)                  # e.g. 196 → 14
            corner_L_patch = _build_corner_L_indices(grid, image_tokens.device)  # indices 0-based *within patches*


            mask = torch.zeros(B, P, dtype=torch.bool, device=image_tokens.device)
            mask[:, corner_L_patch] = True
            tact_out, attn_weights = self.attn_t2i(
                query  = tactile_tokens,  # (B, Q, D)
                key    = patch_tok,       # (B, P, D)
                value  = patch_tok,       # (B, P, D)
                need_weights         = True,
                average_attn_weights = False,
                key_padding_mask=mask
            )
            tactile_tokens = self.ln_tact(tactile_tokens + tact_out)

            # 2) PATCH => TACTILE
            patch_out, _ = self.attn_i2t(
                query  = patch_tok,       # (B, P, D)
                key    = tactile_tokens,   # (B, D)
                value  = tactile_tokens,   # (B, Q, D)
                need_weights         = True,
                average_attn_weights = False,
            )
            patch_tok = self.ln_img(patch_tok + patch_out)

            # Finally, reassemble: keep CLS untouched, put updated patch_tok back
            image_tokens = torch.cat([cls_tok, patch_tok], dim=1)


        return image_tokens, tactile_tokens, attn_weights

class PatchAggregator(nn.Module):
    """
    Learnable attention-pooling over a set of patch embeddings.
    Returns a single (B, D) vector.
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_tok = nn.Parameter(torch.randn(1, 1, embed_dim))  # the aggregator token
        nn.init.trunc_normal_(self.query_tok, std=0.02)

        self.attn   = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.ln     = nn.LayerNorm(embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches : (B, P, D)
        returns : (B, D)
        """
        B = patches.size(0)
        # repeat the learnable token across the batch
        q = self.query_tok.expand(B, -1, -1)           # (B, 1, D)
        pooled, _ = self.attn(query=q, key=patches, value=patches)
        pooled = self.ln(pooled + q)                   # residual + LN
        return pooled.squeeze(1)                       # → (B, D)


class SimpleCNN(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool  = nn.AdaptiveAvgPool2d((1,1))
        self.fc    = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: (B, 3, 24, 32)
        x = F.relu(self.bn1(self.conv1(x)))  # => (B,16,12,16)
        x = F.relu(self.bn2(self.conv2(x)))  # => (B,32,6,8)
        x = F.relu(self.bn3(self.conv3(x)))  # => (B,64,3,4)
        x = self.pool(x)                     # => (B,64,1,1)
        x = x.view(x.size(0), -1)            # => (B,64)
        x = self.fc(x)                       # => (B,out_dim)
        return x


def make_viridis_colormap():
    from matplotlib import cm
    viridis = cm.get_cmap('viridis', 256)
    colormap = viridis(np.arange(256))[:, :3]  # shape (256,3)
    return torch.FloatTensor(colormap)


###############################################################################
# TactileAutoencoderModel
###############################################################################
class TactileAutoencoderModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        tactile_patch_size: int = 4,
        num_heads: int = 8,
        attention_type: str = "cls",
        predict_channel=3,
        mask_ratio_min: float = 0.6,
        mask_ratio_max: float = 0.8,
        unmask_prob: float = 0.05,
        *,
        save_images: bool = False,
        save_dir: str = "saved_tactile_imgs"
    ):
        super().__init__()

        # 1) Vision encoder (CLIP ViT)
        self.clip_encoder = timm.create_model(
            "vit_base_patch16_clip_224.openai",
            pretrained=True,
            global_pool='',
            num_classes=0
        )
        for p in self.clip_encoder.parameters():
            p.requires_grad = True

        # 2) Viridis lookup table
        self.register_buffer("viridis_map", make_viridis_colormap(), persistent=False)

        # 3) CNN for masked tactile colour image
        self.cnn = SimpleCNN(out_dim=embed_dim)

        self.patch_size  = tactile_patch_size

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 3, self.patch_size, self.patch_size))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        # Positional embedding for image tokens
        grid_size   = 224 // 16
        num_patches = grid_size * grid_size
        self.pos_embed = nn.Embedding(num_patches, embed_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

        # Cross-attention
        self.attention_type = attention_type
        self.cross_attn = CrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=0.2, attention_type=self.attention_type)

        self.predict_channel = predict_channel
        # Decide output dimension: either 3 * 24 * 32 or 1 * 24 * 32
        if self.predict_channel == 3:
            self.decoder_out_dim = 3 * 24 * 32
        else:
            self.decoder_out_dim = 1 * 24 * 32

        hidden_dim = embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, self.decoder_out_dim),
        )
        # zero‐init the final bias (optional, but keeps outputs initially near zero)
        nn.init.constant_(self.decoder[-1].bias, 0.0)


        #self.decoder   = nn.Linear(embed_dim * 2, self.decoder_out_dim)
        #self.dropout_dec = nn.Dropout(p=0.1)
        #nn.init.constant_(self.decoder.bias, 0.0)
        self.out_act   = nn.Sigmoid()

        # Image-dump settings
        self.save_images = save_images
        self.save_dir    = save_dir
        if self.save_images:
            os.makedirs(self.save_dir, exist_ok=True)
        self._internal_step = 0

        if attention_type == "patch":
            self.patch_pool = PatchAggregator(embed_dim, num_heads=num_heads)
        else:
            self.patch_pool = None

        self.mask_ratio_min = mask_ratio_min
        self.mask_ratio_max = mask_ratio_max
        self.unmask_prob = unmask_prob

    @torch.no_grad()
    def _dump_batch(self, masked_color: torch.Tensor, gt_color: torch.Tensor, step_id: int):
        masked_color = masked_color.cpu().clamp(0, 1)
        gt_color     = gt_color.cpu().clamp(0, 1)
        for i in range(masked_color.size(0)):
            base = f"step{step_id:06d}_{i}"
            from torchvision.utils import save_image
            save_image(masked_color[i], os.path.join(self.save_dir, f"{base}_masked.png"))
            save_image(gt_color[i],     os.path.join(self.save_dir, f"{base}_gt.png"))
    
    def encode(
        self,
        rgb_cur_unmasked: torch.Tensor,
        tactile_12x64:    torch.Tensor,
        global_step:      int = None
    ) -> torch.Tensor:
        B, n_cams, C, H, W = rgb_cur_unmasked.shape

        # 1) CLIP image tokens
        rgb_flat      = rgb_cur_unmasked.view(B * n_cams, C, H, W)
        camera_tokens = self.clip_encoder(rgb_flat)
        T             = camera_tokens.size(1)
        camera_tokens = camera_tokens.view(B, n_cams * T, -1)

        # positional embed
        cls_tok   = camera_tokens[:, :1]
        patch_tok = camera_tokens[:, 1:]
        pos_ids   = torch.arange(patch_tok.size(1), device=patch_tok.device)
        patch_tok = patch_tok + self.pos_embed(pos_ids).unsqueeze(0)
        camera_tokens = torch.cat([cls_tok, patch_tok], dim=1)

        # 2) build colour images
        gt_color     = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=False)
        masked_color = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=True)

        # 3) CNN embed
        tactile_emb    = self.cnn(masked_color)
        tactile_tokens = tactile_emb.unsqueeze(1)

        # 4) cross‐attention
        camera_tokens, tactile_tokens, _ = self.cross_attn(camera_tokens, tactile_tokens)

        if self.attention_type == "cls":
            # Use the CLS token as the image representation:
            image_cls = camera_tokens[:, :1, :].squeeze(1)  # (B, D)
            tactile_  = tactile_tokens.squeeze(1)           # (B, D)
            fused_in  = torch.cat([tactile_, image_cls], dim=1)  # => (B, 2D)
        else:
            # Use patch embeddings, ignoring CLS
            patch_tok = camera_tokens[:, 1:, :]        # (B, P, D)
            patch_rep = self.patch_pool(patch_tok)     # (B, D)  ← learnable pooling
            tactile_  = tactile_tokens.squeeze(1)      # (B, D)
            fused_in  = torch.cat([tactile_, patch_rep], dim=1)  # (B, 2D)

        return fused_in

    def encode_unmasked(
        self,
        rgb_cur_unmasked: torch.Tensor,
        tactile_12x64:    torch.Tensor,
        global_step:      int = None
    ) -> torch.Tensor:
        B, n_cams, C, H, W = rgb_cur_unmasked.shape

        # 1) CLIP image tokens
        rgb_flat      = rgb_cur_unmasked.view(B * n_cams, C, H, W)
        camera_tokens = self.clip_encoder(rgb_flat)
        T             = camera_tokens.size(1)
        camera_tokens = camera_tokens.view(B, n_cams * T, -1)

        # positional embed
        cls_tok   = camera_tokens[:, :1]
        patch_tok = camera_tokens[:, 1:]
        pos_ids   = torch.arange(patch_tok.size(1), device=patch_tok.device)
        patch_tok = patch_tok + self.pos_embed(pos_ids).unsqueeze(0)
        camera_tokens = torch.cat([cls_tok, patch_tok], dim=1)

        # 2) build colour images
        gt_color     = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=False)
        masked_color = self.convert_12x64_to_3x24x32(tactile_12x64, apply_mask=False)

        # 3) CNN embed
        tactile_emb    = self.cnn(masked_color)
        tactile_tokens = tactile_emb.unsqueeze(1)

        # 4) cross‐attention
        camera_tokens, tactile_tokens, _ = self.cross_attn(camera_tokens, tactile_tokens)

        if self.attention_type == "cls":
            # Use the CLS token as the image representation:
            image_cls = camera_tokens[:, :1, :].squeeze(1)  # (B, D)
            tactile_  = tactile_tokens.squeeze(1)           # (B, D)
            fused_in  = torch.cat([tactile_, image_cls], dim=1)  # => (B, 2D)
        else:
            # Use patch embeddings, ignoring CLS
            patch_tok = camera_tokens[:, 1:, :]        # (B, P, D)
            patch_rep = self.patch_pool(patch_tok)     # (B, D)  ← learnable pooling
            tactile_  = tactile_tokens.squeeze(1)      # (B, D)
            fused_in  = torch.cat([tactile_, patch_rep], dim=1)  # (B, 2D)

        return fused_in
            
    def forward(
        self,
        rgb_cur_unmasked: torch.Tensor,
        tactile_12x64:    torch.Tensor,
        global_step:      int = None
    ) -> torch.Tensor:
        # 1) encode => returns [B, 2 * embed_dim]
        latent = self.encode(rgb_cur_unmasked, tactile_12x64, global_step)

        # 2) decode => (B,3×24×32)
        x    = self.decoder(latent)                    # (B, hidden_dim) → (B, out_dim)
        pred = x.view(latent.size(0), self.predict_channel, 24, 32)
        return self.out_act(pred)

    def convert_12x64_to_3x24x32(self, tactile_map: torch.Tensor, apply_mask: bool) -> torch.Tensor:
        B      = tactile_map.size(0)
        left   = tactile_map[..., :32].clamp(0, 1)
        right  = tactile_map[..., 32:].clamp(0, 1)
        left_i = (left  * 255).long().clamp(0, 255)
        right_i= (right * 255).long().clamp(0, 255)
        left_c = self.viridis_map[left_i]
        right_c= self.viridis_map[right_i]
        left_c = left_c.permute(0, 3, 1, 2)
        right_c= right_c.permute(0, 3, 1, 2)
        color_img = torch.cat([left_c, right_c], dim=2)
        if apply_mask:
            color_img = self.mask_tactile_color_image(color_img)
        return color_img

    @torch.no_grad()
    def convert_12x64_to_1x24x32(self, tactile_map, apply_mask=False):
        B, H, W = tactile_map.shape
        left  = tactile_map[:, : , :32]   # (B,12,32)
        right = tactile_map[:, : , 32:]   # (B,12,32)
        # stack top/bottom
        stacked = torch.cat([left, right], dim=1)  # (B,24,32)
        return stacked.unsqueeze(1)  # (B,1,24,32)

    def mask_tactile_color_image(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() < self.unmask_prob:
            return img

        B, C, H, W = img.shape
        pH         = H // self.patch_size
        pW         = W // self.patch_size
        total      = pH * pW

        ratio    = random.uniform(self.mask_ratio_min, self.mask_ratio_max)
        num_mask = int(total * ratio)

        for b in range(B):
            idxs = list(range(total))
            np.random.shuffle(idxs)
            for pid in idxs[:num_mask]:
                r0 = (pid // pW) * self.patch_size
                c0 = (pid % pW) * self.patch_size
                img[b, :, r0:r0 + self.patch_size, c0:c0 + self.patch_size] = self.mask_token
        return img
