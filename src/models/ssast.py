import torch
import torch.nn as nn
import numpy as np
import timm
from timm.models.layers import to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SSASTModel(nn.Module):
    def __init__(self, label_dim=4, fshape=16, tshape=16, fstride=10, tstride=10,
                 input_fdim=128, input_tdim=1024, model_size='base',
                 load_pretrained_mdl_path=None):
        super().__init__()
        # Override timm's PatchEmbed
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if load_pretrained_mdl_path is None:
            raise ValueError('Please provide load_pretrained_mdl_path')
        sd = torch.load(load_pretrained_mdl_path, map_location=device, weights_only=False)
        # Get pretraining patch shape and input dims from saved weights
        p_fshape = sd['module.v.patch_embed.proj.weight'].shape[2]
        p_tshape = sd['module.v.patch_embed.proj.weight'].shape[3]
        p_input_fdim = sd['module.p_input_fdim'].item()
        p_input_tdim = sd['module.p_input_tdim'].item()
        # Recreate the pretraining model to load weights into
        pretrain_model = self._build_pretrain_model(
            p_fshape, p_tshape, p_input_fdim, p_input_tdim, model_size
        )
        pretrain_model = torch.nn.DataParallel(pretrain_model)
        pretrain_model.load_state_dict(sd, strict=False)
        # Extract the ViT backbone
        self.v = pretrain_model.module.v
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.cls_token_num = pretrain_model.module.cls_token_num
        # Classification head for fine-tuning
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim)
        )
        # Adjust patch embedding stride (overlapping patches for fine-tuning)
        f_dim, t_dim = self._get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
        p_f_dim, p_t_dim = pretrain_model.module.p_f_dim, pretrain_model.module.p_t_dim
        num_patches = f_dim * t_dim
        p_num_patches = p_f_dim * p_t_dim
        self.v.patch_embed.num_patches = num_patches
        if fstride != p_fshape or tstride != p_tshape:
            new_proj = nn.Conv2d(1, self.original_embedding_dim,
                                 kernel_size=(fshape, tshape), stride=(fstride, tstride))
            new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj
        # Interpolate positional embeddings to match new patch count
        new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach()
        new_pos_embed = new_pos_embed.reshape(1, p_num_patches, self.original_embedding_dim)
        new_pos_embed = new_pos_embed.transpose(1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
        if t_dim <= p_t_dim:
            new_pos_embed = new_pos_embed[:, :, :, int(p_t_dim/2) - int(t_dim/2): int(p_t_dim/2) - int(t_dim/2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
        if f_dim <= p_f_dim:
            new_pos_embed = new_pos_embed[:, :, int(p_f_dim/2) - int(f_dim/2): int(p_f_dim/2) - int(f_dim/2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
        self.v.pos_embed = nn.Parameter(
            torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1)
        )
        print(f'SSAST loaded: patches={num_patches}, embed_dim={self.original_embedding_dim}')

    def _build_pretrain_model(self, fshape, tshape, input_fdim, input_tdim, model_size):
        """Build a pretraining-stage model to load weights into."""
        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        if model_size == 'base':
            v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
            embed_dim = v.pos_embed.shape[2]
            cls_token_num = 2
        elif model_size == 'small':
            v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
            embed_dim = v.pos_embed.shape[2]
            cls_token_num = 2
        else:
            v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
            embed_dim = v.pos_embed.shape[2]
            cls_token_num = 2

        class PretrainShell(nn.Module):
            def __init__(self):
                super().__init__()
                self.v = v
                self.cls_token_num = cls_token_num
                self.p_input_fdim = nn.Parameter(torch.tensor(input_fdim), requires_grad=False)
                self.p_input_tdim = nn.Parameter(torch.tensor(input_tdim), requires_grad=False)
                self.v.patch_embed.proj = nn.Conv2d(1, embed_dim,
                    kernel_size=(fshape, tshape), stride=(fshape, tshape))
                f_dim = input_fdim // fshape
                t_dim = input_tdim // tshape
                num_patches = f_dim * t_dim
                self.p_f_dim = f_dim
                self.p_t_dim = t_dim
                self.v.patch_embed.num_patches = num_patches
                self.v.pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches + cls_token_num, embed_dim))
                trunc_normal_(self.v.pos_embed, std=.02)
                self.cpredlayer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 256))
                self.gpredlayer = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 256))
                self.mask_embed = nn.Parameter(torch.zeros([1, 1, embed_dim]))

        return PretrainShell()
    def _get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim if hasattr(self, 'original_embedding_dim') else 768,
                              kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        return test_out.shape[2], test_out.shape[3]
    def forward(self, x):
        # Input: (batch, time_frames, freq_bins) e.g. (8, 1024, 128)
        x = x.unsqueeze(1)      # (batch, 1, time, freq)
        x = x.transpose(2, 3)   # (batch, 1, freq, time)
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        # Mean pooling (skip cls/dist tokens)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    