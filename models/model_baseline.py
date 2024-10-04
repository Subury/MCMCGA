import functools
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from transformers import AutoModel
from timm.models import vision_transformer
from transformers.models.bert import modeling_bert

from utils import misc , pos_embed

class MultiGPU_GlobalNceLoss(nn.Module):

    def __init__(self):
        super(MultiGPU_GlobalNceLoss, self).__init__()

        self.logit_scale = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

    def all_gather(self, input):

        output = misc.GatherLayer.apply(input)
        output = output.view(-1, *output.shape[2:])
        return output

    def forward(self, zg_1, zg_2, is_training=True):
        """

        :param zg_1: (B x d_z)
        :param zg_2: (B x d_z)
        :return:
        """

        device = zg_1.device

        # normalized features
        image_features = zg_1 / (zg_1.norm(dim=-1, keepdim=True) + 1e-10)
        text_features = zg_2 / (zg_2.norm(dim=-1, keepdim=True) + 1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100, min=0.01)

        if is_training:
            gathered_image_features = self.all_gather(image_features)
            gathered_text_features = self.all_gather(text_features)

            logits_per_image = logit_scale * image_features @ gathered_text_features.t()
            logits_per_text = logit_scale * text_features @ gathered_image_features.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

        bs, l_bs = logits_per_image.shape
        if l_bs == bs:
            labels = torch.arange(len(logits_per_image)).to(device)
        else:
            labels = misc.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).to(device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)

        return 0.75 * loss_i + 0.25 * loss_t

class MRM(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., 
                 norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=8,
                 global_project_dim=512, **kwargs):
        super().__init__()

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = vision_transformer.PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            vision_transformer.Block(embed_dim, num_heads, mlp_ratio, 
                                     qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # image decoder specifics
        self.scan_mlp = nn.Sequential(
                            nn.Linear(embed_dim, decoder_embed_dim, bias=True), 
                            norm_layer(decoder_embed_dim), nn.GELU(), 
                            nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
                                    torch.zeros(1, num_patches + 1, decoder_embed_dim), 
                                    requires_grad=True)

        self.decoder_blocks = nn.ModuleList([
            vision_transformer.Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, 
                                     qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size ** 2) * in_channels, bias=True)

        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_encoder = AutoModel.from_pretrained(
                                            '/workspace/Bio_ClinicalBERT', 
                                            trust_remote_code=True)
        self.bert_encoder._set_gradient_checkpointing(self.bert_encoder.encoder, value=True)
        self.bert_head = modeling_bert.BertOnlyMLMHead(self.bert_encoder.config)
        self.bert_mlp = nn.Linear(embed_dim, self.bert_encoder.config.hidden_size, bias=True)

        # --------------------------------------------------------------------------
        # global project
        self.global_scan_mlp = nn.Linear(embed_dim, global_project_dim, bias=True)
        self.global_report_mlp = nn.Linear(embed_dim, global_project_dim, bias=True)

        # --------------------------------------------------------------------------
        # ConVIRT Loss
        self.norm_pix_loss = norm_pix_loss
        self.global_forward_loss = MultiGPU_GlobalNceLoss()
    
        self.initialize_weights()
    
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embedding = pos_embed.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                                          int(self.patch_embed.num_patches**.5), 
                                                          cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embedding).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_scan_encoder(self, x, mask_ratio=0.5, is_training=True):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore = None, None

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            if is_training:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_scan_decoder(self, x, ids_restore, is_training=True):

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            if is_training:
                x = checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def reconstruction_loss(self, imgs, pred, mask):

        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, batch, mask_ratio=0.5, is_training=True, loss_keys=[]):
        
        device = next(self.parameters()).device

        imgs= batch["image"].to(device)
        if is_training:
            ids, attention_mask, type_ids = batch["ids"].to(device), batch["attention_mask"].to(device), batch["type_ids"].to(device)
        else:
            ids, attention_mask, type_ids = batch["labels"].to(device), batch["attention_mask"].to(device), batch["type_ids"].to(device)

        scan_latents, mask, ids_restore = self.forward_scan_encoder(imgs, mask_ratio=mask_ratio, is_training=is_training)
        report_latents = self.bert_encoder(ids, attention_mask, type_ids)[0]
    
        global_report_features = self.global_report_mlp(report_latents[:, 1:, :]).max(dim=1)[0]
        global_scan_features = self.global_scan_mlp(scan_latents[:, 1:, :]).max(dim=1)[0]

        if is_training:
            output_loss = {}
            if 'src' in loss_keys:
                global_contrastive_loss = self.global_forward_loss(global_scan_features, global_report_features, is_training=is_training)
                output_loss['src'] = global_contrastive_loss
            
            if 'mrm' in loss_keys:
                report_pred = self.bert_head(report_latents)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
                masked_lm_loss = loss_fct(report_pred.view(-1, self.bert_encoder.config.vocab_size), batch["labels"].to(device).view(-1))
                output_loss['mrm'] = masked_lm_loss
            
            if 'msm' in loss_keys:
                scan_features = self.scan_mlp(scan_latents)
                scan_pred = self.forward_scan_decoder(scan_features, ids_restore, is_training=is_training)
                masked_im_loss = self.reconstruction_loss(imgs, scan_pred, mask)
                output_loss['msm'] = masked_im_loss

            return output_loss, (global_scan_features, global_scan_features), (global_scan_features, global_report_features)
            
        else:
            return {
                    'global':{'scan': global_scan_features, 
                              'report': global_report_features}
                }

    def get_parameter_group(self):
        
        scale_keys = ['scan_mlp', 'mask_token', 'decoder_pos_embed', 'decoder_blocks', 
                      'decoder_norm', 'decoder_pred', 'bert_head', 'bert_mlp', 
                      'global_scan_mlp', 'global_report_mlp', 'global_forward_loss']

        base_parameters = []
        scale_parameters = []
        for name, param in self.named_parameters():
            if sum([1 for key in scale_keys if key in name]):
                scale_parameters.append(param)
            else:
                base_parameters.append(param)
        
        return [{'params': base_parameters}, {'params': scale_parameters, 'lr_scale': 3.0}]


def baseline(**kwargs):
    model = MRM(
        patch_size=16, in_channels=3, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=functools.partial(nn.LayerNorm, eps=1e-6), 
        decoder_embed_dim=512, decoder_depth=3, decoder_num_heads=8,
        global_project_dim=512, **kwargs)
    return model