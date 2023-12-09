import os.path as osp
from collections import OrderedDict
import math

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import json

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root = os.path.expanduser("/home/ymlei/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model, args):
        super().__init__()
        device = int(args.gpu[0])
        self.transformer = clip_model.transformer.type(torch.FloatTensor).to(device)
        self.positional_embedding = clip_model.positional_embedding.type(torch.FloatTensor).to(device)
        self.ln_final = clip_model.ln_final.type(torch.FloatTensor).to(device)
        self.text_projection = clip_model.text_projection.type(torch.FloatTensor).to(device)
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, attrnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 4
        ctx_init = "a nodule image of"
        # ctx_init = None
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        '''
        Class prompts
        '''
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn) (2, 77)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) # (2, 77, 512)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        '''
        Attribute prompts
        '''
        attr_init = 'the nodule is'
        attr_prompts = [attr_init + " " + attr + "." for attr in attrnames]
        tokenized_attr = torch.cat([clip.tokenize(q) for q in attr_prompts]) # (9, 77)
        with torch.no_grad():
            attr_embedding = clip_model.token_embedding(tokenized_attr).type(dtype) # (9, 77, 512)
        self.register_buffer("token_attr_prefix", attr_embedding[:, :1, :])  # SOS
        self.register_buffer("token_attr_suffix", attr_embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        self.tokenized_attr = tokenized_attr

        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix

        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)

        attr_prefix = self.token_attr_prefix # (9, 1, 512)
        attr_suffix = self.token_attr_suffix # (9, 72, 512)

        # Use instance-conditioned context tokens for all classes
        prompts = []
        attr_prompts = []
        for ctx_shifted_i in ctx_shifted:

            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)

            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            
            attr_ctx_i = ctx_shifted_i.unsqueeze(0).expand(attr_prefix.shape[0], -1, -1)
            attr_pts_i = self.construct_prompts(attr_ctx_i, attr_prefix, attr_suffix)
            
            prompts.append(pts_i)
            attr_prompts.append(attr_pts_i)

        prompts = torch.stack(prompts)
        attr_prompts = torch.stack(attr_prompts)
        
        return prompts, attr_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits

if __name__ == '__main__':

    clip_model = load_clip_to_cpu()
    text_encoder = TextEncoder(clip_model)







