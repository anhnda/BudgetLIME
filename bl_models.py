"""
bl_models.py
============
Black-box model wrappers (the ONLY torch-dependent code). These are the
query-only black boxes explained by the surrogate; they are carried over UNCHANGED
from the original codebase -- their numerics already match the paper and were
validated in prior runs. All certification math lives in bl_core.py; these
classes only turn (input, binary mask) -> model output.

Nothing heavy runs on import: a model is constructed only when a wrapper is
instantiated inside a driver's main(), honoring the "never auto-run torch"
preference.

  TextClassifier  : sentence + token-mask  -> class probability  (sigma_obs > 0)
  ImageClassifier : image   + cell-mask    -> class logit        (sigma_obs ~ 0)
"""
from __future__ import annotations
import numpy as np

NLP_REFERENCES = ("mask", "pad", "zero")
IMAGE_REFERENCES = ("white", "black", "mean")
NLP_BACKBONES = ("distilbert", "roberta", "visobert")
IMAGE_BACKBONES = ("resnet50", "resnet18", "vit_b_16")


class TextClassifier:
    """Query-only black box: sentence + binary token-mask -> class probability.

    Off tokens (z=0) are replaced by a baseline embedding. The model is
    evaluated in chunks under no_grad. sigma_obs is 0 for a deterministic
    forward pass.
    """

    def __init__(self, model="distilbert", dataset="sst2", device=None,
                 chunk_size=64):
        import torch
        import inspect
        from transformers import (AutoTokenizer,
                                   AutoModelForSequenceClassification)
        self.torch = torch
        self.inspect = inspect
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.model_key = model
        self.model_name = self._resolve_model_name(model, dataset)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,
                                                       use_fast=True)
        self.model = (AutoModelForSequenceClassification
                      .from_pretrained(self.model_name)
                      .eval().to(self.device))
        self.embed = self.model.get_input_embeddings()

    def close(self):
        """Free the model so the next backbone has room (one model in RAM)."""
        try:
            del self.model, self.embed
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _resolve_model_name(model, dataset):
        table = {
            ("distilbert", "sst2"):
                "distilbert-base-uncased-finetuned-sst-2-english",
            ("distilbert", "imdb"):
                "textattack/distilbert-base-uncased-imdb",
            ("distilbert", "rotten"):
                "textattack/distilbert-base-uncased-rotten-tomatoes",
            ("bert", "sst2"): "textattack/bert-base-uncased-SST-2",
            ("bert", "imdb"): "textattack/bert-base-uncased-imdb",
            ("bert", "rotten"): "textattack/bert-base-uncased-rotten-tomatoes",
            ("roberta", "sst2"): "textattack/roberta-base-SST-2",
            ("roberta", "imdb"): "textattack/roberta-base-imdb",
            ("roberta", "rotten"): "textattack/roberta-base-rotten-tomatoes",
            # ---- Vietnamese sentiment models --------------------------- #
            # ViSoBERT-based 3-class (0=NEG, 1=POS, 2=NEU); SentencePiece
            # tokenizer -> interpretable units are SUBWORD pieces. Works on
            # RAW text (no external word segmentation, unlike PhoBERT).
            ("visobert", "vsfc"): "5CD-AI/Vietnamese-Sentiment-visobert",
            ("visobert", "vicomment"): "5CD-AI/Vietnamese-Sentiment-visobert",
            ("visobert", "sst2"): "5CD-AI/Vietnamese-Sentiment-visobert",
            # PhoBERT: requires word-segmented input; raw text => degraded.
            ("phobert", "vsfc"): "wonrax/phobert-base-vietnamese-sentiment",
            ("phobert", "vicomment"): "wonrax/phobert-base-vietnamese-sentiment",
        }
        return table.get((model, dataset), model)  # allow raw HF id

    # ----- baseline embedding factory (self-contained) -------------------- #
    def _baseline_embedding(self, X, kind):
        """Return a baseline embedding of shape (1, L, d), detached.
        Supported NLP references: mask | pad | zero (also mean | random)."""
        torch = self.torch
        L, d = X.shape[1], X.shape[2]
        tok = self.tokenizer
        embed = self.embed
        if kind == "mask":
            # `or` is wrong when mask_token_id == 0 (a valid id); test None.
            tid = tok.mask_token_id
            if tid is None:
                tid = tok.pad_token_id
            if tid is None:
                tid = tok.unk_token_id
            if tid is None:
                raise ValueError(
                    "tokenizer exposes no mask/pad/unk id for baseline='mask'; "
                    "use baseline='zero' instead.")
            with torch.no_grad():
                base = embed(torch.tensor([[tid]], device=self.device))
            return base.expand(1, L, d).clone()
        elif kind == "pad":
            tid = tok.pad_token_id
            if tid is None:
                tid = tok.eos_token_id if tok.eos_token_id is not None \
                    else tok.unk_token_id
            if tid is None:
                raise ValueError(
                    "tokenizer exposes no pad id for baseline='pad'; "
                    "use baseline='zero' instead.")
            with torch.no_grad():
                base = embed(torch.tensor([[tid]], device=self.device))
            return base.expand(1, L, d).clone()
        elif kind == "zero":
            return torch.zeros(1, L, d, device=self.device, dtype=X.dtype)
        elif kind == "mean":
            with torch.no_grad():
                mean_vec = embed.weight.mean(dim=0)
            return mean_vec.view(1, 1, d).expand(1, L, d).clone()
        elif kind == "random":
            vocab = embed.weight.shape[0]
            rid = torch.randint(0, vocab, (1,), device=self.device)
            with torch.no_grad():
                base = embed(rid.unsqueeze(0))
            return base.expand(1, L, d).clone()
        else:
            raise ValueError(f"unknown reference '{kind}' "
                             "(mask|pad|zero|mean|random)")

    def make_baseline(self, ctx, kind):
        return self._baseline_embedding(ctx["X"], kind)

    # ----- encode a sentence; expose free token positions ----------------- #
    def encode(self, sentence):
        torch = self.torch
        enc = self.tokenizer(sentence, return_tensors="pt", truncation=True,
                             return_special_tokens_mask=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        special = enc.get("special_tokens_mask",
                          torch.zeros_like(input_ids)).to(self.device)
        token_type_ids = enc.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        fwd_params = self.inspect.signature(self.model.forward).parameters
        extra = {}
        if "token_type_ids" in fwd_params and token_type_ids is not None:
            extra["token_type_ids"] = token_type_ids

        with torch.no_grad():
            X = self.embed(input_ids)                      # (1, L, dmodel)
        L = X.shape[1]

        is_special = special[0].bool()
        is_pad = (attention_mask[0] == 0)
        fixed = (is_special | is_pad)
        free_idx = torch.nonzero(~fixed, as_tuple=False).squeeze(-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        free_tokens = [tokens[i] for i in free_idx.tolist()]

        return {
            "X": X,
            "attention_mask": attention_mask,
            "extra_kwargs": extra,
            "free_idx": free_idx,
            "L": L,
            "tokens": tokens,
            "free_tokens": free_tokens,
        }

    def target_class(self, ctx):
        torch = self.torch
        with torch.no_grad():
            logits = self.model(inputs_embeds=ctx["X"],
                                attention_mask=ctx["attention_mask"],
                                **ctx["extra_kwargs"]).logits[0]
        return int(logits.argmax().item())

    # ----- chunked masked forward (self-contained) ------------------------ #
    def query(self, ctx, X_baseline, Z_free, target):
        """Z_free: N x d binary masks over FREE positions only. Returns
        y: N predicted-class probabilities for `target`. Off-token embeddings
        are replaced by the baseline; pinned positions forced to 1."""
        import torch
        import torch.nn.functional as Fnn

        free_idx = ctx["free_idx"]
        L = ctx["L"]
        X = ctx["X"]
        N = Z_free.shape[0]

        Z_full = torch.ones(N, L, device=self.device, dtype=X.dtype)
        if free_idx.numel() > 0:
            bits = torch.as_tensor(Z_free, device=self.device, dtype=X.dtype)
            Z_full[:, free_idx] = bits

        X_sq = X.squeeze(0)                                # (L, dmodel)
        Xref_sq = X_baseline.squeeze(0)                    # (L, dmodel)
        attn = ctx["attention_mask"]
        extra = ctx["extra_kwargs"]
        cs = self.chunk_size

        y = np.empty(N, dtype=np.float64)
        for i in range(0, N, cs):
            j = min(i + cs, N)
            z = Z_full[i:j].unsqueeze(-1)                  # (b, L, 1)
            X_pert = X_sq * z + Xref_sq * (1.0 - z)        # (b, L, dmodel)
            attn_b = attn.expand(j - i, -1)
            extra_b = {k: v.expand(j - i, -1) for k, v in extra.items()}
            with torch.no_grad():
                out = self.model(inputs_embeds=X_pert,
                                 attention_mask=attn_b, **extra_b)
            probs = Fnn.softmax(out.logits, dim=-1)
            y[i:j] = probs[:, target].double().cpu().numpy()
        return y




class ImageClassifier:
    """Query-only black box: image + binary cell-mask -> class logit.

    Masking fills OFF cells with a reference. Evaluated in batches.
    sigma_obs is 0 for a deterministic forward pass.
    """

    def __init__(self, backbone="resnet50", device=None):
        import torch
        import torchvision.models as tvm
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ctor = {
            "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
            "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
            "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
        }[backbone]
        weights = ctor[1]
        self.model = ctor[0](weights=weights).eval().to(self.device)
        self.preprocess = weights.transforms()
        self.backbone = backbone

    def close(self):
        """Free the model so the next backbone has room (one model in RAM)."""
        try:
            del self.model
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass

    def load_image(self, path, size=224):
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((size, size))
        return np.asarray(img).astype(np.float32) / 255.0   # HxWx3 in [0,1]

    def make_reference(self, img, kind="mean"):
        """Constant OFF-cell fill. Supported: white | black | mean."""
        if kind == "mean":
            return np.ones_like(img) * img.reshape(-1, 3).mean(0)
        if kind == "black":
            return np.zeros_like(img)
        if kind == "white":
            return np.ones_like(img)
        if kind == "gray":
            return np.ones_like(img) * 0.5
        if kind.startswith("blur"):
            from scipy.ndimage import gaussian_filter
            s = float(kind[4:]) if len(kind) > 4 else 8.0
            return np.stack([gaussian_filter(img[..., c], s)
                             for c in range(3)], axis=-1)
        raise ValueError(f"unknown reference {kind} (white|black|mean)")

    def _cell_slices(self, H, W, grid):
        ys = np.linspace(0, H, grid + 1).astype(int)
        xs = np.linspace(0, W, grid + 1).astype(int)
        slices = []
        for i in range(grid):
            for j in range(grid):
                slices.append((slice(ys[i], ys[i + 1]),
                               slice(xs[j], xs[j + 1])))
        return slices   # length d = grid*grid

    def target_class(self, img):
        import torch
        with torch.no_grad():
            x = self._to_tensor(img[None])
            logit = self.model(x)[0]
            return int(logit.argmax().item())

    def _to_tensor(self, imgs):
        import torch
        from PIL import Image
        ts = []
        for im in imgs:
            pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
            ts.append(self.preprocess(pil))
        return torch.stack(ts).to(self.device)

    def query(self, img, ref, slices, Z, target, batch=64):
        """Z: N x d binary masks. Returns y: N logits for `target`."""
        import torch
        H, W, _ = img.shape
        N, d = Z.shape
        cell_map = np.full((H, W), -1, dtype=int)
        for c, (sy, sx) in enumerate(slices):
            cell_map[sy, sx] = c
        y = np.empty(N, dtype=np.float64)
        for b0 in range(0, N, batch):
            b1 = min(b0 + batch, N)
            imgs = np.empty((b1 - b0, H, W, 3), dtype=np.float32)
            for k, t in enumerate(range(b0, b1)):
                on = Z[t][cell_map]
                imgs[k] = np.where(on[..., None], img, ref)
            with torch.no_grad():
                logits = self.model(self._to_tensor(imgs))
                y[b0:b1] = logits[:, target].double().cpu().numpy()
        return y


