# ============================================================
# Appendix G.3 — SAE Loss Added Across Layers (Figure 23)
# Reproduces the loss-added plot for mlp_out and resid_mid SAEs
# on the 6-layer TinyStories bilinear transformer.
#
# Loss added (Eq. 4):  (L_patched − L_clean) / L_clean
# where L_patched runs the model with the SAE reconstruction
# spliced into the stream at the SAE's hook point.
#
# Uses the public SAEs from tdooms/ts-medium-scope
# ({layer}-mlp-out-x4-k30 and {layer}-resid-mid-x4-k30) — the
# same 4x-expansion, k=30 dictionaries described in Appendix G.
#
# CAVEAT: the authors trained model and SAEs on their own cleaned/
# simplified TinyStories dataset, which was never published (the
# public checkpoint scores ~2.9 on raw roneneldan/TinyStories vs
# the paper's 1.34 on their data). On this slightly out-of-
# distribution eval data the absolute loss-added values are
# distorted (patching can even *lower* the loss, since the SAE
# denoises toward the training manifold). The qualitative claim —
# resid_mid adds more loss than mlp_out at every layer — still
# reproduces. Per-layer reconstruction NMSE is reported alongside
# as a data-robust quality metric.
# ============================================================

import json
import os
import types
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

os.chdir(Path(__file__).resolve().parents[3])
HERE = Path(__file__).parent

from language.transformer import Transformer, Attention, Rotary
from sae import SAE

if not hasattr(Transformer, "all_tied_weights_keys"):
    Transformer.all_tied_weights_keys = property(lambda self: {})


def fix_buffers(model):
    for m in model.modules():
        if isinstance(m, Attention):
            dev = next(m.parameters()).device
            m.mask = torch.tril(
                torch.ones(m.config.n_ctx, m.config.n_ctx)
            )[None, None].to(dev)
        if isinstance(m, Rotary):
            m.seq_len_cached = None


torch.set_grad_enabled(False)

N_SEQ = 512
BATCH = 16
SAE_REPO = "tdooms/ts-medium-scope"
RESULTS = HERE / "appendix_g_results.json"

# ── Compute loss added (cached to JSON) ──────────────────────
if RESULTS.exists():
    print(f"Loading cached {RESULTS.name} — delete it to recompute.")
    results = json.load(open(RESULTS))
else:
    print("Loading ts-medium …")
    model = Transformer.from_pretrained("tdooms/ts-medium", device="cuda")
    fix_buffers(model)
    model.eval()
    n_ctx = model.config.n_ctx

    print(f"Tokenising {N_SEQ} validation sequences …")
    tok = AutoTokenizer.from_pretrained(
        "tdooms/ts-tokenizer-4096", pad_token="[EOS]", padding_side="right"
    )
    raw = load_dataset("roneneldan/TinyStories", split=f"validation[:{N_SEQ}]")
    texts = ["[BOS] " + r["text"].lower() for r in raw]
    ids = tok(texts, truncation=True, padding="max_length", max_length=n_ctx,
              return_tensors="pt", add_special_tokens=False)["input_ids"]

    # Padding tokens are excluded from the loss (CrossEntropyLoss default
    # ignore_index=-100), so short stories don't dilute the comparison.
    labels = ids.clone()
    labels[ids == tok.pad_token_id] = -100

    def mean_loss():
        losses = []
        for s in range(0, len(ids), BATCH):
            out = model(ids[s:s + BATCH].cuda(), labels=labels[s:s + BATCH].cuda())
            losses.append(float(out.loss))
        return sum(losses) / len(losses)

    clean = mean_loss()
    print(f"Clean loss: {clean:.4f}")

    # Splice the SAE reconstruction into the layer's forward pass.
    # Layer.forward is:  x = x + scale·attn(n1(x));  x = x + mlp(n2(x))
    #   resid-mid: replace x after the attention residual add
    #   mlp-out:   replace the MLP branch output
    def patched_forward(sae, point):
        def fwd(self, x, attn_mask=None):
            x = x + self.scale * self.attn(self.n1(x), attn_mask)
            if point == "resid-mid":
                x = sae(x)[0]
                x = x + self.mlp(self.n2(x))
            else:  # mlp-out
                x = x + sae(self.mlp(self.n2(x)))[0]
            return x
        return fwd

    results = {"clean_loss": clean, "mlp-out": [], "resid-mid": [],
               "nmse": {"mlp-out": [], "resid-mid": []}}

    def capture_point(layer, point):
        """Collect the activations the SAE reconstructs, for NMSE."""
        acts = []
        block = model.transformer.h[layer]
        if point == "resid-mid":
            h = block.n2.register_forward_pre_hook(
                lambda _, inp: acts.append(inp[0].detach()))
        else:
            h = block.mlp.register_forward_hook(
                lambda _, inp, out: acts.append(out.detach()))
        for s in range(0, min(len(ids), 128), BATCH):
            model(ids[s:s + BATCH].cuda())
        h.remove()
        return torch.cat(acts).flatten(0, 1)

    for point in ["mlp-out", "resid-mid"]:
        for layer in tqdm(range(model.config.n_layer), desc=point):
            sae = SAE.from_pretrained(
                SAE_REPO, point=(point, layer), expansion=4, k=30
            ).cuda()

            block = model.transformer.h[layer]
            original = block.forward
            block.forward = types.MethodType(patched_forward(sae, point), block)
            patched = mean_loss()
            block.forward = original

            added = (patched - clean) / clean
            results[point].append(added)

            x = capture_point(layer, point)
            x_hat, _ = sae(x)
            nmse = float((x - x_hat).pow(2).mean() / x.pow(2).mean())
            results["nmse"][point].append(nmse)

            del sae, x, x_hat
            torch.cuda.empty_cache()

    json.dump(results, open(RESULTS, "w"), indent=2)
    print(f"Saved {RESULTS.name}")

out = results["mlp-out"]
mid = results["resid-mid"]
print(f"mlp-out  loss added: {[f'{v:.4f}' for v in out]}")
print(f"resid-mid loss added: {[f'{v:.4f}' for v in mid]}")

# ============================================================
# Plot configuration
# ============================================================
pio.templates.default = "plotly_white"

layers = list(range(1, len(out) + 1))

# ============================================================
# Figure construction
# ============================================================
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=layers,
        y=out,
        mode="lines+markers",
        name="out"
    )
)

fig.add_trace(
    go.Scatter(
        x=layers,
        y=mid,
        mode="lines+markers",
        name="mid"
    )
)

# ============================================================
# Annotations & layout
# ============================================================
fig.add_annotation(
    x=6.1,
    y=out[-1],
    text="<b>mlp_out</b>",
    showarrow=False,
    xanchor="left"
)

fig.add_annotation(
    x=6.1,
    y=mid[-1],
    text="<b>resid_mid</b>",
    showarrow=False,
    xanchor="left"
)

fig.update_layout(
    showlegend=False,
    width=700,
    height=400
)

fig.update_xaxes(title="Layer")
fig.update_yaxes(title="Loss Added")

# ============================================================
# Save figure
# ============================================================
fig.write_image(HERE / "appendix_g.png", scale=4)
print(f"Saved {HERE / 'appendix_g.png'}")
