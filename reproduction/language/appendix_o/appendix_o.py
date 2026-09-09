# ============================================================
# Appendix O — Input Features of the Negation Feature
# Reproduces Tables 7–9 from Pearce et al. (2025)
#
# Shows the top-activating text examples for each input SAE
# feature contributing to the "not-good" negation circuit.
# Output: appendix_o.html  (highlighted token tables)
#         appendix_o.txt   (plain text version)
# ============================================================

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def get_resid_mid(model, input_ids, layer):
    cap = {}
    h = model.transformer.h[layer].n2.register_forward_pre_hook(
        lambda _, inp: cap.__setitem__("x", inp[0].detach())
    )
    with torch.no_grad():
        model(input_ids)
    h.remove()
    return cap["x"]


torch.set_grad_enabled(False)

LAYER   = 4
TOP_K   = 5      # top-K examples per feature
CONTEXT = 15     # tokens on each side of the active token
N_SEQ   = 4096   # sequences to scan
BATCH   = 16

# Input features and descriptions (from paper Tables 7–9)
FEATURES = {
    326:  "crashing and breaking",
    1376: "dangerous actions",
    1636: "nervous / worried",
    123:  "negative attribute",
    990:  "a bad turn",
    1929: "inability to do something / failure",
    491:  "body parts",
    947:  "bad ending (seriously)",
    882:  "being positive",
    240:  "negation of attribute",
    766:  "inability to perform physical actions",
    1604: "avoiding bad things",
    1395: "positive ending",
}

# ── Load model and input SAE ─────────────────────────────────
print("Loading model …")
model = Transformer.from_pretrained("tdooms/ts-medium", device="cuda")
fix_buffers(model)

print("Loading input SAE (resid-mid, layer 4) …")
inp_sae = SAE.from_pretrained(
    "tdooms/ts-medium-scope",
    point=("resid-mid", LAYER),
    expansion=4, k=30,
).cuda()

# ── Tokenise dataset ─────────────────────────────────────────
print(f"Tokenising {N_SEQ} sequences …")
ts_tok = AutoTokenizer.from_pretrained(
    "tdooms/ts-tokenizer-4096", pad_token="[EOS]", padding_side="right"
)
raw   = load_dataset("roneneldan/TinyStories", split=f"train[:{N_SEQ}]")
texts = ["[BOS] " + r["text"].lower() for r in raw]
ids   = ts_tok(
    texts, truncation=True, padding="max_length",
    max_length=256, return_tensors="pt", add_special_tokens=False,
)["input_ids"]

# ── Scan sequences — store full activation maps ───────────────
feat_ids  = list(FEATURES.keys())
# top_store[f] = list of (max_act, seq_idx, full_act_tensor)
top_store = {f: [] for f in feat_ids}
pad_id    = ts_tok.pad_token_id or 0

print("Scanning sequences …")
for start in tqdm(range(0, len(ids), BATCH)):
    batch = ids[start:start + BATCH].cuda()
    acts  = get_resid_mid(model, batch, LAYER)    # (B, T, d)
    B, T, D = acts.shape
    sae_h = inp_sae.encode(acts.view(B * T, D))   # (B*T, d_feats)
    sae_h = sae_h.view(B, T, -1).cpu()            # (B, T, d_feats)

    for f in feat_ids:
        feat_acts = sae_h[:, :, f]                # (B, T)
        max_vals  = feat_acts.max(dim=1).values   # (B,)
        for b in range(B):
            mv = float(max_vals[b])
            if mv > 0:
                top_store[f].append((mv, start + b, feat_acts[b].clone()))

    torch.cuda.empty_cache()

for f in feat_ids:
    top_store[f] = sorted(top_store[f], key=lambda x: -x[0])[:TOP_K]


# ── Helper: colour a token span using Blues colormap ─────────
def act_to_hex(val, global_max):
    """Map activation value to a blue hex colour (white → dark blue)."""
    if val <= 0 or global_max <= 0:
        return None
    intensity = min(val / global_max, 1.0)
    # Use Blues from 0.2 (light) to 0.9 (dark) to avoid near-white
    r, g, b, _ = plt.cm.Blues(0.2 + 0.7 * intensity)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def render_html_row(seq_idx, feat_acts, global_max):
    """Return an HTML string for one example row with multi-tone highlights."""
    toks = [t for t in ids[seq_idx].tolist() if t != pad_id]
    T    = min(len(toks), len(feat_acts))
    conv = ts_tok.convert_ids_to_tokens

    # Find the context window around the peak
    peak = int(feat_acts[:T].argmax())
    s    = max(0, peak - CONTEXT)
    e    = min(T, peak + CONTEXT + 1)

    parts = []
    if s > 0:
        parts.append("<span style='color:#aaa'>… </span>")
    for i in range(s, e):
        word = ts_tok.convert_tokens_to_string(conv([toks[i]])).strip() or " "
        col  = act_to_hex(float(feat_acts[i]), global_max)
        if col:
            # Darker backgrounds get white text for readability
            intensity = float(feat_acts[i]) / global_max
            fg = "white" if intensity > 0.6 else "black"
            parts.append(
                f"<span style='background:{col};color:{fg};"
                f"padding:1px 3px;border-radius:2px'> {word} </span>"
            )
        else:
            parts.append(f" {word} ")
    if e < T:
        parts.append("<span style='color:#aaa'> …</span>")
    return "".join(parts)


def render_text_row(seq_idx, feat_acts, global_max):
    """Plain-text row: mark tokens above 30% threshold with *asterisks*."""
    toks = [t for t in ids[seq_idx].tolist() if t != pad_id]
    T    = min(len(toks), len(feat_acts))
    conv = ts_tok.convert_ids_to_tokens
    peak = int(feat_acts[:T].argmax())
    s    = max(0, peak - CONTEXT)
    e    = min(T, peak + CONTEXT + 1)
    out  = [] if s == 0 else ["…"]
    for i in range(s, e):
        word = ts_tok.convert_tokens_to_string(conv([toks[i]])).strip() or " "
        v    = float(feat_acts[i])
        thresh = global_max * 0.30
        out.append(f"*{word}*" if v >= thresh else word)
    if e < T:
        out.append("…")
    return " ".join(out)


# ── Plain-text output ─────────────────────────────────────────
txt_lines = []
for f, desc in FEATURES.items():
    txt_lines.append(f"\nInput Feature ({f}): {desc}")
    txt_lines.append("=" * 60)
    for max_act, seq_idx, feat_acts in top_store[f]:
        txt_lines.append(f"  [{max_act:.2f}]  {render_text_row(seq_idx, feat_acts, max_act)}")
(HERE / "appendix_o.txt").write_text("\n".join(txt_lines), encoding="utf-8")
print("  → appendix_o.txt")

# ── HTML output with blue-gradient colour highlights ──────────
html = ["""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
  body { font-family: monospace; font-size:13px; padding:20px; max-width:960px; }
  h2   { color:#2c3e50; }
  .feat{ margin:28px 0; }
  .feat h3 { background:#2c3e50; color:#fff; padding:6px 14px;
             border-radius:4px; margin:0 0 8px; font-size:14px; }
  .ex  { margin:5px 0 5px 10px; line-height:2.2; background:#fafafa;
         padding:4px 8px; border-radius:3px; }
  .sc  { color:#aaa; font-size:11px; margin-right:8px; }
</style></head><body>
<h2>Appendix O — Input Features of the Negation Feature</h2>"""]

for f, desc in FEATURES.items():
    html.append(f"<div class='feat'><h3>Input Feature ({f}): {desc}</h3>")
    for max_act, seq_idx, feat_acts in top_store[f]:
        row = render_html_row(seq_idx, feat_acts, max_act)
        html.append(
            f"<div class='ex'><span class='sc'>[{max_act:.2f}]</span>{row}</div>"
        )
    html.append("</div>")

html.append("</body></html>")
(HERE / "appendix_o.html").write_text("\n".join(html), encoding="utf-8")
print("  → appendix_o.html")
print("Done.")
