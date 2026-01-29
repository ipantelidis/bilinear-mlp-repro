from jaxtyping import Float
from torch import Tensor
from einops import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import torch
from encoder import VAE


def plot_explanation_vae(model, sample, latents=(0,1,2), eigenvalues=10):
    """
    Exact analogue of plot_explanation, but for a VAE.
    latents = which latent dimensions to explain.
    """
    colors = px.colors.qualitative.Plotly

    # get decomposition
    vals, vecs = model.encoder.decompose()
    vals, vecs = vals.cpu(), vecs.cpu()

    # compute eigenvector activations
    acts = einsum(
        sample.flatten().cpu(),
        vecs,
        "inp, lat comp inp -> lat comp"
    ).pow(2) * vals

    contrib, idxs = acts[list(latents)].sort(dim=-1)

    fig = make_subplots(rows=2, cols=1 + len(latents))

    # line plots (same as paper)
    for i in range(len(latents)):
        fig.add_scatter(
            y=contrib[i, -eigenvalues-2:].flip(0),
            mode="lines",
            marker=dict(color=colors[i]),
            row=1, col=1
        )
        fig.add_scatter(
            y=contrib[i, :eigenvalues+2],
            mode="lines",
            marker=dict(color=colors[i]),
            row=2, col=1
        )

    # eigenvector heatmaps
    for i, lat in enumerate(latents):
        fig.add_heatmap(
            z=vecs[lat][idxs[i, -1]].view(28,28).flip(0),
            colorscale="RdBu",
            zmid=0,
            showscale=False,
            row=1, col=i+2
        )
        fig.add_heatmap(
            z=vecs[lat][idxs[i, 0]].view(28,28).flip(0),
            colorscale="RdBu",
            zmid=0,
            showscale=False,
            row=2, col=i+2
        )

    return fig



def plot_eigenspectrum(model, digit, eigenvectors=3, eigenvalues=20, ignore_pos=[], ignore_neg=[]):
    """Plot the eigenspectrum for a given digit."""
    colors = px.colors.qualitative.Plotly
    fig = make_subplots(rows=2, cols=1 + eigenvectors)
    
    vals, vecs = model.decompose()
    vals, vecs = vals[digit].cpu(), vecs[digit].cpu()
    
    negative = torch.arange(eigenvectors)
    positive = -1 - negative

    fig.add_trace(go.Scatter(y=vals[-eigenvalues-2:].flip(0), mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=negative.flip(0), y=vals[positive].flip(0), mode='markers', marker=dict(color=colors[0])), row=1, col=1)

    fig.add_trace(go.Scatter(y=vals[:eigenvalues+2], mode="lines", marker=dict(color=colors[1])), row=2, col=1)
    fig.add_trace(go.Scatter(x=negative, y=vals[negative], mode='markers', marker=dict(color=colors[1])), row=2, col=1)

    for i, idx in enumerate(positive):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=1, col=i+2)

    for i, idx in enumerate(negative):
        fig.add_trace(go.Heatmap(z=vecs[idx].view(28, 28).flip(0), colorscale="RdBu", zmid=0, showscale=False), row=2, col=i+2)

    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_xaxes(visible=True, tickvals=[eigenvalues], ticktext=[f'{eigenvalues}'], zeroline=False, col=1)
    fig.update_yaxes(zeroline=True, rangemode="tozero", col=1)
    
    tickvals = [0] + [x.item() for i, x in enumerate(vals[positive]) if i not in ignore_pos]
    ticktext = [f'{val:.2f}' for val in tickvals]
    
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=1)

    tickvals = [0] + [x.item() for i, x in enumerate(vals[negative]) if i not in ignore_neg]
    ticktext = [f'{val:.2f}' for val in tickvals]
    fig.update_yaxes(visible=True, tickvals=tickvals, ticktext=ticktext, col=1, row=2)

    fig.update_coloraxes(showscale=False)
    fig.update_layout(autosize=False, width=170*(eigenvectors+1), height=300, margin=dict(l=0, r=0, b=0, t=0), template="plotly_white")
    fig.update_legends(visible=False)
    
    return fig


# load trained model
model = VAE(d_input=784, d_hidden=400, d_latent=20)
model.load_state_dict(torch.load("encoder_vae.pt", map_location="cpu"))
model.eval()

# make ONE plot and SAVE it
fig = plot_eigenspectrum(model.encoder, digit=0)
fig.write_html("latent_0.html")

print("Saved latent_0.html")