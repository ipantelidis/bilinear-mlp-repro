# ============================================================
# Imports
# ============================================================
import plotly.graph_objects as go
import plotly.io as pio

# ============================================================
# Plot configuration
# ============================================================
pio.templates.default = "plotly_white"

# Data (precomputed losses per layer)
out = [0.011913, 0.020534, 0.037918, 0.048743, 0.059292, 0.086776]
mid = [0.017323, 0.02781, 0.087793, 0.16479, 0.21956, 0.28609]

layers = list(range(1, 7))

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
fig.write_image(
    "appendix_g.png",
    scale=4
)
