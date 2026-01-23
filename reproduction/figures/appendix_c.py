# =====================================
# Imports and global setup
# =====================================

import plotly.io as pio
import torch
from einops import *
from image import MNIST, Model, plot_explanation
from kornia.augmentation import RandomGaussianNoise
from torch import nn

pio.templates.default = "plotly_white"

# Shared color configuration 
color = dict(
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
)

# =====================================
# Model & training configuration
# =====================================

mnist = Model.from_config(
    epochs=100,
    wd=1.0,
    n_layer=1,
    residual=False,
    seed=420,
).cuda()

transform = nn.Sequential(
    RandomGaussianNoise(mean=0, std=0.5, p=1),
)

train, test = MNIST(train=True), MNIST(train=False)

# =====================================
# Train model
# =====================================

torch.set_grad_enabled(True)
mnist.fit(train, test, transform)
torch.set_grad_enabled(False)

# =====================================
# Explanation: correctly classified example of a '5'
# =====================================

sample = test.x[8]
correctly_classified_5 = plot_explanation(mnist, sample)

correctly_classified_5.write_image(
    "../outputs/figures/correctly_classified_5.png"
)

# =====================================
# Explanation: misclassified example of a '2'
# =====================================

sample = test.x[321]
misclassified_2 = plot_explanation(mnist, sample)

misclassified_2.write_image(
    "../outputs/figures/misclassified_2.png"
)
