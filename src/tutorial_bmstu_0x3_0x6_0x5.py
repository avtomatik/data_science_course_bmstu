#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 15:07:25 2022

@author: alexander
"""

import numpy as np
import plotly.graph_objs as go

# =============================================================================
# penguins = sns.load_dataset('penguins')
#
# _species = pd.unique(penguins.species)
#
# fig = go.Figure()
#
# for specie in _species:
#     hist = go.Histogram(
#         x=penguins[penguins.species == specie]['flipper_length_mm']
#     )
#     fig.add_trace(hist)
#
# fig.update_layout(barmode='overlay')
# fig.update_traces(opacity=.75)
# fig.show()
# =============================================================================


t = np.linspace(0, 4 * np.pi, 100)
theta, phi = np.meshgrid(t, t)
r = .2
x, y, z = (
    (1 + r * np.cos(phi)) * np.cos(theta),
    (1 + r * np.cos(phi)) * np.sin(theta),
    r * np.sin(phi)
)

surface = go.Surface(x=x, y=y, z=z)
data = [surface]
axparams = dict(
    gridcolor='rgb(255, 255, 255)',
    zerolinecolor='rgb(255, 255, 255)',
    showbackground=True,
    backgroundcolor='rgb(230, 230, 230)',
)
layout = go.Layout(
    title='Torus',
    scene=dict(
        xaxis=axparams,
        yaxis=axparams,
        zaxis=axparams,
    ),
    scene_aspectmode='data',
)

fig = go.Figure(data=data, layout=layout)
fig.show()
