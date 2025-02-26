#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:16:54 2022

@author: alexander
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sinplot(flip: int = 1) -> None:
    _n = 7
    x = np.linspace(0, 14, 100)
    for _ in range(1, _n):
        plt.plot(x, np.sin(x + .5 * _) * (_n - _) * flip)


# =============================================================================
# Plot
# =============================================================================
plt.figure(figsize=(15, 9))
for _, context in enumerate(['notebook', 'paper', 'talk', 'poster'], start=1):
    sns.set(context=context)
    plt.subplot(2, 2, _)
    sinplot()
    plt.title(context)


# =============================================================================
# Plot
# =============================================================================
plt.figure(figsize=(15, 12))
for _, style in enumerate(
    ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'],
    start=1
):
    sns.set(style=style)
    plt.subplot(3, 2, _)
    sinplot()
    plt.title(style)


# =============================================================================
# Plot
# =============================================================================
with sns.plotting_context('notebook'), sns.axes_style('ticks'):
    sinplot()
    sns.despine()


# =============================================================================
# Plot
# =============================================================================
count = 100
colors = sns.color_palette('rainbow', count)
layers = np.linspace(1, 2, count)
plt.figure(figsize=(8, 4))
plt.title('Rainbow')
for _ in np.arange(count):
    nodes_x = np.linspace(-layers[_], layers[_], count)
    nodes_y = np.sqrt(layers[_] ** 2 - nodes_x ** 2)
    sns.lineplot(x=nodes_x, y=nodes_y, color=colors[_])

plt.xticks([])
plt.yticks([])
sns.despine(left=True, bottom=True)
