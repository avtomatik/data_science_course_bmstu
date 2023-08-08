#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:54:37 2022

@author: alexander
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns

# =============================================================================
# penguins = sns.load_dataset('penguins')
# sns.displot(x='flipper_length_mm', kde=True, data=penguins)
# sns.displot(x='flipper_length_mm', kde=True, hue='species', data=penguins)
# sns.displot(x='flipper_length_mm', col='species', data=penguins)
# sns.displot(x='flipper_length_mm', row='species', data=penguins)
# =============================================================================


# =============================================================================
# x, y = sps.multivariate_normal(cov=[[2, 1], [1, 2]]).rvs(size=1000).transpose()
# plt.figure(figsize=(12, 7))
# plt.title(
#     r"Ядерная оценка плотности $\mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$")
# sns.kdeplot(x=x, y=y, n_levels=15, shade=True, cmap='magma')
# =============================================================================


iris = sns.load_dataset('iris')
setosa = iris.loc[iris.species == 'setosa']
virginica = iris.loc[iris.species == 'virginica']
versicolor = iris.loc[iris.species == 'versicolor']

plt.figure(figsize=(12, 8))
with sns.axes_style('darkgrid'):
    ax = sns.kdeplot(
        x=setosa.sepal_length, y=setosa.sepal_width, label='setosa', cmap='Blues'
    )
    ax = sns.kdeplot(
        x=virginica.sepal_length, y=virginica.sepal_width, label='virginica', cmap='Greens'
    )
    ax = sns.kdeplot(
        x=versicolor.sepal_length, y=versicolor.sepal_width, label='versicolor', cmap='Reds'
    )
    ax.set_title('Fisher\'s Iris')


# =============================================================================
# data = sps.norm.rvs(size=(1000, 6)) + np.arange(6) / 2
# plt.figure(figsize=(8, 7))
# sns.boxplot(data=data, palette='Set2')
# =============================================================================


# =============================================================================
# tips = sns.load_dataset('tips')
#
# plt.figure(figsize=(15, 6))
#
# plt.subplot(121)
# sns.boxplot(x='day', y='total_bill', data=tips, palette='Set3')
# plt.ylabel('Сумма счёта, долл. США')
#
# plt.subplot(122)
# sns.boxplot(x='day', y='tip', data=tips, palette='Set3')
# plt.ylabel('Размер чаевых, долл. США')
# =============================================================================


# =============================================================================
# tips = sns.load_dataset('tips')
#
# plt.figure(figsize=(15, 6))
#
# plt.subplot(121)
# ax = sns.boxplot(
#     x='day', y='total_bill', hue='smoker', data=tips, palette='Set3'
# )
# ax.legend().get_frame().set_facecolor('white')
# plt.ylabel('Сумма счёта, долл. США')
#
# plt.subplot(122)
# ax = sns.boxplot(x='day', y='tip', hue='smoker', data=tips, palette='Set3')
# ax.legend().get_frame().set_facecolor('white')
# plt.ylabel('Размер чаевых, долл. США')
# =============================================================================


# =============================================================================
# tips = sns.load_dataset('tips')
#
# plt.figure(figsize=(15, 6))
#
# plt.subplot(121)
# ax = sns.boxplot(
#     x='day', y='total_bill', hue='sex', data=tips, palette='Set3'
# )
# ax.legend().get_frame().set_facecolor('white')
# plt.ylabel('Сумма счёта, долл. США')
#
# plt.subplot(122)
# ax = sns.boxplot(x='day', y='tip', hue='sex', data=tips, palette='Set3')
# ax.legend().get_frame().set_facecolor('white')
# plt.ylabel('Размер чаевых, долл. США')
# =============================================================================


# =============================================================================
# data = sps.norm.rvs(size=(20, 6)) + np.arange(6) / 2
# plt.figure(figsize=(15, 7))
#
# with sns.plotting_context(font_scale=1.5), sns.axes_style('darkgrid'):
#     plt.subplot(121)
#     sns.boxenplot(data=data, palette='Set2')
#
# with sns.plotting_context(font_scale=1.5), sns.axes_style('whitegrid'):
#     plt.subplot(122)
#     sns.boxenplot(data=data, palette='Set2')
# =============================================================================


# =============================================================================
# data = sps.norm.rvs(size=(20, 6)) + np.arange(6) / 2
# plt.figure(figsize=(8, 5))
# sns.violinplot(data=data, palette='Set2', bw=.2, cut=1, linewidth=1)
# =============================================================================


# tips = sns.load_dataset('tips')

# with sns.plotting_context('notebook', font_scale=1.5):
#     plt.figure(figsize=(8, 5))
#     sns.violinplot(
#         x='day', y='tip', hue='smoker', data=tips, palette='Set2', split=True
#     )
#     plt.ylabel('Размер чаевых, долл. США')
#     plt.xlabel('День недели')


# =============================================================================
# sns.set(style='white', font_scale=1.3)
#
# df = sns.load_dataset('iris')
#
# g = sns.PairGrid(df, diag_sharey=False)
# g.map_lower(sns.kdeplot)
# g.map_upper(plt.scatter)
# g.map_diag(sns.kdeplot)
# =============================================================================


# =============================================================================
# sns.set(style='white', font_scale=1.3)
#
# df = sns.load_dataset('iris')
#
# g = sns.PairGrid(df, hue='species', height=4, diag_sharey=False)
# g.map_lower(sns.kdeplot)
# g.map_upper(plt.scatter)
# g.map_diag(sns.histplot, kde=True)
# =============================================================================

# =============================================================================
# flights_flat = sns.load_dataset('flights')
# flights = flights_flat.pivot_table(
#     index='month',
#     columns='year',
#     values='passengers'
# )
#
# sns.set(font_scale=1.3)
# f, ax = plt.subplots(figsize=(12, 7))
# sns.heatmap(flights, annot=True, fmt='d', ax=ax, cmap='viridis')
# plt.ylim((0, 12))
# =============================================================================
