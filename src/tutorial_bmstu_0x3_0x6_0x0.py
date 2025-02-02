#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 11:21:54 2022

@author: alexander
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 2 * np.pi, 100)

plt.figure(figsize=(10, 5))
plt.plot(
    x,
    np.sin(x),
    linewidth=2,
    color='g',
    dashes=[8, 4],
    label=r'$\sin x$'
)
plt.plot(
    x,
    np.cos(x),
    linewidth=2,
    color='r',
    dashes=[8, 4, 2, 4],
    label=r'$\cos x$'
)
plt.axis([0, 2 * np.pi, -1, 1])
plt.xticks(
    np.linspace(0, 2 * np.pi, 9),
    (
        '0',
        r'$\frac{1}{4}\pi$',
        r'$\frac{1}{2}\pi$',
        r'$\frac{3}{4}\pi$',
        r'$\pi$',
        r'$\frac{5}{4}\pi$',
        r'$\frac{3}{2}\pi$',
        r'$\frac{7}{4}\pi$',
        r'$2\pi$'
    ),
    fontsize=20
)
plt.yticks(fontsize=12)
plt.title(r'$\sin x$, $\cos x$', fontsize=20)
plt.xlabel(r'$x$', fontsize=20)
plt.ylabel(r'$y$', fontsize=20)
plt.legend(loc=0, fontsize=20)
plt.grid(ls=':')
plt.show()


# =============================================================================
# plt.xscale('log')
# =============================================================================
# =============================================================================
# plt.polar(args, kwargs)
# =============================================================================

# =============================================================================
# x = np.linspace(-1, 1, 50)
# y = x
# z = np.outer(x, y)
#
# plt.figure(figsize=(5, 5))
# plt.contour(x, y, z)
# plt.show()
# =============================================================================


# =============================================================================
# x = np.linspace(-1, 1, 50)
# y = x
# z = np.outer(x, y)
#
# plt.figure(figsize=(5, 5))
# curves = plt.contour(x, y, z, np.linspace(-1, 1, 11))
# plt.clabel(curves)
# plt.title(r'$z=xy$', fontsize=10)
# plt.show()
# =============================================================================


# =============================================================================
# x = np.linspace(-1, 1, 50)
# y = x
# z = np.outer(x, y)
#
# plt.figure(figsize=(5, 5))
# plt.contourf(x, y, z, np.linspace(-1, 1, 11))
# plt.colorbar()
# plt.show()
# =============================================================================


# =============================================================================
# t = np.linspace(0, 4 * np.pi, 100)
# x = np.cos(t)
# y = np.sin(t)
# z = t / (4 * np.pi)
#
# fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False)
# fig.add_axes(ax)
# ax.plot(x, y, z)
# =============================================================================


# =============================================================================
# t = np.linspace(0, 4 * np.pi, 100)
# x = np.cos(t)
# y = np.sin(t)
# z = t / (4 * np.pi)
#
# fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False)
# ax.elev, ax.azim = 45, 30
# fig.add_axes(ax)
# ax.plot(x, y, z)
# =============================================================================


# =============================================================================
# X = 10
# N = 50
# u = np.linspace(-X, X, N)
# x, y = np.meshgrid(u, u)
# r = np.sqrt(x ** 2 + y ** 2)
# z = np.sin(r) / r
#
# fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False)
# ax.plot_surface(x, y, z, rstride=10, cstride=10)
# fig.add_axes(ax)
# plt.show()
# =============================================================================


# =============================================================================
# X = 10
# N = 50
# u = np.linspace(-X, X, N)
# x, y = np.meshgrid(u, u)
# r = np.sqrt(x ** 2 + y ** 2)
# z = np.sin(r) / r
#
# fig = plt.figure()
# ax = Axes3D(fig, auto_add_to_figure=False)
# ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='gnuplot')
# fig.add_axes(ax)
# plt.show()
# =============================================================================


t = np.linspace(0, 2 * np.pi, 100)
theta, phi = np.meshgrid(t, t)
r = .2
x, y, z = (
    (1 + r * np.cos(phi)) * np.cos(theta),
    (1 + r * np.cos(phi)) * np.sin(theta),
    r * np.sin(phi)
)
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
ax.elev = 45
ax.plot_surface(x, y, z, rstride=2, cstride=1)
fig.add_axes(ax)
plt.show()
