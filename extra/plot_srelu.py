#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot SReLU activation function.

Example:
  ./plot_srelu.py ./plots/func_srelu_tanh.pdf ./plots/func_srelu_relu.pdf
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_srelu(fname, tl, al, tr, ar):
    print("plot srelu activation function")

    z = np.arange(-2, 2, .05)
    tl_i = np.where((tl - 0.00001 <= z) & (z <= tl + 0.00001))[0][0]
    tr_i = np.where((tr - 0.00001 <= z) & (z <= tr + 0.00001))[0][0]

    yl = tl + al * (z - tl)
    ym = z
    yr = tr + ar * (z - tr)

    y = np.concatenate([yl[:tl_i], ym[tl_i:tr_i], yr[tr_i:]], axis=0)

    # prepare
    fig, ax = plt.subplots()
    linestyle_lines = 'solid'
    linestyle_thresholds = (0, (5, 5))  # dashed

    ax.plot(z, y, 'r-', linewidth=1)
    #ax.plot(z, yl, 'g-', linewidth=0.5)
    #ax.plot(z, ym, 'g-', linewidth=0.5)
    #ax.plot(z, yr, 'g-', linewidth=0.5)

    # plot zero line
    ax.axhline(y=0, color='k', linestyle=linestyle_lines, linewidth=1)

    # plot tl and tr thresholds
    ax.axvline(x=tl, color='k', linestyle=linestyle_thresholds, linewidth=1)
    ax.axvline(x=tr, color='k', linestyle=linestyle_thresholds, linewidth=1)

    ax.set_ylim([-2.0, 2.0])
    ax.set_xlim([-2.0, 2.0])
    fig.set_size_inches(4.8, 4.8)  # default [6.4, 4.8]

    # save plot
    if fname is not None:
        print("- saving to '{}'".format(fname))
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    import sys
    tanh_fname = None
    relu_fname = None
    if len(sys.argv) > 2:
        tanh_fname = sys.argv[1]
        relu_fname = sys.argv[2]

    plot_srelu(tanh_fname, tl=-0.75, al=0.2, tr=0.75, ar=0.2)
    plot_srelu(relu_fname, tl=-0.0, al=0.1, tr=0.75, ar=3.0)
    print("")

