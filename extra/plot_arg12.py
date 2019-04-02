#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot argument lengths in CoNLL 2016 dataset.

Example:
  ./plot_arg12.py en ../data/conll16st-en-03-29-16-train/ ./plots/en_arg12w_lengths.pdf ./plots/en_arg12c_lengths.pdf
  ./plot_arg12.py zh ../data/conll16st-zh-01-08-2016-train/ ./plots/zh_arg12w_lengths.pdf ./plots/zh_arg12c_lengths.pdf
"""

import numpy as np
import matplotlib.pyplot as plt

from conll16st_data.files import load_relations_gold


def count_arg12w(relations_gold, xcrop, xmax):
    print("count argument lengths at word level")

    # count
    arg1w_cnts = {}
    arg2w_cnts = {}
    relw_cnts = {}
    crop_cnt = 0
    for rel_id, r in relations_gold.iteritems():
        # argument lengths at word level
        arg1w_len = len(r['Arg1']['TokenList'])
        arg2w_len = len(r['Arg2']['TokenList'])
        relw_len = arg1w_len + arg2w_len + len(r['Connective']['TokenList']) + len(r['Punctuation']['TokenList'])

        if arg1w_len > xcrop or arg2w_len > xcrop:  # count cropped strings
            crop_cnt += 1
        if arg1w_len > xmax:  # limit too long relations
            arg1w_len = xmax
        if arg2w_len > xmax:  # limit too long relations
            arg2w_len = xmax
        if relw_len > xmax:  # limit too long relations
            relw_len = xmax

        try:
            arg1w_cnts[arg1w_len] += 1
        except KeyError:
            arg1w_cnts[arg1w_len] = 1
        try:
            arg2w_cnts[arg2w_len] += 1
        except KeyError:
            arg2w_cnts[arg2w_len] = 1
        try:
            relw_cnts[relw_len] += 1
        except KeyError:
            relw_cnts[relw_len] = 1

    # prepare
    arg1_x, arg1_y = zip(*sorted(arg1w_cnts.iteritems(), key=lambda a: a[0]))
    arg2_x, arg2_y = zip(*sorted(arg2w_cnts.iteritems(), key=lambda a: a[0]))
    rel_x, rel_y = zip(*sorted(relw_cnts.iteritems(), key=lambda a: a[0]))

    print("- arg1_x range: {}-{}".format(min(arg1_x), max(arg1_x)))
    print("- arg2_x range: {}-{}".format(min(arg2_x), max(arg2_x)))
    print("- rel_x range: {}-{}".format(min(rel_x), max(rel_x)))
    print("- relations: {}".format(len(relations_gold.keys())))
    print("- cropped relations: {}".format(crop_cnt))
    return arg1_x, arg1_y, arg2_x, arg2_y, rel_x, rel_y


def count_arg12c(relations_gold, xcrop, xmax):
    print("count argument lengths at char level")

    # count
    arg1c_cnts = {}
    arg2c_cnts = {}
    relc_cnts = {}
    crop_cnt = 0
    for rel_id, r in relations_gold.iteritems():
        # argument lengths at character level
        arg1c_len = len(r['Arg1']['RawText'])
        arg2c_len = len(r['Arg2']['RawText'])
        relc_len = arg1c_len + arg2c_len + len(r['Connective']['RawText']) + len(r['Punctuation']['RawText'])

        if arg1c_len > xcrop or arg2c_len > xcrop:  # count cropped strings
            crop_cnt += 1
        if arg1c_len > xmax:  # limit too long relations
            arg1c_len = xmax
        if arg2c_len > xmax:  # limit too long relations
            arg2c_len = xmax
        if relc_len > xmax:  # limit too long relations
            relc_len = xmax

        try:
            arg1c_cnts[arg1c_len] += 1
        except KeyError:
            arg1c_cnts[arg1c_len] = 1
        try:
            arg2c_cnts[arg2c_len] += 1
        except KeyError:
            arg2c_cnts[arg2c_len] = 1
        try:
            relc_cnts[relc_len] += 1
        except KeyError:
            relc_cnts[relc_len] = 1

    # prepare
    arg1_x, arg1_y = zip(*sorted(arg1c_cnts.iteritems(), key=lambda a: a[0]))
    arg2_x, arg2_y = zip(*sorted(arg2c_cnts.iteritems(), key=lambda a: a[0]))
    rel_x, rel_y = zip(*sorted(relc_cnts.iteritems(), key=lambda a: a[0]))

    print("- arg1_x range: {}-{}".format(min(arg1_x), max(arg1_x)))
    print("- arg2_x range: {}-{}".format(min(arg2_x), max(arg2_x)))
    print("- rel_x range: {}-{}".format(min(rel_x), max(rel_x)))
    print("- relations: {}".format(len(relations_gold.keys())))
    print("- cropped relations: {}".format(crop_cnt))
    return arg1_x, arg1_y, arg2_x, arg2_y, rel_x, rel_y


def plot_arg12(fname, arg1_x, arg1_y, arg2_x, arg2_y, rel_x, rel_y, xcrop, xmax, xticks=[], yticks=[]):
    print("plot argument lengths")

    # prepare
    fig, ax = plt.subplots()
    linestyle_lines = 'solid'
    linestyle_median = (0, (5, 5))  # dashed

    # plot zero line
    ax.axhline(y=0, color='k', linestyle=linestyle_lines, linewidth=1)
    
    # plot crop line
    ax.axvline(x=xcrop, color='k', linestyle=linestyle_lines, linewidth=1)

    # plot median
    arg1_xmedian = arg1_x[arg1_y.index(max(arg1_y))]
    arg2_xmedian = arg2_x[arg2_y.index(max(arg2_y))]
    ax.axvline(x=arg1_xmedian, color='r', linestyle=linestyle_median, linewidth=1)
    ax.axvline(x=arg2_xmedian, color='b', linestyle=linestyle_median, linewidth=1)
    text_x = arg1_xmedian + xmax * 0.04
    text_y = max(max(arg1_y), max(arg2_y))
    plt.text(text_x, text_y * 0.98, "median: {}".format(arg1_xmedian), color='r')
    plt.text(text_x, text_y * 0.93, "median: {}".format(arg2_xmedian), color='b')

    # plot raw points
    ax.plot(arg1_x, arg1_y, 'r.-', linewidth=0.5)
    ax.plot(arg2_x, arg2_y, 'b.-', linewidth=0.5)
    #ax.plot(rel_x, rel_y, 'k-', linewidth=0.5)

    # plot smoothed points
    # arg1_xi = []
    # arg1_yi = []
    # for i in range(len(arg1_x)):
    #     arg1_xi.append(arg1_x[i])
    #     y = arg1_y[i]
    #     yc = 1
    #     try:
    #         y += arg1_y[i - 1]
    #         yc += 1
    #     except IndexError:
    #         pass
    #     try:
    #         y += arg1_y[i + 1]
    #         yc += 1
    #     except IndexError:
    #         pass
    #     arg1_yi.append(float(y) / yc)
    # ax.plot(arg1_xi, arg1_yi, 'r-')

    xbound = ax.get_xbound()
    ybound = ax.get_ybound()
    # ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    # ax.set_ylim(-text_y * 0.06, None)
    ax.set_xticks(list(ax.get_xticks()) + xticks)  # extra ticks on x-axis
    ax.set_yticks(list(ax.get_yticks()) + yticks)  # extra ticks on x-axis
    ax.set_xlim(*xbound)
    ax.set_ylim(*ybound)

    # save plot
    if fname is not None:
        print("- saving to '{}'".format(fname))
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    import sys
    lang = sys.argv[1]
    data_dir = sys.argv[2]
    arg12w_fname = None
    arg12c_fname = None
    if len(sys.argv) > 4:
        arg12w_fname = sys.argv[3]
        arg12c_fname = sys.argv[4]

    print("load dataset '{}'".format(data_dir))
    relations_gold = load_relations_gold(data_dir, with_senses=True, with_rawtext=True)

    if lang == 'en':
        xcrop, xmax = 100, 200
        params = count_arg12w(relations_gold, xcrop=xcrop, xmax=xmax)
        plot_arg12(arg12w_fname, *params, xcrop=xcrop, xmax=xmax, xticks=[25, 50], yticks=[50, 100])
        xcrop, xmax = 400, 1000
        params = count_arg12c(relations_gold, xcrop=xcrop, xmax=xmax)
        plot_arg12(arg12c_fname, *params, xcrop=xcrop, xmax=xmax, xticks=[100, 200], yticks=[25, 50])
    elif lang == 'zh':
        xcrop, xmax = 500, 800
        params = count_arg12w(relations_gold, xcrop=xcrop, xmax=xmax)
        plot_arg12(arg12w_fname, *params, xcrop=xcrop, xmax=xmax, xticks=[50, 100], yticks=[50, 100])
        xcrop, xmax = 900, 1500
        params = count_arg12c(relations_gold, xcrop=xcrop, xmax=xmax)
        plot_arg12(arg12c_fname, *params, xcrop=xcrop, xmax=xmax, xticks=[100], yticks=[25, 50])
    else:
        raise Exception("unknown language '{}'".format(lang))
    print("")

