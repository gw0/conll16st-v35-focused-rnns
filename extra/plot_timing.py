#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot empirical time complexity from console logs of FR system.

Example:
  ./plot_timing.py ./plots/timing_al_train.pdf ./plots/timing_al_classify.pdf arg1_len "max argument length" ../models-timing/conll16st-v35-*-al-*-0
  ./plot_timing.py ./plots/timing_is_train.pdf ./plots/timing_is_classify.pdf input_size "num. of samples" ../models-timing/conll16st-v35-*-is-*-0
  ./plot_timing.py ./plots/timing_fr_train.pdf ./plots/timing_fr_classify.pdf rnn_num "num. of focused RNNs" ../models-timing/conll16st-v35-*-fr-*-0
"""

import codecs
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_logs(field, model_dirs, fname_fmt):
    print("parse FR system console logs")

    # parse from console logs
    data = []
    for d in model_dirs:
        fname = fname_fmt.format(d)
        f = codecs.open(fname, 'r', encoding='utf8')

        # iterate over all lines
        lang = None
        mode = None
        first_rel_ids = None
        field_val = None
        time_epoch = None
        time_epoch_samples = None
        time_predicted = None
        for line in f:
            # from configuration dump
            m = re.match(r".*  config 'lang': (.+)$", line)
            if m:
                lang = m.group(1)
            m = re.match(r".*  config 'mode': (.+)$", line)
            if m:
                mode = m.group(1)
            m = re.match(r".*  config '{}': ([0-9\.]+)$".format(field), line)
            if m:
                field_val = float(m.group(1))

            # from first loaded dataset
            m = re.match(r".*  lang:.* rel_ids: ([0-9]+).*$", line)
            if m and first_rel_ids is None:
                first_rel_ids = float(m.group(1))

            # from timing footer
            m = re.match(r".*time:.* epoch: ([0-9\.]+).*$", line)
            if m:
                time_epoch = float(m.group(1))
            m = re.match(r".*time:.* epoch_samples: ([0-9\.]+).*$", line)
            if m:
                time_epoch_samples = float(m.group(1))
            m = re.match(r".*time:.* predicted: ([0-9\.]+).*$", line)
            if m:
                time_predicted = float(m.group(1))

        # convert format
        if mode == 'word':
            langm = lang
        elif mode == 'char':
            langm = lang + 'ch'
        else:
            raise Exception("Unknown language mode '{}' in '{}'".format(mode, fname))
        d = (langm, field_val, time_epoch, time_predicted, first_rel_ids, time_epoch_samples, fname)
        print(d)
        data.append(d)

        f.close()

    return data


def plot_timing(fname, xlabel, ylabel, xylists, lang_colors):
    print("plot timing: '{}'-'{}'".format(xlabel, ylabel))

    # prepare
    fig, ax = plt.subplots()
    linestyle_lines = 'solid'

    # plot zero line
    ax.axhline(y=0, color='k', linestyle=linestyle_lines, linewidth=1)

    # plot raw points
    xmax = max([ xylist[-1][0] for xylist in xylists.itervalues() ])
    ymax = max([ xylist[-1][1] for xylist in xylists.itervalues() ])
    for lang, xylist in xylists.iteritems():
        x, y = zip(*xylist)
        ax.plot(x, y, lang_colors[lang], linewidth=1)

        # add text
        text_x = x[-1] + xmax * 0.02
        text_y = y[-1] - ymax * 0.02
        plt.text(text_x, text_y, lang, color=lang_colors[lang][0:1])

    # xbound = ax.get_xbound()
    # ybound = ax.get_ybound()
    # # ax.set_xlim(-xmax * 0.03, xmax * 1.03)
    # # ax.set_ylim(-text_y * 0.06, None)
    # ax.set_xticks(list(ax.get_xticks()) + xticks)  # extra ticks on x-axis
    # ax.set_yticks(list(ax.get_yticks()) + yticks)  # extra ticks on x-axis
    # ax.set_xlim(*xbound)
    # ax.set_ylim(*ybound)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # save plot
    if fname is not None:
        print("- saving to '{}'".format(fname))
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    import sys
    train_fname = sys.argv[1]
    train_ylabel = "training time per epoch [s]"
    classify_fname = sys.argv[2]
    classify_ylabel = "classification time per sample [s]"
    field = sys.argv[3]

    field_xlabel = sys.argv[4]
    model_dirs = sys.argv[5:]
    if train_fname == '-':
        train_fname = None
    if classify_fname == '-':
        classify_fname = None

    lang_colors = {
        'en': 'r.-',
        'ench': 'r.--',
        'zh': 'b.-',
        'zhch': 'b.--',
    }

    # train timing
    data = parse_logs(field, model_dirs, fname_fmt="{}/console.log")
    train_xylists = {}
    for d in sorted(data):
        if d[0] not in train_xylists:
            train_xylists[d[0]] = []
        train_xylists[d[0]].append((d[1], d[2]))
    plot_timing(train_fname, field_xlabel, train_ylabel, train_xylists, lang_colors=lang_colors)
    print("")

    # classification timing
    data = parse_logs(field, model_dirs, fname_fmt="{}/blind/console.log")
    classify_xylists = {}
    for d in sorted(data):
        if d[0] not in classify_xylists:
            classify_xylists[d[0]] = []
        classify_xylists[d[0]].append((d[1], d[3] / d[4]))
    plot_timing(classify_fname, field_xlabel, classify_ylabel, classify_xylists, lang_colors=lang_colors)
    print("")
