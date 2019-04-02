#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print out samples with sense probabilities of given sense labels.

Usage:
  ./extra_proba_print.py <dataset> <model_output> <label1> [<label2>...]

  ./conll16st_evaluation/extra_proba_print.py ../data/conll15st-en-03-29-16-blind-test ./models-v35/conll16st-v35b1-en-fd-8-rn-8-3/blind Expansion.Conjunction EntRel > models-v35/conll16st-v35b1-en-fd-8-rn-8-3/blind/extra_proba_Conj-EntRel.log
  ./conll16st_evaluation/extra_proba_print.py ../data/conll15st-en-03-29-16-blind-test ./models-v35/conll16st-v35b1-en-fd-8-rn-8-3/blind Expansion.Restatement EntRel > models-v35/conll16st-v35b1-en-fd-8-rn-8-3/blind/extra_proba_Resta-EntRel.log
  ./conll16st_evaluation/extra_proba_print.py ../data/conll15st-en-03-29-16-blind-test ./models-v35/conll16st-v35b1-en-fd-8-rn-8-3/blind Contingency.Cause.Result Expansion.Conjunction > models-v35/conll16st-v35b1-en-fd-8-rn-8-3/blind/extra_proba_CausResult-Conj.log
"""
import codecs
import json
import sys


def main(args):
    dataset_dir = args[1]
    run_output_dir = args[2]
    senses = args[3:]

    f = codecs.open('{}/relations.json'.format(dataset_dir), 'r', encoding='utf8')
    gold_relations = [json.loads(x) for x in f]
    f.close()
    f = codecs.open('{}/output_proba.json'.format(run_output_dir), 'r', encoding='utf8')
    predicted_relations = [json.loads(x) for x in f]
    f.close()
    if len(gold_relations) != len(predicted_relations):
        print('Number of instances mismatch ({} != {})'.format(len(gold_relations), len(predicted_relations)))
        exit(1)

    # sort relations by ID
    gold_relations = sorted(gold_relations, key=lambda x: x['ID'])
    predicted_relations = sorted(predicted_relations, key=lambda x: x['ID'])

    # extract interesting relations
    samples = []
    for gr, pr in zip(gold_relations, predicted_relations):
        if gr['ID'] != pr['ID']:
            print('ID mismatch ({} != {})'.format(gr['ID'], pr['ID']))
            exit(1)
        pr['Type'] = gr['Type']  # use gold discourse relation type

        try:
            gold_sense = list(set(gr['Sense']).intersection(senses))[0]
        except IndexError:  # no sense matches
            continue
        try:
            predicted_sense = list(set(pr['Sense']).intersection(senses))[0]
        except IndexError:  # no sense matches
            continue

        if pr['Type'] == 'Explicit':  #NOTE: filter only non-explicit
            continue

        if gold_sense != senses[0]:  #NOTE: filter only gold first sense
            continue

        # predicted sense probabilities
        probas = {}
        for s in senses:
            probas[s] = pr['SenseProba'][s]

        # extract discourse relation
        sample = {
            'Arg1RawText': gr['Arg1']['RawText'].encode('utf8'),
            'Arg2RawText': gr['Arg2']['RawText'].encode('utf8'),
            'ConnRawText': gr['Connective']['RawText'].encode('utf8'),
            'PuncRawText': (gr['Punctuation']['RawText'] if 'Punctuation' in gr else ""),
            'GoldType': gr['Type'],
            'GoldSense': gold_sense,
            'PredSense': predicted_sense,
            'SenseProba': probas,
        }
        samples.append(sample)

    # sort relations by first given sense label
    samples = sorted(samples, key=lambda x: x['SenseProba'][senses[0]])

    print("\nTop few samples:")
    print(samples[0])
    print(samples[1])

    print("\nBottom few samples:")
    print(samples[-2])
    print(samples[-1])

    print("\nLaTeX output:")
    proba_str = ""
    for s in senses:
        proba_str += " & \\kwd{{{}}}".format(s)
    #print("Arg1 & Arg2 & GoldSense & PredSense{proba_str} \\\\".format(proba_str=proba_str))
    print("Arg1 & Arg2 & Conn{proba_str} \\\\".format(proba_str=proba_str))
    print("\\hline")

    samples_latex = []
    for sample in samples:
        proba_str = ""
        for s in senses:
            proba_str += " & ${:.4f}$".format(sample['SenseProba'][s])
        arg1_str = sample['Arg1RawText'].replace('&', '\\&').replace('$', '\\$').replace('%', '\\%').replace(r"’", "'").replace("\r", "").replace("\n", " ")
        arg2_str = sample['Arg2RawText'].replace('&', '\\&').replace('$', '\\$').replace('%', '\\%').replace(r"’", "'").replace("\r", "").replace("\n", " ")
        conn_str = sample['ConnRawText'].replace('&', '\\&').replace('$', '\\$').replace('%', '\\%').replace(r"’", "'").replace("\r", "").replace("\n", " ")
        #sample_str = "{} & {} & \\kwd{{{}}} & \\kwd{{{}}}{proba_str} \\\\".format(arg1_str, arg2_str, sample['GoldSense'], sample['PredSense'], proba_str=proba_str)
        sample_str = "{} & {} & {}{proba_str} \\\\".format(arg1_str, arg2_str, conn_str, proba_str=proba_str)

        print_maxlen = 180
        #print_maxlen = 240
        if len(arg1_str) + len(arg2_str) < print_maxlen:  #NOTE: print only shorter lines in bytes
            samples_latex.append(sample_str)
            samples_latex.append(sample_str)  #XXX: duplicate experiment for better printing

    #NOTE: print only a few samples
    print_first = 2
    print_mid = 9
    print_last = 2
    #print_mid_step = float(len(samples_latex) - print_first - print_last) / print_mid
    #samples_only = samples_latex[:print_first] + samples_latex[print_first:-print_last:int((len(samples_latex) - print_first - print_last) / print_mid)] + samples_latex[-print_last:]
    #XXX: duplicate experiment for better printing
    print_mid_step = float(len(samples_latex) - print_first*2 - print_last*2) / print_mid
    if print_mid_step < 1.:
        print_mid_step = 1
    samples_only = samples_latex[:print_first*2:2] + samples_latex[print_first*2+int(print_mid_step/2):-print_last*2:int(print_mid_step)] + samples_latex[-print_last*2::2]
    #samples_only = samples_latex[:print_first] + [ samples_latex[print_first + int((i + 0.5) * print_mid_step)] for i in range(print_mid) ] + samples_latex[-print_last:]
    sample_str_prev = None
    #i = 0
    for sample_str in samples_only:
        if sample_str == sample_str_prev:  # skip duplicates
            continue
        sample_str_prev = sample_str
        print(sample_str)
        #if i % 2 == 0:  # for exchanging row colors
        #    print("\\RO " + sample_str)
        #else:
        #    print("\\RE " + sample_str)
        #i += 1

if __name__ == '__main__':
    main(sys.argv)

