#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Custom Keras callbacks.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2017@ena.one>"
__license__ = "GPLv3+"

from keras.callbacks import Callback

from word_utils import conv_to_output
from conll16st_evaluation.scorer import evaluate_sense



def use_gold_standard_types(sorted_gold_relations, sorted_predicted_relations):
    """Copy gold standard types to predicted relations."""
    i = 0
    j = 0
    prev_id = None
    while i < len(sorted_gold_relations) and j < len(sorted_predicted_relations):
        gr = sorted_gold_relations[i]
        pr = sorted_predicted_relations[j]
        if pr['ID'] > gr['ID']:  # predicted relations may skip some
            i += 1
            continue
        elif gr['ID'] == pr['ID']:  # matching relation
            pr['Type'] = gr['Type']
        else:  # error
            print("ID mismatch ([{}]={} != [{}]={}). Make sure you copy the ID from gold standard.".format(i, gr['Type'], j, pr['Type']))
        prev_id = pr['ID']
        i += 1
        j += 1


class CoNLL16stMetrics(Callback):
    """Evaluate using CoNLL16st metrics for sense classification."""

    def __init__(self, dataset, indexes, indexes_size, x, y, batch_size=1):
        super(CoNLL16stMetrics, self).__init__()
        self.dataset = dataset
        self.indexes = indexes
        self.indexes_size = indexes_size
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs={}):
        """Trigger evaluation on epoch end."""
        print("")

        # evaluate metrics
        if 'val_loss' not in logs:
            outs = self.model.evaluate(self.x, self.y, sample_weight=None, verbose=0)
            if type(outs) is not list:
                outs = [outs]
            for k, o in zip(self.model.metrics_names, outs):
                logs['val_' + k] = o
        print("- " + " - ".join([ "val_{}: {:.4f}".format(k, logs['val_' + k]) for k in self.model.metrics_names ]))

        # make predictions
        y_pred = self.model.predict(self.x, batch_size=self.batch_size)

        # prepare sorted relations list
        gold_list = [ self.dataset['relations_gold'][rel_id] for rel_id in sorted(self.dataset['relations_gold'].keys()) ]
        predicted_list = conv_to_output(self.dataset, self.x, y_pred, self.indexes, self.indexes_size)
        del y_pred  # release memory
        predicted_list = sorted(predicted_list, key=lambda r: r['ID'])
        use_gold_standard_types(gold_list, predicted_list)

        # evaluate all discourse relations
        sense_cm = evaluate_sense(gold_list, predicted_list)
        precision, recall, f1 = sense_cm.compute_micro_average_f1()
        logs['val_all_f1'] = f1
        print("- val_all_f1:  {:1.4f} {:1.4f} {:1.4f}".format(precision, recall, f1))
        #sense_cm.print_summary()

        # evaluate only explicit discourse relations
        exp_gold_list = [ r for r in gold_list if r['Type'] == 'Explicit' ]
        exp_predicted_list = [ r for r in predicted_list if r['Type'] == 'Explicit' ]
        exp_sense_cm = evaluate_sense(exp_gold_list, exp_predicted_list)
        exp_precision, exp_recall, exp_f1 = exp_sense_cm.compute_micro_average_f1()
        logs['val_exp_f1'] = exp_f1
        print("- val_exp_f1:  {:1.4f} {:1.4f} {:1.4f}".format(exp_precision, exp_recall, exp_f1))
        #exp_sense_cm.print_summary()

        # evaluate only non-explicit discourse relations
        non_gold_list = [ r for r in gold_list if r['Type'] != 'Explicit' ]
        non_predicted_list = [ r for r in predicted_list if r['Type'] != 'Explicit' ]
        non_sense_cm = evaluate_sense(non_gold_list, non_predicted_list)
        non_precision, non_recall, non_f1 = non_sense_cm.compute_micro_average_f1()
        logs['val_non_f1'] = non_f1
        print("- val_non_f1:  {:1.4f} {:1.4f} {:1.4f}".format(non_precision, non_recall, non_f1))
        #non_sense_cm.print_summary()

        print("")
