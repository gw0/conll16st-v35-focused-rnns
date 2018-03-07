#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Utils for character-level mode dataset preparation.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2017@ena.one>"
__license__ = "GPLv3+"

import numpy as np

from conll16st_data.files import load_relations_gold
from conll16st_data.relations import get_rel_parts, get_rel_types, get_rel_senses_all
from word_utils import build_index, map_sequence, pad_sequence
from conll16st_evaluation.validator import EN_SENSES, ZH_SENSES


def load_data(dataset_dir, lang, filter_fn_name):
    """Load dataset for character-level mode."""

    # filtering
    filter_types = None  # ["Explicit"], ["Implicit", "EntRel", "AltLex"]
    filter_senses = None  # ["Contingency.Condition"]
    if filter_fn_name == "conn_eq_0":  # connective length not equals 0
        filter_fn = lambda r: len(r['Connective']['TokenList']) == 0
    elif filter_fn_name == "conn_gt_0":  # connective length not greater than 0
        filter_fn = lambda r: len(r['Connective']['TokenList']) > 0
    else:  # no filter
        filter_fn = None

    # load data
    class Conll16stRawDataset(dict):
        pass
    dataset = Conll16stRawDataset()
    dataset['relations_gold'] = load_relations_gold(dataset_dir, with_senses=True, with_rawtext=True, filter_types=filter_types, filter_senses=filter_senses, filter_fn=filter_fn)
    for relation in dataset['relations_gold'].itervalues():  # remove not-explicitly stated raw text
        if not relation['Arg1']['TokenList']:
            relation['Arg1']['RawText'] = ''
        if not relation['Arg2']['TokenList']:
            relation['Arg2']['RawText'] = ''
        if not relation['Connective']['TokenList']:
            relation['Connective']['RawText'] = ''
        if not relation['Punctuation']['TokenList']:
            relation['Punctuation']['RawText'] = ''
    dataset['rel_parts'] = get_rel_parts(dataset['relations_gold'])  # for classification
    dataset['rel_ids'] = sorted(dataset['relations_gold'].keys())
    dataset['rel_types'] = get_rel_types(dataset['relations_gold'])
    dataset['rel_senses'] = get_rel_senses_all(dataset['relations_gold'])
    dataset['lang'] = lang
    dataset.summary = lambda: "lang: {}, rel_ids: {}".format(dataset['lang'], len(dataset['rel_ids']))

    # predict only sense
    #dataset['target'] = dataset['rel_senses']
    # predict sense with prepended type
    #target = {}
    #for rel_id, rel_senses in dataset['rel_senses'].iteritems():
    #    rel_type = dataset['rel_types'][rel_id]
    #    target[rel_id] = tuple("{}:{}".format(rel_type, s)  for s in rel_senses)
    #dataset['target'] = target
    # predict only valid senses
    SENSES = EN_SENSES
    if dataset['lang'] == 'zh':
        SENSES = ZH_SENSES
    target = {}
    for rel_id, rel_senses in dataset['rel_senses'].iteritems():
        rel_type = dataset['rel_types'][rel_id]
        senses = []
        for s in rel_senses:
            if s in SENSES:
                senses.append(s)
                continue
            # for partial senses mark all subsenses
            for k in SENSES:
                if k.startswith(s):
                    senses.append(s)
        # only sense
        target[rel_id] = senses
        # with prepended type
        #target[rel_id] = tuple("{}:{}".format(rel_type, s)  for s in senses)
    dataset['target'] = target

    return dataset


def build_indexes(dataset):
    """Build indexes for character-level mode."""

    indexes = {}
    indexes_size = {}
    indexes_cnts = {}
    indexes['words2id'], indexes_size['words2id'], indexes_cnts['words2id'] = build_index([ list(r['Arg1']['RawText'] + r['Arg2']['RawText'] + r['Connective']['RawText'] + r['Punctuation']['RawText']) for r in dataset['relations_gold'].values() ], min_count=1, vocab_start=2)  # use all characters
    indexes['target2id'], indexes_size['target2id'], indexes_cnts['target2id'] = build_index(dataset['target'], min_count=2)  # ignore less frequent and partial senses
    return indexes, indexes_size, indexes_cnts


def batch_generator(dataset, indexes, indexes_size, indexes_cnts, arg1_max_len, arg2_max_len, conn_max_len, punc_max_len, batch_size, original_positives=1, random_positives=0, random_negatives=1, random_proba_1=0.1, random_proba_2=0.1, random_proba_3=0.7, curriculum_end=0.):
    """Batch generator where each sample represents a different discourse relation for character-level mode."""

    def text_np(text, max_len):
        ids = map_sequence(text, indexes['words2id'])
        x_np = pad_sequence(ids, max_len, value=0)
        return x_np

    def target_np(cats, oov_key=""):
        if isinstance(cats, (str, unicode)):  # first sense only
            cats = [cats]

        x_np = np.zeros((indexes_size['target2id'],), dtype=np.float32)
        for cat in cats:
            try:
                i = indexes['target2id'][cat]
            except KeyError:  # missing in vocabulary
                i = indexes['target2id'][oov_key]
                # for partial senses mark all subsenses
                for k in indexes['target2id'].keys():
                    if k is not None and k.startswith(cat):
                        i = indexes['target2id'][k]
                        x_np[i] = 1
            x_np[i] = 1
        return x_np

    none_key = None
    none_id = indexes['words2id'][none_key]
    oov_key = ""
    oov_id = indexes['words2id'][oov_key]
    vocab_start = 2

    rel_ids = list(dataset['rel_ids'])  # copy list

    curriculum_i = 0.05 * curriculum_end  # curriculum learning counter starts at 5%

    # reset batch
    _rel_id = []
    batch_x = {'arg1_ids':[], 'arg2_ids':[], 'conn_ids':[], 'punc_ids':[]}
    batch_y = {'target':[]}

    while True:
        # shuffle relations on each epoch
        np.random.shuffle(rel_ids)

        for rel_id in rel_ids:
            # original sequence lengths
            arg1_len = min(len(dataset['relations_gold'][rel_id]['Arg1']['RawText']), arg1_max_len)
            arg2_len = min(len(dataset['relations_gold'][rel_id]['Arg2']['RawText']), arg2_max_len)
            conn_len = min(len(dataset['relations_gold'][rel_id]['Connective']['RawText']), conn_max_len)
            punc_len = min(len(dataset['relations_gold'][rel_id]['Punctuation']['RawText']), punc_max_len)

            # skip hard samples in curriculum learning
            if curriculum_i < curriculum_end and np.random.uniform() < 0.9:
                if arg1_len > arg1_max_len * curriculum_i / curriculum_end or arg2_len > arg2_max_len * curriculum_i / curriculum_end:
                    continue
            curriculum_i += 1

            # original sequences of ids
            arg1_np = text_np(dataset['relations_gold'][rel_id]['Arg1']['RawText'], arg1_max_len)
            arg2_np = text_np(dataset['relations_gold'][rel_id]['Arg2']['RawText'], arg2_max_len)
            conn_np = text_np(dataset['relations_gold'][rel_id]['Connective']['RawText'], conn_max_len)
            punc_np = text_np(dataset['relations_gold'][rel_id]['Punctuation']['RawText'], punc_max_len)

            # original relation senses
            if dataset['target']:
                target = target_np(dataset['target'][rel_id])
            else:
                target = target_np(oov_key)

            # append original samples
            for _ in range(original_positives):
                _rel_id.append(rel_id)
                batch_x['arg1_ids'].append(arg1_np)
                batch_x['arg2_ids'].append(arg2_np)
                batch_x['conn_ids'].append(conn_np)
                batch_x['punc_ids'].append(punc_np)
                batch_y['target'].append(target)

            # positive random samples for each original sample
            for _ in range(random_positives):
                pos_arg1_len, pos_arg1_np = arg1_len, np.array(arg1_np)
                pos_arg2_len, pos_arg2_np = arg2_len, np.array(arg2_np)
                pos_conn_len, pos_conn_np = conn_len, np.array(conn_np)
                pos_punc_len, pos_punc_np = punc_len, np.array(punc_np)
                pos_target = target  # mark same as original sample

                # word mutations, eg. 10% probability that 10% gets modified by each modificator
                arg1_proba = random_proba_1 ** (random_proba_2 * arg1_max_len / pos_arg1_len)
                arg2_proba = random_proba_1 ** (random_proba_2 * arg2_max_len / pos_arg2_len)
                while np.random.uniform() < arg1_proba:
                    # duplicate random word in arg1
                    i = np.random.randint(0, pos_arg1_len)
                    pos_arg1_np[i + 1:arg1_max_len] = pos_arg1_np[i:arg1_max_len - 1]
                    pos_arg1_len = min(pos_arg1_len + 1, arg1_max_len)
                while np.random.uniform() < arg1_proba:
                    # insert random oov in arg1
                    i = np.random.randint(0, pos_arg1_len)
                    pos_arg1_np[i + 1:arg1_max_len] = pos_arg1_np[i:arg1_max_len - 1]
                    pos_arg1_len = min(pos_arg1_len + 1, arg1_max_len)
                    pos_arg1_np[i] = oov_id
                while np.random.uniform() < arg1_proba:
                    # forget random word in arg1
                    i = np.random.randint(0, pos_arg1_len)
                    pos_arg1_np[i] = oov_id
                while np.random.uniform() < arg2_proba:
                    # duplicate random word in arg2
                    i = np.random.randint(0, pos_arg2_len)
                    pos_arg2_np[i + 1:arg2_max_len] = pos_arg2_np[i:arg2_max_len - 1]
                    pos_arg2_len = min(pos_arg2_len + 1, arg2_max_len)
                while np.random.uniform() < arg2_proba:
                    # insert random oov in arg2
                    i = np.random.randint(0, pos_arg2_len)
                    pos_arg2_np[i + 1:arg2_max_len] = pos_arg2_np[i:arg2_max_len - 1]
                    pos_arg2_len = min(pos_arg2_len + 1, arg2_max_len)
                    pos_arg2_np[i] = oov_id
                while np.random.uniform() < arg2_proba:
                    # forget random word in arg2
                    i = np.random.randint(0, pos_arg2_len)
                    pos_arg2_np[i] = oov_id

                # prepend some none padding (unnecessary when masking=True)
                #i = np.random.randint(0, arg1_max_len - pos_arg1_len + 1)
                #pos_arg1_np[i:] = pos_arg1_np[:arg1_max_len - i]
                #pos_arg1_np[:i] = none_id
                #pos_arg1_len += i
                #i = np.random.randint(0, arg2_max_len - pos_arg2_len + 1)
                #pos_arg2_np[i:] = pos_arg2_np[:arg2_max_len - i]
                #pos_arg2_np[:i] = none_id
                #pos_arg2_len += i
                #i = np.random.randint(0, conn_max_len - pos_conn_len + 1)
                #pos_conn_np[i:] = pos_conn_np[:conn_max_len - i]
                #pos_conn_np[:i] = none_id
                #pos_conn_len += i
                #i = np.random.randint(0, punc_max_len - pos_punc_len + 1)
                #pos_punc_np[i:] = pos_punc_np[:punc_max_len - i]
                #pos_punc_np[:i] = none_id
                #pos_punc_len += i

                # append positive sample
                _rel_id.append(rel_id)
                batch_x['arg1_ids'].append(pos_arg1_np)
                batch_x['arg2_ids'].append(pos_arg2_np)
                batch_x['conn_ids'].append(pos_conn_np)
                batch_x['punc_ids'].append(pos_punc_np)
                batch_y['target'].append(pos_target)

            # negative random samples for each original sample
            for _ in range(random_negatives):
                neg_arg1_np = arg1_np
                neg_arg2_np = arg2_np
                neg_conn_np = conn_np
                neg_punc_np = punc_np
                neg_target = target_np(oov_key)  # mark as out-of-vocabulary

                # always randomize connective and punctiation ids
                if conn_max_len > 0 and np.max(conn_np) > vocab_start:
                    neg_conn_np = np.random.randint(vocab_start, len(indexes['words2id']), size=conn_np.shape)
                    neg_conn_np[conn_len:] = none_id
                if punc_max_len > 0 and np.max(punc_np) > vocab_start:
                    neg_punc_np = np.random.randint(vocab_start, len(indexes['words2id']), size=punc_np.shape)
                    neg_punc_np[punc_len:] = none_id

                # span mutations, eg. 70% probability that it gets randomized by each modificator
                if np.random.uniform() < random_proba_3:
                    # randomize arg1 ids
                    neg_arg1_np = np.random.randint(vocab_start, len(indexes['words2id']), size=arg1_np.shape)
                    neg_arg1_np[arg1_len:] = none_id
                if np.random.uniform() < random_proba_3:
                    # randomize arg2 ids
                    neg_arg2_np = np.random.randint(vocab_start, len(indexes['words2id']), size=arg2_np.shape)
                    neg_arg2_np[arg2_len:] = none_id
                if np.random.uniform() < random_proba_3:
                    # swap arg1 and arg2
                    neg_arg1_np, neg_arg2_np = np.array(neg_arg2_np), np.array(neg_arg1_np)
                if np.all(neg_arg1_np == arg1_np) and np.all(neg_arg2_np == arg2_np) and np.all(neg_conn_np == conn_np) and np.all(neg_punc_np == punc_np):
                    # else randomize all
                    neg_arg1_np = np.random.randint(vocab_start, len(indexes['words2id']), size=arg1_np.shape)
                    neg_arg2_np = np.random.randint(vocab_start, len(indexes['words2id']), size=arg2_np.shape)
                    if conn_max_len > 0 and np.max(conn_np) > vocab_start:
                        neg_conn_np = np.random.randint(vocab_start, len(indexes['words2id']), size=conn_np.shape)
                    if punc_max_len > 0 and np.max(punc_np) > vocab_start:
                        neg_punc_np = np.random.randint(vocab_start, len(indexes['words2id']), size=punc_np.shape)

                # append negative sample
                _rel_id.append(rel_id)
                batch_x['arg1_ids'].append(neg_arg1_np)
                batch_x['arg2_ids'].append(neg_arg2_np)
                batch_x['conn_ids'].append(neg_conn_np)
                batch_x['punc_ids'].append(neg_punc_np)
                batch_y['target'].append(neg_target)

            while len(_rel_id) >= batch_size:
                # convert to numpy arrays
                batch_x_np = {}
                batch_y_np = {}
                for k, v in batch_x.items():
                    batch_x_np[k] = np.asarray(v[:batch_size])
                    batch_x[k] = v[batch_size:]
                for k, v in batch_y.items():
                    batch_y_np[k] = np.asarray(v[:batch_size])
                    batch_y[k] = v[batch_size:]

                # append meta data
                batch_x_np['_rel_id'] = _rel_id[:batch_size]
                _rel_id = _rel_id[batch_size:]

                # yield batch
                yield (batch_x_np, batch_y_np)
