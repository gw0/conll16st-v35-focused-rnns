#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Utils for word-level mode dataset preparation.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2017@ena.one>"
__license__ = "GPLv3+"

import numpy as np

from conll16st_data.load import Conll16stDataset
from conll16st_evaluation.validator import RELATION_TYPES, EN_SENSES, ZH_SENSES
EN_SENSES_DEFAULT = 'Expansion.Conjunction'
ZH_SENSES_DEFAULT = 'Conjunction'


def build_index(sequences, max_new=None, min_count=1, index=None, vocab_start=2, none_key=None, none_ids=0, oov_key="", oov_ids=1):
    """Build vocabulary index from dicts or lists of strings (reserved ids: 0 = none/padding, 1 = out-of-vocabulary)."""
    if index is None:
        index = {}

    def _traverse_cnt(obj, cnts):
        """Recursively traverse dicts and lists of strings."""
        if isinstance(obj, dict):
            for s in obj.itervalues():
                _traverse_cnt(s, cnts)
        elif isinstance(obj, list) or isinstance(obj, tuple):
            for s in obj:
                _traverse_cnt(s, cnts)
        else:
            try:
                cnts[obj] += 1
            except KeyError:
                cnts[obj] = 1

    # count string occurrences
    cnts = {}
    _traverse_cnt(sequences, cnts)

    # ignore strings with low occurrences
    for k, cnt in cnts.items():
        if cnt < min_count:
            del cnts[k]

    # rank strings by decreasing occurrences and use as index
    index_rev = sorted(cnts, key=cnts.get, reverse=True)
    if max_new is not None:
        index_rev = index_rev[:max_new]  # limit amount of added strings

    # mapping of strings to vocabulary ids
    index.update([ (k, i) for i, k in enumerate(index_rev, start=vocab_start) ])
    index_size = vocab_start + len(index)  # largest vocabulary id + 1

    # add none/padding and out-of-vocabulary ids
    index[none_key] = none_ids
    index[oov_key] = oov_ids
    cnts[none_key] = None
    cnts[oov_key] = None
    return index, index_size, cnts


def map_sequence(sequence, index, oov_key=""):
    """Map sequence of strings to vocabulary ids."""

    ids = []
    for s in sequence:
        try:
            ids.append(index[s])
        except KeyError:  # missing in vocabulary
            ids.append(index[oov_key])
    return ids


def pad_sequence(sequence, max_len, value=0, max_rand=None):
    """Post-pad sequence of ids as numpy array."""

    # crop sequence if needed
    sequence = sequence[:max_len]

    # convert to numpy array with masked and random post-padding
    if isinstance(value, int):
        x = np.hstack([sequence, np.ones((max_len - len(sequence),), dtype=np.int) * value])
    elif isinstance(value, float):
        x = np.hstack([sequence, np.ones((max_len - len(sequence),), dtype=np.float32) * value])
    elif value == 'rand' and isinstance(max_rand, int):
        x = np.hstack([sequence, np.random.randint(1, max_rand, size=max_len - len(sequence),)])
    else:
        raise ValueError("Padding value '{}' not understood".format(value))
    return x


def decode_category(y_np, cats2id, cats2id_size):
    """Decode category from one-hot vector (cats2id)."""

    # normalize by rows to [-1,1] interval
    y_sum = np.abs(y_np).sum(axis=0)
    #y_sum[y_sum == 0.] = 1.  # prevent NaN
    if y_sum != 0.:
        totals = y_np / np.expand_dims(y_sum, axis=-1)

    # return most probable category
    cat = None  #= none_key
    max_total = -1.
    for t, j in cats2id.items():
        if totals[j] > max_total:
            max_total = totals[j]
            cat = t
    return cat, totals


def load_word2vec(words2id, words2id_size, words_dim, words2vec_bin=None, words2vec_txt=None):

    from gensim.models import word2vec
    if words2vec_bin:
        model = word2vec.Word2Vec.load_word2vec_format(words2vec_bin, binary=True)
    else:
        model = word2vec.Word2Vec.load_word2vec_format(words2vec_txt)
    init_weights = np.zeros((words2id_size, words_dim), dtype=np.float32)
    for k, i in words2id.iteritems():
        if not isinstance(k, str):
            continue
        try:
            init_weights[i] = model[k][:words_dim]
        except KeyError:  # missing in word2vec
            pass
    return [init_weights]


def conv_to_output(dataset, x, y, indexes, indexes_size):
    """Convert model output to CoNLL16st relations output format."""

    SENSES = EN_SENSES
    SENSES_DEFAULT = EN_SENSES_DEFAULT
    if dataset['lang'] == 'zh':
        SENSES = ZH_SENSES
        SENSES_DEFAULT = ZH_SENSES_DEFAULT
    if isinstance(y, (list, tuple)):
        y = y[0]  # get only sense

    relations = []
    seen_ids = set()
    none_key = None
    oov_key = ""
    for rel_id, y_np in zip(x['_rel_id'], y):
        if rel_id in seen_ids:  # prevent duplicates
            continue
        seen_ids.add(rel_id)

        # decode relation type and sense
        target, target_totals = decode_category(y_np, indexes['target2id'], indexes_size['target2id'])
        if target == none_key or target == oov_key:
            # fallback for invalid senses
            rel_type = RELATION_TYPES[0]  # ignored by official evaluation
            rel_sense = SENSES_DEFAULT
        elif ':' in target:
            # predict sense with prepended type
            rel_type, rel_sense = target.split(':', 1)
        else:
            # predict only sense
            rel_type = RELATION_TYPES[0]  # ignored by official evaluation
            rel_sense = target

        if rel_sense not in SENSES:
            # fallback for invalid senses
            for s in SENSES:
                if s.startswith(rel_sense):  # first matching lower level sense
                    rel_sense = s
                    break
            if rel_sense not in SENSES:
                rel_sense = SENSES_DEFAULT
            #print "fallback {} to '{}' ({})".format(rel_id, rel_sense, target_totals)  #XXX

        # relation output format
        doc_id = dataset['relations_gold'][rel_id]['DocID']
        arg1_list = [ t[2]  for t in dataset['relations_gold'][rel_id]['Arg1']['TokenList'] ]
        arg2_list = [ t[2]  for t in dataset['relations_gold'][rel_id]['Arg2']['TokenList'] ]
        conn_list = [ t[2]  for t in dataset['relations_gold'][rel_id]['Connective']['TokenList'] ]
        punc_list = [ t[2]  for t in dataset['relations_gold'][rel_id]['Punctuation']['TokenList'] ]
        relation = {
            'Arg1': {'TokenList': arg1_list},
            'Arg2': {'TokenList': arg2_list},
            'Connective': {'TokenList': conn_list},
            'Punctuation': {'TokenList': punc_list},
            'DocID': doc_id,
            'ID': rel_id,
            'Type': rel_type,
            'Sense': [rel_sense],
        }
        relations.append(relation)
    return relations


def load_data(dataset_dir, lang, filter_fn_name):
    """Load dataset for word-level mode."""

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
    dataset = Conll16stDataset(dataset_dir, lang=lang, filter_types=filter_types, filter_senses=filter_senses, filter_fn=filter_fn, with_rel_senses_all=True)

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

    # strip unneeded data to release memory
    dataset['word_metas'] = None
    dataset['dependencies'] = None
    dataset['parsetrees'] = None
    import gc
    gc.collect()
    return dataset


def build_indexes(dataset):
    """Build indexes for word-level mode."""

    indexes = {}
    indexes_size = {}
    indexes_cnts = {}
    indexes['words2id'], indexes_size['words2id'], indexes_cnts['words2id'] = build_index(dataset['words'], min_count=1, vocab_start=2)  # use all words
    indexes['target2id'], indexes_size['target2id'], indexes_cnts['target2id'] = build_index(dataset['target'], min_count=2)  # ignore less frequent and partial senses
    return indexes, indexes_size, indexes_cnts


def batch_generator(dataset, indexes, indexes_size, indexes_cnts, arg1_max_len, arg2_max_len, conn_max_len, punc_max_len, batch_size, original_positives=1, random_positives=0, random_negatives=1, random_proba_1=0.1, random_proba_2=0.1, random_proba_3=0.7, curriculum_end=0.):
    """Batch generator where each sample represents a different discourse relation for word-level mode."""

    def tokens_np(words_ids, token_ids, max_len):
        words_slice = [ words_ids[i]  for i in token_ids ]

        ids = map_sequence(words_slice, indexes['words2id'])
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
            doc_id = dataset['rel_parts'][rel_id]['DocID']

            # original sequence lengths
            arg1_len = min(arg1_max_len, len(dataset['rel_parts'][rel_id]['Arg1']))
            arg2_len = min(arg2_max_len, len(dataset['rel_parts'][rel_id]['Arg2']))
            conn_len = min(conn_max_len, len(dataset['rel_parts'][rel_id]['Connective']))
            punc_len = min(punc_max_len, len(dataset['rel_parts'][rel_id]['Punctuation']))

            # skip hard samples in curriculum learning
            if curriculum_i < curriculum_end and np.random.uniform() < 0.9:
                if arg1_len > arg1_max_len * curriculum_i / curriculum_end or arg2_len > arg2_max_len * curriculum_i / curriculum_end:
                    continue
            curriculum_i += 1

            # original sequences of ids
            arg1_np = tokens_np(dataset['words'][doc_id], dataset['rel_parts'][rel_id]['Arg1'], arg1_max_len)
            arg2_np = tokens_np(dataset['words'][doc_id], dataset['rel_parts'][rel_id]['Arg2'], arg2_max_len)
            conn_np = tokens_np(dataset['words'][doc_id], dataset['rel_parts'][rel_id]['Connective'], conn_max_len)
            punc_np = tokens_np(dataset['words'][doc_id], dataset['rel_parts'][rel_id]['Punctuation'], punc_max_len)

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
                    neg_conn_np = np.random.randint(vocab_start, indexes_size['words2id'], size=conn_np.shape)
                    neg_conn_np[conn_len:] = none_id
                if punc_max_len > 0 and np.max(punc_np) > vocab_start:
                    neg_punc_np = np.random.randint(vocab_start, indexes_size['words2id'], size=punc_np.shape)
                    neg_punc_np[punc_len:] = none_id

                # span mutations, eg. 90% probability that it gets randomized by each modificator
                if np.random.uniform() < random_proba_3:
                    # randomize arg1 ids
                    neg_arg1_np = np.random.randint(vocab_start, indexes_size['words2id'], size=arg1_np.shape)
                    neg_arg1_np[arg1_len:] = none_id
                if np.random.uniform() < random_proba_3:
                    # randomize arg2 ids
                    neg_arg2_np = np.random.randint(vocab_start, indexes_size['words2id'], size=arg2_np.shape)
                    neg_arg2_np[arg2_len:] = none_id
                if np.random.uniform() < random_proba_3:
                    # swap arg1 and arg2
                    neg_arg1_np, neg_arg2_np = np.array(neg_arg2_np), np.array(neg_arg1_np)
                if np.all(neg_arg1_np == arg1_np) and np.all(neg_arg2_np == arg2_np) and np.all(neg_conn_np == conn_np) and np.all(neg_punc_np == punc_np):
                    # else randomize all
                    neg_arg1_np = np.random.randint(vocab_start, indexes_size['words2id'], size=arg1_np.shape)
                    neg_arg2_np = np.random.randint(vocab_start, indexes_size['words2id'], size=arg2_np.shape)
                    if conn_max_len > 0 and np.max(conn_np) > vocab_start:
                        neg_conn_np = np.random.randint(vocab_start, indexes_size['words2id'], size=conn_np.shape)
                    if punc_max_len > 0 and np.max(punc_np) > vocab_start:
                        neg_punc_np = np.random.randint(vocab_start, indexes_size['words2id'], size=punc_np.shape)

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
