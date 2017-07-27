#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Apply a trained Keras model for discourse relation sense classification (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2017@ena.one>"
__license__ = "GPLv3+"

import argparse
import codecs
import json
import logging
import os
import sys
import numpy as np
from keras.models import model_from_yaml
from keras.layers import K

from generic_utils import Tee, debugger, load_from_pkl
from model import CUSTOM_LAYERS, peek_filters
from word_utils import conv_to_output, decode_category


### debugging
sys.excepthook = debugger  # attach debugger on error
sys.stdout = Tee([sys.stdout])
sys.stderr = Tee([sys.stderr])

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

### parse arguments
argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
argp.add_argument('model_dir',
    help="pre-trained model directory")
argp.add_argument('lang',
    choices=["en", "zh"],
    help="dataset language (en/zh)")
argp.add_argument('dataset_dir',
    help="CoNLL16st dataset for prediction (directory with 'parses.json', 'relations-no-senses.json')")
argp.add_argument('output_dir',
    help="output directory for system predictions ('output.json')")
argp.add_argument('--filter_fn_name', default=None,
    choices=["conn_eq_0", "conn_gt_0", None],
    help="dataset filtering function name (%(default)s)")

argp.add_argument('--mode', default="word",
    choices=["word", "char"],
    help="word- or character-level mode (%(default)s)")
argp.add_argument('--batch_size', type=int, default=64,
    help="mini-batch size should match training (%(default)s)")
argp.add_argument('--snapshot_size', type=int, default=2048,
    help="snapshot size of validation data (%(default)s)")

argp.add_argument('--arg1_len', type=int, default=100,  # word: en=100, zh=500; char: en=400, zh=900
    help="length of argument 1 text span (en=100, zh=500) (%(default)s)")
argp.add_argument('--arg2_len', type=int, default=100,  # word: en=100, zh=500; char: en=400, zh=900
    help="length of argument 2 text span (en=100, zh=500) (%(default)s)")
argp.add_argument('--conn_len', type=int, default=10,  # word: en/zh=100; char: en/zh=20
    help="length of connective text span (en/zh=10) (%(default)s)")
argp.add_argument('--punc_len', type=int, default=0,
    help="length of punctuation text span (en=0, zh=2) (%(default)s)")

argp.add_argument('--normalization_mode', default="sample_aspect",
    choices=["none", "sample", "sample_aspect", "span", "aspect", "word"],
    help="normalization axis for visualization (%(default)s)")

args, unknown_args = argp.parse_known_args()
args.backend = K._backend
args.backend_theano = os.getenv("THEANO_FLAGS")
args.rest = unknown_args

### initialize experiment
indexes_pkl = "{}/indexes.pkl".format(args.model_dir)
indexes_size_pkl = "{}/indexes_size.pkl".format(args.model_dir)
indexes_cnts_pkl = "{}/indexes_cnts.pkl".format(args.model_dir)
model_yaml = "{}/model.yaml".format(args.model_dir)
model_png = "{}/model.png".format(args.model_dir)
weights_hdf5 = "{}/weights.hdf5".format(args.model_dir)
weights_val_hdf5 = "{}/weights_val.hdf5".format(args.model_dir)
weights_val_all_f1_hdf5 = "{}/weights_val_all_f1.hdf5".format(args.model_dir)
config_json = "{}/config.json".format(args.output_dir)
console_log = "{}/console.log".format(args.output_dir)
output_json = "{}/output.json".format(args.output_dir)
peek_html = "{}/peek.html".format(args.output_dir)

if not os.path.isdir(args.output_dir):  # create directory
    os.makedirs(args.output_dir)

f_log = codecs.open(console_log, mode='a', encoding='utf8')
try:
    sys.stdout.files.append(f_log)
    sys.stderr.files.append(f_log)
except AttributeError:
    f_log.close()

log.info("configuration ({})".format(args.output_dir))
for k, v in sorted(vars(args).iteritems()):
    log.debug("  config '{}': {}".format(k, v))
with open(config_json, 'a') as f:  # dump configuration
    json.dump(vars(args), f)

if args.mode == "word":  # import word-level mode
    from word_utils import load_data, batch_generator
elif args.mode == "char":  # import character-level mode
    from char_utils import load_data, batch_generator

### load datasets
log.info("load dataset for prediction ({})".format(args.dataset_dir))
dataset = load_data(args.dataset_dir, args.lang, args.filter_fn_name)
log.info("  " + dataset.summary())

### load indexes
log.info("load previous indexes ({})".format(indexes_pkl))
indexes = load_from_pkl(indexes_pkl)
indexes_size = load_from_pkl(indexes_size_pkl)
indexes_cnts = load_from_pkl(indexes_cnts_pkl)
log.info("  " + ", ".join([ "{}: {}".format(k, v) for k, v in indexes_size.items() ]))

### load model
log.info("load model ({})".format(args.model_dir))
model = model_from_yaml(open(model_yaml, 'r').read(), custom_objects=CUSTOM_LAYERS)

# load weights
log.info("load previous weights ({})".format(weights_val_all_f1_hdf5))
model.load_weights(weights_val_all_f1_hdf5)  # weights of best validation loss

### prediction
log.info("prediction preparation")

# prepare data
test_size = (len(dataset['rel_ids']) + args.batch_size - 1) / args.batch_size * args.batch_size  # round up to multiple of batch_size
x, _ = next(batch_generator(dataset, indexes, indexes_size, indexes_cnts, args.arg1_len, args.arg2_len, args.conn_len, args.punc_len, test_size, original_positives=1, random_positives=0, random_negatives=0))

# make predictions
log.info("make predictions ({})".format(output_json))
y_pred = model.predict(x, batch_size=args.batch_size)
relations = conv_to_output(dataset, x, y_pred, indexes, indexes_size)

f_out = codecs.open(output_json, mode='w', encoding='utf8')
for relation in relations:
    f_out.write(json.dumps(relation) + "\n")
f_out.close()

# peek into filtering weights
log.info("peek into filtering weights ({})".format(peek_html))
peek_funcs = peek_filters(model)

try:  # after Keras 1.2.2
    try:
        from keras.engine.training import _standardize_input_data
    except ImportError:
        from keras.engine.training import standardize_input_data as _standardize_input_data
    x_np = _standardize_input_data(x, model.input_names, model.internal_input_shapes, check_batch_axis=False)
except TypeError:  # older versions of Keras
    from keras.engine.training import standardize_input_data
    x_np = standardize_input_data(x, model.input_names, model.internal_input_shapes, check_batch_dim=False)


def filters_to_html(words, extras, vocab_masks, filters_np, span_height, normalization_mode=None, filters_min=None, filters_max=None):
    """Convert words with filtering weights to HTML lines."""

    # per-span normalization
    if normalization_mode == "span" and filters_np.size > 0:
        # normalize filtering weights of each span to [0,1]
        filters_min = filters_np.min(keepdims=True)
        filters_max = filters_np.max(keepdims=True)

    elif normalization_mode == "aspect" and filters_np.size > 0:
        # normalize filtering weights of each aspect in each span to [0,1]
        filters_min = filters_np.min(axis=0, keepdims=True)
        filters_max = filters_np.max(axis=0, keepdims=True)

    elif normalization_mode == "word" and filters_np.size > 0:
        # normalize filtering weights of each word in each span to [0,1]
        filters_min = filters_np.min(axis=1, keepdims=True)
        filters_max = filters_np.max(axis=1, keepdims=True)

    if filters_min is not None and filters_max is not None:
        eps = 0.00000001
        filters_np = (filters_np - filters_min) / (filters_max - filters_min + eps)

    line = ''
    for w, e, m, focus_v in zip(words, extras, vocab_masks, filters_np):
        if m:
            line += u'\n<span class="w">{}<small>{}</small>'.format(w, e)
        else:
            line += u'\n<span class="w"><s>{}<small>{}</small></s>'.format(w, e)
        for i, f in enumerate(focus_v):
            r = g = b = 0
            if f > 1. or f < -1.:  # blue for overflow
                b = 255
            if f > 0.:  # green for positive intensity
                g = int(f * 255) % 256
            elif f < 0.:  # red for negative intensity
                r = int(-f * 255) % 256
            line += '<span style="bottom: {:d}px; background: #{:02x}{:02x}{:02x};"></span>'.format(i * span_height, r, g, b)
        line += '</span>'
    return line

ins = [ [a[0]] for a in x_np ] + [0.]
filters_shape = peek_funcs[0][1](ins)[0].shape
span_height = 5
f_peek = codecs.open(peek_html, mode='a', encoding='utf8')
f_peek.write('<style type="text/css">\n')
f_peek.write('.w {{ display: inline-block; position: relative; padding-bottom: {}px; }}\n'.format(filters_shape[-1] * span_height))
f_peek.write('.w span {{ position: absolute; left: 0; width: 100%; height: {}px; }}'.format(span_height - 1))
f_peek.write('small { color:#999; }')
f_peek.write('</style>\n')

filter_to_span = {'arg1_filters': 'Arg1', 'arg2_filters': 'Arg2', 'conn_filters': 'Connective', 'punc_filters': 'Punctuation'}
seen_ids = set()
for i, rel_id in enumerate(x['_rel_id']):  # for each relation/sample
    if rel_id in seen_ids:  # prevent duplicates
        continue
    seen_ids.add(rel_id)

    # get basic relation information
    rel_parts = dataset['rel_parts'][rel_id]
    doc_id = rel_parts['DocID']
    rel_type = dataset['rel_types'][rel_id]
    rel_sense_all = dataset['rel_senses'][rel_id]
    if relations[i]['DocID'] == doc_id and relations[i]['ID'] == rel_id:
        pred_type = ''  # ignored by official evaluation
        pred_sense = relations[i]['Sense'][0]
    else:
        pred_sense = pred_type = "?"
    _, pred_totals = decode_category(y_pred[i], indexes['target2id'], indexes_size['target2id'])
    pred_max_total = np.max(pred_totals)

    # print relation/sample summary
    if any(( s == pred_sense for s in rel_sense_all )):
        equal_sign = '<b style="background:#0f0;">&nbsp;&nbsp;== <small>{:.4f}</small>&nbsp;&nbsp;</b>'.format(pred_max_total)
    else:
        equal_sign = '<b style="background:#f00;">&nbsp;&nbsp;!= <small>{:.4f}</small>&nbsp;&nbsp;</b>'.format(pred_max_total)
    f_peek.write("<b>{} ({})</b>: <i>gold</i> {}:{} {} <i>pred</i> {}:{}</h3>\n".format(doc_id, rel_id, rel_type, rel_sense_all, equal_sign, pred_type, pred_sense))

    # get text spans
    words = {}
    extras = {}
    vocab_masks = {}
    for span in rel_parts.keys():
        if span not in filter_to_span.values():
            continue
        if 'words' in dataset:  # for word mode
            words[span] = [ dataset['words'][doc_id][o] for o in rel_parts[span] ]
            extras[span] = [ '/' + dataset['pos_tags'][doc_id][o] for o in rel_parts[span] ]
        else:  # for char mode
            words[span] = list(dataset['relations_gold'][rel_id][span]['RawText'])
            extras[span] = [''] * len(words[span])
        vocab_masks[span] = [ w in indexes['words2id'] for w in words[span] ]

    # compute filtering weights for current relation/sample
    ins = [ [a[i]] for a in x_np ] + [0.]
    peek_values = {}
    for peek_name, peek_func in peek_funcs:
        span = filter_to_span[peek_name]
        peek_values[span] = peek_func(ins)[0][0][:len(words[span])]

    # global normalization
    filters_min = None
    filters_max = None
    if args.normalization_mode == "sample":
        # normalize filtering weights of whole relation/sample to [0,1]
        filters_min = np.asarray([ filters_np.min(keepdims=True) for filters_np in peek_values.values() if filters_np.size > 0 ]).min(axis=0)
        filters_max = np.asarray([ filters_np.max(keepdims=True) for filters_np in peek_values.values() if filters_np.size > 0 ]).max(axis=0)

    elif args.normalization_mode == "sample_aspect":
        # normalize filtering weights of each aspect for whole relation/sample to [0,1]
        filters_min = np.asarray([ filters_np.min(axis=0, keepdims=True) for filters_np in peek_values.values() if filters_np.size > 0 ]).min(axis=0)
        filters_max = np.asarray([ filters_np.max(axis=0, keepdims=True) for filters_np in peek_values.values() if filters_np.size > 0 ]).max(axis=0)

    # print each text span with filtering weights
    for span in peek_values.keys():
        filters_html = filters_to_html(words[span], extras[span], vocab_masks[span], peek_values[span], span_height=span_height, normalization_mode=args.normalization_mode, filters_min=filters_min, filters_max=filters_max)
        f_peek.write("<p><b>" + span + "</b>:" + filters_html + "</p>\n")

    f_peek.write("<hr />\n\n")
f_peek.close()

log.info("finished ({})".format(args.output_dir))
