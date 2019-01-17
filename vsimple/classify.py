#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Apply a trained baseline model with simple LSTMs to discourse relation sense classification (CoNLL16st).
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
from model import CUSTOM_LAYERS
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
    choices=["en", "ench", "zh", "zhch"],
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

args, unknown_args = argp.parse_known_args()
args.backend = K._backend
args.backend_theano = os.getenv("THEANO_FLAGS")
args.rest = unknown_args
if args.lang.endswith("ch"):
    args.mode = 'char'
    args.lang = args.lang[:-2]

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
output_proba_json = "{}/output_proba.json".format(args.output_dir)

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
log.info("make predictions")

# prepare data
test_size = (len(dataset['rel_ids']) + args.batch_size - 1) / args.batch_size * args.batch_size  # round up to multiple of batch_size
x, _ = next(batch_generator(dataset, indexes, indexes_size, indexes_cnts, args.arg1_len, args.arg2_len, args.conn_len, args.punc_len, test_size, original_positives=1, random_positives=0, random_negatives=0))

# make predictions
y_pred = model.predict(x, batch_size=args.batch_size)

# save predictions
log.info("save predictions ({})".format(output_json))
relations = conv_to_output(dataset, x, y_pred, indexes, indexes_size, with_proba=False)
f_out = codecs.open(output_json, mode='w', encoding='utf8')
for relation in relations:
    f_out.write(json.dumps(relation) + "\n")
f_out.close()

log.info("save predictions ({})".format(output_proba_json))
relations = conv_to_output(dataset, x, y_pred, indexes, indexes_size, with_proba=True)
f_out = codecs.open(output_proba_json, mode='w', encoding='utf8')
for relation in relations:
    f_out.write(json.dumps(relation) + "\n")
f_out.close()

log.info("finished ({})".format(args.output_dir))
