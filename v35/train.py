#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103,W0621
"""
Training of our model with focused RNNs for discourse relation sense classification (CoNLL16st).
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2017@ena.one>"
__license__ = "GPLv3+"

import argparse
import codecs
import json
import logging
import os
import sys
import time
from keras.utils.visualize_util import plot as plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import K

from generic_utils import Tee, debugger, load_from_pkl, save_to_pkl
from callbacks import CoNLL16stMetrics
from model import build_model


### debugging
time_start = time.time()  # for time measurement
sys.excepthook = debugger  # attach debugger on error
sys.stdout = Tee([sys.stdout])
sys.stderr = Tee([sys.stderr])

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M", level=logging.DEBUG)
log = logging.getLogger(__name__)

### parse arguments
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
rnn_choices = ["rnn-fwd", "rnn-fb", "rnn-fbconcat", "rnn-bi", "rnn-biconcat", "gru-fwd", "gru-fb", "gru-fbconcat", "gru-bi", "gru-biconcat", "lstm-fwd", "lstm-fb", "lstm-fbconcat", "lstm-bi", "lstm-biconcat"]

argp = argparse.ArgumentParser(description=__doc__.strip().split("\n", 1)[0])
argp.add_argument('model_dir',
    help="model directory to store configuration and best models")
argp.add_argument('lang',
    choices=["en", "ench", "zh", "zhch"],
    help="dataset language (en/zh)")
argp.add_argument('train_dir',
    help="CoNLL16st dataset for training (directory with 'parses.json', 'relations.json', 'raw/')")
argp.add_argument('valid_dir',
    help="CoNLL16st dataset for validation (directory with 'parses.json', 'relations.json', 'raw/')")
argp.add_argument('--continue', action='store_true', dest='continue_',
    help="continue training of a previous experiment")
argp.add_argument('--filter_fn_name', default=None,
    choices=["conn_eq_0", "conn_gt_0", None],
    help="dataset filtering function name (%(default)s)")

argp.add_argument('--mode', default="word",
    choices=["word", "char"],
    help="word- or character-level mode (%(default)s)")
argp.add_argument('--batch_size', type=int, default=64,
    help="mini-batch size (%(default)s)")
argp.add_argument('--snapshot_size', type=int, default=2048,
    help="snapshot size of validation data (%(default)s)")
argp.add_argument('--epochs', type=int, default=1000,
    help="number of partial epochs (evaluation checkpoints) (%(default)s)")
argp.add_argument('--epochs_ratio', type=float, default=0.1,
    help="ratio of real to partial epochs (evaluate more often) (%(default)s)")
argp.add_argument('--epochs_patience', type=int, default=100,
    help="number of epochs with no improvement for early stopping (-1=disable) (%(default)s)")
argp.add_argument('--curriculum_end', type=float, default=0.0,
    help="curriculum learning ends after this many real epochs (%(default)s)")
argp.add_argument('--optimizer', default="adam",
    choices=["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"],
    help="optimization algorithm (%(default)s)")

argp.add_argument('--arg1_len', type=int, default=500,  # word: en=100, zh=500; char: en=400, zh=900
    help="length of argument 1 text span (en=100, zh=500) (%(default)s)")
argp.add_argument('--arg2_len', type=int, default=500,  # word: en=100, zh=500; char: en=400, zh=900
    help="length of argument 2 text span (en=100, zh=500) (%(default)s)")
argp.add_argument('--conn_len', type=int, default=10,  # word: en/zh=10; char: en/zh=20
    help="length of connective text span (en/zh=10) (%(default)s)")
argp.add_argument('--punc_len', type=int, default=2,  # en=0, zh=2
    help="length of punctuation text span (en=0, zh=2) (%(default)s)")
argp.add_argument('--masking', type=str2bool, nargs='?', default=True,
    help="use masking to process variable-length inputs (%(default)s)")
argp.add_argument('--original_positives', type=int, default=0,
    help="add untouched original samples (%(default)s)")
argp.add_argument('--random_positives', type=int, default=2,
    help="add positive random samples per original sample (%(default)s)")
argp.add_argument('--random_negatives', type=int, default=2,
    help="add negative random samples per original sample (%(default)s)")
argp.add_argument('--random_proba_1', type=float, default=0.1,
    help="probability 1 for random samples (see source) (%(default)s)")
argp.add_argument('--random_proba_2', type=float, default=0.1,
    help="probability 2 for random samples (see source) (%(default)s)")
argp.add_argument('--random_proba_3', type=float, default=0.9,
    help="probability 3 for random samples (see source) (%(default)s)")

argp.add_argument('--words_dim', type=int, default=20,  # en/zh: 20
    help="size of word embeddings (%(default)s)")
argp.add_argument('--words_dropout', type=float, default=0.3,
    help="dropout at word embeddings layer (%(default)s)")
argp.add_argument('--words2vec_bin', default=None,
    help="initialize with pre-trained word2vec embeddings (.gz) (en='GoogleNews-vectors-negative300.bin.gz')")
argp.add_argument('--words2vec_txt', default=None,
    help="initialize with pre-trained word2vec embeddings (.txt) (zh='zh-Gigaword-300.txt')")
argp.add_argument('--words_trainable', type=str2bool, nargs='?', default=True,
    help="trainable word embeddings (%(default)s)")
argp.add_argument('--words_shared', default="global",
    choices=["global", "span"],
    help="siamese sharing of word embeddings layer (%(default)s)")

argp.add_argument('--filter_type', default="lstm-fb",
    choices=rnn_choices,
    help="recurrent layer for filtering RNN (%(default)s)")
argp.add_argument('--filter_dim', type=int, default=12,  # en: 8, zh: 12
    help="size of filtering RNN (%(default)s)")
argp.add_argument('--filter_act', default="sigmoid",
    choices=["relu", "tanh", "sigmoid", "softmax", "softplus", "linear"],
    help="activation function at output gates of filtering RNN (%(default)s)")
argp.add_argument('--filter_dropout_W', type=float, default=0.0,
    help="dropout at input gates of filtering RNN (%(default)s)")
argp.add_argument('--filter_dropout_U', type=float, default=0.0,
    help="dropout at recurrent gates of filtering RNN (%(default)s)")
argp.add_argument('--filter_zoneout_h', type=float, default=0.0,
    help="zoneout at hidden unit of filtering RNN (%(default)s)")
argp.add_argument('--filter_zoneout_c', type=float, default=0.0,
    help="zoneout at cell unit of filtering RNN (%(default)s)")
argp.add_argument('--filter_append_dim', type=int, default=0,
    help="append fully-connected layer after filtering RNN (%(default)s)")
argp.add_argument('--filter_append_act', default="linear",
    choices=["relu", "tanh", "sigmoid", "softmax", "softplus", "linear", "softmax-time", "LeakyReLU-time", "PReLU-time", "ELU-time", "SReLU-time"],
    help="append activation function after filtering RNN (%(default)s)")
argp.add_argument('--filter_append_batchnorm', type=str2bool, nargs='?', default=False,
    help="append batch normalization after filtering RNN (%(default)s)")
argp.add_argument('--filter_append_dropout', type=float, default=0.0,
    help="append dropout after filtering RNN (%(default)s)")
argp.add_argument('--filter_shared', default="global",
    choices=["global", "span"],
    help="siamese sharing of filtering RNN layer (%(default)s)")
argp.add_argument('--filter_apply', default="mul",
    choices=["mul", "sum", "mulsum"],
    help="how to apply the filtering RNN (%(default)s)")

argp.add_argument('--rnn_num', type=int, default=12,  # zh: 12 for 10+1 senses, en: 8
    help="number of focused RNNs (<= filter_dim) (%(default)s)")
argp.add_argument('--rnn_type', default="lstm-fwd",
    choices=rnn_choices,
    help="recurrent layer for focused RNNs (%(default)s)")
argp.add_argument('--rnn_dim', type=int, default=20,  # en/zh: 20
    help="size of focused RNNs (%(default)s)")
argp.add_argument('--rnn_act', default="tanh",
    choices=["relu", "tanh", "sigmoid", "softmax", "softplus", "linear"],
    help="activation function at output gates of focused RNNs (%(default)s)")
argp.add_argument('--rnn_dropout_W', type=float, default=0.0,
    help="dropout at input gates of focused RNNs (%(default)s)")
argp.add_argument('--rnn_dropout_U', type=float, default=0.0,
    help="dropout at recurrent gates of focused RNNs (%(default)s)")
argp.add_argument('--rnn_zoneout_h', type=float, default=0.0,
    help="zoneout at hidden units of focused RNNs (%(default)s)")
argp.add_argument('--rnn_zoneout_c', type=float, default=0.0,
    help="zoneout at cell units of focused RNNs (%(default)s)")
argp.add_argument('--rnn_shared', default="aspect",
    choices=["global", "aspect", "span", "none"],
    help="siamese sharing of focused RNNs layers (%(default)s)")
argp.add_argument('--rnn_merge', default="concat",
    choices=["concat", "sum", "mul", "max"],
    help="merge output of focused RNNs layers (%(default)s)")
argp.add_argument('--rnn_dropout_merge', type=float, default=0.3,
    help="dropout at merge output of focused RNNs (%(default)s)")

argp.add_argument('--final_dim', type=int, default=80,  # zh: 80 (4*rnn_dim)
    help="size of additional fully-connected hidden layer (%(default)s)")
argp.add_argument('--final_act', default="SReLU",
    choices=["relu", "tanh", "sigmoid", "softmax", "softplus", "linear", "LeakyReLU", "PReLU", "ELU", "SReLU"],
    help="activation function at additional fully-connected hidden layer (%(default)s)")
argp.add_argument('--final_dropout', type=float, default=0.3,
    help="dropout at additional fully-connected hidden layer (%(default)s)")

argp.add_argument('--input_size', type=int, default=None,
    help="limit number of input samples (%(default)s)")
argp.add_argument('--words_crop', type=int, default=None,
    help="limit words vocabulary size (%(default)s)")

args = argp.parse_args()
args.backend = K._backend
args.backend_theano = os.getenv("THEANO_FLAGS")
if args.lang.endswith("ch"):
    args.mode = 'char'
    args.lang = args.lang[:-2]

if args.filter_apply == "mulsum" and args.rnn_num > args.filter_dim / 2 and args.rnn_num > args.filter_append_dim / 2:
    raise Exception("Configuration error for 'mulsum': rnn_num ({}) <= filter_dim/2 ({})".format(args.rnn_num, args.filter_dim))
elif args.rnn_num > args.filter_dim and args.rnn_num > args.filter_append_dim:
    raise Exception("Configuration error: rnn_num ({}) <= filter_dim ({})".format(args.rnn_num, args.filter_dim))

if args.filter_zoneout_h > 0.0 or args.filter_zoneout_c > 0.0 or args.rnn_zoneout_h > 0.0 or args.rnn_zoneout_c > 0.0:
    try:  # check for RNN zoneout support in Keras
        from keras.layers.recurrent import zoneout
    except ImportError:
        raise Exception("Error: RNN zoneout support in Keras is missing!")

### initialize experiment
indexes_pkl = "{}/indexes.pkl".format(args.model_dir)
indexes_size_pkl = "{}/indexes_size.pkl".format(args.model_dir)
indexes_cnts_pkl = "{}/indexes_cnts.pkl".format(args.model_dir)
model_yaml = "{}/model.yaml".format(args.model_dir)
model_png = "{}/model.png".format(args.model_dir)
weights_hdf5 = "{}/weights.hdf5".format(args.model_dir)
weights_val_hdf5 = "{}/weights_val.hdf5".format(args.model_dir)
weights_val_all_f1_hdf5 = "{}/weights_val_all_f1.hdf5".format(args.model_dir)
config_json = "{}/config.json".format(args.model_dir)
history_json = "{}/history.json".format(args.model_dir)
console_log = "{}/console.log".format(args.model_dir)

if not args.continue_ and os.path.isdir(args.model_dir):
    raise Exception("Model directory already exists: {}".format(args.model_dir))
if not os.path.isdir(args.model_dir):  # create directory
    os.makedirs(args.model_dir)

f_log = codecs.open(console_log, mode='a', encoding='utf8')
try:
    sys.stdout.files.append(f_log)
    sys.stderr.files.append(f_log)
except AttributeError:
    f_log.close()

log.info("configuration ({})".format(args.model_dir))
for k, v in sorted(vars(args).iteritems()):
    log.debug("  config '{}': {}".format(k, v))
with open(config_json, 'a') as f:  # dump configuration
    json.dump(vars(args), f)

if args.mode == "word":  # import word-level mode
    from word_utils import load_data, build_indexes, batch_generator, load_word2vec
elif args.mode == "char":  # import character-level mode
    from char_utils import load_data, build_indexes, batch_generator

### load datasets
log.info("load dataset for training ({})".format(args.train_dir))
train = load_data(args.train_dir, args.lang, args.filter_fn_name)
if args.input_size is not None:
    train['rel_ids'] = train['rel_ids'][:args.input_size]
log.info("  " + train.summary())

log.info("load dataset for validation ({})".format(args.valid_dir))
valid = load_data(args.valid_dir, args.lang, args.filter_fn_name)
log.info("  " + valid.summary())

### build indexes
if not os.path.isfile(indexes_pkl) or not os.path.isfile(indexes_size_pkl):
    log.info("build indexes from training")
    indexes, indexes_size, indexes_cnts = build_indexes(train, words_crop=args.words_crop)
    save_to_pkl(indexes_pkl, indexes)
    save_to_pkl(indexes_size_pkl, indexes_size)
    save_to_pkl(indexes_cnts_pkl, indexes_cnts)
else:
    log.info("load previous indexes ({})".format(indexes_pkl))
    indexes = load_from_pkl(indexes_pkl)
    indexes_size = load_from_pkl(indexes_size_pkl)
    indexes_cnts = load_from_pkl(indexes_cnts_pkl)
log.info("  " + ", ".join([ "{}: {}".format(k, v) for k, v in indexes_size.items() ]))
del indexes_cnts['words2id']   # free some memory

init_weights_emb = None
if args.words2vec_bin or args.words2vec_txt:
    log.info("load pre-trained word2vec embeddings")
    init_weights_emb = load_word2vec(indexes['words2id'], indexes_size['words2id'], args.words_dim, args.words2vec_bin, args.words2vec_txt)

# show target class distribution
log.info("target class distribution from training")
for k, i in sorted(indexes['target2id'].items(), key=lambda (k, i): i):
    log.info("  class {} = '{}': {}".format(i, k, indexes_cnts['target2id'][k]))
time_loaded = time.time()  # for time measurement

### build model
log.info("build model")
model = build_model(args, init_weights_emb, indexes_size['words2id'], indexes_size['target2id'])
plot_model(model, to_file=model_png, show_shapes=True)
with open(model_yaml, 'w') as f:
    model.to_yaml(stream=f)
model.summary()

model.compile(optimizer=args.optimizer, loss='categorical_crossentropy')
model._make_train_function()
model._make_predict_function()

# initialize weights
if not os.path.isfile(weights_hdf5):
    log.info("initialize weights")
else:
    log.info("load previous weights ({})".format(weights_hdf5))
    model.load_weights(weights_hdf5)

### training
log.info("training preparation")

# prepare data
curriculum_end = args.curriculum_end * len(train['rel_ids'])
samples_per_epoch = int((len(train['rel_ids']) * (args.original_positives + args.random_positives + args.random_negatives) * args.epochs_ratio + args.batch_size - 1) / args.batch_size) * args.batch_size
valid_size = (min(len(valid['rel_ids']), args.snapshot_size) + args.batch_size - 1) / args.batch_size * args.batch_size  # round up to multiple of batch_size
valid_snapshot = next(batch_generator(valid, indexes, indexes_size, None, args.arg1_len, args.arg2_len, args.conn_len, args.punc_len, valid_size, original_positives=1, random_positives=0, random_negatives=0))
train_iter = batch_generator(train, indexes, indexes_size, indexes_cnts, args.arg1_len, args.arg2_len, args.conn_len, args.punc_len, args.batch_size, original_positives=args.original_positives, random_positives=args.random_positives, random_negatives=args.random_negatives, random_proba_1=args.random_proba_1, random_proba_2=args.random_proba_2, random_proba_3=args.random_proba_3, curriculum_end=curriculum_end)

# prepare callbacks
callbacks = [
    ModelCheckpoint(monitor='loss', mode='min', filepath=weights_hdf5, save_best_only=True),
    ModelCheckpoint(monitor='val_loss', mode='min', filepath=weights_val_hdf5, save_best_only=True),
    CoNLL16stMetrics(valid, indexes, indexes_size, valid_snapshot[0], valid_snapshot[1], batch_size=args.batch_size),
    ModelCheckpoint(monitor='val_all_f1', mode='max', filepath=weights_val_all_f1_hdf5, save_best_only=True),
]
if args.epochs_patience >= 0:
    callbacks.append(EarlyStopping(monitor='val_all_f1', mode='max', patience=args.epochs_patience))

# train model
time_prepared = time.time()  # for time measurement
log.info("train model ({})".format(args.model_dir))
history = model.fit_generator(train_iter, nb_epoch=args.epochs, samples_per_epoch=samples_per_epoch, validation_data=valid_snapshot, callbacks=callbacks, verbose=1)
log.info("training finished ({})".format(args.model_dir))
time_trained = time.time()  # for time measurement
time_epoch_cnt = float(len(history.history['loss'])) * args.epochs_ratio  # number of real epochs

# display best results again
with open(history_json, 'a') as f:  # dump training history
    json.dump(history.history, f)

best_order = ['loss', 'val_loss', 'val_all_f1', 'val_exp_f1', 'val_non_f1']
if all(k in best_order for k in history.history.keys()):
    for k in sorted(history.history.keys()):
        if k not in best_order:
            best_order.append(k)
    best_i = min(range(len(history.history['val_loss'])), key=history.history["val_loss"].__getitem__)
    log.info("best_val_loss: {} - ".format(best_i) + " - ".join([ "{}: {:.4f}".format(k, history.history[k][best_i]) for k in best_order ]))
    best_i = max(range(len(history.history['val_all_f1'])), key=history.history["val_all_f1"].__getitem__)
    log.info("best_val_all_f1: {} - ".format(best_i) + " - ".join([ "{}: {:.4f}".format(k, history.history[k][best_i]) for k in best_order ]))

# display time measurement
time_info = {
    'loaded': time_loaded - time_start,
    'prepared': time_prepared - time_loaded,
    'trained': time_trained - time_prepared,
    'epoch': (time_trained - time_prepared) / time_epoch_cnt,
    'epoch_cnt': time_epoch_cnt,
    'epoch_samples': float(samples_per_epoch) / args.epochs_ratio,
}
time_order = ['loaded', 'prepared', 'trained', 'epoch', 'epoch_cnt', 'epoch_samples']
log.info("time: " + " - ".join([ "{}: {:.4f}".format(k, time_info[k]) for k in time_order ]))
