#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
"""
Baseline model with simple LSTMs in Keras.
"""
__author__ = "GW [http://gw.tnode.com/] <gw.2017@ena.one>"
__license__ = "GPLv3+"

from keras.engine import Model, Input
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.layers import advanced_activations, merge, InputSpec
from keras import backend as K

try:  # check for RNN zoneout support in Keras
    from keras.layers.recurrent import zoneout
except ImportError:
    zoneout = None


### custom layers

# shareable variable-size RNN layers (only Theano)

class ShareableSimpleRNN(SimpleRNN):
    def __init__(self, *args, **kwargs):
        super(ShareableSimpleRNN, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        res = super(ShareableSimpleRNN, self).__call__(*args, **kwargs)
        self.input_spec = [InputSpec(shape=(self.get_input_shape_at(0)[0], None, self.get_input_shape_at(0)[2]))]
        return res

class ShareableGRU(GRU):
    def __init__(self, *args, **kwargs):
        super(ShareableGRU, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        res = super(ShareableGRU, self).__call__(*args, **kwargs)
        self.input_spec = [InputSpec(shape=(self.get_input_shape_at(0)[0], None, self.get_input_shape_at(0)[2]))]
        return res

class ShareableLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        super(ShareableLSTM, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        res = super(ShareableLSTM, self).__call__(*args, **kwargs)
        self.input_spec = [InputSpec(shape=(self.get_input_shape_at(0)[0], None, self.get_input_shape_at(0)[2]))]
        return res

if K._backend == "tensorflow":  # fallback workaround for TensorFlow
    ShareableSimpleRNN = SimpleRNN
    ShareableGRU = GRU
    ShareableLSTM = LSTM

CUSTOM_LAYERS = {
    'ShareableSimpleRNN': ShareableSimpleRNN,
    'ShareableGRU': ShareableGRU,
    'ShareableLSTM': ShareableLSTM,
}


### building blocks

def select_embedding(input_dim, output_dim, weights=None, dropout=0.0, trainable=True, mask_zero=True):

    # embedding block
    embedding = Embedding(input_dim=input_dim, output_dim=output_dim, weights=weights, dropout=dropout, trainable=trainable, mask_zero=mask_zero)

    def emb(x):
        y = embedding(x)
        return y
    return emb


def select_rnn(type, dim, return_sequences=False, activation='tanh', dropout_U=0.0, dropout_W=0.0, zoneout_h=0.0, zoneout_c=0.0, append_dim=0, append_activation='linear', append_batchnorm=False, append_dropout=0.0):
    """Wrapper providing the selected RNN layer type and direction."""

    type = type.split('-')

    # RNN layer
    rnn_kwargs = {
        'return_sequences': return_sequences,
        'activation': activation,
        'dropout_U': dropout_U,
        'dropout_W': dropout_W,
    }
    if zoneout_h > 0.0:  # only with RNN zoneout support
        rnn_kwargs['zoneout_h'] = zoneout_h
    if zoneout_c > 0.0:  # only for LSTM with RNN zoneout support
        rnn_kwargs['zoneout_c'] = zoneout_c

    if type[0] == 'rnn':
        rnn_fwd = ShareableSimpleRNN(dim, **rnn_kwargs)
        rnn_bck = ShareableSimpleRNN(dim, go_backwards=True, **rnn_kwargs)
    elif type[0] == 'gru':
        rnn_fwd = ShareableGRU(dim, **rnn_kwargs)
        rnn_bck = ShareableGRU(dim, go_backwards=True, **rnn_kwargs)
    elif type[0] == 'lstm':
        rnn_fwd = ShareableLSTM(dim, **rnn_kwargs)
        rnn_bck = ShareableLSTM(dim, go_backwards=True, **rnn_kwargs)

    if 'bi' in type:  # same bidirectional RNN with average
        rnn_bi = Bidirectional(rnn_fwd, merge_mode='ave')
    elif 'biconcat' in type:  # same bidirectional RNN with concat
        rnn_bi = Bidirectional(rnn_fwd, merge_mode='concat')

    # append layers
    if append_dim > 0:  # append time-distributed fully-connected layer
        dense_layer = Dense(append_dim)
    if append_activation == 'linear':  # append no activation
        activation_layer = None
    elif append_activation.endswith('-time') and append_activation[:-5] in advanced_activations.__dict__:  # append time-distributed advanced activation
        activation_layer = advanced_activations.__dict__.get(append_activation[:-5])()
    elif append_activation.endswith('-time'):  # append time-distributed simple activation
        activation_layer = Activation(append_activation[:-5])
    else:  # append simple activation
        activation_layer = Activation(append_activation)
    if append_dropout > 0.0:  # append dropout layer
        dropout_layer = Dropout(append_dropout)

    # RNN block
    def rnn(x):
        if 'fwd' in type:  # forward-only RNN
            y = rnn_fwd(x)
        elif 'fb' in type:  # separate forward/backward RNNs with average
            y_fwd = rnn_fwd(x)
            y_bck = rnn_bck(x)
            y = merge([y_fwd, y_bck], mode='ave')
        elif 'fbconcat' in type:  # separate forward/backward RNNs with concat
            y_fwd = rnn_fwd(x)
            y_bck = rnn_bck(x)
            y = merge([y_fwd, y_bck], mode='concat')
        elif 'bi' in type:  # same bidirectional RNN with average
            y = rnn_bi(x)
            #y = Bidirectional(rnn_fwd, merge_mode='ave')(x)
        elif 'biconcat' in type:  # same bidirectional RNN with concat
            y = rnn_bi(x)
            #y = Bidirectional(rnn_fwd, merge_mode='concat')(x)

        if append_dim > 0:
            y = TimeDistributed(dense_layer)(y)

        if activation_layer:
            if append_activation.endswith('-time'):
                y = TimeDistributed(activation_layer)(y)
            else:
                y = activation_layer(y)

        if append_batchnorm:
            y = BatchNormalization(mode=0, axis=1)(y)

        if append_dropout > 0.0:
            y = dropout_layer(y)
        return y

    return rnn


def simple_rnns(args, inputs, shared_rnn=None):
    """Build simple RNNs layer."""

    # additional stacked simple RNN layers
    for i in range(args.rnn_num - 1):
        #XXX: not shared, due to different dimensions
        shared_rnn = select_rnn(args.rnn_type, args.rnn_dim, return_sequences=True, activation=args.rnn_act, dropout_U=args.rnn_dropout_U, dropout_W=args.rnn_dropout_W, zoneout_h=args.rnn_zoneout_h, zoneout_c=args.rnn_zoneout_c)

        # individual stacked simple RNN
        inputs = shared_rnn(inputs)
        # shape: (samples, arg1_len, rnn_dim)

    # last simple RNN with one output
    if args.rnn_shared == "none":
        # no shared simple RNNs
        shared_rnn = select_rnn(args.rnn_type, args.rnn_dim, return_sequences=False, activation=args.rnn_act, dropout_U=args.rnn_dropout_U, dropout_W=args.rnn_dropout_W, zoneout_h=args.rnn_zoneout_h, zoneout_c=args.rnn_zoneout_c)

    # individual simple RNN
    outputs = shared_rnn(inputs)
    # shape: (samples, rnn_dim)
    return outputs


### build model

def build_model(args, init_weights_emb, words2id_size, target2id_size):
    """Build full Keras model with simple RNNs."""

    shared_emb = None
    shared_rnn = None
    if args.words_shared == "global":
        # globally shared word embeddings
        shared_emb = select_embedding(input_dim=words2id_size, output_dim=args.words_dim, weights=init_weights_emb, dropout=args.words_dropout, trainable=args.words_trainable, mask_zero=args.masking)
    if args.rnn_shared == "global":
        # globally shared simple RNNs
        shared_rnn = select_rnn(args.rnn_type, args.rnn_dim, return_sequences=False, activation=args.rnn_act, dropout_U=args.rnn_dropout_U, dropout_W=args.rnn_dropout_W, zoneout_h=args.rnn_zoneout_h, zoneout_c=args.rnn_zoneout_c)

    all_ids = []
    all_x = []

    # argument 1 text span
    if args.arg1_len > 0:
        if args.words_shared == "span":
            # per-span shared word embeddings
            shared_emb = select_embedding(input_dim=words2id_size, output_dim=args.words_dim, weights=init_weights_emb, dropout=args.words_dropout, trainable=args.words_trainable, mask_zero=args.masking)

        # input: arg1 word/token ids
        arg1_ids = Input(shape=(args.arg1_len,), dtype='int32', name="arg1_ids")
        # shape: (sample, arg1_len) of words2id_size
        # embed arg1 input sequence
        arg1_emb = shared_emb(arg1_ids)
        # shape: (sample, arg1_len, words_dim)
        # simple RNNs for arg1
        arg1_x = simple_rnns(args, arg1_emb, shared_rnn=shared_rnn)
        # shape: (sample, filter_dim*rnn_dim)

        all_ids.append(arg1_ids)
        all_x.append(arg1_x)

    # argument 2 text span
    if args.arg2_len > 0:
        if args.words_shared == "span":
            # per-span shared word embeddings
            shared_emb = select_embedding(input_dim=words2id_size, output_dim=args.words_dim, weights=init_weights_emb, dropout=args.words_dropout, trainable=args.words_trainable, mask_zero=args.masking)

        # input: arg2 word/token ids
        arg2_ids = Input(shape=(args.arg2_len,), dtype='int32', name="arg2_ids")
        # shape: (sample, arg2_len) of words2id_size
        # embed arg2 input sequence
        arg2_emb = shared_emb(arg2_ids)
        # shape: (sample, arg2_len, words_dim)
        # simple RNNs for arg2
        arg2_x = simple_rnns(args, arg2_emb, shared_rnn=shared_rnn)
        # shape: (sample, filter_dim*rnn_dim)

        all_ids.append(arg2_ids)
        all_x.append(arg2_x)

    # connective text span
    if args.conn_len > 0:
        if args.words_shared == "span":
            # per-span shared word embeddings
            shared_emb = select_embedding(input_dim=words2id_size, output_dim=args.words_dim, weights=init_weights_emb, dropout=args.words_dropout, trainable=args.words_trainable, mask_zero=args.masking)

        # input: conn word/token ids
        conn_ids = Input(shape=(args.conn_len,), dtype='int32', name="conn_ids")
        # shape: (sample, conn_len) of words2id_size
        # embed conn input sequence
        conn_emb = shared_emb(conn_ids)
        # shape: (sample, conn_len, words_dim)
        # simple RNNs for conn
        conn_x = simple_rnns(args, conn_emb, shared_rnn=shared_rnn)
        # shape: (sample, filter_dim*rnn_dim)

        all_ids.append(conn_ids)
        all_x.append(conn_x)

    # punctuation text span
    if args.punc_len > 0:
        if args.words_shared == "span":
            # per-span shared word embeddings
            shared_emb = select_embedding(input_dim=words2id_size, output_dim=args.words_dim, weights=init_weights_emb, dropout=args.words_dropout, trainable=args.words_trainable, mask_zero=args.masking)

        # input: punc word/token ids
        punc_ids = Input(shape=(args.punc_len,), dtype='int32', name="punc_ids")
        # shape: (sample, punc_len) of words2id_size
        # embed punc input sequence
        punc_emb = shared_emb(punc_ids)
        # shape: (sample, punc_len, words_dim)
        # simple RNNs for punc
        punc_x = simple_rnns(args, punc_emb, shared_rnn=shared_rnn)
        # shape: (sample, filter_dim*rnn_dim)

        all_ids.append(punc_ids)
        all_x.append(punc_x)

    # fully-connected layers
    x = merge(all_x, mode='concat')

    if args.final_dim > 0:
        # one hidden layer
        x = Dense(args.final_dim)(x)
        if args.final_act in advanced_activations.__dict__:  # advanced activation
            x = advanced_activations.__dict__.get(args.final_act)()(x)
        else:  # simple activation
            x = Activation(args.final_act)(x)
        x = Dropout(args.final_dropout)(x)
        # shape: (samples, final_dim)

    # logistic regression for classification
    x = Dense(target2id_size)(x)
    target = Activation('softmax', name='target')(x)
    # shape: (samples, target2id_size)

    # return model
    model = Model(input=all_ids, output=[target])
    return model
