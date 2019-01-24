#!/bin/sh
# Re-run the classification step of all FR system variants on CoNLL16st datasets.
#
# Author: gw0 [http://gw.tnode.com/] <gw.2019@ena.one>
# License: All rights reserved

###
TRAIN_DIR="../data/conll16st-zh-01-08-2016-train"
VALID_DIR="../data/conll16st-zh-01-08-2016-dev"
TEST_DIR="../data/conll16st-zh-01-08-2016-test"
BLIND_DIR="../data/conll16st-zh-04-27-2016-blind-test"

LANG=zh LANG_CONFIG='--arg1_len=500 --arg2_len=500 --conn_len=10 --punc_len=2'

# [v35] output probabilities
EXTRA_CONFIG=''
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [v35noaugm] output probabilities
EXTRA_CONFIG='--original_positives=1 --random_positives=0 --random_negatives=0'
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [v35word2vec] output probabilities
EXTRA_CONFIG='--words_trainable=True --words_dim=300 --words2vec_txt=./data/word2vec-zh/zh-Gigaword-300.txt'
for MODEL_DIR in models-v35word2vec/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35word2vec/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35word2vec/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [vsimple] output probabilities
EXTRA_CONFIG="--rnn_dim=$((12*20))"
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done


###
LANG=zhch LANG_CONFIG='--mode=char --arg1_len=900 --arg2_len=900 --conn_len=20 --punc_len=2'

# [v35] output probabilities
EXTRA_CONFIG=''
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [v35noaugm] output probabilities
EXTRA_CONFIG='--original_positives=1 --random_positives=0 --random_negatives=0'
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [vsimple] output probabilities
EXTRA_CONFIG="--rnn_dim=$((12*20))"
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done


###
TRAIN_DIR="../data/conll16st-en-03-29-16-train"
VALID_DIR="../data/conll16st-en-03-29-16-dev"
TEST_DIR="../data/conll16st-en-03-29-16-test"
BLIND_DIR="../data/conll15st-en-03-29-16-blind-test"

LANG=en LANG_CONFIG='--arg1_len=100 --arg2_len=100 --conn_len=10 --punc_len=0'

# [v35] output probabilities
EXTRA_CONFIG='--filter_dim=8 --rnn_num=8'
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [v35noaugm] output probabilities
EXTRA_CONFIG='--original_positives=1 --random_positives=0 --random_negatives=0 --filter_dim=8 --rnn_num=8'
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [v35word2vec] output probabilities
EXTRA_CONFIG='--words_trainable=True --words_dim=300 --words2vec_bin=./data/word2vec-en/GoogleNews-vectors-negative300.bin.gz --filter_dim=8 --rnn_num=8'
for MODEL_DIR in models-v35word2vec/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35word2vec/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35word2vec/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [vsimple] output probabilities
EXTRA_CONFIG="--rnn_dim=$((8*20))"
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done


###
LANG=ench LANG_CONFIG='--mode=char --arg1_len=400 --arg2_len=400 --conn_len=20 --punc_len=0'

# [v35] output probabilities
EXTRA_CONFIG='--filter_dim=8 --rnn_num=8'
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35/*-$LANG-*; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [v35noaugm] output probabilities
EXTRA_CONFIG='--original_positives=1 --random_positives=0 --random_negatives=0 --filter_dim=8 --rnn_num=8'
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-v35noaugm/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./v35/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done

# [vsimple] output probabilities
EXTRA_CONFIG="--rnn_dim=$((8*20))"
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/valid"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/test"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
for MODEL_DIR in models-vsimple/*-$LANG; do OUTPUT_DIR="$MODEL_DIR/blind"; ./vsimple/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR" $LANG_CONFIG $EXTRA_CONFIG; done
