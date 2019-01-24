#!/bin/sh
# Evaluate simple baseline models for discourse relation sense classifier (CoNLL16st).
#
# Author: gw0 [http://gw.tnode.com/] <gw.2019@ena.one>
# License: All rights reserved

###
TRAIN_DIR="../data/conll16st-zh-01-08-2016-train"
VALID_DIR="../data/conll16st-zh-01-08-2016-dev"
TEST_DIR="../data/conll16st-zh-01-08-2016-test"
BLIND_DIR="../data/conll16st-zh-04-27-2016-blind-test"

LANG=zh LANG_CONFIG='--arg1_len=500 --arg2_len=500 --conn_len=10 --punc_len=2'

# [majority] output
MODEL_DIR="models-baselines/majority-$LANG"
OUTPUT_DIR="$MODEL_DIR/valid"; mkdir -p "$OUTPUT_DIR"
./baselines/majority_sup_parser.py "$LANG" "$VALID_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/test"; mkdir -p "$OUTPUT_DIR"
./baselines/majority_sup_parser.py "$LANG" "$TEST_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/blind"; mkdir -p "$OUTPUT_DIR"
./baselines/majority_sup_parser.py "$LANG" "$BLIND_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

# [random] output
MODEL_DIR="models-baselines/random-$LANG"
OUTPUT_DIR="$MODEL_DIR/valid"; mkdir -p "$OUTPUT_DIR"
./baselines/random_sup_parser.py "$LANG" "$VALID_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/test"; mkdir -p "$OUTPUT_DIR"
./baselines/random_sup_parser.py "$LANG" "$TEST_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/blind"; mkdir -p "$OUTPUT_DIR"
./baselines/random_sup_parser.py "$LANG" "$BLIND_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

# compute TIRA evaluation output
for OUTPUT_DIR in models-baselines/*-$LANG/valid; do ./conll16st_evaluation/tira_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/tira_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/test; do ./conll16st_evaluation/tira_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/tira_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/blind; do ./conll16st_evaluation/tira_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/tira_sup_eval.log; done

# compute confusion matrix and Brier score
for OUTPUT_DIR in models-baselines/*-$LANG/valid; do ./conll16st_evaluation/extra_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/test; do ./conll16st_evaluation/extra_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/blind; do ./conll16st_evaluation/extra_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done


###
TRAIN_DIR="../data/conll16st-en-03-29-16-train"
VALID_DIR="../data/conll16st-en-03-29-16-dev"
TEST_DIR="../data/conll16st-en-03-29-16-test"
BLIND_DIR="../data/conll15st-en-03-29-16-blind-test"

LANG=en LANG_CONFIG='--arg1_len=100 --arg2_len=100 --conn_len=10 --punc_len=0'

# [majority] output
MODEL_DIR="models-baselines/majority-$LANG"
OUTPUT_DIR="$MODEL_DIR/valid"; mkdir -p "$OUTPUT_DIR"
./baselines/majority_sup_parser.py "$LANG" "$VALID_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/test"; mkdir -p "$OUTPUT_DIR"
./baselines/majority_sup_parser.py "$LANG" "$TEST_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/blind"; mkdir -p "$OUTPUT_DIR"
./baselines/majority_sup_parser.py "$LANG" "$BLIND_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

# [random] output
MODEL_DIR="models-baselines/random-$LANG"
OUTPUT_DIR="$MODEL_DIR/valid"; mkdir -p "$OUTPUT_DIR"
./baselines/random_sup_parser.py "$LANG" "$VALID_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/test"; mkdir -p "$OUTPUT_DIR"
./baselines/random_sup_parser.py "$LANG" "$TEST_DIR" "$MODEL_DIR" "$OUTPUT_DIR"
OUTPUT_DIR="$MODEL_DIR/blind"; mkdir -p "$OUTPUT_DIR"
./baselines/random_sup_parser.py "$LANG" "$BLIND_DIR" "$MODEL_DIR" "$OUTPUT_DIR"

# compute TIRA evaluation output
for OUTPUT_DIR in models-baselines/*-$LANG/valid; do ./conll16st_evaluation/tira_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/tira_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/test; do ./conll16st_evaluation/tira_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/tira_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/blind; do ./conll16st_evaluation/tira_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/tira_sup_eval.log; done

# compute confusion matrix and Brier score
for OUTPUT_DIR in models-baselines/*-$LANG/valid; do ./conll16st_evaluation/extra_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/test; do ./conll16st_evaluation/extra_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in models-baselines/*-$LANG/blind; do ./conll16st_evaluation/extra_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
