#!/bin/sh
# Evaluate additional scores, confusion matrices, and Brier score of all FR system variants on CoNLL16st datasets.
#
# Author: gw0 [http://gw.tnode.com/] <gw.2019@ena.one>
# License: All rights reserved

###
TRAIN_DIR="../data/conll16st-zh-01-08-2016-train"
VALID_DIR="../data/conll16st-zh-01-08-2016-dev"
TEST_DIR="../data/conll16st-zh-01-08-2016-test"
BLIND_DIR="../data/conll16st-zh-04-27-2016-blind-test"

LANG=zh LANG_CONFIG='--arg1_len=500 --arg2_len=500 --conn_len=10 --punc_len=2'

# compute confusion matrix and Brier score
for OUTPUT_DIR in ../models-*/*-$LANG-*/valid ../models-*/*-$LANG/valid; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/test ../models-*/*-$LANG/test; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/blind ../models-*/*-$LANG/blind; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done


###
LANG=zhch LANG_CONFIG='--mode=char --arg1_len=900 --arg2_len=900 --conn_len=20 --punc_len=2'

# compute confusion matrix and Brier score
for OUTPUT_DIR in ../models-*/*-$LANG-*/valid ../models-*/*-$LANG/valid; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/test ../models-*/*-$LANG/test; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/blind ../models-*/*-$LANG/blind; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done


###
TRAIN_DIR="../data/conll16st-en-03-29-16-train"
VALID_DIR="../data/conll16st-en-03-29-16-dev"
TEST_DIR="../data/conll16st-en-03-29-16-test"
BLIND_DIR="../data/conll15st-en-03-29-16-blind-test"

LANG=en LANG_CONFIG='--arg1_len=100 --arg2_len=100 --conn_len=10 --punc_len=0'

# compute confusion matrix and Brier score
for OUTPUT_DIR in ../models-*/*-$LANG-*/valid ../models-*/*-$LANG/valid; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/test ../models-*/*-$LANG/test; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/blind ../models-*/*-$LANG/blind; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done


###
LANG=ench LANG_CONFIG='--mode=char --arg1_len=400 --arg2_len=400 --conn_len=20 --punc_len=0'

# compute confusion matrix and Brier score
for OUTPUT_DIR in ../models-*/*-$LANG-*/valid ../models-*/*-$LANG/valid; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/test ../models-*/*-$LANG/test; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
for OUTPUT_DIR in ../models-*/*-$LANG-*/blind ../models-*/*-$LANG/blind; do [ ! -f $OUTPUT_DIR/extra_sup_eval.log ] && ../conll16st_evaluation/extra_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR" "$OUTPUT_DIR" 2>&1 > $OUTPUT_DIR/extra_sup_eval.log; done
