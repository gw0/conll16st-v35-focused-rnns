#!/bin/bash
# Full run of baseline model with simple LSTMs for discourse relation sense classifier (CoNLL16st).
#
# Consisting of:
#   - trainer of the focused RNN model from scratch
#   - classifier of the trained model
#   - official validator of the output format
#   - official scorer for the final evaluation
#
# Usage:
#   mkdir data/
#   tar -C data/ -vzxf ./conll16st-en-zh-dev-train-test_LDC2016E50.tgz
#   git clone https://github.com/attapol/conll16st conll16st_evaluation
#   ./vsimple/run.sh [en|zh] <model_dir> [extra_param]
#
# Author: gw0 [http://gw.tnode.com/] <gw.2017@ena.one>
# License: All rights reserved

MODEL_DIR="$1"
LANG="${2:0:2}"  # use only first two chars
shift; shift
EXTRA="$@"
OUTPUT_DIR="$MODEL_DIR"

# preset configuration
if [ "$LANG" == 'en' ]; then
  TRAIN_DIR="data/conll16st-en-03-29-16-train"
  VALID_DIR="data/conll16st-en-03-29-16-dev"
  TEST_DIR="data/conll16st-en-03-29-16-test"
  BLIND_DIR="data/conll15st-en-03-29-16-blind-test"
elif [ "$LANG" == 'zh' ]; then
  TRAIN_DIR="data/conll16st-zh-01-08-2016-train"
  VALID_DIR="data/conll16st-zh-01-08-2016-dev"
  TEST_DIR="data/conll16st-zh-01-08-2016-test"
  BLIND_DIR="data/conll16st-zh-04-27-2016-blind-test"
fi
if [ -z "$TRAIN_DIR" -a -z "$MODEL_DIR" ]; then
  echo "Usage: ./vsimple/run.sh <model_dir> [en|zh] [extra_param]"
  exit -1
fi

# train a focused RNN model on training dataset
sleep 1
if [ -d "$MODEL_DIR" ]; then
  echo
  echo "=== skip training, use existing model : $MODEL_DIR ==="
else
  echo
  echo "=== train model on training dataset : $MODEL_DIR ==="
  ./vsimple/train.py "$MODEL_DIR" "$LANG" "$TRAIN_DIR" "$VALID_DIR" $EXTRA
fi
set -e -o pipefail

# evaluate on validation dataset
echo
echo "=== evaluate on validation dataset : $MODEL_DIR ==="
./vsimple/classify.py "$MODEL_DIR" "$LANG" "$VALID_DIR" "$OUTPUT_DIR/valid" $EXTRA
./conll16st_evaluation/validator.py "$LANG" "$OUTPUT_DIR/valid/output.json" 2>&1 | tee "$OUTPUT_DIR/valid/validator.log" | (grep -v 'Validating line' || true)
#./conll16st_evaluation/scorer.py "$VALID_DIR/relations.json" "$OUTPUT_DIR/valid/output.json" 2>&1 | tee "$OUTPUT_DIR/valid/scorer.log"
./conll16st_evaluation/tira_sup_eval.py "$VALID_DIR" "$OUTPUT_DIR/valid" "$OUTPUT_DIR/valid" 2>&1 | tee "$OUTPUT_DIR/valid/tira_sup_eval.log"

# evaluate on test dataset (DO NOT LOOK AT THIS)
echo
echo "=== evaluate on test dataset (DO NOT LOOK AT THIS) : $MODEL_DIR ==="
./vsimple/classify.py "$MODEL_DIR" "$LANG" "$TEST_DIR" "$OUTPUT_DIR/test" $EXTRA
./conll16st_evaluation/tira_sup_eval.py "$TEST_DIR" "$OUTPUT_DIR/test" "$OUTPUT_DIR/test" 2>&1 | tee "$OUTPUT_DIR/test/tira_sup_eval.log"

# evaluate on blind test dataset (DO NOT LOOK AT THIS)
echo
echo "=== evaluate on blind test dataset (DO NOT LOOK AT THIS) : $MODEL_DIR ==="
./vsimple/classify.py "$MODEL_DIR" "$LANG" "$BLIND_DIR" "$OUTPUT_DIR/blind" $EXTRA
./conll16st_evaluation/tira_sup_eval.py "$BLIND_DIR" "$OUTPUT_DIR/blind" "$OUTPUT_DIR/blind" 2>&1 | tee "$OUTPUT_DIR/blind/tira_sup_eval.log"

echo
echo "=== done : $MODEL_DIR ==="
