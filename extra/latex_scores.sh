#!/bin/bash
# Helper script for LaTeX output of scores in tabular environment.
#
#   for DIR in ../models-ablation/conll16st-v35*-en; do ./latex_scores.sh $DIR; done
#
# Author: gw0 [http://gw.tnode.com/] <gw.2019@ena.one>
# License: All rights reserved

DIR="$1"

echo
echo "F1-score"

VALID=$(egrep '^Precision' $DIR/valid/extra_sup_eval.log | cut -d ' ' -f 6)
VALID=(${VALID[@]})
TEST=$(egrep '^Precision' $DIR/test/extra_sup_eval.log | cut -d ' ' -f 6)
TEST=(${TEST[@]})
BLIND=$(egrep '^Precision' $DIR/blind/extra_sup_eval.log | cut -d ' ' -f 6)
BLIND=(${BLIND[@]})

echo "    %SRC: $DIR"
echo -n "    - XXX                         "
echo -n " &   \$${VALID[0]}\$ &   \$${VALID[1]}\$ &   \$${VALID[2]}\$"
echo -n " &   \$${TEST[0]}\$ &   \$${TEST[1]}\$ &   \$${TEST[2]}\$"
echo -n " &   \$${BLIND[0]}\$ &   \$${BLIND[1]}\$ &   \$${BLIND[2]}\$"
echo " \\\\"


echo "Brier-score"

VALID=$(egrep '^Brier' $DIR/valid/extra_sup_eval.log | cut -d ' ' -f 3)
VALID=(${VALID[@]})
TEST=$(egrep '^Brier' $DIR/test/extra_sup_eval.log | cut -d ' ' -f 3)
TEST=(${TEST[@]})
BLIND=$(egrep '^Brier' $DIR/blind/extra_sup_eval.log | cut -d ' ' -f 3)
BLIND=(${BLIND[@]})

echo "    %SRC: $DIR"
echo -n "    - XXX                         "
echo -n " &   \$${VALID[0]}\$ &   \$${VALID[1]}\$ &   \$${VALID[2]}\$"
echo -n " &   \$${TEST[0]}\$ &   \$${TEST[1]}\$ &   \$${TEST[2]}\$"
echo -n " &   \$${BLIND[0]}\$ &   \$${BLIND[1]}\$ &   \$${BLIND[2]}\$"
echo " \\\\"

