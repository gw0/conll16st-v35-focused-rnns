#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Majority sense classifier of discourse relations (CoNLL16st).

Ready for evaluation on TIRA evaluation system (supplementary evaluation).
The parse should take three arguments:

    $language = specify language (en/zh)

    $inputDataset = the folder of the dataset to parse.
        The folder structure is the same as in the tar file
        $inputDataset/parses.json
        $inputDataset/relations-no-senses.json

    $inputRun = the folder that contains the model file or other resources

    $outputDir = the folder that the parser will output 'output.json' to

Note that we have to fill in 'Type' field as Explict and Implicit, 
but that will be overridden by the evaluator.

Based on <https://github.com/attapol/conll16st>.
"""
import codecs
import json
import random
import sys

from conll16st_evaluation.validator import EN_SENSES, ZH_SENSES


class DiscourseParser(object):
    """Sample discourse relation sense classifier
    
    This simply classifies each instance to the most common class for language.
    """

    def __init__(self):
        pass

    def classify_sense(self, data_dir, output_dir, valid_senses, select_from_senses):
        relation_file = '%s/relations-no-senses.json' % data_dir
        parse_file = '%s/parses.json' % data_dir
        parse = json.load(codecs.open(parse_file, encoding='utf8'))

        relation_dicts = []
        for line in codecs.open(relation_file, 'r', encoding='utf8'):
            if line.startswith('\x1b[?1034h'):
                line = line[8:]
            relation_dicts.append(json.loads(line))

        # sort relations
        relation_dicts.sort(key=lambda r: "{}:{}".format(r['DocID'], r['ID']))

        output = codecs.open('%s/output.json' % output_dir, 'wb', encoding ='utf8')
        output_proba = codecs.open('%s/output_proba.json' % output_dir, 'wb', encoding ='utf8')
        random.seed(10)
        for i, relation_dict in enumerate(relation_dicts):
            # fill in discourse relation text spans (not evaluated in supplementary task)
            relation_dict['Arg1']['TokenList'] = \
                    [x[2] for x in relation_dict['Arg1']['TokenList']]
            relation_dict['Arg2']['TokenList'] = \
                    [x[2] for x in relation_dict['Arg2']['TokenList']]
            relation_dict['Connective']['TokenList'] = \
                    [x[2] for x in relation_dict['Connective']['TokenList']]
            if len(relation_dict['Connective']['TokenList']) > 0:
                relation_dict['Type'] = 'Explicit'
            else:
                relation_dict['Type'] = 'Implicit'

            # output for one sense
            sense = select_from_senses[random.randint(0, len(select_from_senses) - 1)]
            relation_dict['Sense'] = [sense]
            output.write(json.dumps(relation_dict) + '\n')

            # output for sense probabilities
            proba = {}
            for t in valid_senses:
                if t == sense:
                    proba[t] = 1.
                else:
                    proba[t] = 0.
            relation_dict['SenseProba'] = proba
            output_proba.write(json.dumps(relation_dict) + '\n')

        output_proba.close()
        output.close()


if __name__ == '__main__':
    language = sys.argv[1]
    input_dataset = sys.argv[2]
    input_run = sys.argv[3]
    output_dir = sys.argv[4]
    if language == 'en':
        valid_senses = EN_SENSES
        select_from_senses = ['Expansion.Conjunction']  # select only most common class
    elif language == 'zh':
        valid_senses = ZH_SENSES
        select_from_senses = ['Conjunction']  # select only most common class
    parser = DiscourseParser()
    parser.classify_sense(input_dataset, output_dir, valid_senses, select_from_senses)

