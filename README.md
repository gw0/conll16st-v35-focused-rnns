conll16st-v35-focused-rnns
==========================

System implementation of the PLOS ONE paper "*Sense Classification of Shallow Discourse Relations with focused RNNs*", that follows the official CoNLL 2016 Shared Task definition. Implemented in Python 2.7 using Keras 1.2.2 and Numpy.

**Note:** Same implementation is used on English and Chinese datasets on word and character levels.

- <http://github.com/gw0/conll16st-v35-focused-rnns/>
- <http://www.cs.brandeis.edu/~clp/conll16st/>


Usage
-----

Although the source code can run on *Ubuntu 16.04* in *Python 2.7* with *Keras 1.2.2*, *Numpy*, and some other Python libraries, we prepared a *Docker* image to simplify the preparation of the environment (see `./Dockerfile` for details).

Requirements:

- *Ubuntu 16.04* or similar
- [*Docker Engine*](https://www.docker.com/get-docker)
- official CoNLL 2016 Shared Task datasets for English and Chinese:
  - `data/conll16st-en-03-29-16-train`: English training dataset from PDTB
  - `data/conll16st-en-03-29-16-dev`: English validation dataset from PDTB
  - `data/conll16st-en-03-29-16-test`: English test dataset from PDTB
  - `data/conll15st-en-03-29-16-blind-test`: English blind test dataset from English Wikinews
  - `data/conll16st-zh-01-08-2016-train`: Chinese training dataset from CDTB
  - `data/conll16st-zh-01-08-2016-dev`: Chinese validation dataset from CDTB
  - `data/conll16st-zh-01-08-2016-test`: Chinese test dataset from CDTB
  - `data/conll16st-zh-04-27-2016-blind-test`: Chinese blind test dataset from Chinese Wikinews

Preparation:

```bash
$ sudo apt-get install docker-engine
$ git clone --recursive https://github.com/gw0/conll16st-v35-focused-rnns.git
$ cd ./conll16st-v35-focused-rnns
$ tar -C data/ -vzxf ../conll16st-en-zh-dev-train-test_LDC2016E50.tgz
```

We prepared the script `./v35/run.sh` to perform a full run of sense classification of discourse relations on all datasets for a given language. Its syntax is:

```bash
$ ./v35/run.sh [en|zh] <model_dir> [extra_param]
```

It consist of running (see `./v35/run.sh` for details):

- train a focused RNN model from scratch on training dataset (`./v35/train.py`)
- classification and evaluation on validation dataset:
    - classification using the trained model (`./v35/classify.py`)
    - official validation of the output format (`./conll16st_evaluation/validator.py`)
    - official TIRA evaluation scorer (`./conll16st_evaluation/tira_sup_eval.py`)
- same for classification and evaluation on test dataset
- same for classification and evaluation on blind test dataset

Usage example with Docker (set environment variables according to your setup):

```bash
$ PRE=conll16st-v35b1 VOLDIR=/srv/conll16st/volume REPODIR="/srv/conll16st/conll16st" MEM=11000M DOCKER_ARGS="-m $MEM --memory-swap $MEM -v $VOLDIR/data:/srv/data -v $VOLDIR/models-v35:/srv/models-v35"
$ LANG=zh CONFIGB='--arg1_len=500 --arg2_len=500 --punc_len=2'
$ NAME=$PRE-$LANG-default-0

# build image
$ docker build -t $PRE $REPODIR

# run container
$ docker run -d $DOCKER_ARGS --name $NAME $PRE ./v35/run.sh models-v35/$NAME $LANG $CONFIGB && echo -ne "\ek${NAME:10}\e\\" && docker attach --sig-proxy=false $NAME

# remove container
$ docker rm -f $NAME
```

Pre-trained models for English and Chinese datasets at word and character levels are available in the folder `./models-v35`. These can be used for classification like this:

```bash
$ ./v35/classify.py <model_dir> [en|zh] <dataset_dir> <output_dir> [extra_param]
```


License
=======

Copyright &copy; 2016-2018 *gw0* [<http://gw.tnode.com/>] &lt;<gw.2018@ena.one>&gt;

This code is licensed under the [GNU Affero General Public License 3.0+](LICENSE_AGPL-3.0.txt) (*AGPL-3.0+*). Note that it is mandatory to make all modifications and complete source code publicly available to any user.

THIS SOURCE CODE IS SUPPLIED “AS IS” WITHOUT WARRANTY
OF ANY KIND, AND ITS AUTHOR AND THE JOURNAL OF
MACHINE LEARNING RESEARCH (JMLR) AND JMLR’S PUBLISHERS
AND DISTRIBUTORS, DISCLAIM ANY AND ALL WARRANTIES,
INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE,
AND ANY WARRANTIES OR NON INFRINGEMENT. THE USER
ASSUMES ALL LIABILITY AND RESPONSIBILITY FOR USE OF THIS
SOURCE CODE, AND NEITHER THE AUTHOR NOR JMLR, NOR
JMLR’S PUBLISHERS AND DISTRIBUTORS, WILL BE LIABLE FOR
DAMAGES OF ANY KIND RESULTING FROM ITS USE. Without limiting
the generality of the foregoing, neither the author, nor JMLR, nor
JMLR’s publishers and distributors, warrant that the Source Code will be
error-free, will operate without interruption, or will meet the needs of the
user.
