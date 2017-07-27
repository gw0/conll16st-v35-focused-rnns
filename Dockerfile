# Sense classification of discourse relations (CoNLL16st).
#
# Configuration example:
#   PRE=conll16st-v35b1 VOLDIR=/srv/conll16st/volume REPODIR="/srv/conll16st/conll16st" MEM=11000M DOCKER_ARGS="-m $MEM --memory-swap $MEM -v $VOLDIR/data:/srv/data -v $VOLDIR/ex:/srv/ex"
#   LANG=en CONFIGB='' CONFIG_WORDS2VEC='--words_dim=300 --words2vec_bin=./data/word2vec-en/GoogleNews-vectors-negative300.bin.gz'
#   LANG=zh CONFIGB='--arg1_len=500 --arg2_len=500 --punc_len=2' CONFIG_WORDS2VEC='--words_dim=300 --words2vec_txt=./data/word2vec-zh/zh-Gigaword-300.txt'
#   LANG=en CONFIGB='--mode=char --arg1_len=400 --arg2_len=400 --conn_len=20' CONFIG_WORDS2VEC=''
#   LANG=zh CONFIGB='--mode=char --arg1_len=900 --arg2_len=900 --conn_len=20 --punc_len=2' CONFIG_WORDS2VEC=''
#   docker build -t $PRE $REPODIR
#
# Development environment:
#   sudo docker run -it --rm -v $(pwd):/srv -v $(pwd)/home:/root -e KERAS_BACKEND='theano' -e THEANO_FLAGS='mode=FAST_COMPILE' gw000/keras-full:1.2.0 bash
#   ./v35/train.py ex/$(date +%Y%m%d-%H%M%S) en data/conll16st-en-03-29-16-trial data/conll16st-en-03-29-16-trial --epochs_ratio=1 --words_dim=2 --filter_dim=2 --rnn_num=2 --rnn_type=rnn-fwd --rnn_dim=10
#
# Usage example with Theano on GPU:
#   NAME=$PRE-$LANG
#   docker run -d $DOCKER_ARGS $(ls /dev/nvidia* | xargs -I{} echo '--device={}') -e THEANO_FLAGS='device=gpu,floatX=float32,nvcc.fastmath=True,lib.cnmem=0.45' --name $NAME $PRE ./v35/run.sh ex/$NAME $LANG $CONFIGB && echo -ne "\ek${NAME:10}\e\\" && docker attach --sig-proxy=false $NAME
#     docker rm -f $NAME; rm -rf $VOLDIR/ex/$NAME
#

FROM gw000/keras:1.2.2-py2-th-gpu
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2017@ena.one>

# requirements (for project)
RUN pip install gensim pattern
RUN git clone https://github.com/gw0/conll16st_data.git ./conll16st_data
RUN git clone https://github.com/attapol/conll16st.git ./conll16st_evaluation

# setup parser
ADD v35/ ./v35/
RUN groupadd -g 1000 app \
 && useradd -m -u 1000 -g app -G video app \
 && chown -R app:app /srv

# expose interfaces
#VOLUME /srv/data /srv/ex

USER app
