# Sense classification of discourse relations (CoNLL16st).
#
# Quick development environment in Docker:
#   cd $REPODIR
#   sudo docker run -it --rm -v $(pwd):/srv -v $(pwd)/home:/root -e KERAS_BACKEND='theano' -e THEANO_FLAGS='device=cpu' gw000/keras-full:1.2.0 bash
#   ./v35/train.py ex/test1 en data/conll16st-en-03-29-16-trial data/conll16st-en-03-29-16-trial --epochs_ratio=1 --words_dim=2 --filter_dim=2 --rnn_num=2 --rnn_type=rnn-fwd --rnn_dim=10
#   (you may also run any other command or full model training here)
#
# Advanced environment with GPU support in Docker and screen:
#   VOLDIR=/srv/conll16st/volume REPODIR="/srv/conll16st/conll16st-v35-focused-rnns" MEM=11000M DOCKER_ARGS="-m $MEM --memory-swap $MEM -v $VOLDIR/data:/srv/data -v $VOLDIR/ex:/srv/ex"
#   docker build -t conll16st-v35 $REPODIR
#   (set language configuration here)
#   NAME=conll16st-v35-$LANG
#   docker run -d $DOCKER_ARGS $(ls /dev/nvidia* | xargs -I{} echo '--device={}') $(ls /usr/lib/*-linux-gnu/{libcuda,libnvidia}* | xargs -I{} echo '-v {}:{}:ro') -e THEANO_FLAGS='device=gpu,floatX=float32,nvcc.fastmath=True,lib.cnmem=0.45' --name $NAME conll16st-v35 ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG && echo -ne "\ek${NAME:10}\e\\" && docker attach --sig-proxy=false $NAME
#     docker rm -f $NAME; rm -rf $VOLDIR/ex/$NAME
#
#
# Language configuration (pick one line):
#   LANG=zh LANG_CONFIG='--arg1_len=500 --arg2_len=500 --conn_len=10 --punc_len=2'
#   LANG=en LANG_CONFIG='--arg1_len=100 --arg2_len=100 --conn_len=10 --punc_len=0'
#   LANG=zhch LANG_CONFIG='--mode=char --arg1_len=900 --arg2_len=900 --conn_len=20 --punc_len=2'
#   LANG=ench LANG_CONFIG='--mode=char --arg1_len=400 --arg2_len=400 --conn_len=20 --punc_len=0'
#
# Our model [v35]:
#   (first set language configuration)
#   NAME=conll16st-v35-$LANG
#   zh: ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG
#   en: ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG --filter_dim=8 --rnn_num=8
#
# Our model without data augmentation [v35noaugm]:
#   (first set language configuration)
#   NAME=conll16st-v35noaugm-$LANG
#   zh: ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG --original_positives=1 --random_positives=0 --random_negatives=0
#   en: ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG --original_positives=1 --random_positives=0 --random_negatives=0 --filter_dim=8 --rnn_num=8
#
# Our model with pre-trained word embeddings (trainable, 300-dim) [v35word2vec]:
#   (first set language configuration)
#   NAME=conll16st-v35word2vec-$LANG
#   zh: ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG --words_trainable=True --words_dim=300 --words2vec_txt=./data/word2vec-zh/zh-Gigaword-300.txt
#   en: ./v35/run.sh ex/$NAME $LANG $LANG_CONFIG --words_trainable=True --words_dim=300 --words2vec_bin=./data/word2vec-en/GoogleNews-vectors-negative300.bin.gz --filter_dim=8 --rnn_num=8
#
# Baseline model with simple LSTMs [vsimple]:
#   (first set language configuration)
#   NAME=conll16st-vsimple-$LANG
#   zh: ./vsimple/run.sh ex/$NAME $LANG $LANG_CONFIG --rnn_dim=$((12*20))
#   en: ./vsimple/run.sh ex/$NAME $LANG $LANG_CONFIG --rnn_dim=$((8*20))
#

FROM gw000/keras:1.2.2-py2-th-gpu
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2017@ena.one>

# requirements (for project)
RUN pip install gensim==0.13.4.1 pattern==2.6
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
