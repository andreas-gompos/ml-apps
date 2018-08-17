FROM python:3.6

RUN apt-get update -y \
 && apt-get install -y \
    zip

# set up non-root user for jupyter notebook
ENV NB_USER jupyter_user
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# download bbc data and glove embeddings
WORKDIR ${HOME}
RUN wget http://nlp.stanford.edu/data/glove.6B.zip \
 && unzip -j glove.6B.zip glove.6B.300d.txt -d . \
 && rm glove.6B.zip

RUN wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip \
 && unzip bbc-fulltext.zip -d ./data \
 && cp -r ./data/bbc/* ./data \
 && rm -r ./data/bbc/ \
 && rm ./data/README.TXT \
 && rm bbc-fulltext.zip

# install python requirements
COPY ./requirements.txt ./requirements.txt
RUN set -ex \
 && python -m venv ./venv \
 && . ./venv/bin/activate \
 && pip install -U pip \
 && pip install -r requirements.txt \
 && python -m nltk.downloader stopwords punkt wordnet

COPY . ${HOME}
CMD . ./venv/bin/activate \
 && jupyter notebook --ip=0.0.0.0
