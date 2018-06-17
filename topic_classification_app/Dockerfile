FROM python:3.6.5 as base
WORKDIR /usr/src/
RUN apt-get update -y \
 && apt-get install -y \
    zip

RUN wget http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip \
 && unzip bbc-fulltext.zip -d ./data \
 && cp -r ./data/bbc/* ./data \
 && rm -r ./data/bbc/ \
 && rm ./data/README.TXT \
 && rm bbc-fulltext.zip

RUN wget http://nlp.stanford.edu/data/glove.6B.zip \
 && unzip -j glove.6B.zip glove.6B.300d.txt -d . \
 && rm glove.6B.zip

COPY ./requirements.txt ./requirements.txt
RUN python -m venv ./venv \
 && . ./venv/bin/activate \
 && pip install -U pip \
 && pip install -r requirements.txt \
 && python -m nltk.downloader stopwords punkt wordnet

COPY ./app ./app
RUN pip freeze > ./app/artifacts/requirements.txt

WORKDIR /usr/src/app
RUN . ../venv/bin/activate \
 && python ./train.py


FROM python:3.6.5 
WORKDIR /usr/src/
COPY --from=base /usr/src/venv /usr/src/venv
COPY --from=base /root/nltk_data /root/nltk_data
COPY --from=base /usr/src/app /usr/src/app

WORKDIR /usr/src/app
CMD . ../venv/bin/activate && python main.py
