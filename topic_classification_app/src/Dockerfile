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
 && python -m nltk.downloader stopwords punkt wordnet \
 && mkdir ./artifacts \
 && pip freeze > ./artifacts/requirements.txt

COPY ./dependencies.py ./dependencies.py
COPY ./train.py ./train.py
RUN . ./venv/bin/activate && python ./train.py

COPY ./main.py ./main.py

FROM python:3.6.5
WORKDIR /usr/src/

COPY --from=base /usr/src/venv /usr/src/venv
COPY --from=base /root/nltk_data /root/nltk_data
COPY --from=base /usr/src/dependencies.py /usr/src/dependencies.py
COPY --from=base /usr/src/artifacts /usr/src/artifacts
COPY --from=base /usr/src/main.py /usr/src/main.py

CMD . ./venv/bin/activate && python main.py
