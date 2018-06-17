In this project a topic classification app was created. More specifically an LSTM network was trained to distinguish between 5 different categories of articles (business, entertainment, politics, sport, tech) and a minimal Flask API was created to serve that app. The dataset used, for training the network, was the [BBC articles dataset](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip), which consists of 2225 documents, from the BBC news website corresponding to stories from 2004-2005.

**Requirements**

* [Docker](https://docs.docker.com/install/)

**Run the app**

1.  Option 1 
You can build and run the app using the following commands:
    * `docker build -t topic_classification_image_app .`
    * `docker container run -p 5000:5000 --rm -it --name topic_classification_container_app topic_classification_image_app`
    
    Note that you can use the provided Makefile to run the command `make deploy`, which will run the above commands for you.

2.  Option 2
You can pull and run the image from [Docker Hub](https://hub.docker.com/r/datagusto/topic_classification/) directly. Simply run:
    * `docker container run --rm  -p 5000:5000 -d --name flask_app_container datagusto/topic_classification`


3.The app should is now reachable on `localhost:5000/predict`.

Any feedback is welcome! :)
[LinkedIn](https://www.linkedin.com/in/andreas-gompos/)
