In this project a topic classification app was created. More specifically an LSTM network was trained to distinguish between 5 different categories of articles (business, entertainment, politics, sport, tech) and a minimal Sanic API was created to serve the app. The dataset used, for training the network, was the [BBC articles dataset](http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip), which consists of 2225 documents, from the BBC news website corresponding to stories from 2004-2005.

**Requirements**

* [Docker](https://docs.docker.com/install/)

**Run the app**

1.  Run the containerised app by doing either a or b.

      a.   Build docker image locally

          1. `docker build -t topic_classification_image_app .`
          2. `docker container run --rm -it -p 5000:5000 topic_classification_image_app`
    
    
      b.   Run container directly from [Docker Hub](https://hub.docker.com/r/datagusto/topic_classification_app/)
          1. `docker container run --rm -it -p 5000:5000 datagusto/topic_classification_app`
      
      
2.  Post a json file of the following format to localhost:5000.
    {"text":"made with love"}
    

Any feedback is welcome! :)
[LinkedIn](https://www.linkedin.com/in/andreas-gompos/)
