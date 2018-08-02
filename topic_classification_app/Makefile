help:
	@cat Makefile

build:
	docker build -t topic_classification_app ./src/

run:
	docker container run -d -p 5000:5000 --rm -it --name topic_classification_app topic_classification_app


