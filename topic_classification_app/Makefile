help:
	@cat Makefile

build:
	docker build -t topic_classification_image_app .

bash: 
	docker container run --rm -it --name topic_classification_container_app topic_classification_image_app bash

run_app:
	docker container run -p 5000:5000 --rm -it --name topic_classification_container_app topic_classification_image_app

deploy:
	build run_app	
