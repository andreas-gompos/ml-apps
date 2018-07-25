help:
	@cat Makefile

jupyter_image:
	docker image build -t topic_classification_image_dev .

jupyter_container:
	docker container run -p 8888:8888 --rm -it --name topic_classification_container_dev topic_classification_image_dev

all:
	jupyter_image jupyter_container	
