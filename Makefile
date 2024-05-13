PYTHON = python3
PIP = pip3

IMAGE=ravihammond/hanabi-project

default:
	@make -s build-cpp

build-cpp:
	sh scripts/build_cpp.sh

build-cudaconda: 
	docker build $(FLAGS) -t ravihammond/cuda-conda -f dockerfiles/Dockerfile.cudaconda .

build-plain: 
	docker build $(FLAGS) -t ${IMAGE}:plain -f dockerfiles/Dockerfile.projectnew --target plain .

build-dev: 
	docker build $(FLAGS) -t ${IMAGE}:dev -f dockerfiles/Dockerfile.projectnew --target dev .

build-temp: 
	docker build $(FLAGS) -t ${IMAGE}:temp -f dockerfiles/Dockerfile.projectnew --target dev .

build-prod:
	docker build $(FLAGS) -t ${IMAGE}:prod -f dockerfiles/Dockerfile.projectnew --target prod .

push-prod:
	docker push ${IMAGE}:prod

build-push-prod: build-prod push-prod

run-old:
	bash scripts/run_docker.bash

run-dev:
	bash scripts/run_docker_dev.bash

run-prod:
	bash scripts/run_docker_prod.bash $(SCRIPT) $(FLAGS)

run-plot:
	bash scripts/run_docker_plot.bash

buildrun-prod:
	make -s build-prod
	make -s run-prod

jupyter:
	bash scripts/jupyter.bash

