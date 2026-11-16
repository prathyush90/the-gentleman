PY := .venv/bin/python
MODEL ?= llama8b

.PHONY: help install chat probes data train

help:
	@echo "make install                 create .venv and install dependencies"
	@echo "make chat [MODEL=ritchie-3b]  chat with the bot in the terminal (default: llama8b)"
	@echo "make probes [MODEL=...]      run the 20 test prompts, before vs after"
	@echo "make data                    rebuild the train/eval split"
	@echo "make train NAME=my-run [BASE=unsloth/Llama-3.1-8B-Instruct]"

install:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.txt

chat:
	$(PY) train/chat.py $(MODEL)

probes:
	$(PY) train/eval_probes.py $(MODEL)

data:
	$(PY) train/prepare_data.py

BASE ?= unsloth/Llama-3.2-3B-Instruct
train:
	@test -n "$(NAME)" || (echo "usage: make train NAME=my-run [BASE=...]" && exit 1)
	$(PY) train/train.py --name $(NAME) --model $(BASE)
