# 2-TPG

GPT-2, but fine-tuned on its own output to predict the _previous_ token.

## Setup

1. `pyenv local 3.13.2`
2. `python -m venv env`
3. `. env/bin/activate`
4. `pip install -r requirements.txt`

## Train

1. Get [gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset)
    ```bash
    cd data
    wget https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/small-117M-k40.train.jsonl
    ```
2. `python train.py`

## Evaluate

1. Get [gpt-2-output-dataset](https://github.com/openai/gpt-2-output-dataset)
    ```bash
    cd data
    wget https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/small-117M-k40.valid.jsonl
    ```
2. `python evaluate.py`

## Inference

1. `python inference.py --prompt "So we beat on, boats against the current, borne back ceaselessly into the past."`

## Push to Hugging Face

1. `pip install -U "huggingface_hub[cli]"`
2. `huggingface-cli login`
3. `cd 2tpg`
4. `huggingface-cli upload drwahl/2tpg .`
