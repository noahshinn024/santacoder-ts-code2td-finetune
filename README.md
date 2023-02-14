# Fine-tuning for TypeScript Code to Type Declaration Code Generation

## Setup

First, clone this repo:

```
git clone https://github.com/noahshinn024/santacoder-ts-code2td-finetune.git
cd santacoder-ts-code2td-finetune
```

Then, install the required packages

```
pip install -r requirements.txt
```

Then, login to HuggingFace Hub and Weights & Biases

```
huggingface-cli login
wandb login
```

## To Run
To run on 1 GPU

```bash
python train.py \
        --model_path="noahshinn024/santacoder-ts" \
        --dataset_name="noahshinn024/ts-code2td" \
        --seq_length 2048 \
        --learning_rate 2e-5 \
        --batch_size 6 \
        --max_steps 50000 \
	      --no_fp16 \
        --log_freq 1 \
        --eval_freq 1000 \
        --save_freq 1000
```

To run on multiple GPUs
```bash
python -m torch.distributed.launch \
        --nproc_per_node number_of_gpus train.py \
        --model_path="noahshinn024/santacoder-ts" \
        --dataset_name="noahshinn024/ts-code2td" \
        --seq_length 2048 \
        --learning_rate 2e-5 \
        --batch_size 6 \
        --max_steps 50000 \
	      --no_fp16 \
        --log_freq 1 \
        --eval_freq 1000 \
        --save_freq 1000
```
