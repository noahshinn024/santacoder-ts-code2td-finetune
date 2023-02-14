"""
Fine-Tune a TypeScript fine-tuned SantaCoder model for code to type declaration generation.
"""

import os
import torch
import argparse
import evaluate
import numpy as np
from datasets.load import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging,
    set_seed,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)

TASK_PREFIX = 'Generate type definitions for the following section of code:\n\nCode:\n'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="noahshinn024/santacoder-ts")
    parser.add_argument("--dataset_name", type=str, default="noahshinn024/ts-code2td")

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=10000)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)
    return parser.parse_args()


def create_datasets(args, tokenizer):
    def preprocess_function(examples):
        inputs = [TASK_PREFIX + code for code in examples["code"]]
        model_inputs = tokenizer(inputs, max_length=2048, truncation=True)

        labels = tokenizer(text_target=examples["declarations"], max_length=2048, truncation=True)
        
        # Only keep samples with token sizes less than 2048
        indices = [i for i, input_ids in enumerate(model_inputs["input_ids"]) if np.count_nonzero(input_ids != tokenizer.pad_token_id) < 2048]
        model_inputs = {k: [v[i] for i in indices] for k, v in model_inputs.items()}
        labels = {k: [v[i] for i in indices] for k, v in labels.items()}

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    ds = load_dataset(args.dataset_name)
    tokenized_ds = ds.map(preprocess_function, batched=True)
    return tokenized_ds["train"], tokenized_ds["test"]

def run_training(args, model, tokenizer, train_set, eval_set, metric):
    def compute_metrics(eval_pred, tokenizer):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}
    
    train_set.start_iteration = 0

    print(f"Starting main loop")
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=args.no_fp16,
        run_name=f'code2td-{args.model_path.split("/")[-1]}',
        report_to="wandb",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))

def main(args):
    print("Loading the model")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    print("Loading the tokenzizer")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/santacoder", use_auth_token=True)
    print("Loading the train and eval datasets")
    train_set, eval_set = create_datasets(args, tokenizer)
    metric = evaluate.load("google_bleu")
    run_training(args, model, tokenizer, train_set, eval_set, metric)

if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
