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

TASK_PREFIX = "// Generate type definitions for the following sections of code"
FEW_SHOT_EXAMPLES = """Code:
function combineFuncs(a: TestType, b: TestType): TestType { return (elem: AnyNode) => a(elem) || b(elem); }

Type Declaration:
type TestType = (elem: AnyNode) => boolean;

Code:
function createMap(schema: Schema, obj: unknown, ctx: CreateNodeContext) { const { keepUndefined, replacer } = ctx const map = new YAMLMap(schema) const add = (key: unknown, value: unknown) => { if (typeof replacer === 'function') value = replacer.call(obj, key, value) else if (Array.isArray(replacer) && !replacer.includes(key)) return if (value !== undefined || keepUndefined) map.items.push(createPair(key, value, ctx)) } if (obj instanceof Map) { for (const [key, value] of obj) add(key, value) } else if (obj && typeof obj === 'object') { for (const key of Object.keys(obj)) add(key, (obj as any)[key]) } if (typeof schema.sortMapEntries === 'function') { map.items.sort(schema.sortMapEntries) } return map }

Type Declaration:
class Schema { compat: Array<CollectionTag | ScalarTag> | null knownTags: Record<string, CollectionTag | ScalarTag> merge: boolean name: string sortMapEntries: ((a: Pair, b: Pair) => number) | null tags: Array<CollectionTag | ScalarTag> toStringOptions: Readonly<ToStringOptions> | null; // Used by createNode() and composeScalar() declare readonly [MAP]: CollectionTag; declare readonly [SCALAR]: ScalarTag; declare readonly [SEQ]: CollectionTag constructor({ compat, customTags, merge, resolveKnownTags, schema, sortMapEntries, toStringDefaults }: SchemaOptions) { this.compat = Array.isArray(compat) ? getTags(compat, 'compat') : compat ? getTags(null, compat) : null this.merge = !!merge this.name = (typeof schema === 'string' && schema) || 'core' this.knownTags = resolveKnownTags ? coreKnownTags : {} this.tags = getTags(customTags, this.name) this.toStringOptions = toStringDefaults ?? null Object.defineProperty(this, MAP, { value: map }) Object.defineProperty(this, SCALAR, { value: string }) Object.defineProperty(this, SEQ, { value: seq }) // Used by createMap() this.sortMapEntries = typeof sortMapEntries === 'function' ? sortMapEntries : sortMapEntries === true ? sortMapEntriesByKey : null } clone(): Schema { const copy: Schema = Object.create( Schema.prototype, Object.getOwnPropertyDescriptors(this) ) copy.tags = this.tags.slice() return copy } }"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="noahshinn024/santacoder-ts")
    parser.add_argument("--dataset_name", type=str, default="noahshinn024/ts-code2td")

    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    return parser.parse_args()


def create_datasets(args, tokenizer):
    def preprocess_function(examples):
        if "code" not in examples or "declarations" not in examples:
            raise ValueError("The dataset must contain 'code' and 'declarations' columns.")
        
        inputs = [f'{TASK_PREFIX}\n\n{FEW_SHOT_EXAMPLES}\n\nCode:\n{code}\n\nType Declaration:\n' for code in examples["code"]]
        inputs_tokenized = tokenizer(inputs, padding="max_length", max_length=args.seq_length, truncation=True)

        labels_tokenized = tokenizer(text_target=examples["declarations"], padding="max_length", max_length=args.seq_length, truncation=True)
        
        return {
            "input_ids": inputs_tokenized["input_ids"],
            "attention_mask": inputs_tokenized["attention_mask"],
            "labels": labels_tokenized["input_ids"]
        }
    ds = load_dataset(args.dataset_name)
    tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)
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
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=500,
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        predict_with_generate=True,
        fp16=args.no_fp16,
        run_name=f'code2td-{args.model_path.split("/")[-1]}',
        report_to=["wandb"],
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
    tokenizer.pad_token = tokenizer.eos_token
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
