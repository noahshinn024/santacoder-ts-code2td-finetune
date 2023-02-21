python train.py \
        --model_path="noahshinn024/santacoder-ts" \
        --dataset_name="noahshinn024/ts-code2td" \
        --seq_length 1024 \
        --learning_rate 2e-5 \
        --batch_size 4 \
        --no_fp16 \
        --num_epochs 10
