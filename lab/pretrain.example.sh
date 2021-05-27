#!/bin/bash

# Example of train from pretrain model

export PRETRAINED_PATH="/path/to/pretrained/model/dir"
export CFG_PATH="$PRETRAINED_PATH/config.json"
export VOCAB_PATH="$PRETRAINED_PATH/vocab.txt"
export DATA_PATH="/path/to/finetune/dataset.json"
export OUTPUT_PATH="/path/to/output/dir"

python train.py \
  --pretrained_model $PRETRAINED_PATH \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/large/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 20 \
  --log_step 10 \
  --stride 768 \
  --output_dir $OUTPUT_PATH \
  --num_pieces 128 \
  --batch_size 2 \
  --epoch_save 5 \
  --raw
