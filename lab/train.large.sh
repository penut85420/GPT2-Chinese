#!/bin/bash
export CODE="200k"
export PRETRAINED_PATH="../models/CKIP-GPT2"
export CFG_PATH="$PRETRAINED_PATH/config.json"
export VOCAB_PATH="$PRETRAINED_PATH/vocab.txt"
export DATA_PATH="./lab/dataset/corpus.$CODE.json"
export OUTPUT_PATH="./lab/models/$CODE"

python train.py \
  --pretrained_model $PRETRAINED_PATH \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/large/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 5 \
  --log_step 20 \
  --stride 768 \
  --output_dir $OUTPUT_PATH \
  --device 0,1 \
  --num_pieces 10 \
  --batch_size 3 \
  --epoch_save 1 \
  --raw
