#!/bin/bash
export CODE="100k"
export PRETRAINED_PATH="../models/CKIP-GPT2"
export CFG_PATH="$PRETRAINED_PATH/config.json"
export VOCAB_PATH="$PRETRAINED_PATH/vocab.txt"
export DATA_PATH="./lab/dataset/corpus.$CODE.json"
export OUTPUT_PATH="./lab/models/$CODE/3"

python train.py \
  --pretrained_model $PRETRAINED_PATH \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/large/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 1 \
  --log_step 1 \
  --stride 768 \
  --output_dir $OUTPUT_PATH \
  --device 0,1 \
  --num_pieces 128 \
  --batch_size 2 \
  --epoch_save 100 \
  --raw
