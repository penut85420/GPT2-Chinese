export CFG_PATH="/path/to/config.json"
export VOCAB_PATH="/path/to/vocab.txt"
export DATA_PATH="/path/to/data.json"
export OUTPUT_PATH='/path/to/model/dir'

python train.py \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 30 \
  --log_step 10 \
  --stride 512 \
  --output_dir $OUTPUT_PATH \
  --device 0 \
  --num_pieces 128 \
  --raw
