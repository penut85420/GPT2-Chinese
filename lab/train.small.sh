export CFG_PATH="./lab/model_config_small.json"
export VOCAB_PATH="./lab/bert-base-chinese.vocab.txt"
export DATA_PATH="./lab/dataset/corpus.small.json"
export OUTPUT_PATH='./lab/models/small'

python train.py \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/small/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 200 \
  --log_step 10 \
  --stride 512 \
  --output_dir $OUTPUT_PATH \
  --device 0 \
  --num_pieces 100 \
  --batch_size 16 \
  --raw
