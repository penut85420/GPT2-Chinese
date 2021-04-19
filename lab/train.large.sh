export CFG_PATH="./lab/model_config_large.json"
export VOCAB_PATH="./lab/bert-base-chinese.vocab.txt"
export DATA_PATH="./lab/dataset/corpus.large.json"
export OUTPUT_PATH='./lab/models/large'

python train.py \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/large/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 5 \
  --log_step 100 \
  --stride 512 \
  --output_dir $OUTPUT_PATH \
  --device 0 \
  --num_pieces 128 \
  --batch_size 2 \
  --raw
