export CFG_PATH="./lab/config.small.json"
export VOCAB_PATH="../models/CKIP-GPT2/vocab.txt"
export DATA_PATH="./lab/dataset/corpus.50k.json"
export OUTPUT_PATH='./lab/models/small'

python train.py \
  --model_config $CFG_PATH \
  --tokenized_data_path ./tmp/small/tokenized/ \
  --tokenizer_path $VOCAB_PATH \
  --raw_data_path $DATA_PATH \
  --epochs 1 \
  --log_step 10 \
  --stride 768 \
  --output_dir $OUTPUT_PATH \
  --device 0,1 \
  --num_pieces 10 \
  --batch_size 16 \
  --epoch_save 100 \
  --raw
