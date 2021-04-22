import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder

def build_files(data_path, tokenized_data_path, num_pieces, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='UTF-8') as f:
        print('Reading Lines')
        lines = json.load(f)
        # 用 [SEP] 取代換行，段落之間使用 [SEP] 表示段落結束
        lines = [line.replace('\n', ' [SEP] ') for line in lines]
    all_len = len(lines)

    os.makedirs(tokenized_data_path, exist_ok=True)

    show = []
    for i in tqdm(range(num_pieces)):
        sublines = lines[all_len // num_pieces * i: all_len // num_pieces * (i + 1)]
        if i == num_pieces - 1:
            # 將最後一個 Example 放到最後一個 Piece
            sublines.extend(lines[all_len // num_pieces * (i + 1):])
        # 只留下長度超過 min_length 的句子
        sublines = [
            full_tokenizer.tokenize(line) for line in sublines if len(line) > min_length]
        show.append(random.choice(sublines))
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            # 使用 [MASK] 表示文章開頭
            full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))
            full_line.extend(subline)
            # 使用 [CLS] 表示文章結束
            full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))

        with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'w') as f:
            for id in full_line:
                f.write(str(id) + ' ')
    print('Tokenized Data Sample')
    print('\n'.join([' '.join(s) for s in show[:5]]))
    print('Building from Raw Data Done')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='設定要使用的顯卡，以逗號區隔')
    parser.add_argument(
        '--model_config', type=str, required=True, help='模型參數設定檔的路徑')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='選擇字典檔的路徑')
    parser.add_argument('--raw_data_path', type=str, required=True, help='訓練用語料庫的路徑')
    parser.add_argument(
        '--tokenized_data_path', default='data/tokenized/', type=str, required=False, help='語料庫 Tokenized 後的存放路徑')
    parser.add_argument('--raw', action='store_true', help='是否已做過 Tokenization')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='設定 Epochs')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='設定 Batch Size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='設定 Learning Rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='設定 Optimizer 的 Warmup Steps')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='Loss 紀錄的間隔，必須是 Gradient Accumulation 的整數倍')
    parser.add_argument('--stride', default=768, type=int, required=False, help='設定訓練語料庫的窗口大小')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度累積')
    parser.add_argument('--fp16', action='store_true', help='是否使用半精度浮點數')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False, help='將訓練語料庫分成多少份')
    parser.add_argument('--min_length', default=1, type=int, required=False, help='文章最短長度，若文章長度不足將被捨棄')
    parser.add_argument('--output_dir', type=str, required=True, help='模型輸出路徑')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型起始路徑')
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False, help='Tensorboard 輸出路徑')
    parser.add_argument('--segment', action='store_true', help='是否以詞為單位')
    parser.add_argument('--bpe_token', action='store_true', help='使用 Byte Pair Encoding')
    parser.add_argument('--encoder_json', default='tokenizations/encoder.json', type=str, help='encoder.json')
    parser.add_argument('--vocab_bpe', default='tokenizations/vocab.bpe', type=str, help='vocab.bpe')
    parser.add_argument('--timezone', default=8, type=int, help='手動指定時區，預設為 GMT+8')
    parser.add_argument('--epoch_save', default=1, type=int, help='每隔幾個 Epoch 就存一次權重')

    args = parser.parse_args()
    print(f'Arguments: {args.__repr__()}')

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    # 設定要使用的顯卡
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('Config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using Device: {device.upper()}')

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    # 不支援半精度浮點數的顯卡不要打開
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    tz = args.timezone
    get_time = lambda: datetime.utcnow() + timedelta(hours=tz)
    tb_writer = SummaryWriter(log_dir=args.writer_dir)
    assert log_step % gradient_accumulation == 0

    os.makedirs(output_dir, exist_ok=True)

    if raw:
        print('Building from Raw Data')
        build_files(
            data_path=raw_data_path,
            tokenized_data_path=tokenized_data_path,
            num_pieces=num_pieces,
            full_tokenizer=full_tokenizer,
            min_length=min_length
        )

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('Number of Parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('Calculating Total Steps')
    for i in tqdm(range(num_pieces)):
        with open(tokenized_data_path + f'tokenized_train_{i}.txt', 'r') as f:
            full_len += len([int(item) for item in f.read().strip().split()])
    total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
    print('Total Steps: {total_steps}')

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True

    print('Training Begin')
    overall_step = 0
    running_loss = 0

    for epoch in range(epochs):
        now = get_time()
        print(f'Epoch {epoch + 1} - Time: {now}')
        x = np.linspace(0, num_pieces - 1, num_pieces, dtype=np.int32)
        random.shuffle(x)
        piece_num = 0
        for i in x:
            with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(i), 'r') as f:
                line = f.read().strip()
            tokens = line.split()
            tokens = [int(token) for token in tokens]
            start_point = 0
            samples = []
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride
            if start_point < len(tokens):
                samples.append(tokens[len(tokens)-n_ctx:])
            random.shuffle(samples)
            # 捨棄最後一個不足一個完整 Batch 的 Step
            _steps = len(samples) // batch_size
            for step in range(_steps):
                # prepare data
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                batch_inputs = torch.tensor(batch_inputs).long().to(device)

                # forward pass
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                loss, logits = outputs[:2]

                # get loss
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                # loss backward
                if fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # optimizer step
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % log_step == 0:
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                    ts = datetime.utcnow() + timedelta(hours=8)
                    ts = ts.strftime('%H:%M:%S')
                    display_loss = running_loss * gradient_accumulation / (log_step / gradient_accumulation)
                    print(
                        f'Time {ts} - '
                        f'Epoch {epoch + 1:{slen(epochs)}d}/{epochs} - '
                        f'Step {step + 1:{slen(_steps)}d}/{_steps} - '
                        f'Piece {piece_num + 1:{slen(num_pieces)}d}/{num_pieces} - '
                        f'Loss {display_loss:.4f}'
                    )
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        if (epoch + 1) % args.epoch_save == 0:
            print(f'Saving Model of Epoch {epoch + 1}')
            model_output_dir = os.path.join(output_dir, f'model_epoch{epoch + 1}')
            os.makedirs(model_output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_output_dir)

        then = get_time()
        print(f'Epoch {epoch + 1} Finished - Time: {then}')
        delta = (then - now).total_seconds()
        mm, ss = delta // 60, delta % 60
        hh, mm = mm // 60, mm % 60
        print(f'Time Cost of the Epoch {epoch + 1} - {hh:.0f}:{mm:.0f}:{ss:.2f}')

    print('Training Done')
    model_output_dir = os.path.join(output_dir, 'final_model')
    os.makedirs(model_output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(model_output_dir)

def slen(n):
    return len(str(n))

if __name__ == '__main__':
    main()
