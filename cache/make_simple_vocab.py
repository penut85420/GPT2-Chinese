import argparse
import penut.io as pio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inn', '-i')
    parser.add_argument('--out', '-o')
    args = parser.parse_args()

    ds = pio.load(args.inn)

    vocab = set()
    for sent in ds:
        for ch in sent:
            vocab.add(ch)

    vocab = '\n'.join(sorted(list(vocab))).strip()
    pre = '\n'.join(['[SEP]', '[CLS]', '[MASK]', '[PAD]', '[UNK]'])
    open(args.out, 'w', encoding='UTF-8').write(f'{pre}\n{vocab}')

if __name__ == "__main__":
    main()
