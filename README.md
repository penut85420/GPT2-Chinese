# GPT2-Chinese

## Introduction
+ 本專案 Fork 自 [Morizeyao/GPT2-Chinese](https://git.io/JUYOs)，主要根據個人的使用習慣與用途修改了部份程式碼
+ [原始 README](README_ORG.md)

## Usage
+ 使用前請先根據 CUDA 的版本修改 `install.sh` 並執行此腳本來安裝相關套件

## Note
+ 本章節紀錄一些檔案與參數相關的說明
+ 設定檔，例如：`config/model_config_small.json`
    + 可以參考[這份程式碼](https://tinyurl.com/yexdykko)的說明
    + `initializer_range` 參數初始話的浮動範圍
    + `layer_norm_epsilon` The epsilon to use in the layer normalization layers.
    + `n_positions` 模型單次可以輸入的最大文章長度
    + `n_ctx` 影響模型可以輸入的文長，通常會與 `n_positions` 相同
    + `n_embd` 模型使用的 Embedding Size
    + `n_head` Attention Heads 的數量
    + `n_layer` Attention Layers 的數量
    + `vocab_size` 字典的大小，數量務必與字典檔相同
+ 字典檔，例如：`cache/vocab.txt`
    + 至少需要包含 `[SEP]`, `[CLS]`, `[MASK]`, `[PAD]`, `[UNK]` 這五個特殊 Token
    + 自行製作的字典檔請記得將字詞進行排序
+ 訓練參數，用於 `train.py`
    + `--device` 設定需要使用的顯卡 ID，多個顯卡以半形逗號區隔，如果只有單張顯卡就設定成 `--device 0`
    + `--num_pieces` 會把 Corpus 切割成指定數量，不太影響效能跟訓練時間的參數
    + `--min_length` 會過濾掉文本過短的文章
    + `--epoch_save` 指定每訓練多少個 Epoch 就存一次模型參數，最後一個 Epoch 必定會存一次

## FAQ
+ 發生以下錯誤：
    + `CUDA error: CUBLAS_STATUS_NOT_INITIALIZED when calling cublasCreate(handle)`
    + `CUDA error: device-side assert triggered`
    + 可能的原因：設定檔內的 vocab_size 與字典檔的大小並不一致
