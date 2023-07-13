# choso_module

## finetune.py:クラス分類　ファインチューニングファイル

2~5分割交差検証もできます．
パラメータなどはコードの中で設定します．

## classification.py:クラス分類実行ファイル

使い方：

python classification.py "model_name" "model_path" "number of classes" "binary" "image_dir_path" "save_file_path"

model_name efficientnet-b5など

model_path 訓練済みモデルを保存したパス

binary 対象画像がグレースケール化否か　グレースケールならTrue カラーならFalse

## repeat_finetune.py:finetuneを指定回数repeat
試行回数毎に正解率が変動する場合，真の正解率がわかりません．そこで，何回も実行して，正解率の確かさを検証します．

## ai.py：AI関連の部品
## cv.py：画像処理関連の部品
