# choso_module
プログラム部品
classification.py:クラス分類実行ファイル
使い方：
python classification.py "model_name" "model_path" "number of classes" "binary" "image_dir_path" "save_file_path"
model_name efficientnet-b5など
model_path 訓練済みモデルを保存したパス
binary 対象画像がグレースケール化否か　グレースケールならTrue カラーならFalse

ai.py：AI関連の部品
cv.py：画像処理関連の部品
