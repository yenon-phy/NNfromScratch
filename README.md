## Neural Networks from Scratch

### 概要
ニューラルネットワークの包括的理解と今後の応用に向けて, スクラッチでの実装を行った.

作成したモデルのアルゴリズム部分は,すべてnumpyのみ.

モデルテストには,赤ワインデータを使用した.

[ワインデータ参照先](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/)

### 実装内容

- シンプルな三層構造（入力層・中間層・出力層）
- 前段: シグモイドレイヤ, 後段: ソフトマックスレイヤ
- 損失関数: 交差エントロピー誤差
- L2およびL1正則化項
- 最適化手法: Adam
- 学習率の減衰定数

### ディレクトリ構成
アルゴリズム部分は、それぞれ`notebooks`フォルダ直下の

`functions.ipynb`, `layers.ipynb`, `neuralnet.ipynb`

に記述されており, `nbconvert.ipynb`で上記ノートブックファイルと同名のpythonスクリプトを`src`配下に出力する.

モデルテストは`winequality.ipynb`で実施.