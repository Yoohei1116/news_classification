# 概要
本プロジェクトの目的は、Livedoorニュース記事をカテゴリに応じて自動分類する機械学習モデルを構築することです。小規模データに対する実験から手法を検討し、最終的には深層学習モデル(BERT)に匹敵する`F1=0.85`のロジスティック回帰を用いた分類モデルを作成しました。

プロジェクトの流れは下記の通りです：
1. 小規模データによる実験で、モデル・特徴量・ハイパーパラメータを決定
2. 実験結果をもとにSparkで機械学習モデルを作成
3. 深層学習(BERT)と精度と計算コストを比較



## 環境
- **環境**: Windows11
- **言語**: Python 3.10.5
- **主要ライブラリ**: `scikit-learn`, `MeCab`, `PySpark`, `Transformers(BERT)` 

## フォルダ構成
```
.
├── experiment/ ................. 小規模データでの実験
│   ├── results ................. 結果のグラフを格納したフォルダ
│   ├── config.py ............... 実験の設定ファイル
│   ├── feature_experiment.py ... TF-IDF以外の特徴量を評価
│   ├── model_experiment.py  .... ロジスティック回帰とランダムフォレストを比較
│   ├── tfidf_experiment.py ..... TF-IDFのトークナイザーの影響を確認
│   └── tuning_experiment.py .... ロジスティック回帰の正則化パラメータのチューニング     
├── README.md
├── news_data.csv ............... ニュースデータ
├── bert.py ..................... BERTによるカテゴリ予測
└── spark_logistc.py ............ 実験結果をもとに作成したメインモデル
```

## データ (`news_data.csv`)
データは[こちら](https://www.rondhuit.com/download.html)より引用したLivedoorニュース記事(7376件)を使用しました。ここではテキスト (`sentence`) とカテゴリラベル (`label`)のみを抽出したデータを用いました。カテゴリラベルは0~8の9つあります。 

| label | sentence |
|-------|--------------------------------------------------------------|
| 1     | やっぱいいわ！レッツノート！  SX2の中身はコレだ！最強モバイルノートPC開封フォトレポ |
| 2     | 元日本代表・本田泰人、テレビ番組で衝撃の告白 |
| 7     | ソニーモバイル、Xperia GXおよびXperia SXを東京・名古屋・大阪にて6月20日から先行展示開始 |
...

ラベルごとの項目数です。概ね同じ項目数になっています。
| Label | 項目数 |
|------|------|
| 0    | 101  |
| 1    | 113  |
| 2    | 129  |
| 3    | 133  |
| 4    | 82   |
| 5    | 107  |
| 6    | 115  |
| 7    | 117  |
| 8    | 103  |


# 実施内容
前述した流れに沿って処理の概要を示します。以下は共通の処理項目です：
- 訓練データと検証データは8:2としています。
- 評価指標はF1スコア(カテゴリのサンプル数の重み付き平均)を用いています。
- 今回のデータはHTMLタグなどの不要な記述が含まれていないため、テキストの前処理は空白処理だけを行っています。

## 1. 小規模データによる実験
1000件の小規模データを抽出して、そのデータを用いて構築するモデルの方針を決めました。ライブラリは`sckit-learn`を用いました。

### 1.1 TF-IDFを用いた単純なモデルでの精度確認(`tfidf_experiment.py`)
まずはシンプルにTF-IDFとロジスティック回帰を用いて精度を確認しました。TF-IDFは次元を考慮して2-gramまでを適用し、ロジスティック回帰はデフォルトの設定(正則化パラメータ`C=1.0`)としました。トークナイザーをデフォルトのものから`MeCab`に変更することでスコアを大幅に改善できることが分かりました(`F1=0.60`から`F1=0.79`に向上)。実際に分割の様子を見てみると、デフォルトでは空白や記号を含まない文章はうまく分割できていませんでした。これがTF-IDFの質を大きく左右したと考えられます。

分割の例：
| 分類 | トークン化結果 |
|------|----------------------------------------------------------------|
| **元の文** | 生活保護受給をめぐり物議を醸す河本親子に法的措置の可能性を示唆 |
| **デフォルト** | ['生活保護受給をめぐり物議を醸す河本親子に法的措置の可能性を示唆'] |
| **MeCab** | ['生活', '保護', '受給', 'を', 'めぐり', '物議', 'を', '醸す', '河本', '親子', 'に', '法的', '措置', 'の', '可能', '性', 'を', '示唆'] |

<img src="experiment/results/トークナイザーを変えたときのF1スコア.png">


### 1.2 ランダムフォレストを用いたTF-IDF以外の特徴量の評価(`feature_experiment.py`)
次にTF-IDF以外に重要な特徴量を探しました。カテゴリと関係しそうな特徴量の候補として文章の長さ、品詞(名詞・固有名詞・動詞)の割合、単語数を考えました。これらの重要度をランダムフォレストを用いて計算したところ、文章の長さが最も重要であると分かりました。
| 特徴量（日本語） | 変数名 | 重要度 |
|-----------------|----------------|--------|
| 文章の長さ | sentence_length | 0.316953 |
| 名詞の割合 | noun_ratio | 0.241879 |
| 固有名詞の割合 | proper_noun_ratio | 0.192443 |
| 動詞の割合 | verb_ratio | 0.174053 |
| 単語数 | word_count | 0.074672 |

<img src="experiment/results/ランダムフォレストによる特徴量の重要度.png">


### 1.3 モデル(線形・非線形)と特徴量の選定(`model_experiment.py`)
線形モデルとしてロジスティック回帰、非線形モデルとしてランダムフォレストを用い最適なモデルを探しました。特徴量はこれまでの結果を踏まえ、`MeCab`を使用したTF-IDFだけの場合と、文章の長さも加えた場合の計4種類としました。最も優れたスコアを出したのは、TF-IDFのみを特徴量としたロジスティック回帰モデルとなりました。センテンス長とカテゴリは線形関係にはなく、線形モデルであるロジスティック回帰にはノイズになってしまったと考えられます。

| モデル | 特徴量 | F1スコア |
|--------|----------------------|--------|
| ロジスティック回帰 | TF-IDF | 0.7927 |
| ランダムフォレスト | TF-IDF | 0.7084 |
| ロジスティック回帰 | TF-IDF + センテンス長 | 0.6714 |
| ランダムフォレスト | TF-IDF + センテンス長 | 0.7248 |

<img src="experiment/results/モデルと特徴量を変えたときのF1スコア.png">

### 1.4 ハイパーパラメータのチューニング(`tuning_experiment.py`)
最後にロジスティック回帰モデルの正則化パラメータをグリッドサーチと効果検証で最適化しました。結果としてはC=100が最適となり、F1スコアとしては0.79(C=1.0)から0.80(C=100)程度の僅かな改善となりました。

<img src="experiment/results/ロジスティック回帰のチューニング前後のF1スコア.png">

### 1.5 小規模データ実験のまとめ
- `MeCab`を用いたTF-IDFを特徴量としたロジスティック回帰が最も優れたF1スコア
- 正則化パラメータは`C=100`が最適


## 2. Spark を用いたメインモデルの作成(`spark_logistic.py`)
小規模データでの検証をもとに、Sparkで`MeCab`を用いたTF-IDFを特徴量としたロジスティック回帰モデル(`C=100`)を作成しました。今回のデータセットは数百MB程度なので分散処理の恩恵は感じられませんでした。評価指標としてF1スコア以外にも学習時間と推論時間を測定しました。`F1=0.8457`と非常に高いスコアを得ることができました。

| 項目 | 値 |
|------|------|
| 訓練データ数 | 5898 |
| 検証データ数 | 1478 |
| 学習時間 | 81.96 秒 |
| 推論時間 | 0.27 秒 |
| F1スコア | 0.8457 |


## 3. BERT ファインチューニングによる比較(`bert.py`)
最後に比較対象として、日本語対応の事前学習済みのBERTのファインチューニングモデルを検証しました。精度は`F1=`とかなり良好ですが、私の実行環境にGPUがないということを踏まえても計算コストが非常に大きいです。

| 項目 | 値 |
|------|------|
| 訓練データ数 | 5898 |
| 検証データ数 | 1478 |
| 学習時間 |  秒 |
| 推論時間 |  秒 |
| F1スコア |  |

## 4. 結果の比較と考察
- TF-IDF + ロジスティック回帰 (C=100)モデルは、BERTと同程度の優れた分類精度(`F1~0.85`)
- 一方で学習や推論速度は TF-IDF + ロジスティック回帰 (C=100)モデルのほうが圧倒的に早く計算コストで有利
- 計算コストの低いTF-IDF + ロジスティック回帰 (C=100)モデルは、大規模データを扱う場合や定期的な調整にも対応しやすい

# 今後の展望
- 数十GB以上の大規模データの分散処理を通じた機械学習モデルの作成
- LLMのembeddingを活用した特徴量やストップワード・特殊記号の影響調査
- GPU環境でのBERTや他の深層学習モデルとの比較


