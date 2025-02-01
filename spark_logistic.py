import re
import time
import MeCab
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml import Pipeline
from pyspark.ml.feature import NGram, CountVectorizer, IDF, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# トークナイザーはMeCabを使用する
def mecab_tokenize(text):
    if text is None:
        return []
    mecab = MeCab.Tagger("-Owakati")
    parsed_text = mecab.parse(text)
    return parsed_text.strip().split() if parsed_text else []

# テキストの前処理関数
def clean_text(text):
    text = re.sub(r"\t", " ", text)  # タブを空白に置換
    text = re.sub(r"\n", " ", text)  # 改行を空白に置換
    text = re.sub(r"\s+", " ", text) # 連続する空白を一つに統一
    return text.strip()

if __name__ == "__main__":
    conf = SparkConf().set("spark.driver.host", "127.0.0.1").set("spark.driver.bindAddress", "127.0.0.1")
    spark = SparkSession.builder.appName("TextClassification").config(conf=conf).getOrCreate()

    mecab_udf = udf(mecab_tokenize, ArrayType(StringType()))
    df = spark.read.csv("news_data.csv", header=True, inferSchema=True)
    df = df.withColumn("sentence", udf(clean_text, StringType())(col("sentence")))
    df = df.withColumn("words", mecab_udf(col("sentence")))

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=1)
    print(f"Train data count: {train_df.count()}, Test data count: {test_df.count()}")

    # 特徴量は1-gram, 2-gramを用いたTF-IDF（トークナイザーはMeCab）
    ngram_1 = NGram(n=1, inputCol="words", outputCol="unigrams")
    ngram_2 = NGram(n=2, inputCol="words", outputCol="bigrams")
    cv_unigram = CountVectorizer(inputCol="unigrams", outputCol="cv_unigram_features", vocabSize=10000)
    cv_bigram = CountVectorizer(inputCol="bigrams", outputCol="cv_bigram_features", vocabSize=10000)
    vector_assembler = VectorAssembler(inputCols=["cv_unigram_features", "cv_bigram_features"], outputCol="features")
    idf = IDF(inputCol="features", outputCol="idf_features", minDocFreq=1)
    lr = LogisticRegression(featuresCol="idf_features", labelCol="label", regParam=0.01, maxIter=500) # 正則化パラメータのチューニングは小規模データで実行済み

    pipeline = Pipeline(stages=[ngram_1, ngram_2, cv_unigram, cv_bigram, vector_assembler, idf, lr])

    # 学習時間計測
    start_train = time.time()
    model = pipeline.fit(train_df)
    end_train = time.time()
    train_time = end_train - start_train
    print(f"学習時間: {train_time:.2f} 秒")

    # 推論時間計測
    start_pred = time.time()
    predictions = model.transform(test_df)
    end_pred = time.time()
    pred_time = end_pred - start_pred
    print(f"推論時間: {pred_time:.2f} 秒")

    # F1スコアで評価
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print(f"F1スコア: {f1_score:.4f}")

    spark.stop()
    
    """ 出力
    Train data count: 5898, Test data count: 1478
    学習時間: 81.96 秒
    推論時間: 0.27 秒
    F1スコア: 0.8457
    """
