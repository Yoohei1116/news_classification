import config
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tfidf_experiment import clean_text, get_tfidf, calc_f1_score_with_logistic_regression
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier

def calc_f1_score_with_random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=config.random_seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1スコア: {f1:.4f}")

if __name__ == "__main__":
    data = pd.read_csv("../news_data.csv").sample(config.sample_num, random_state=config.random_seed)
    data["sentence"] = data["sentence"].apply(clean_text) # テキスト前処理(不要な空白の削除)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=config.random_seed)
    
    # 特徴量1:トークナイザーをmecabに指定したTF-IDF
    X_train_tfidf, X_test_tfidf = get_tfidf(train_sentence=train_data["sentence"], 
                                            test_sentence=test_data["sentence"], 
                                            tokenizer_type="mecab")

    # 特徴量2:センテンス長
    X_train_sentence_length = train_data["sentence"].apply(lambda x: len(x)).values.reshape(-1, 1)
    X_test_sentence_length = test_data["sentence"].apply(lambda x: len(x)).values.reshape(-1, 1)

    y_train = train_data["label"]
    y_test = test_data["label"]

    # TF-IDFのみのF1スコア
    calc_f1_score_with_logistic_regression(X_train_tfidf, X_test_tfidf, y_train, y_test)
    calc_f1_score_with_random_forest(X_train_tfidf, X_test_tfidf, y_train, y_test)

    # TF-IDF + 文長の特徴量を組み合わせたF1スコア
    X_train_combined = hstack([X_train_tfidf, X_train_sentence_length])
    X_test_combined = hstack([X_test_tfidf, X_test_sentence_length])
    calc_f1_score_with_logistic_regression(X_train_combined, X_test_combined, y_train, y_test)
    calc_f1_score_with_random_forest(X_train_combined, X_test_combined, y_train, y_test)
    
    """出力
    F1スコア: 0.7927 (ロジスティック回帰 / TF-IDF)
    F1スコア: 0.7084 (ランダムフォレスト / TF-IDF)
    F1スコア: 0.6714 (ロジスティック回帰 / TF-IDF + センテンス長)
    F1スコア: 0.7248 (ランダムフォレスト / TF-IDF + センテンス長)
    """
    
