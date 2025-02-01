import config
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from tfidf_experiment import clean_text, get_tfidf, calc_f1_score_with_logistic_regression
from sklearn.linear_model import LogisticRegression

def calc_f1_score_with_logistic_regression_tuning(param_grid, X_train, X_test, y_train, y_test):
    """ロジスティック回帰でパラメーターチューニング"""
    grid_search = GridSearchCV(LogisticRegression(max_iter=config.lr_max_iter, random_state=config.random_seed), 
                               param_grid, cv=config.cv, scoring="f1_weighted", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"最適なパラメータ: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1スコア: {f1:.4f}")

if __name__ == "__main__":
    data = pd.read_csv("../news_data.csv").sample(config.sample_num, random_state=config.random_seed)
    data["sentence"] = data["sentence"].apply(clean_text)  # テキスト前処理(不要な空白の削除)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=config.random_seed)
    X_train_tfidf, X_test_tfidf = get_tfidf(train_sentence=train_data["sentence"], 
                                            test_sentence=test_data["sentence"], 
                                            tokenizer_type="mecab")
    y_train = train_data["label"]
    y_test = test_data["label"]

    print("ロジスティック回帰(デフォルトパラメータ(C=1.0)でのスコア)")
    calc_f1_score_with_logistic_regression(X_train_tfidf, X_test_tfidf, y_train, y_test)

    print("\nロジスティック回帰(ハイパーパラメータのチューニング後のスコア)")
    calc_f1_score_with_logistic_regression_tuning(config.lr_param_grid, X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    """ 出力
    ロジスティック回帰(デフォルトパラメータ(C=1.0)でのスコア)
    F1スコア: 0.7927
    
    ロジスティック回帰(ハイパーパラメータのチューニング後のスコア)
    最適なパラメータ: {'C': 100}
    F1スコア: 0.7978
    """
