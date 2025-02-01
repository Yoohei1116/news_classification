import config 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import MeCab

random_seed = 1

def clean_text(text):
    text = re.sub(r"\t", " ", text)  # タブを空白に置換
    text = re.sub(r"\n", " ", text)  # 改行を空白に置換
    text = re.sub(r"\s+", " ", text) # 連続する空白を一つに統一
    return text.strip()

def mecab_tokenize(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip().split()

def get_tfidf(train_sentence, test_sentence, tokenizer_type="default"):
    """TF-IDFのベクトル化（デフォルトかMeCabかを選択可能）"""
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    if tokenizer_type == "mecab":
        tfidf_vectorizer = TfidfVectorizer(tokenizer=mecab_tokenize, ngram_range=(1, 2))
    train_tfidf = tfidf_vectorizer.fit_transform(train_sentence)
    test_tfidf = tfidf_vectorizer.transform(test_sentence)

    return train_tfidf, test_tfidf

def calc_f1_score_with_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=config.lr_max_iter)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"F1スコア: {f1:.4f}")

def display_tokenization_results(sample_sentences):
    tokenizer_default = TfidfVectorizer().build_tokenizer()
    tokenizer_mecab = mecab_tokenize

    for sentence in sample_sentences:
        tokens_default = tokenizer_default(sentence)
        tokens_mecab = tokenizer_mecab(sentence)
        print(f"元の文: {sentence}")
        print(f"Default: {tokens_default}")
        print(f"Mecab: {tokens_mecab}\n")

if __name__ == "__main__":
    sample_num = config.sample_num  # ランダムに1000件抽出したデータを準備
    data = pd.read_csv("../news_data.csv").sample(sample_num, random_state=random_seed)
    data["sentence"] = data["sentence"].apply(clean_text)  # テキスト前処理(不要な空白の削除)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed)

    # デフォルトのTF-IDF
    X_train_default, X_test_default = get_tfidf(train_sentence=train_data["sentence"], 
                                                test_sentence=test_data["sentence"])

    # MeCabを使用したTF-IDF
    X_train_mecab, X_test_mecab = get_tfidf(train_sentence=train_data["sentence"], 
                                            test_sentence=test_data["sentence"], 
                                            tokenizer_type="mecab")

    y_train = train_data["label"]
    y_test = test_data["label"]

    # ロジスティック回帰を適用
    calc_f1_score_with_logistic_regression(X_train_default, X_test_default, y_train, y_test)  # デフォルトのトークナイザー
    calc_f1_score_with_logistic_regression(X_train_mecab, X_test_mecab, y_train, y_test)  # MeCabトークナイザー
    
    # トークナイザーの違い(デフォルト or Mecab)を確認
    print("\n----- トークナイザーごとの文章分割の例 -----\n")
    display_tokenization_results(sample_sentences=data["sentence"][:3])
    
    
    """ 出力
    
    F1スコア: 0.5953  <- デフォルトのトークナイザーを使った場合
    F1スコア: 0.7927  <- MeCabを使った場合
    
    ----- トークナイザーごとの文章分割の例 -----
    
    元の文: 永遠の恋愛テーマにハマった杉崎アナ、涌井投手が破局
    Default: ['永遠の恋愛テーマにハマった杉崎アナ', '涌井投手が破局']
    Mecab: ['永遠', 'の', '恋愛', 'テーマ', 'に', 'ハマっ', 'た', '杉崎', 'アナ', '、', '涌井', '投手', 'が', '破局']       

    元の文: 生活保護受給をめぐり物議を醸す河本親子に法的措置の可能性を示唆
    Default: ['生活保護受給をめぐり物議を醸す河本親子に法的措置の可能性を示唆']
    Mecab: ['生活', '保護', '受給', 'を', 'めぐり', '物議', 'を', '醸す', '河本', '親子', 'に', '法的', '措置', 'の', '可能', '性', 'を', '示唆']

    元の文: 愛される？ 愛する？ 幸せなのはどっち!?
    Default: ['愛される', '愛する', '幸せなのはどっち']
    Mecab: ['愛さ', 'れる', '？', '愛する', '？', '幸せ', 'な', 'の', 'は', 'どっち', '!?']
    
    """