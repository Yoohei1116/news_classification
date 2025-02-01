import config
import pandas as pd
import MeCab
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tfidf_experiment import clean_text

def extract_features(data):
    mecab = MeCab.Tagger()
    
    features = pd.DataFrame()
    features["word_count"] = data["sentence"].apply(lambda x: len(x.split()))
    features["sentence_length"] = data["sentence"].apply(lambda x: len(x))
    
    # 品詞の割合を計算
    def calculate_pos_ratios(sentence):
        parsed = mecab.parse(sentence)
        lines = parsed.splitlines()[:-1]  # 形態素解析結果を行ごとに分割
        
        noun_count = 0  # 名詞の数
        verb_count = 0  # 動詞の数
        proper_noun_count = 0  # 固有名詞の数
        total_count = len(lines)  # 形態素の総数
        
        for line in lines:
            if line == "EOS" or line == "":
                continue
            
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            
            pos_tags = parts[1].split(",")
            
            if "名詞" in pos_tags:
                noun_count += 1
            if "動詞" in pos_tags:
                verb_count += 1
            if "固有名詞" in pos_tags:
                proper_noun_count += 1
        
        return pd.Series({
            "noun_ratio": noun_count / total_count if total_count > 0 else 0,
            "verb_ratio": verb_count / total_count if total_count > 0 else 0,
            "proper_noun_ratio": proper_noun_count / total_count if total_count > 0 else 0
        })

    pos_ratios = data["sentence"].apply(calculate_pos_ratios)
    features = pd.concat([features, pos_ratios], axis=1)
    return features

if __name__ == "__main__":
    data = pd.read_csv("../news_data.csv").sample(config.sample_num, random_state=config.random_seed)
    data["sentence"] = data["sentence"].apply(clean_text)  # テキスト前処理(不要な空白の削除)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=config.random_seed)
    X_train = extract_features(train_data)
    X_test = extract_features(test_data)
    y_train = train_data["label"]
    y_test = test_data["label"]

    model = RandomForestClassifier(random_state=config.random_seed)
    model.fit(X_train, y_train)

    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=X_train.columns,
                                       columns=["importance"]).sort_values("importance", ascending=False)
    
    print("Feature Importances:")
    print(feature_importances)
    
    """出力
    Feature Importances:
                    importance
    sentence_length      0.316953
    noun_ratio           0.241879
    proper_noun_ratio    0.192443
    verb_ratio           0.174053
    word_count           0.074672
    """
