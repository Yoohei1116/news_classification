random_seed = 1
sample_num = 1000  # サンプルデータ数
cv = 5 # チューニングの交差検証の分割数

""" ロジスティック回帰 """
lr_max_iter = 5000 # 最大反復数
lr_param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]} # 探索値