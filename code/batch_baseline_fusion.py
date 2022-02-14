import os
import pandas as pd
import sys

from baseline_fusion import *
from data_loader import ALL_TARGETS
from datetime import datetime
from glob import glob
from imblearn.over_sampling import RandomOverSampler

speakers = 'both'
erisk_codes_file = 'data/ERisk_coded_data_02Sep21.csv'
transcript_file = 'data/fmss_transcripts.csv'
acoustic_features_dirs = glob('data/output_base_line/type*-avg')
# models = ['lr', 'mlp', 'knn', 'rf', 'svr', 'gb']
models = ['svm']
task = 'regression'
# features_to_use = ['acoustic_full', 'acoustic', 'tfidf', 'glove', 'word2vec', 'acoustic tfidf', 'acoustic glove', 'acoustic word2vec']
features_to_use = ['acoustic', 'tfidf', 'glove', 'word2vec', 'acoustic tfidf', 'acoustic glove', 'acoustic word2vec']
features_to_use = ['acoustic']
timestamp = datetime.today().strftime('%Y%m%d') #-%H:%M:%S')

target_labels = ['warme5_cat']
if task == 'regression':
    target_labels = ['warme5', 'warmy5', 'disse5', 'dissy5']

df_results = pd.DataFrame()
output_file = 'data/fmss_full_data_batch.csv'

ros = RandomOverSampler()

for f in features_to_use:
    print('-- using features:', f)
    acoustic_features = []
    text_features = []
    embedding_model = None

    for a in acoustic_features_dirs:
        dl = DataLoader(transcript_file, speakers, pin_acoustics=a, pin_targets=erisk_codes_file)

        if 'glove' in f:
            embedding_model = 'glove'
        if 'word2vec' in f:
            embedding_model = 'word2vec'

        if 'tfidf' in f or 'glove' in f or 'word2vec' in f:
            text_features = ['TEXT']
        if 'acoustic_full' in f:
            df_ac = dl.aggregate_acoustic_features()
            df_data = dl.merge_acoustic_features_and_targets(df_ac)
            acoustic_features = dl.get_acoustic_feature_names()
            df_data.sort_values(by='FILENAME', inplace=True)
            df_data.reset_index(inplace=True, drop=True)
        elif 'acoustic' in f:
            df_data = dl.prepare_data(pout=output_file)
            acoustic_features = dl.get_acoustic_feature_names()
            df_data.sort_values(by='FILENAME', inplace=True)
            df_data.reset_index(inplace=True, drop=True)
        else:
            df_data = dl.prepare_data(pout=output_file)
            df_data.sort_values(by='FILENAME', inplace=True)
            df_data.reset_index(inplace=True, drop=True)

        ac = a.split('/')[-1]

        for r in models:
            if task == 'regression':
                rp = RegressionPipeline(df_data, target_labels, r, acoustic_features, text_features)
            else:
                X = df_data.drop(ALL_TARGETS, axis=1)
                y = df_data[target_labels]
                rp = ClassificationPipeline(df_data, target_labels, r, acoustic_features, text_features)
            res, preds = rp.process(tfidf=('tfidf' in f), embedding_model=embedding_model)
            for target_label in res:
                scores = list(res[target_label].values())
                df_res = pd.DataFrame([target_label, ac, f, r] + scores).T
                df_results = df_results.append(df_res)
                df_results.to_csv('results/baseline_fusion_' + task + '_results_' + timestamp + '.csv')
                key = target_label + '_' + ac + '_' + f + '_' + r
                print('-- processing', key)
                pd.DataFrame(preds).to_csv('results/' + task + '_' + key + '_preds_' + timestamp + '.csv')

        # avoid duplicate runs when not using acoustic features
        if 'acoustic' not in f:
            break

df_results.columns = ['target', 'acoustic', 'used_features', 'regressor', 'mae', 'mse', 'rmse']
df_results.to_csv('results/baseline_fusion_' + task + '_results_' + timestamp + '_acoustic.csv')
