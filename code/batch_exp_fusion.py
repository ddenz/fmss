import os
import pandas as pd
import sys

from exp_fusion import *
from data_loader import ALL_TARGETS
from datetime import datetime
from glob import glob

erisk_codes_file = '../../data/ERisk/ERisk_coded_data_02Sep21.csv'
transcript_file = '../data/fmss_transcripts_labels_twinids.csv'
acoustic_features_dirs = glob('../../exp/acoustic/type-*-stats/')
models = ['lr', 'knn', 'rf', 'svm']
task = 'classification'
# features_to_use = ['acoustic_full', 'acoustic', 'tfidf', 'glove', 'word2vec', 'acoustic tfidf', 'acoustic glove', 'acoustic word2vec']
features_to_use = ['acoustic', 'tfidf', 'glove', 'word2vec', 'fasttext', 'acoustic tfidf', 'acoustic glove', 'acoustic word2vec', 'acoustic fasttext']
timestamp = datetime.today().strftime('%Y%m%d') #-%H:%M:%S')

target_labels = ['warme5_cat']
if task == 'regression':
    target_labels = ['warme5', 'warmy5', 'disse5', 'dissy5']

df_results = pd.DataFrame()
output_file = '../data/fmss_exp_full_data_batch.csv'

for f in features_to_use:
    print('-- using features:', f)
    acoustic_features = []
    text_features = []
    embedding_model = None
    word_embedding_dir = None

    for a in acoustic_features_dirs:

        if 'glove' in f:
            embedding_model = 'glove'
            word_embedding_dir = '../../exp/nlp/text_features/glove-twitter-25_tokens_mum_mean/'
        if 'fasttext' in f:
            embedding_model = 'fasttext'
            word_embedding_dir = '../../exp/nlp/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/'
        if 'word2vec' in f:
            embedding_model = 'word2vec'
            word_embedding_dir = '../../exp/nlp/text_features/word2vec-google-news-300_tokens_mum_mean/'

        if 'tfidf' in f or 'glove' in f or 'fasttext' in f or 'word2vec' in f:
            text_features = ['TEXT']

        dl = DataLoader(transcript_file, a, erisk_codes_file, word_embedding_dir, load_utterances_with_both_twins=False, load_both_speakers=False, merge_on='speaker')
        df_data = dl.process()
        acoustic_features = dl.get_acoustic_feature_names()
        embed_features = dl.get_embed_feature_names()

        ac = a.split('/')[-2]

        for r in models:
            if task == 'regression':
                rp = RegressionPipeline(df_data, target_labels, r, embed_features, acoustic_features, text_features)
            else:
                X = df_data.drop(ALL_TARGETS, axis=1, errors='ignore')
                y = df_data[target_labels]
                rp = ClassificationPipeline(df_data, target_labels, r, embed_features, acoustic_features, text_features)
            res, preds = rp.process(tfidf=('tfidf' in f), embedding_model=embedding_model)
            for target_label in res:
                scores = list(res[target_label].values())
                df_res = pd.DataFrame([target_label, ac, f, r] + scores).T
                df_results = df_results.append(df_res)
                df_results.to_csv('../results/exp_fusion_' + task + '_results_' + timestamp + '.csv')
                key = target_label + '_' + ac + '_' + f + '_' + r
                print('-- processing', key)
                pd.DataFrame(preds).to_csv('../results/exp_' + task + '_' + key + '_preds_' + timestamp + '.csv')

        # avoid duplicate runs when not using acoustic features
        if 'acoustic' not in f:
            break

df_results.columns = ['target', 'acoustic', 'used_features', 'classifier', 'accuracy', 'precision', 'recall', 'f-score']
if task == 'regression':
    df_results.columns = ['target', 'acoustic', 'used_features', 'regressor', 'mae', 'mse', 'rmse']

df_results.to_csv('../results/exp_fusion_' + task + '_results_' + timestamp + '_acoustic.csv')
