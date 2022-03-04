#!/bin/bash

# knn
python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u fasttext

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u glove

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m knn -u word2vec

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m knn -u tfidf

# logistic regression
python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u fasttext

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u glove

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m lr -u word2vec

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m lr -u tfidf

# random forest
python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u fasttext

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u glove

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m rf -u word2vec

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m rf -u tfidf


# svm
python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u fasttext

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u glove

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m svm -u word2vec

python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a ../../exp/acoustic/type-avec2013-stats/ -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m svm -u tfidf
