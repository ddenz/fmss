#!/bin/bash

for D in ../../exp/acoustic/*-stats
do
  #######
	# knn #
	#######
	# acoustic only
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u acoustic

	# text only
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m knn -u word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m knn -u tfidf -p

  # acoustic + text
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u acoustic fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m knn -u acoustic glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m knn -u acoustic word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m knn -u acoustic tfidf -p

  #######################
	# logistic regression #
	#######################
	# acoustic only
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u acoustic

	# text only
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m lr -u word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m lr -u tfidf -p

  # acoustic + text
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u acoustic fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m lr -u acoustic glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m lr -u acoustic word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m lr -u acoustic tfidf -p

  #################
	# random forest #
	#################
  # acoustic only
  python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u acoustic

	# text only
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m rf -u word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m rf -u tfidf -p

  # acoustic + text
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u acoustic fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m rf -u acoustic glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m rf -u acoustic word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m rf -u acoustic tfidf -p

  #######
	# svm #
	#######
  # acoustic only
  python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u acoustic

	# text only
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m svm -u word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m svm -u tfidf -p

  # acoustic + text
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/fasttext-wiki-news-subwords-300_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u acoustic fasttext
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/glove-twitter-25_tokens_mum_mean/ -l warme5_cat -x classification -m svm -u acoustic glove
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -w ../data/text_features/word2vec-google-news-300_tokens_mum_mean -l warme5_cat -x classification -m svm -u acoustic word2vec
	python exp_fusion.py -t ../data/fmss_transcripts_labels_twinids.csv -a $D -e ../../data/ERisk/ERisk_coded_data_02Sep21.csv -l warme5_cat -x classification -m svm -u acoustic tfidf -p
done
