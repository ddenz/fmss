for EM in "~/gensim-data/glove-twitter-25/glove-twitter-25.gz" "~/gensim-data/fasttext-wiki-news-subwords-300/fasttext-wiki-news-subwords-300.gz" "~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"
do
	for UNIT in "word" "sentence"
	do
		echo "$EM $UNIT"
		python exp_text_embedding.py -m "$EM" -s mum -t ../data/fmss_transcripts_labels_twinids.csv -u "$UNIT" -o ../data/text_features
		python exp_text_embedding.py -m "$EM" -s mum -t ../data/fmss_transcripts_labels_twinids.csv -u "$UNIT" -o ../data/text_features -a
	done
done