import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import spacy
import sys

from data_loader import ALL_TARGETS, DataLoader
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from wordcloud import WordCloud

nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    """
    Text preprocessing: tokenisation, lower-casing, removal of stop words and punctuation
    :param text: input text to process
    :return: a single text string of the preprocessed text
    """
    tokens = ['']
    for token in nlp(text):
        if not (token.is_punct or token.is_space or token.is_stop or token.lower_ == '--'):
            tokens.append(token.lower_.replace('--', ''))
            #print(token, token.lower_, token.is_space, token.is_stop)
    return ' '.join(tokens)
    #return ' '.join([token.lower_ for token in nlp(text) if not (token.is_punct or token.is_space or token.is_stop or token.lower_ == '--')])


def do_preprocessing(df, pout=None):
    """
    Preprocess texts.
    :param df: a Pandas data frame containing a text column (header name must be 'TEXT')
    :param pout: a path to output the resulting data frame to in CSV format
    :return: the resulting data frame with a new column containing the preprocessed text (column name 'TEXT_PP')
    """
    print('-- preprocessing text corpus...', end='')
    df['TEXT'] = df.TEXT.apply(lambda x: re.sub('\[[^\]]+\]', '', x, flags=re.I))
    df['TEXT'] = df.TEXT.replace('`', '')
    df['TEXT_PP'] = df.TEXT.apply(preprocess_text)
    df['TEXT_PP'] = df.TEXT_PP.apply(str.strip)
    df['TEXT_PP'] = df.TEXT_PP.fillna(value='')
    if pout is not None:
        df.to_csv(pout, sep=';', index=False)
    print('Done.')
    return df


def calculate_tfidf_freqs(corpus, vocab=None, token_pattern=r"(?u)\b\w\w+\b", max_features=None):
    """
    Calculate TFIDF vectors for corpus
    :param corpus: a list of word strings
    :param vocab: a list of words to include
    :param token_pattern: a regular expression pattern to use to recognise tokens
    :param max_features: maximum number of features to detect - ignored if vocab is not None
    :return: a Pandas data frame containing the corpus words sorted in descending order of TFIDF score
    """
    if vocab is not None and token_pattern is not None:
        raise Exception('specify either vocab or token_pattern, not both')

    vec = TfidfVectorizer(vocabulary=vocab, token_pattern=token_pattern, max_features=max_features)
    try:
        vecs = vec.fit_transform(corpus)
        feature_names = vec.get_feature_names()
        vecs = vecs.todense().tolist()
    except ValueError as e:
        print(e, file=sys.stderr)
        print(corpus.tolist(), file=sys.stderr)
        vecs = []
        feature_names = []
    df_freqs = pd.DataFrame(vecs, columns=feature_names).T.sum(axis=1)
    return df_freqs


def calculate_counts(corpus, vocab=None, token_pattern=r"(?u)\b\w\w+\b", max_features=None):
    """
    Calculate word frequencies for corpus
    :param corpus: a list of word strings
    :param vocab: a list of words to include
    :param token_pattern: a regular expression pattern to use to recognise tokens
    :param max_features: maximum number of features to detect - ignored if vocab is not None
    :return: a Pandas data frame containing the corpus words sorted in descending order of absolute frequency
    """
    if vocab is not None and token_pattern is not None:
        raise Exception('specify either vocab or token_pattern, not both')

    vec = CountVectorizer(vocabulary=vocab, token_pattern=token_pattern, max_features=max_features)
    try:
        vecs = vec.fit_transform(corpus)
        feature_names = vec.get_feature_names()
        vecs = vecs.todense().tolist()
    except ValueError as e:
        print(e, file=sys.stderr)
        print(corpus.tolist(), file=sys.stderr)
        vecs = []
        feature_names = []
    df_freqs = pd.DataFrame(vecs, columns=feature_names).T.sum(axis=1)
    return df_freqs


def load_lexicon(path):
    """
    Load a dictionary of terms. The format of the lexicon must be term<tab>sentiment<tab>truth_value,
    where truth_value is a binary (0 or 1) value indicating presence or absence of the sentiment for the given term.
    :param path: path to the lexicon file
    :return: a Pandas data frame containing the terms and their associated sentiment (positive/negative)
    """
    df = pd.read_csv(path, sep='\t')
    df.columns = ['term', 'sentiment', 'value']
    df = df.loc[(df.sentiment.isin(['positive', 'negative'])) & (df.value == 1)]
    df = df[['term', 'sentiment']]
    df.dropna(axis=0, inplace=True)
    return df


def build_regex_from_lexicon(df_lex, sentiment=None):
    regex = '.'
    if sentiment is not None:
        regex = r'\b(' + '|'.join(df_lex.loc[df_lex.sentiment == sentiment].term.tolist()) + r')\b'
    return regex


def build_wordcloud(df, show=True, pout=None):
    """
    Build a word cloud.
    :param df: a Pandas data frame containing ordered term frequencies
    :param pout: the file path to output a wordcloud image to
    :return: None
    """
    wc = WordCloud(background_color='white', height=800, width=1600)
    wc.generate_from_frequencies(df)
    plt.imshow(wc, interpolation='bilinear')
    if pout is not None:
        pdir = Path(pout).parent.absolute()
        os.makedirs(pdir, exist_ok=True)
        wc.to_file(pout)
        print('-- Wrote image:', pout)
    plt.axis('off')
    if show:
        plt.show()


def create_wordclouds(df_data, lexicon='liwc'):
    """
    Create wordclouds for the corpus.
    :param df_data: a Pandas data frame containing the corpus
    :param lexicon: the name of the lexicon to use to filter terms in the corpus
    :return: None
    """
    if lexicon == 'liwc':
        df_lex = load_lexicon('../resources/LIWC_2015_pos_neg_emo.txt')
    elif lexicon == 'emolex':
        df_lex = load_lexicon('../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    else:
        raise Exception('unknown lexicon: ' + lexicon)

    doc_type = 'interview'
    if 'SPEAKER' in df_data.columns:
        doc_type = 'turns'

    df_freqs = calculate_counts(df_data.TEXT_PP.tolist())
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_counts_all_terms_' + doc_type + '.png')

    df_freqs = calculate_tfidf_freqs(df_data.TEXT_PP.tolist())
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_tfidf_all_terms_' + doc_type + '.png')

    pos_regex = r'\b(' + '|'.join(df_lex.loc[df_lex.sentiment == 'positive'].term.tolist()) + r')\b'

    df_freqs = calculate_counts(df_data.TEXT_PP.tolist(), vocab=None, token_pattern=pos_regex)
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_counts_pos_' + lexicon + '_' + doc_type + '.png')

    neg_regex = r'\b(' + '|'.join(df_lex.loc[df_lex.sentiment == 'negative'].term.tolist()) + r')\b'

    df_freqs = calculate_counts(df_data.TEXT_PP.tolist(), vocab=None, token_pattern=neg_regex)
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_counts_neg_' + lexicon + '_' + doc_type + '.png')

    pos_neg_regex = r'\b(' + '|'.join(df_lex.term.tolist()) + r')\b'

    df_freqs = calculate_counts(df_data.TEXT_PP.tolist(), vocab=None, token_pattern=pos_neg_regex)
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_counts_pos_neg_' + lexicon + '_' + doc_type + '.png')

    df_freqs = calculate_tfidf_freqs(df_data.TEXT_PP.tolist(), vocab=None, token_pattern=pos_regex)
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_tfidf_pos_' + lexicon + '_' + doc_type + '.png')

    df_freqs = calculate_tfidf_freqs(df_data.TEXT_PP.tolist(), vocab=None, token_pattern=neg_regex)
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_tfidf_neg_' + lexicon + '_' + doc_type + '.png')

    df_freqs = calculate_tfidf_freqs(df_data.TEXT_PP.tolist(), vocab=None, token_pattern=pos_neg_regex)
    build_wordcloud(df_freqs, show=False, pout='data/text_exploration/word_clouds/wc_tfidf_pos_neg_' + lexicon + '_' + doc_type + '.png')


def build_heatmap(df_data, target, lexicon, sentiment, freq_type='tfidf', token_regex=None, max_features=20, show=False, pout=None):
    d = {}

    for g in df_data.groupby('FILENAME'):
        texts_pp = g[1].TEXT_PP
        if freq_type == 'counts':
            df_counts = calculate_counts(texts_pp, token_pattern=token_regex, max_features=max_features)
        else:
            df_counts = calculate_tfidf_freqs(texts_pp, token_pattern=token_regex, max_features=max_features)
        d_counts = df_counts.to_dict()
        target_value = g[1][target].iloc[0]
        tmp = d.get(target_value, {})
        for term in d_counts:
            counts = np.mean([tmp.get(term, 0.0), d_counts[term]])
            tmp[term] = counts
            d[target_value] = tmp
    df_values = pd.DataFrame(d)
    df_values.dropna(inplace=True)
    df_values = df_values.reindex(sorted(df_values), axis=1)
    sns.set(font_scale=3.0)
    fig, ax = plt.subplots(figsize=(20, 20))
    hm = sns.heatmap(data=df_values, ax=ax, xticklabels=True, yticklabels=True, annot=True)
    title = freq_type + ' important ' + sentiment + ' words'
    if lexicon is not None:
        title +=  ' (' + lexicon + ')'
    hm.set_title(title)
    plt.yticks(rotation=0)
    if pout is not None:
        pdir = Path(pout).parent.absolute()
        os.makedirs(pdir, exist_ok=True)
        plt.savefig(pout)
        print('-- Wrote image:', pout)
    if show:
        plt.show()


def create_heatmaps(df_data, target, lexicon, freq_type='tfidf', show=False):
    if lexicon == 'liwc':
        df_lex = load_lexicon('../resources/LIWC_2015_pos_neg_emo.txt')
    elif lexicon == 'emolex':
        df_lex = load_lexicon('../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    else:
        raise Exception('unknown lexicon: ' + lexicon)

    pos_regex = r'\b(' + '|'.join(df_lex.loc[df_lex.sentiment == 'positive'].term.tolist()) + r')\b'
    neg_regex = r'\b(' + '|'.join(df_lex.loc[df_lex.sentiment == 'negative'].term.tolist()) + r')\b'
    pos_neg_regex = r'\b(' + '|'.join(df_lex.term.tolist()) + r')\b'

    build_heatmap(df_data, target, None, '', freq_type=freq_type, token_regex=r'(?u)\b\w\w+\b', max_features=20, show=show, pout='data/text_exploration/heatmaps/' + target + '/hm_' + freq_type + '_all.png')
    build_heatmap(df_data, target, lexicon, 'positive', freq_type=freq_type, token_regex=pos_regex, max_features=20, show=show, pout='data/text_exploration/heatmaps/' + target + '/hm_' + freq_type + '_pos_' + lexicon + '.png')
    build_heatmap(df_data, target, lexicon, 'negative', freq_type=freq_type, token_regex=neg_regex, max_features=20, show=show, pout='data/text_exploration/heatmaps/' + target + '/hm_' + freq_type + '_neg_' + lexicon + '.png')
    build_heatmap(df_data, target, lexicon, 'positive & negative', freq_type=freq_type, token_regex=pos_neg_regex, max_features=20, show=show, pout='data/text_exploration/heatmaps/' + target + '/hm_' + freq_type + '_pos_neg_' + lexicon + '.png')



if __name__ == '__main__':
    # df_data = do_preprocessing(pd.read_csv('data/fmss_text.csv', sep=';'))
    # df_data = do_preprocessing(pd.read_csv('data/fmss_text_targets.csv', sep=';'))
    # df_data = pd.read_csv('data/fmss_text_pp.csv', sep=';', keep_default_na=False)
    # df_data = pd.read_csv('data/fmss_text_pp.csv', sep=';', keep_default_na=False)
    # create_wordclouds(df_data, doc_type='interview', lexicon='liwc')
    # df_data = pd.read_csv('data/fmss_text_pp.csv', sep=';', keep_default_na=False)
    # df_lex = load_lexicon('resources/LIWC_2015_pos_neg_emo.txt')
    # df_data = do_preprocessing(df_data, pout='data/fmss_text_pp_targets_P_merged.csv')

    """ prepare data
    dl = DataLoader('data/ERisk_coded_data_02Sep21.xlsx', 'data/fmss_text.csv', 'data/output_base_line/type-avec2013-seg-3600-hop-3600-avg', 'P')

    df_data = dl.merge_transcripts_and_targets(merge_on=True)
    df_data = do_preprocessing(df_data, pout='data/fmss_text_pp_targets_P_merged.csv')

    df_data = dl.merge_transcripts_and_targets(merge_on=False)
    df_data = do_preprocessing(df_data, pout='data/fmss_text_pp_targets_P_unmerged.csv')
    end prepare data"""

    df = pd.read_csv('../data/fmss_text_pp_targets_P_unmerged.csv', sep=';', keep_default_na=False)

    create_wordclouds(df, lexicon='liwc')
    create_wordclouds(df, lexicon='emolex')

    # see notebook word-score correlations
    for target in ALL_TARGETS:
        print('-- Processing', target, end='...')
        try:
            create_heatmaps(df, target, 'liwc', show=False)
            create_heatmaps(df, target, 'emolex', show=False)
            print('Done.')
        except Exception as e:
            print('Skipped.')
