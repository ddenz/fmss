import pandas as pd
import spacy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def pattern_sentiment(df_data, sort_key='polarity'):
    df_results = pd.DataFrame(columns=['FILENAME', 'TEXT', 'polarity', 'sentiment'])
    n = 0
    for i, row in df_data.iterrows():
        filename = row.FILENAME
        text = row.TEXT
        text = text.replace('[unknown]', 'whatstheirname')
        for sent in nlp(text).sents:
            tokens = [token.text for token in sent]
            if len(sent) > 0 and ('she' in tokens or 'he' in tokens in tokens) and '?' not in tokens:
                t = TextBlob(sent.text, analyzer=PatternAnalyzer())
                sentiment = '-'
                if t.sentiment.polarity > 0:
                    sentiment = 'positive'
                elif t.sentiment.polarity < 0:
                    sentiment = 'negative'
                df_results.loc[n] = [filename, sent.text, t.sentiment.polarity, sentiment]
                n += 1
    df_results.sort_values(by=['FILENAME', sort_key], ascending=False, inplace=True)
    return df_results


def vader_sentiment(df_data, sort_key='pos'):
    df_results = pd.DataFrame(columns=['FILENAME', 'TEXT', 'pos', 'neu', 'neg'])
    vader = SentimentIntensityAnalyzer()
    d = []
    n = 0
    for i, row in df_data.iterrows():
        filename = row.FILENAME
        text = row.TEXT
        text = text.replace('[unknown]', 'whatstheirname')
        for sent in nlp(text).sents:
            tokens = [token.text for token in sent]
            if len(sent) > 0 and ('she' in tokens or 'he' in tokens in tokens) and '?' not in tokens:
                res = vader.polarity_scores(sent.text)
                max(res['pos'], res['neu'], res['neg'])
                df_results.loc[n] = [filename, sent.text, res['pos'], res['neu'], res['neg']]
                n += 1
    df_results.sort_values(by=['FILENAME', sort_key], ascending=False, inplace=True)
    return df_results


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    df_data = pd.read_csv('../data/fmss_text.csv', sep=';')
    df_data = df_data.loc[df_data.SPEAKER == 'P']
    df_sentiment_pattern = pattern_sentiment(df_data, sort_key='polarity')
    df_sentiment_pattern.to_html('../data/sentiment_analysis/fmss_sentiment_pattern.html')
    df_sentiment_vader = vader_sentiment(df_data, sort_key='pos')
    df_sentiment_vader.to_html('../data/sentiment_analysis/fmss_sentiment_vader.html')
