import argparse
import os

import numpy as np
import pandas as pd
import spacy
import sys

from exp_data_loader import *
from pathlib import Path
from text_exploration import do_preprocessing, preprocess_text
from gensim.models.keyedvectors import KeyedVectors

# en_core_web_sm or en_core_web_trf
nlp = spacy.load('en_core_web_sm')


def prepare_data(transcript_path, unit='token'):
    """
    Prepare data for extraction of word embedding features
    :param transcript_path: path to CSV file containing all transcripts
    :param speakers: speakers to use (P or both)
    :param unit: linguistic unit to use (token/word or sentence)
    :return: Pandas data frame containing transcripts with merged turns for chosen speaker(s), mean sentence length,
    mean number of sentences per interview (None if token/word is used as a unit)
    """
    print('Preparing data...')
    dl = DataLoader(transcript_path, None, None, load_utterances_with_both_twins=False, load_both_speakers=False, merge_on='speaker')
    df = dl.load_transcripts()

    if unit in ['token', 'word']:
        df = do_preprocessing(df)
        mean_sent_len = int(round(df['TEXT_PP'].apply(lambda x: len(x.split())).mean(), 0))
        mean_num_sent = None
    else:
        df = do_preprocessing_sentences(df)
        mean_num_sent = int(np.round(df['TEXT_PP_SENT'].apply(lambda x: len(x)).mean(), 0))
        mean_sent_len = int(round(df['TEXT_PP_SENT'].apply(lambda x: np.mean([len(y) for y in x])).mean(), 0))

    return df, mean_sent_len, mean_num_sent


def do_preprocessing_sentences(df, pout=None):
    """

    :param df:
    :param pout:
    :return:
    """
    def preprocess_text(text):
        sents = []
        for sent in nlp(text).sents:
            tokens = ' '.join([token.lower_ for token in sent if not (token.is_punct or token.is_space or token.is_stop or token.lower_ == '--')])
            sents.append(tokens)
        return sents

    df['TEXT_PP_SENT'] = df[TEXT_HDR].apply(preprocess_text)
    return df


def create_unknown_embedding(model):
    """
    Create an embedding for unknown words by averaging all embeddings for known words
    :param model: embedding model
    :return: unknown word vector
    """
    new_index = len(model)
    hidden_dim = model[0].shape[0]
    vecs = np.zeros((new_index, hidden_dim), dtype=np.float32)
    for i in range(len(model)):
        vecs[i] = model[i]
    average_vec = np.mean(vecs, axis=0)
    return average_vec


def embed_text(model, text, unknown_vec=None):
    """
    Covert text to embeddings
    :param model: embedding model
    :param text: text as a list of tokens
    :param unknown_vec: a vector to use for unknown words
    :return: list of word vectors for the text
    """
    vectors = []
    for word in text:
        if word in model:
            vectors.append(model.get_vector(word))
        elif unknown_vec is not None:
            vectors.append(unknown_vec)
    return vectors


def embed_sentences(model, sents, unknown_vec, max_len=None, aggregation=None):
    """

    :param model:
    :param sents:
    :param unknown_vec:
    :param max_len:
    :param aggregation:
    :return:
    """
    vectors = []
    for sent in sents:
        if max_len is not None:
            sent = sent[0:max_len]
        sent_vec = embed_text(model, sent, unknown_vec)
        if aggregation in ['average', 'mean']:
            sent_vec = np.mean(sent_vec, axis=0)
        vectors.append(sent_vec)
    return vectors


def batch_embed_tokens(model_path, df, output_dir_path, speakers, max_len=None, create_unknown=None, average_all=False):
    """

    :param model_path:
    :param df:
    :param output_dir_path:
    :param speakers: the speakers to include (mum or both)
    :param max_len:
    :return:
    """
    try:
        embs = KeyedVectors.load_word2vec_format(model_path)
    except UnicodeDecodeError:
        embs = KeyedVectors.load_word2vec_format(model_path, binary=True)
    pad_vec = np.zeros((len(embs), embs[0].shape[0]))
    model_name = Path(model_path).parent.name
    pout_dir = os.path.join(output_dir_path, model_name + '_tokens_' + speakers)
    if average_all:
        pout_dir += '_mean'
    unk_vec = None
    if create_unknown:
        unk_vec = create_unknown_embedding(embs)
        pout_dir += '_unk'
    os.makedirs(pout_dir, exist_ok=True)

    for i, row in df.iterrows():
        filename = row[FILE_HDR]
        twinid = str(row[TWINID_HDR])
        text = row['TEXT_PP'].split()
        if max_len is not None and len(text) > max_len:
            text = text[0:max_len]
        emb_text = embed_text(embs, text, unk_vec)
        if max_len is not None and len(emb_text) < max_len:
            emb_text += [pad_vec for i in range(max_len - len(emb_text))]
        pout = pout_dir + '/' + filename + '_twin-' + twinid + '.csv'
        df_out = pd.DataFrame(emb_text)
        if average_all:
            df_out = pd.DataFrame(df_out.mean(axis=0)).T
        df_out.insert(0, 'name', len(df_out) * [filename])
        df_out.to_csv(pout, index=False)
        print('-- wrote file:', pout)


def batch_embed_sentences_with_padding(model_path, df, max_num_sent, max_sent_len, output_dir_path, speakers, create_unknown=None):
    """
    Embed tokens in all sentences as-is and adjust for sentence length and number of sentences if required
    :param model_path:
    :param df: a Pandas data frame containing all transcripts
    :param output_dir_path: directory to output all files to
    :param max_num_sent: the maximum number of sentences to include in a document
    :param max_sent_len: the maximum allowable sentence length in number of tokens
    :param speakers: the speakers to include (mum or both)
    :return: None
    """
    try:
        embs = KeyedVectors.load_word2vec_format(model_path)
    except UnicodeDecodeError:
        embs = KeyedVectors.load_word2vec_format(model_path, binary=True)
    pad_vec = np.zeros((len(embs), embs[0].shape[0]))
    model_name = Path(model_path).parent.name
    pout_dir = os.path.join(output_dir_path, model_name + '_sents_full_' + str(max_num_sent) + '_' + str(max_sent_len) + '_' + speakers)

    unk_vec = None
    if create_unknown:
        unk_vec = create_unknown_embedding(embs)
        pout_dir += '_unk'

    os.makedirs(pout_dir, exist_ok=True)

    for i, row in df.iterrows():
        filename = row[FILE_HDR]
        twinid = str(row[TWINID_HDR])
        sents = row['TEXT_PP_SENT']
        if len(sents) > max_num_sent:
            sents = sents[0:max_num_sent]
        emb_sents = embed_sentences(embs, sents, unk_vec, max_len=max_sent_len)
        if len(emb_sents) < max_num_sent:
            for i in range(max_num_sent - len(emb_sents)):
                emb_sents.append([pad_vec for j in range(max_sent_len)])
        pout = pout_dir + '/' + filename + '_twin-' + twinid + '.pickle'
        pd.to_pickle(emb_sents, pout)
        print('-- wrote sentence embeddings with padding to', pout)


def batch_embed_sentences(model_path, df, output_dir_path, speakers, create_unknown=None, average_all=False):
    """
    Embed tokens in all sentences and average for sentence embeddings
    :param model_path:
    :param df: a Pandas data frame containing all transcripts
    :param output_dir_path: directory to output all files to
    :param speakers: the speakers to include (mum or both)
    :return: None
    """
    try:
        embs = KeyedVectors.load_word2vec_format(model_path)
    except UnicodeDecodeError:
        embs = KeyedVectors.load_word2vec_format(model_path, binary=True)

    model_name = Path(model_path).parent.name
    pout_dir = os.path.join(output_dir_path, model_name + '_sents_mean_' + speakers)
    if average_all:
        pout_dir += '_mean'
    unk_vec = None
    if create_unknown:
        unk_vec = create_unknown_embedding(embs)
        pout_dir += '_unk'
    os.makedirs(pout_dir, exist_ok=True)

    for i, row in df.iterrows():
        filename = row[FILE_HDR]
        twinid = str(row[TWINID_HDR])
        sents = [s for s in row['TEXT_PP_SENT'] if s != '']
        emb_sents = embed_sentences(embs, sents, unk_vec, max_len=None, aggregation='mean')
        pout = pout_dir + '/' + filename + '_twin-' + twinid + '.csv'
        df_out = pd.DataFrame(emb_sents)
        if average_all:
            df_out = pd.DataFrame(df_out.mean(axis=0)).T
        df_out.insert(0, 'name', len(df_out) * [filename])
        df_out.to_csv(pout, index=False)
        print('-- wrote sentence embeddings without padding to', pout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMSS: apply text embeddings')
    parser.add_argument('-m', '--model_path', type=str, nargs=1, help='path to file containing word embedding model', required=True)
    parser.add_argument('-s', '--speakers', type=str, nargs=1, help='speakers to include', choices=['mum', 'both'],
                        required=True)
    parser.add_argument('-t', '--transcript_file', type=str, nargs=1,
                        help='path to turn-based text transcripts per audio file in CSV format', required=True)
    parser.add_argument('-a', '--average_all', action='store_true',
                        help='take mean of all rows to produce final output (same number of columns)', required=False)
    parser.add_argument('-g', '--generate_unknown', action='store_true',
                        help='generate a word vector for unknown words (mean of all embeddings in model)', required=False)
    parser.add_argument('-u', '--unit', type=str, nargs=1, help='linguistic units to use (token/word or sentence)',
                        choices=['token', 'word', 'sentence'], required=True)
    parser.add_argument('-o', '--output_directory', type=str, nargs=1, help='directory to output embedded transcripts '
                        'to (a subdirectory for output files will be created here)', required=True)
    args = parser.parse_args()

    model_path = args.model_path[0]
    speakers = args.speakers[0]
    transcript_file = args.transcript_file[0]
    unit = args.unit[0]
    pout = args.output_directory[0]

    df_data, max_len, max_num_sent = prepare_data(transcript_file, unit=unit)

    if unit in ['token', 'word']:
        batch_embed_tokens(model_path, df_data, pout, speakers, create_unknown=args.generate_unknown, average_all=args.average_all)
    elif unit == 'sentence':
        batch_embed_sentences(model_path, df_data, pout, speakers, create_unknown=args.generate_unknown, average_all=args.average_all)
    else:
        raise ValueError('-- invalid unit: ' + unit + ". Use 'token', 'word' or 'sentence'.")
