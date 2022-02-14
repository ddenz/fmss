import glob

import docx
import logging
import numpy as np
import os
import pandas as pd
import re
import spacy

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

from sklearn.preprocessing import LabelBinarizer

env = 'local'

if env == 'local':
    NORM_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Audio/Normalized'
    CLEAN_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Audio/Cleaned'
    FEATS_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Audio/Cleaned/Features'
    CODING_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Coding'
    TEXT_DIR = '/Users/Andre/workspace/PycharmProjects/asfmss/Quest ASFMSS/Data/Transcripts/TXT/Cleaned'
else:
    CODING_DIR = './data'
    FEATS_DIR = './data'

GENSIM_DATA_DIR = '~/gensim-data/'

SEED = 7
UNK = '<UNKNOWN>'
MAX_LENGTH = 400

LABELS = ['t2_fmss_IS', 't2_fmss_war', 't2_fmss_eoi', 't2_rel', 't2_ee']

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

nlp = spacy.load('en_core_web_sm')


class SentenceEncoder(Sequential):
    def __init__(self, embedding_matrix):
        super().__init__()
        self.embedding_matrix = embedding_matrix

    def build_model(self, optimizer=Adam(lr=0.001), loss='categorical_crossentropy'):
        # input_dim = vocab size, output_dim = embedding size, input_length = sentence length
        self.add(Embedding(input_dim=self.embedding_matrix.shape[0], output_dim=self.embedding_matrix[0].shape[0],
                           input_length=MAX_LENGTH, weights=[self.emb_matrix], trainable=False))
        self.add(Conv1D(filters=64, kernel_size=7))
        self.add(MaxPooling1D())
        self.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


def word2txt():
    """
    Extract text from Word documents and save as text.
    :return:
    """
    pin = 'data/test.docx'
    pout = os.path.splitext(pin)[0] + '.txt'

    doc = docx.Document(pin)
    text = '\n'.join([p.text for p in doc.paragraphs])

    with open(pout, 'w') as fout:
        print(text, file=fout)
    fout.close()


def load_data_to_dataframe():
    """
    Load all data to a dataframe.
    :return:
    """
    df_coding = pd.read_excel(os.path.join(CODING_DIR, 'Quest_ASFMSS_coding.xlsx'))

    inds = df_coding[df_coding.columns[1:]].dropna(how='all').index
    df_coding = df_coding.loc[inds]
    # Remove 69
    df_coding = df_coding.loc[df_coding.idnum != 69]

    """
    Only retain useful features
    IS = initial statement
    war = warmth
    eoi = emotional over-involvement
    rel = rela
    """
    df_coding = df_coding[['idnum', 't2_fmss_IS', 't2_fmss_war', 't2_fmss_eoi', 't2_rel', 't2_ee']]

    # Insert text for each id
    id_map = {re.sub('(Q[0-9]+)[\_ ].+', '\g<1>', f): f for f in os.listdir(TEXT_DIR)}
    df_coding['text'] = df_coding.idnum.apply(
        lambda x: open(os.path.join(TEXT_DIR, id_map['Q' + str(x)]), 'r', encoding='latin-1').read())

    return df_coding.reset_index(drop=True)


def create_token_index_mappings(texts, sentence_tokenized_input=False):
    logging.info('Creating token-idx mappings...')
    # create mappings of words to indices and indices to words
    UNK = '<UNKNOWN>'
    # PAD = '<PAD>'
    token_counts = {}

    if sentence_tokenized_input:
        for doc in texts:
            for sent in doc:
                for token in sent:
                    c = token_counts.get(token, 0) + 1
                    token_counts[token] = c
    else:
        for doc in texts:
            for token in doc:
                c = token_counts.get(token, 0) + 1
                token_counts[token] = c

    vocab = sorted(token_counts.keys())
    # start indexing at 1 as 0 is reserved for padding
    token2index = dict(zip(vocab, list(range(1, len(vocab) + 1))))
    token2index[UNK] = len(vocab) + 1
    # token2index[PAD] = len(vocab) + 2
    index2token = {value: key for (key, value) in token2index.items()}
    assert index2token[token2index['help']] == 'help'

    return token_counts, index2token, token2index


def load_data(test=False):
    if test:
        return pd.read_csv(CODING_DIR + '/asfmss_dummy_data_text.csv', dtype={'idnum': str})
    return pd.read_csv(CODING_DIR + '/Quest_ASFMSS_all_data.csv', dtype={'idnum': str})


def load_embeddings(emb_path):
    model = None
    w2v_path = emb_path
    if 'glove.6B' in emb_path:
        w2v_path = os.path.splitext(emb_path)[0] + '_w2v.txt'
        glove2word2vec(glove_input_file=emb_path, word2vec_output_file=w2v_path)
    model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)
    return model


def load_audio_features(idnum, ftype='mfcc', dim=400, test=False):
    if test:
        audio_features = pd.read_pickle('data/' + str(idnum) + '.wav_' + ftype + '.pickle')
    else:
        audio_features = pd.read_pickle(FEATS_DIR + '/Q' + str(idnum) + '.wav_' + ftype + '.pickle')
    if dim >= 0:
        logging.info('-- Truncating audio features to length ' + str(dim))
        return audio_features[:dim]
    return audio_features


def load_sequential_features(audio_type=None, text_type=None):
    """
    Load from disk to avoid preprocessing.
    :param audio_type: type of audio features to load (e.g. mfcc)
    :param text_type: type of text embedding features to load (e.g. gensim-csg-gigaword-300)
    :return: word embeddings, padded audio feature matrices, target labels, word embedding matrix
    """
    logging.info('Loading pre-calculated features from disk:')
    if audio_type is None and text_type is None:
        raise Exception('-- Unspecified feature types. Choose audio and text inputs.')
    t = None
    a = None
    e = None
    if text_type is not None:
        tp = 'data/tf_embeddings_' + text_type + '.pickle'
        logging.info('-- text (' + tp + ')')
        t = pd.read_pickle(tp)
        e = pd.read_pickle('data/embedding_matrix_' + text_type + '.pickle')
    if audio_type is not None:
        ap = 'data/af_padded_' + audio_type + '.pickle'
        logging.info('-- audio (' + ap + ')')
        a = pd.read_pickle(ap)
    y = pd.read_pickle('data/labels_binarized.pickle')
    return t, a, y, e


def prepare_sequential_features(emb_path, sentence_tokenize=False, test=False, save=False):
    logging.info('Preparing sequential data (' + emb_path + ')...')

    df_data = load_data(test=test)

    texts = []

    for doc in nlp.pipe(df_data.text):
        texts.append(spacy_tokenize(doc, sentence_tokenize=sentence_tokenize))

    audio_features = []
    for idnum in df_data.idnum:
        af = load_audio_features(idnum, ftype='mfcc', dim=400, test=test)
        audio_features.append(af)

    audio_features_padded = [pad_sequences(seq, padding='post') for seq in audio_features]

    embedding_vectors = load_embeddings(emb_path)

    token_counts, index2token, token2index = create_token_index_mappings(texts, sentence_tokenized_input=sentence_tokenize)

    # create mapping of words to their embeddings
    emb_map = {}
    for w in embedding_vectors.key_to_index:
        emb_map[w] = embedding_vectors.get_vector(w)

    vocab_size = len(token_counts)
    embed_len = embedding_vectors[list(emb_map.keys())[0]].shape[0]
    embedding_matrix = np.zeros((vocab_size + 1, embed_len))

    # initialize the embedding matrix
    logging.info('Initializing embeddings matrix...')
    for word, i in token2index.items():
        if i >= vocab_size:
            continue
        if word in embedding_vectors:
            embedding_vector = embedding_vectors.get_vector(word)
            # words not found in embedding idx will be all-zeros.
            embedding_matrix[i] = embedding_vector

    logging.info('Preparing labels...')

    lb = LabelBinarizer()
    labels = {}

    for label in LABELS:
        # df_data[label] = lb.fit_transform(df_data[label])
        labels[label] = lb.fit_transform(df_data[label])

    logging.info('Preparing features...')

    if sentence_tokenize:
        x = [[[token2index.get(token, token2index[UNK]) for token in sent] for sent in doc] for doc in texts]
        x = [pad_sequences(sent, maxlen=MAX_LENGTH, padding='post') for sent in x]
    else:
        x = [[token2index.get(token, token2index[UNK]) for token in doc] for doc in texts]
        x = pad_sequences(x, maxlen=MAX_LENGTH, padding='post')

    x = np.asarray(x)
    audio_features_padded = np.asarray(audio_features_padded)

    if save:
        emb_name = os.path.basename(os.path.dirname(emb_path))
        pd.to_pickle(x, 'data/tf_embeddings_' + emb_name + '.pickle')
        pd.to_pickle(audio_features_padded, 'data/af_padded_mfcc.pickle')
        # pd.to_pickle(df_data[LABELS], 'data/labels_binarized.pickle')
        pd.to_pickle(labels, 'data/labels_binarized.pickle')
        pd.to_pickle(embedding_matrix, 'data/embedding_matrix_' + emb_name + '.pickle')

    # return x, audio_features_padded, df_data[LABELS], embedding_matrix
    return x, audio_features_padded, labels, embedding_matrix


def process_token(token):
    if token.like_url:
        return '__URL__'
    elif token.like_num:
        return '__NUM__'
    else:
        form = token.lower_
        form = re.sub('[\!\"#\$%&\(\)\*\+,\./:;<=>\?@\[\\]\^_`\{\|\}\~]+', '', form)
        form = re.sub('([^\-,]+)[\-,]', '\g<1>', form)
        form = re.sub('^([^\.]+)\.', '\g<1>', form)
        return form


def spacy_tokenize(doc, sentence_tokenize=False):
    if isinstance(doc, str):
        doc = nlp(doc)
    if sentence_tokenize:
        return [[process_token(token) for token in sent if (not token.is_punct or token.is_space)] for sent in doc.sents]
    return [process_token(token) for token in doc]


def prepare_sequential_audio(feature_type='mfcc'):
    return [pd.read_pickle(f) for f in glob.glob('./data/*mfcc.pickle')]


"""
if __name__ == '__main__':
    #df = load_data_to_dataframe()
    #df.to_csv(CODING_DIR + '/Quest_ASFMSS_all_data.csv', encoding='utf-8', idx=False)
    tf, af, labels, emb_matrix = prepare_sequential_features(GENSIM_DATA_DIR + '/glove.6B/glove.6B.300d.txt', sentence_tokenize=False, test=True)
"""
