import logging
import os
import numpy as np
import pandas as pd
import spacy
import sys
import tensorflow

from data_loader import DataLoader
from tensorflow.keras.layers import Bidirectional, Dense, Embedding, Dropout, Input, LSTM, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import spacy_tokenize, load_embeddings, create_token_index_mappings
from utils import SEED, MAX_LENGTH, UNK
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelBinarizer

OUTPUT_DIR = './output'


def build_model(params, emb_matrix, n_labels, audio=True, text=True):
    if not (text or audio):
        raise Exception('-- Please specify input type (text/audio).')

    logging.info('-- Loading audio:' + str(audio))
    logging.info('-- Loading text:' + str(text))

    tf_dim = params['text_feature_dim']
    af_dim = params['audio_feature_dim']
    nunits1 = params['text_lstm_nunits']
    nunits2 = 2 * nunits1
    dropout = params['dropout']
    lr = params['lr']

    tf_inputs = None
    af_inputs = None
    tf_emb = None
    tf_bilstm = None
    af_lstm = None
    merged = None
    model = None

    if text:
        tf_inputs = Input(shape=(tf_dim,), name='tf_inputs')
        tf_emb = Embedding(input_dim=emb_matrix.shape[0], output_dim=emb_matrix[0].shape[0], input_length=tf_dim,
                           weights=[emb_matrix], trainable=False)(tf_inputs)
        tf_bilstm = Bidirectional(LSTM(nunits1, activation='sigmoid', recurrent_dropout=0.2, recurrent_activation='sigmoid',
                                       return_sequences=False))(tf_emb)
    if audio:
        af_inputs = Input(shape=af_dim, name='af_inputs')
        af_lstm = LSTM(nunits2, return_sequences=False)(af_inputs)

    if text and audio:
        merged = concatenate([tf_bilstm, af_lstm])
        merged = Dropout(dropout)(merged)
    elif text:
        merged = tf_bilstm
    elif audio:
        merged = af_lstm

    output = Dense(n_labels, activation='softmax', name='output')(merged)

    if text and audio:
        model = Model(inputs=[tf_inputs, af_inputs], outputs=[output], name='ee_bimodal')
    elif text:
        model = Model(inputs=[tf_inputs], outputs=[output], name='ee_text')
    elif audio:
        model = Model(inputs=[af_inputs], outputs=[output], name='ee_audio')

    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy')

    return model


def prepare_data(pin_transcripts, pin_targets, ac_dir, emb_path, target='warme5_cat', save=False):
    dl = DataLoader(pin_transcripts, 'P', None, pin_targets)
    df_data = dl.prepare_data()
    #ac_dir = '/Users/andre/workspace/PycharmProjects/fmss/data/output_base_line/type-egemaps-seg-3600-hop-3600'
    #emb_path = '/Users/andre/gensim-data/glove.6B/glove.6B.300d.txt'

    audio_features = dl.load_sequential_acoustic_features(ac_dir)
    audio_features_padded = [pad_sequences(seq, padding='post') for seq in audio_features]
    audio_features_padded = np.asarray(audio_features_padded)

    nlp = spacy.load('en_core_web_sm')
    texts = []

    for doc in nlp.pipe(df_data['TEXT']):
        texts.append(spacy_tokenize(doc, sentence_tokenize=False))

    embedding_vectors = load_embeddings(emb_path)

    token_counts, index2token, token2index = create_token_index_mappings(texts, sentence_tokenized_input=False)

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
    y = lb.fit_transform(df_data[target])

    logging.info('Preparing features...')
    x = [[token2index.get(token, token2index[UNK]) for token in doc] for doc in texts]
    x = pad_sequences(x, maxlen=MAX_LENGTH, padding='post')
    x = np.asarray(x)

    if save:
        emb_name = os.path.basename(os.path.dirname(emb_path))
        pd.to_pickle(x, '../data/tf_embeddings_' + emb_name + '.pickle')
        pd.to_pickle(y, '../data/labels_binarized.pickle')
        pd.to_pickle(embedding_matrix, '../data/embedding_matrix_' + emb_name + '.pickle')
        pd.to_pickle(audio_features_padded, '../data/audio_features_padded.pickle')

    return x, audio_features_padded, embedding_matrix, y


def load_data(emb_name):
    logging.info('Loading data...')
    x = pd.read_pickle('../data/tf_embeddings_' + emb_name + '.pickle')
    audio_features_padded = pd.read_pickle('../data/audio_features_padded.pickle')
    embedding_matrix = pd.read_pickle('../data/embedding_matrix_' + emb_name + '.pickle')
    y = pd.read_pickle('../data/labels_binarized.pickle')
    logging.info('Done.')
    return x, audio_features_padded, embedding_matrix, y


if __name__ == '__main__':
    #pin_transcripts = sys.argv[1]
    #pin_targets = sys.argv[2]
    #ac_dir = sys.argv[3]
    #emb_path = sys.argv[4]

    #x, audio_features_padded, embedding_matrix, y = prepare_data(pin_transcripts, pin_targets, ac_dir, emb_path, target='warme5_cat', save=True)
    x, audio_features_padded, embedding_matrix, y = load_data('glove.6B')

    params = {
        'text_feature_dim': MAX_LENGTH,
        'audio_feature_dim': audio_features_padded[0].shape[0],
        'text_lstm_nunits': 150,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 1
    }

    m = build_model(params, embedding_matrix, 3, audio=True, text=True)

    print(m.summary())
    m.fit([x, audio_features_padded], y)

