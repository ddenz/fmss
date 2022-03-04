import argparse
import numpy as np
import pandas as pd

from exp_data_loader import DataLoader

from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict, GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import LinearSVR, LinearSVC

from sklearn.preprocessing import StandardScaler

from zeugma import EmbeddingTransformer, ItemSelector

SEED = 42


class BasePipeline(object):
    def __init__(self, data: pd.DataFrame, target_names: list, estimator_name: str, embed_cols:list, acoustic_cols:list, text_cols:list):
        self.data = data
        self.target_names = target_names
        self.estimator_name = estimator_name
        self.vectorizer_params = None
        self.acoustic_cols = acoustic_cols
        self.embedding_cols = embed_cols
        self.text_cols = text_cols
        self.task = None
        self.metrics = {}

    def init_estimator(self):
        pass

    def select_embedding_feature_item(self):
        """
        Choose appropriate embedding features - on-the-fly or loaded from disk
        :return:
        """
        if self.embedding_cols is not None and self.embedding_cols != []:
            print('-- using precalculated embeddings')
            return ('embedding_features', Pipeline([('selector', ItemSelector(key=self.embedding_cols))]))
        else:
            print('-- using on-the-fly embeddings')
            return ('embedding_features', Pipeline([
                ('selector', ItemSelector(key='TEXT')),
                ('embedding_vectorizer', EmbeddingTransformer(embedding_model, aggregation='average'))
            ]))

    def init_pipeline(self, tfidf=True, embedding_model=None):
        if self.acoustic_cols is not None and self.acoustic_cols != []:
            print('-- number of acoustic features:', len(self.acoustic_cols))
            if self.text_cols is not None and self.text_cols != []:
                print('-- using text features', self.text_cols)
                if tfidf and embedding_model is not None:
                    if self.vectorizer_params is not None:
                        vectorizer = TfidfVectorizer(**self.vectorizer_params)
                    else:
                        vectorizer = TfidfVectorizer()
                    print('-- using acoustic, TFIDF and', embedding_model)
                    feature_union = FeatureUnion([
                        ('acoustic_features', Pipeline([
                            ('selector', ItemSelector(key=self.acoustic_cols)),
                            ('scaler', StandardScaler())
                        ])),
                        ('tfidf_features', Pipeline([
                            ('selector', ItemSelector(key='TEXT')),
                            ('vectorizer', vectorizer)
                        ])),
                        self.select_embedding_feature_item()
                    ])
                elif tfidf:
                    if self.vectorizer_params is not None:
                        vectorizer = TfidfVectorizer(**self.vectorizer_params)
                    else:
                        vectorizer = TfidfVectorizer()
                    print('-- using acoustic and TFIDF')
                    feature_union = FeatureUnion([
                        ('acoustic_features', Pipeline([
                            ('selector', ItemSelector(key=self.acoustic_cols)),
                            ('scaler', StandardScaler())
                        ])),
                        ('tfidf_features', Pipeline([
                            ('selector', ItemSelector(key='TEXT')),
                            ('vectorizer', vectorizer)
                        ]))
                    ])
                else:
                    print('-- using acoustic and', embedding_model)
                    feature_union = FeatureUnion([
                        ('acoustic_features', Pipeline([
                            ('selector', ItemSelector(key=self.acoustic_cols)),
                            ('scaler', StandardScaler())
                        ])),
                        self.select_embedding_feature_item()
                    ])
            else:
                print('-- using acoustic features only', sorted(self.acoustic_cols))
                feature_union = FeatureUnion([
                    ('acoustic_features', Pipeline([
                        ('selector', ItemSelector(key=self.acoustic_cols)),
                        ('scaler', StandardScaler())
                    ]))
                ])
        else:
            print('-- using text features only', self.text_cols)
            if tfidf and embedding_model is not None:
                if self.vectorizer_params is not None:
                    vectorizer = TfidfVectorizer(**self.vectorizer_params)
                else:
                    vectorizer = TfidfVectorizer()
                print('-- using TFIDF and', embedding_model)
                feature_union = FeatureUnion([
                    ('tfidf_features', Pipeline([
                        ('selector', ItemSelector(key='TEXT')),
                        ('vectorizer', vectorizer)
                    ])),
                    self.select_embedding_feature_item()
                ])
            elif tfidf:
                print('-- using TFIDF only', embedding_model)
                if self.vectorizer_params is not None:
                    vectorizer = TfidfVectorizer(**self.vectorizer_params)
                else:
                    vectorizer = TfidfVectorizer()
                feature_union = FeatureUnion([
                    ('tfidf_features', Pipeline([
                        ('selector', ItemSelector(key='TEXT')),
                        ('vectorizer', vectorizer)
                    ]))
                ])
            else:
                print('-- using', embedding_model, 'only')
                feature_union = FeatureUnion([
                    self.select_embedding_feature_item()
                ])
        estimator = self.init_estimator()
        pipeline = Pipeline([
            ('features', feature_union),
            ('estimator', estimator)
        ])
        self.set_metrics()
        return pipeline

    def load_lexicon_pattern(self, pin):
        lines = open(pin, 'r').read().split('\n')
        lines = [line.split()[0] for line in lines]
        regex = '(' + '|'.join(lines) + ')'
        if self.vectorizer_params is not None:
            self.vectorizer_params.update({'token_pattern': regex})
        else:
            self.vectorizer_params = {'token_pattern': regex}
        return regex

    def set_metrics(self):
        pass

    def get_Xy(self, target: str):
        feature_names = self.acoustic_cols + self.embedding_cols + self.text_cols
        X = self.data[feature_names]
        y = self.data[target]
        return X, y

    def grid_search(self, tfidf, embedding_model, n_splits):
        """
        Determine parameters for the TfIdfVectorizer
        NOT USING THIS
        :param tfidf:
        :param embedding_model:
        :param cv:
        :return:
        """
        params = {'features__tfidf_features__vectorizer__ngram_range': [(1, 1),
                                                                        (1, 2),
                                                                        (2, 2),
                                                                        (1, 3)],
                  'features__tfidf_features__vectorizer__max_features': [100, 500, 1000],
                  'features__tfidf_features__vectorizer__stop_words': ['english', None]
                  }
        pipeline = self.init_pipeline(tfidf=tfidf, embedding_model=embedding_model)
        gs_dict = {}
        for target in self.target_names:
            print('-- doing grid search for', target)
            X, y = self.get_Xy(target)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            gs = GridSearchCV(pipeline, params, scoring='f1_weighted', cv=skf, n_jobs=-1)
            gs.fit(X_train, y_train)
            gs_dict[target] = gs.best_params_
        return gs_dict

    def print_target_distribution(self):
        print('-- class distributions:')
        for t in self.target_names:
            print(t)
            print(self.data[t].value_counts())

    def process(self, tfidf=True, embedding_model=None, pout=None):
        print('-- number of rows:', len(self.data))
        pipeline = self.init_pipeline(tfidf=tfidf, embedding_model=embedding_model)
        results = {}
        all_preds = {}
        for t in self.target_names:
            X, y = self.get_Xy(t)
            # running cv twice is inefficient, but cross_val_predict is not reliable for evaluating generalisability
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            cv_scores = cross_validate(pipeline, X, y, cv=skf, scoring=self.metrics, n_jobs=-1)
            preds = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=-1)
            all_preds[t] = preds
            for m in self.metrics:
                scores = cv_scores['test_' + m]
                scores_mean = np.mean(scores)
                scores_std = np.std(scores)
                if self.task == 'regression':
                    scores_mean = -scores_mean
                    scores_std = -scores_std
                tmp = results.get(t, {})
                tmp[m] = scores_mean
                results[t] = tmp
        return results, all_preds


class RegressionPipeline(BasePipeline):
    def __init__(self, data: pd.DataFrame, target_names: list, estimator_name: str, embed_cols:list, acoustic_cols: list, text_cols: list):
        super().__init__(data, target_names, estimator_name, embed_cols, acoustic_cols, text_cols)
        self.task = 'regression'

    def init_estimator(self):
        print('-- regressor:', self.estimator_name)
        if self.estimator_name == 'lr':
            estimator = LinearRegression()
        elif self.estimator_name == 'mlp':
            estimator = MLPRegressor(random_state=SEED, max_iter=500)
        elif self.estimator_name == 'knn':
            estimator = KNeighborsRegressor(3, n_jobs=-1)
        elif self.estimator_name == 'rf':
            estimator = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        elif self.estimator_name == 'gb':
            estimator = GradientBoostingRegressor(random_state=SEED)
        else:
            estimator = LinearSVR(random_state=SEED, max_iter=3000)
        return estimator

    def set_metrics(self):
        self.metrics = {'mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
                        'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
                        'root_mean_squared_error': make_scorer(mean_squared_error, squared=False, greater_is_better=False)}


class ClassificationPipeline(BasePipeline):
    def __init__(self, data: pd.DataFrame, target_names: list, estimator_name: str, embed_cols:list, acoustic_cols: list, text_cols: list):
        super().__init__(data, target_names, estimator_name, embed_cols, acoustic_cols, text_cols)
        self.task = 'classification'

    def init_estimator(self):
        print('-- classifier:', self.estimator_name)
        if self.estimator_name == 'lr':
            estimator = LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=-1)
        elif self.estimator_name == 'mlp':
            estimator = MLPClassifier(random_state=SEED, max_iter=500)
        elif self.estimator_name == 'knn':
            estimator = KNeighborsClassifier(3, n_jobs=-1)
        elif self.estimator_name == 'rf':
            estimator = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        elif self.estimator_name == 'gb':
            estimator = GradientBoostingClassifier(random_state=SEED)
        else:
            estimator = LinearSVC(random_state=SEED, max_iter=3000)
        return estimator

    def set_metrics(self, average='weighted'):
        self.metrics = {'accuracy': make_scorer(accuracy_score, greater_is_better=True),
                        'precision': make_scorer(precision_score, greater_is_better=True, average=average),
                        'recall': make_scorer(recall_score, greater_is_better=True, average=average),
                        'f1-score': make_scorer(f1_score, greater_is_better=True, average=average)}


if __name__ == '__main__':
    """
    Example:
        python exp_fusion.py -t data/fmss_text.csv -e data/ERisk_coded_data_02Sep21.xlsx -a data/output_base_line/type-avec2013-seg-3600-hop-3600-avg -o data/fmss_text_type-avec2013-seg-3600-hop-3600-avg.csv -l warmey5 -m svm - x classification -u acoustic tfidf
    """

    parser = argparse.ArgumentParser(description='FMSS: baseline data preparation')
    parser.add_argument('-t', '--transcript_file', type=str, nargs=1,
                        help='path to turn-based text transcripts per audio file in CSV format', required=True)
    parser.add_argument('-e', '--erisk_codes_file', type=str, nargs=1,
                        help='path to Excel spreadsheet containing ERisk coded data', required=True)
    parser.add_argument('-a', '--acoustic_feature_dir', type=str, nargs=1,
                        help='path to directory containing baseline acoustic features', required=True)
    parser.add_argument('-w', '--word_embedding_dir', type=str, nargs=1,
                        help='path to directory containing word embedding features', required=False)
    parser.add_argument('-o', '--output_file', type=str, nargs=1,
                        help='path of CSV file to save all features to', required=False)
    parser.add_argument('-l', '--labels', type=str, nargs='+', help='names of the target label(s) to predict', required=True)
    parser.add_argument('-x', '--task', type=str, nargs=1, help='task type', choices=['classification', 'regression'],
                        required=True)
    parser.add_argument('-m', '--model', type=str, nargs=1, help='model to use (lr not valid for classification)',
                        choices=['lr', 'mlp', 'knn', 'rf', 'svm', 'gb'], required=True)
    parser.add_argument('-u', '--use_features', type=str, nargs='+', help='types of features to use. Choose from '
                                                                          '"acoustic" and/or one of "tfidf", "glove", '
                                                                          'or "word2vec"', required=True)
    #parser.add_argument('-g', '--grid_search', action='store_true', help='perform hyperparameter tuning using grid search', required=False)

    args = parser.parse_args()

    erisk_codes_file = args.erisk_codes_file[0]
    transcript_file = args.transcript_file[0]
    acoustic_features_dir = args.acoustic_feature_dir[0]
    word_embedding_dir = None
    if args.word_embedding_dir is not None:
        word_embedding_dir = args.word_embedding_dir[0]
    output_file = None
    #grid_search = args.grid_search
    if args.output_file is not None:
        output_file = args.output_file[0]
    target_labels = args.labels
    task = args.task[0]
    model = args.model[0]

    if task == 'classification':
        for t in target_labels:
            if not t.endswith('_cat'):
                raise ValueError("-- invalid feature name: " + t + ". Feature names for classification must end in '_cat' (e.g. 'warme5_cat')")
    elif task == 'regression':
        for t in target_labels:
            if t.endswith('_cat'):
                raise ValueError("-- invalid feature name:" + t + ". Feature names for regression must not end in '_cat' (e.g. 'warme5')")
    else:
        raise ValueError("-- invalid task:" + task + ". Choose 'classification' or 'regression'")

    dl = DataLoader(transcript_file, acoustic_features_dir, erisk_codes_file, word_embedding_dir, load_utterances_with_both_twins=True, load_both_speakers=True, merge_on='speaker')
    df_data = dl.process()

    text_features = []
    acoustic_features = []
    embed_features = []
    tfidf = 'tfidf' in args.use_features
    embedding_model = None

    if tfidf or 'glove' in args.use_features or 'word2vec' in args.use_features or 'fasttext' in args.use_features or 'bert' in args.use_features:
        text_features = ['TEXT']
    if 'acoustic' in args.use_features:
        acoustic_features = dl.get_acoustic_feature_names()
    if 'glove' in args.use_features:
        embedding_model = 'glove'
        embed_features = dl.get_embed_feature_names()
    elif 'word2vec' in args.use_features:
        embedding_model = 'word2vec'
        embed_features = dl.get_embed_feature_names()
    elif 'fasttext' in args.use_features:
        embedding_model = 'fasttext'
        embed_features = dl.get_embed_feature_names()
    #elif 'bert' in args.use_features:
    #    # TODO implement this
    #    embedding_model = 'bert'

    if task == 'classification':
        rp = ClassificationPipeline(df_data, target_labels, model, embed_features, acoustic_features, text_features)
    elif task == 'regression':
        rp = RegressionPipeline(df_data, target_labels, model, embed_features, acoustic_features, text_features)
    else:
        raise ValueError('-- incorrect task', task)

    if len(args.use_features) > 2:
        raise Exception('-- too many feature types specified. Choose at most 2, including only 1 text feature type.')
    if len([a for a in args.use_features if a != 'acoustic']) > 1:
        raise Exception('-- too many text feature types specified. Choose 1 only from "tfidf", "glove", "word2vec", "bert".')

    rp.print_target_distribution()

    df_results, preds = rp.process(tfidf=tfidf, embedding_model=embedding_model)
    print(df_results)
    print(preds)
    pd.DataFrame(preds).to_csv('../results/preds_exp.csv')