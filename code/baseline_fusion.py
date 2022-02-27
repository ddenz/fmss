import argparse
import numpy as np
import pandas as pd

from bert_transformer import *
from data_loader import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC

from zeugma import EmbeddingTransformer, ItemSelector


class BasePipeline(object):
    def __init__(self, data: pd.DataFrame, target_names: list, estimator_name: str, acoustic_cols:list, text_cols:list):
        self.data = data
        self.target_names = target_names
        self.estimator_name = estimator_name
        self.acoustic_cols = acoustic_cols
        self.text_cols = text_cols
        self.task = None
        self.metrics = {}

    def init_estimator(self):
        pass

    def init_pipeline(self, tfidf=True, embedding_model=None):
        if self.acoustic_cols is not None and self.acoustic_cols != []:
            print('-- number of acoustic features:', len(self.acoustic_cols))
            if self.text_cols is not None and self.text_cols != []:
                print('-- using text features', self.text_cols)
                if tfidf and embedding_model is not None:
                    print('-- using acoustic, TFIDF and', embedding_model)
                    feature_union = FeatureUnion([
                        ('acoustic_features', Pipeline([
                            ('selector', ItemSelector(key=self.acoustic_cols))
                        ])),
                        ('tfidf_features', Pipeline([
                            ('selector', ItemSelector(key='TEXT')),
                            ('vectorizer', TfidfVectorizer())
                        ])),
                        ('embedding_features', Pipeline([
                            ('selector', ItemSelector(key='TEXT')),
                            ('embedding_vectorizer', EmbeddingTransformer(embedding_model, aggregation='average'))
                        ]))
                    ])
                elif tfidf:
                    print('-- using acoustic and TFIDF')
                    feature_union = FeatureUnion([
                        ('acoustic_features', Pipeline([
                            ('selector', ItemSelector(key=self.acoustic_cols))
                        ])),
                        ('tfidf_features', Pipeline([
                            ('selector', ItemSelector(key='TEXT')),
                            ('vectorizer', TfidfVectorizer())
                        ]))
                    ])
                else:
                    print('-- using acoustic and', embedding_model)
                    feature_union = FeatureUnion([
                        ('acoustic_features', Pipeline([
                            ('selector', ItemSelector(key=self.acoustic_cols))
                        ])),
                        ('embedding_features', Pipeline([
                            ('selector', ItemSelector(key='TEXT')),
                            ('embedding_vectorizer', EmbeddingTransformer(embedding_model, aggregation='average'))
                        ]))
                    ])
            else:
                print('-- using acoustic features only', self.acoustic_cols)
                feature_union = FeatureUnion([
                    ('acoustic_features', Pipeline([
                        ('selector', ItemSelector(key=self.acoustic_cols))
                    ]))
                ])
        else:
            print('-- using text features only', self.text_cols)
            if tfidf and embedding_model is not None:
                print('-- using TFIDF and', embedding_model)
                feature_union = FeatureUnion([
                    ('tfidf_features', Pipeline([
                        ('selector', ItemSelector(key='TEXT')),
                        ('vectorizer', TfidfVectorizer())
                    ])),
                    ('embedding_features', Pipeline([
                        ('selector', ItemSelector(key='TEXT')),
                        ('embedding_vectorizer', EmbeddingTransformer(embedding_model, aggregation='average'))
                    ]))
                ])
            elif tfidf:
                print('-- using TFIDF only', embedding_model)
                feature_union = FeatureUnion([
                    ('tfidf_features', Pipeline([
                        ('selector', ItemSelector(key='TEXT')),
                        ('vectorizer', TfidfVectorizer())
                    ]))
                ])
            else:
                print('-- using', embedding_model, 'only')
                feature_union = FeatureUnion([
                    ('embedding_features', Pipeline([
                        ('selector', ItemSelector(key='TEXT')),
                        ('embedding_vectorizer', EmbeddingTransformer(embedding_model, aggregation='average'))
                    ]))
                ])
        estimator = self.init_estimator()
        pipeline = Pipeline([
            ('features', feature_union),
            ('estimator', estimator)
        ])
        self.set_metrics()
        return pipeline

    def set_metrics(self):
        pass

    def get_Xy(self, target: str):
        feature_names = self.acoustic_cols + self.text_cols
        X = self.data[feature_names]
        y = self.data[target]
        return X, y

    def process(self, tfidf=True, embedding_model=None, pout=None):
        print('-- number of rows:', len(self.data))
        pipeline = self.init_pipeline(tfidf=tfidf, embedding_model=embedding_model)
        results = {}
        all_preds = {}
        for t in self.target_names:
            X, y = self.get_Xy(t)
            # running cv twice is inefficient, but cross_val_predict is not reliable for evaluating generalisability
            cv_scores = cross_validate(pipeline, X, y, cv=10, scoring=self.metrics)
            preds = cross_val_predict(pipeline, X, y, cv=10)
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
    def __init__(self, data: pd.DataFrame, target_names: list, estimator_name: str, acoustic_cols: list, text_cols: list):
        super().__init__(data, target_names, estimator_name, acoustic_cols, text_cols)
        self.task = 'regression'

    def init_estimator(self):
        print('-- regressor:', self.estimator_name)
        if self.estimator_name == 'lr':
            estimator = LinearRegression()
        elif self.estimator_name == 'mlp':
            estimator = MLPRegressor(random_state=0, max_iter=500)
        elif self.estimator_name == 'knn':
            estimator = KNeighborsRegressor(3)
        elif self.estimator_name == 'rf':
            estimator = RandomForestRegressor()
        elif self.estimator_name == 'gb':
            estimator = GradientBoostingRegressor()
        else:
            estimator = SVR()
        return estimator

    def set_metrics(self):
        self.metrics = {'mean_absolute_error': make_scorer(mean_absolute_error, greater_is_better=False),
                        'mean_squared_error': make_scorer(mean_squared_error, greater_is_better=False),
                        'root_mean_squared_error': make_scorer(mean_squared_error, squared=False, greater_is_better=False)}


class ClassificationPipeline(BasePipeline):
    def __init__(self, data: pd.DataFrame, target_names: list, estimator_name: str, acoustic_cols: list, text_cols: list):
        super().__init__(data, target_names, estimator_name, acoustic_cols, text_cols)
        self.task = 'classification'

    def init_estimator(self):
        print('-- classifier:', self.estimator_name)
        if self.estimator_name == 'mlp':
            estimator = MLPClassifier(random_state=0, max_iter=500)
        elif self.estimator_name == 'knn':
            estimator = KNeighborsClassifier(3)
        elif self.estimator_name == 'rf':
            estimator = RandomForestClassifier()
        elif self.estimator_name == 'gb':
            estimator = GradientBoostingClassifier()
        else:
            estimator = SVC()
        return estimator

    def set_metrics(self):
        self.metrics = {'accuracy': make_scorer(accuracy_score, greater_is_better=True),
                   'precision': make_scorer(precision_score, greater_is_better=True, average='macro'),
                   'recall': make_scorer(recall_score, greater_is_better=True, average='macro'),
                   'f1-score': make_scorer(f1_score, greater_is_better=True, average='macro')}


if __name__ == '__main__':
    """
    Example:
        python baseline_fusion.py -s P -t data/fmss_text.csv -e data/ERisk_coded_data_02Sep21.xlsx -a data/output_base_line/type-avec2013-seg-3600-hop-3600-avg -o data/fmss_text_type-avec2013-seg-3600-hop-3600-avg.csv -l warmey5 -r svm
    """

    parser = argparse.ArgumentParser(description='FMSS: baseline data preparation')
    parser.add_argument('-s', '--speakers', type=str, nargs=1, help='speakers to include', choices=['P', 'both'],
                        required=True)
    parser.add_argument('-t', '--transcript_file', type=str, nargs=1,
                        help='path to turn-based text transcripts per audio file in CSV format', required=True)
    parser.add_argument('-e', '--erisk_codes_file', type=str, nargs=1,
                        help='path to Excel spreadsheet containing ERisk coded data', required=True)
    parser.add_argument('-a', '--acoustic_feature_dir', type=str, nargs=1,
                        help='path to directory containing baseline acoustic features', required=True)
    parser.add_argument('-o', '--output_file', type=str, nargs=1,
                        help='path of CSV file to save all features to', required=False)
    parser.add_argument('-l', '--labels', type=str, nargs='+', help='names of the target label(s) to predict', required=True)
    parser.add_argument('-x', '--task', type=str, nargs=1, help='task type', choices=['classification', 'regression'],
                        required=True)
    parser.add_argument('-m', '--model', type=str, nargs=1, help='model to use (lr not valid for classification)',
                        choices=['lr', 'mlp', 'knn', 'rf', 'svm', 'gb'], required=True)
    parser.add_argument('-u', '--use_features', type=str, nargs='+', help='types of features to use. Choose from '
                                                                          '"acoustic" and/or one of "tfidf", "glove", '
                                                                          '"word2vec", or "bert"', required=True)

    args = parser.parse_args()

    speakers = args.speakers[0]
    erisk_codes_file = args.erisk_codes_file[0]
    transcript_file = args.transcript_file[0]
    acoustic_features_dir = args.acoustic_feature_dir[0]
    output_file = None
    if args.output_file is not None:
        output_file = args.output_file[0]
    target_labels = args.labels
    task = args.task[0]
    model = args.model[0]

    if task == 'classification':
        if model == 'lr':
            raise ValueError('-- invalid model for classification:', model)
        for t in target_labels:
            if not t.endswith('_cat'):
                raise ValueError("-- invalid feature name: " + t + ". Feature names for classification must end in '_cat' (e.g. 'warme5_cat')")
    elif task == 'regression':
        for t in target_labels:
            if t.endswith('_cat'):
                raise ValueError("-- invalid feature name:" + t + ". Feature names for regression must not end in '_cat' (e.g. 'warme5')")
    else:
        raise ValueError("-- invalid task:" + task + ". Choose 'classification' or 'regression'")

    dl = DataLoader(transcript_file, speakers, pin_acoustics=acoustic_features_dir, pin_targets=erisk_codes_file)

    text_features = []
    acoustic_features = []
    df_data = pd.DataFrame()

    if 'tfidf' in args.use_features or 'glove' in args.use_features or 'word2vec' in args.use_features or 'bert' in args.use_features:
        text_features = ['TEXT']
    if 'acoustic_full' in args.use_features:
        df_ac = dl.aggregate_acoustic_features()
        df_data = dl.merge_acoustic_features_and_targets(df_ac)
        acoustic_features = dl.get_acoustic_feature_names()
    elif 'acoustic' in args.use_features:
        df_data = dl.prepare_data(pout=output_file)
        acoustic_features = dl.get_acoustic_feature_names()
    else:
        df_data = dl.prepare_data(pout=output_file)

    if task == 'classification':
        rp = ClassificationPipeline(df_data, target_labels, model, acoustic_features, text_features)
    else:
        rp = RegressionPipeline(df_data, target_labels, model, acoustic_features, text_features)

    if len(args.use_features) > 2:
        raise Exception('-- too many feature types specified. Choose at most 2, including only 1 text feature type.')
    if len([a for a in args.use_features if a != 'acoustic']) > 1:
        raise Exception('-- too many text feature types specified. Choose 1 only from "tfidf", "glove", "word2vec", "bert".')

    # TODO get text vectorizer type

    df_results, preds = rp.process(tfidf='tfidf' in args.use_features, embedding_model='glove')
    print(df_results)
    print(preds)
    pd.DataFrame(preds).to_csv('../results/preds.csv')