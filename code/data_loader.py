import argparse
import numpy as np
import pandas as pd

from glob import glob

ALL_TARGETS = ['totexte5', 'totinte5', 'negate5', 'posite5', 'warme5', 'disse5', 'favorite5', 'totee5', 'warmee5',
               'wrm50e5', 'negaty5', 'posity5', 'warmy5', 'dissy5', 'favority5', 'totey5', 'warmey5', 'wrm50y5',
               'fhanypm12', 'cdie12', 'masce12', 'totadde12']


class DataLoader(object):
    """"
    Manage loading and merging of data sources
    """
    def __init__(self, pin_transcripts, speakers, pin_acoustics=None, pin_targets=None):
        self.pin_targets = pin_targets
        self.pin_transcripts = pin_transcripts
        self.pin_acoustics = pin_acoustics
        self.speakers = speakers.lower()
        self.acoustic_feature_names = []

    def load_sequential_acoustic_features(self, pin):
        """
        Load acoustic features for sequential models (i.e. not aggregated per audio file)
        :param pin: input directory
        :return: Pandas data frame containing audio features
        """
        print('-- loading sequential acoustic features')
        ac_features = []
        csvs = glob(pin + '/A*.csv')
        self.acoustic_feature_names = pd.read_csv(csvs[0], sep=';').drop('name', axis=1).columns
        for csv in csvs:
            af = pd.read_csv(csv, sep=';', skiprows=[0], header=None)
            ac_features.append(af.iloc[: , 1:].to_numpy())
        return ac_features

    def load_transcripts(self, merge_speakers):
        """
        Load all transcripts into a Pandas data frame
        :param merge_speakers:
        :return: a Pandas data frame containing the transcripts
        """
        try:
            df_transcripts = pd.read_csv(self.pin_transcripts)
        except pd.errors.ParserError:
            df_transcripts = pd.read_csv(self.pin_transcripts, sep=';')
        if self.speakers == 'p':
            if merge_speakers:
                df_merged = df_transcripts.loc[df_transcripts.SPEAKER == 'P'].groupby('FILENAME')['TEXT']\
                    .apply(' '.join).reset_index()
            else:
                df_merged = df_transcripts.loc[df_transcripts.SPEAKER == 'P']
        elif self.speakers == 'both':
            if merge_speakers:
                df_merged = df_transcripts.groupby('FILENAME')['TEXT'].apply(' '.join).reset_index()
            else:
                return df_transcripts
        else:
            raise Exception("incorrect speaker (must be 'p' or 'both'): " + self.speakers)
        return df_merged

    def merge_transcripts_and_targets(self, merge_speakers=False, pout=None):
        """
        Merge transcript text data with ERisk target data
        :param pout: path of output CSV file containing text and targets
        :return: Pandas data frame of merged data
        """
        try:
            df_transcripts = pd.read_csv(self.pin_transcripts)
        except pd.errors.ParserError:
            df_transcripts = pd.read_csv(self.pin_transcripts, sep=';')
        # df_targets = pd.read_excel(self.pin_targets)
        try:
            df_targets = pd.read_csv(self.pin_targets)
        except pd.errors.ParserError:
            df_targets = pd.read_csv(self.pin_targets, sep=';')

        # set keys and actions for aggregation (i.e. save first values of non-numeric columns and average numeric ones)
        # agg_keys = {k: 'first' for k in df_targets.columns.drop(ALL_TARGETS).tolist()}
        # agg_keys.update({k: 'mean' for k in ALL_TARGETS})
        # select first row for all variables
        agg_keys = {k: 'first' for k in df_targets.columns}
        df_targets = df_targets.groupby('familyid').agg(agg_keys).reset_index(drop=True)

        if self.speakers == 'p':
            if merge_speakers:
                df_merged = df_transcripts.loc[df_transcripts.SPEAKER == 'P'].groupby('FILENAME')['TEXT']\
                    .apply(' '.join).reset_index().merge(df_targets, on='FILENAME')
            else:
                df_merged = df_transcripts.loc[df_transcripts.SPEAKER == 'P'].merge(df_targets, on='FILENAME')
        elif self.speakers == 'both':
            if merge_speakers:
                df_merged = df_transcripts.groupby('FILENAME')['TEXT'].apply(' '.join).reset_index()\
                    .merge(df_targets, on='FILENAME')
            else:
                df_merged = df_transcripts.merge(df_targets, on='FILENAME')
        else:
            raise Exception("incorrect speaker (must be 'p' or 'both'): " + self.speakers)

        if pout is not None:
            df_merged.to_csv(pout,sep=';', index=False)
            print('-- Wrote merged data:', pout)

        return df_merged

    def aggregate_acoustic_features(self):
        """
        Aggregate all acoustic features into single CSV files
        :return: Pandas data frame containing all aggregated acoustic features for all samples
        """
        if self.pin_acoustics is None:
            return pd.DataFrame()
        csvs = glob(self.pin_acoustics + '/A*.csv')
        df_af = pd.DataFrame()
        for csv in csvs:
            df = pd.read_csv(csv, sep=';', index_col=None)
            df_af = df_af.append(df)
        df_af.reset_index(inplace=True)
        df_af.drop(['idx', 'segNo'], axis=1, inplace=True)
        df_af = df_af.rename(columns={'name': 'FILENAME'})
        df_af['FILENAME'] = df_af.FILENAME.str.replace("'", '')
        cols = df_af.drop('FILENAME', axis=1).columns
        df_af[cols] = df_af[cols].apply(pd.to_numeric)
        self.acoustic_feature_names = df_af.columns.drop('FILENAME').tolist()
        return df_af

    def merge_acoustic_features_and_targets(self, df_ac, pout=None):
        """
        Merge acoustic features with ERisk target data
        :param df_ac: Pandas data frame containing aggregated acoustic features
        :param pout: path of output CSV file containing acoustic features and targets
        :return: Pandas data frame of merged data
        """
        try:
            df_targets = pd.read_csv(self.pin_targets)
        except pd.errors.ParserError:
            df_targets = pd.read_csv(self.pin_targets, sep=';')

        # set keys and actions for aggregation (i.e. save first values of non-numeric columns and average numeric ones)
        agg_keys = {k: 'first' for k in df_targets.columns.drop(ALL_TARGETS).tolist()}
        agg_keys.update({k: 'first' for k in ALL_TARGETS})
        df_targets = df_targets.groupby('familyid').agg(agg_keys).reset_index(drop=True)

        df_merged = df_ac.merge(df_targets, on='FILENAME')

        return df_merged

    def merge_all_features(self, df_audio, df_all, pout=None):
        """
        Merge acoustic, text and targets into a single Pandas data frame
        :param df_audio: Pandas data frame containing acoustic features
        :param df_all: Pandas data frame containing text features and targets
        :param pout: path to save CSV file to
        :return: Pandas data frame containing all features
        """
        cols_all = list(df_all.drop(ALL_TARGETS, axis=1).columns)
        cols_af = []
        if 'FILENAME' in df_audio.columns:
            cols_af = list(df_audio.drop('FILENAME', axis=1).columns)
            df_all = df_all.merge(df_audio, on='FILENAME')[cols_all + cols_af + ALL_TARGETS]
        else:
            df_all = df_all[cols_all + cols_af + ALL_TARGETS]

        if pout is not None:
            df_all.to_csv(pout, sep=';', index=False)
            print('-- Wrote all aggregated features:', pout)

        return df_all

    def to_categorical(self, df, feature, bins, labels):
        """
        Make a continuous variable categorical and add to the data
        :param df: Pandas data frame containing the data with continuous feature
        :param feature: continuous feature
        :param bins: numerical ranges to indicate categorie divisions
        :param labels: labels to assign to each category in the range
        :return: the name of the new categorical column
        """
        cat_name = feature + '_cat'
        category = pd.cut(df[feature], bins=bins, labels=labels)
        df.insert(df.columns.get_loc(feature) + 1, cat_name, category)
        print('-- added categorical variable:', cat_name)
        return cat_name

    def prepare_data(self, pout=None):
        """
        Prepare all data
        :param pout: output file path for all features
        :return: Pandas data frame containing all features and target variables
        """
        df_merged = self.merge_transcripts_and_targets(merge_speakers=True)
        df_acc = self.aggregate_acoustic_features()
        df_all = self.merge_all_features(df_acc, df_merged, pout)
        # add categorical variables here - NB using 0 as lower bound converts to NaN so using -0.000001
        _ = self.to_categorical(df_all, 'warme5', bins=[-0.000001, 2.999, 3.999, 5], labels=['low', 'moderate', 'high'])
        return df_all

    def get_acoustic_feature_names(self):
        return self.acoustic_feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMSS: data preparation')
    parser.add_argument('-s', '--speakers', type=str, nargs=1, help='speakers to include', choices=['P', 'both'],
                        required=True)
    parser.add_argument('-t', '--transcript_file', type=str, nargs=1,
                        help='path to turn-based text transcripts per audio file in CSV format', required=True)
    parser.add_argument('-e', '--erisk_codes_file', type=str, nargs=1,
                        help='path to Excel spreadsheet containing ERisk coded data', required=False)
    parser.add_argument('-a', '--acoustic_feature_dir', type=str, nargs=1,
                        help='path to directory containing baseline acoustic features', required=False)
    parser.add_argument('-o', '--output_file', type=str, nargs=1,
                        help='path of CSV file to save all features to', required=False)
    # parser.add_argument('-l', '--labels', type=str, nargs='+', help='list of names of the target labels to predict', required=True)

    args = parser.parse_args()

    speakers = args.speakers[0]
    erisk_codes_file = None
    if args.erisk_codes_file is not None:
        erisk_codes_file = args.erisk_codes_file[0]
    transcript_file = args.transcript_file[0]
    acoustic_features_dir = None
    if args.acoustic_feature_dir is not None:
        acoustic_features_dir = args.acoustic_feature_dir[0]
    output_file = None
    if args.output_file is not None:
        output_file = args.output_file[0]
    # target_label = args.labels

    dl = DataLoader(transcript_file, speakers, pin_acoustics=acoustic_features_dir, pin_targets=erisk_codes_file)
    df_data = dl.prepare_data(pout=output_file)
    print(df_data)