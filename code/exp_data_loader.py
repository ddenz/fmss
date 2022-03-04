import argparse
import os
import sys

import pandas as pd
import re

from glob import glob

FILE_HDR = 'FILENAME'
SPEAKER_HDR = 'SPEAKER'
TEXT_HDR = 'TEXT'
TWINID_HDR = 'atwinid'

REQ_COLUMNS = [FILE_HDR, TWINID_HDR]
TARGET_NAMES = ['warme5', 'disse5']


class DataLoader(object):
    """

    """
    def __init__(self, pin_transcripts, pin_audio, pin_targets, pin_embed, load_utterances_with_both_twins=False, load_both_speakers=False, merge_on='speaker'):
        self.pin_transcripts = pin_transcripts
        self.pin_audio = pin_audio
        self.pin_targets = pin_targets
        self.pin_embed = pin_embed
        self.load_utterances_with_both_twins = load_utterances_with_both_twins
        self.load_both_speakers = load_both_speakers
        self.merge_on = merge_on
        self.acoustic_feature_names = []
        self.embed_feature_names = []

    def set_pin_embed(self, pin):
        self.pin_embed = pin

    def get_acoustic_feature_names(self):
        return self.acoustic_feature_names

    def get_embed_feature_names(self):
        return self.embed_feature_names

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

    def load_targets(self):
        """

        :return:
        """
        df = pd.read_csv(self.pin_targets)
        df = df[REQ_COLUMNS + TARGET_NAMES]
        # convert all twin ids to 1 (elder twin) or 2 (younger twin)
        df.loc[(df.atwinid % 2 != 0), TWINID_HDR] = 1
        df.loc[(df.atwinid % 2 == 0), TWINID_HDR] = 2
        return df

    def load_transcripts(self):
        """

        :return:
        """
        try:
            df = pd.read_csv(self.pin_transcripts)
        except pd.errors.ParserError:
            df = pd.read_csv(self.pin_transcripts, sep=';')

        df[TEXT_HDR].fillna(value='', inplace=True)

        if self.merge_on == 'speaker':
            # merge all utterances for a speaker (i.e. int, mum) for each twin
            if self.load_both_speakers:
                df_int = pd.DataFrame(df.loc[df[SPEAKER_HDR].str.contains('int')]
                                      .groupby([FILE_HDR, TWINID_HDR])[TEXT_HDR]
                                      .apply(' '.join))
            else:
                # create empty data frame if we are not including interviewer utterances
                df_int = pd.DataFrame(columns=[TEXT_HDR])
            df_mum = pd.DataFrame(df.loc[df[SPEAKER_HDR].str.contains('mum')]
                                  .groupby([FILE_HDR, TWINID_HDR])[TEXT_HDR]
                                  .apply(' '.join))

            if self.load_utterances_with_both_twins:
                df_both = pd.DataFrame(df.loc[df[TWINID_HDR] == 3]
                                       .groupby([FILE_HDR])[TEXT_HDR]
                                       .apply(' '.join))
                if self.load_both_speakers:
                    df_int[TEXT_HDR] = df_both[TEXT_HDR] + ' ' + df_int[TEXT_HDR]
                df_mum[TEXT_HDR] = df_both[TEXT_HDR] + ' ' + df_mum[TEXT_HDR]

            if self.load_both_speakers:
                df_int[SPEAKER_HDR] = 'int'
            df_mum[SPEAKER_HDR] = 'mum'

            df = df_mum.append(df_int).reset_index()
            # merge int and mum for each twin to get one row per twin
            df = df.groupby([FILE_HDR, TWINID_HDR])[TEXT_HDR].apply(' '.join).reset_index()
            # remove rows with atwinid == 3 if present
            df = df.loc[df[TWINID_HDR] != 3]
        elif self.merge_on == 'utterance':
            # merge all utterances of a given tag type (e.g. int-both-general, mum-t1-general) for each twin
            df_utt = pd.DataFrame(df.groupby([FILE_HDR, TWINID_HDR, SPEAKER_HDR])[TEXT_HDR].apply(' '.join))
            df_utt.reset_index(inplace=True)
            # remove rows with atwinid == 3 if present
            df_utt = df_utt.loc[df_utt[TWINID_HDR] != 3]

            if self.load_utterances_with_both_twins:
                # retrieve and join text for both-type utterances
                df_both = pd.DataFrame(df.loc[df[TWINID_HDR] == 3].groupby([FILE_HDR, SPEAKER_HDR])[TEXT_HDR]
                                       .apply(' '.join))
                df_both.reset_index(inplace=True)
                # append the rows for each twin
                df_tmp = df_both.copy()
                df_tmp[TWINID_HDR] = 1
                df_utt = df_utt.append(df_tmp)
                df_tmp[TWINID_HDR] = 2
                df_utt = df_utt.append(df_tmp)

            df = df_utt
            df.reset_index(drop=True, inplace=True)

        return df

    def load_audio_features(self):
        """

        :return:
        """

        def transpose_audio_features(file, twinid=-1):
            """

            :param file:
            :param twinid:
            :return:
            """
            df = pd.read_csv(file, sep=';')
            vals_new = []
            cols_new = []
            features = [f for f in df.stat.unique() if 'AVG' in f or 'STD' in f] # only use AVE and STD features
            for s in features:
                row = df.loc[df.stat == s].iloc[:, 2:]
                vals = row.values[0]
                new_cols = [c + '_' + re.sub('-t[12]', '-twin', s) for c in df.columns[2:]]
                self.acoustic_feature_names += new_cols
                vals_new.extend(vals)
                cols_new.extend(new_cols)
            d = {FILE_HDR: df['name'], TWINID_HDR: twinid}
            d.update({k: float(v) for (k, v) in zip(cols_new, vals_new)})
            df_tmp = pd.DataFrame(d, index=[0])
            return df_tmp

        files_t1 = glob(self.pin_audio + '/A*twin-0*.csv')
        files_t2 = glob(self.pin_audio + '/A*twin-1*.csv')

        df_global = pd.DataFrame()

        # elder twin
        for file in files_t1:
            df_tmp = transpose_audio_features(file, twinid=1)
            df_global = df_global.append(df_tmp)

        # younger twin
        for file in files_t2:
            df_tmp = transpose_audio_features(file, twinid=2)
            df_global = df_global.append(df_tmp)

        # both twins
        if self.load_utterances_with_both_twins:
            files_both = glob(self.pin_audio + '/A*both*.csv')
            for file in files_both:
                df_tmp = transpose_audio_features(file, twinid=3)
                df_global = df_global.append(df_tmp)

        # remove both-type features and global stats which are identical for both twins
        #if not self.load_utterances_with_both_twins:
        #    cols_to_drop = [col for col in df_global.columns if 'both' in col] + \
        #                   [col for col in df_global.columns if 'twin' not in col and
        #                    col not in (REQ_COLUMNS + TARGET_NAMES)]
        #    df_global.drop(cols_to_drop, axis=1, inplace=True)
        #    self.acoustic_feature_names = list(set(self.acoustic_feature_names) - set(cols_to_drop))

        df_global.fillna(value=0, inplace=True)
        df_global.sort_values(by=[FILE_HDR, TWINID_HDR], inplace=True)
        df_global.reset_index(inplace=True, drop=True)

        return df_global

    def load_word_embedding_features(self):
        files_t1 = glob(self.pin_embed + '/A*twin-1*.csv')
        files_t2 = glob(self.pin_embed + '/A*twin-2*.csv')

        df_global = pd.DataFrame()

        # elder twin
        for file in files_t1:
            df_tmp = pd.read_csv(file)
            df_tmp[TWINID_HDR] = 1
            df_tmp[FILE_HDR] = os.path.basename(re.sub('_twin.+', '', file))
            df_global = df_global.append(df_tmp)

        # younger twin
        for file in files_t2:
            df_tmp = pd.read_csv(file)
            df_tmp[TWINID_HDR] = 2
            df_tmp[FILE_HDR] = os.path.basename(re.sub('_twin.+', '', file))
            df_global = df_global.append(df_tmp)

        self.embed_feature_names = [c for c in df_global.columns if 'dim_' in c]
        df_global.fillna(value=0, inplace=True)
        df_global.sort_values(by=[FILE_HDR, TWINID_HDR], inplace=True)
        df_global.reset_index(inplace=True, drop=True)

        return df_global

    def merge_transcripts_and_targets(self):
        """

        :return:
        """
        df_targets = self.load_targets()
        df_transcripts = self.load_transcripts()
        return df_transcripts.merge(df_targets, on=[FILE_HDR, TWINID_HDR])

    def process(self, pout=None):
        """

        :return:
        """
        df_targets = self.load_targets()
        df_transcripts = self.load_transcripts()
        df_merged = self.merge_transcripts_and_targets()

        df_audio = self.load_audio_features()
        df_all = df_merged.merge(df_audio, on=[FILE_HDR, TWINID_HDR])

        if self.pin_embed is not None:
            df_embed = self.load_word_embedding_features()
            df_all = df_all.merge(df_embed, on=[FILE_HDR, TWINID_HDR])

        # remove features that have identical values across the data set
        cols_to_drop = df_all.columns[df_all.nunique() <= 1]
        df_all.drop(cols_to_drop, axis=1, inplace=True)
        # sort this to ensure reproducibility as some models are affected by column/feature order
        self.acoustic_feature_names = set(self.acoustic_feature_names) - set(self.embed_feature_names)
        self.acoustic_feature_names = sorted(list(set(self.acoustic_feature_names) - set(cols_to_drop)))

        # add categorical variables here - NB using 0 as lower bound converts to NaN so using -0.000001
        _ = self.to_categorical(df_all, 'warme5', bins=[-0.000001, 3.999, 4.999, 5], labels=['low', 'moderate', 'high'])
        #_ = self.to_categorical(df_all, 'warme5', bins=[-0.000001, 0.999, 1.999, 2.999, 3.999, 5], labels=['1', '2', '3', '4', '5'])

        df_all.sort_index(axis=1, inplace=True)
        df_all.reset_index(drop=True, inplace=True)

        if pout is not None:
            df_all.to_csv(pout, sep=';', index=False)
            print('-- wrote all aggregated features:', pout)

        return df_all


if __name__ == '__main__':
    # display options
    pd.set_option('display.width', 100)
    pd.set_option('display.max_columns', 10)

    parser = argparse.ArgumentParser(description='FMSS: experiment data preparation')
    parser.add_argument('-t', '--transcript_file', type=str, nargs=1,
                        help='path to turn-based text transcripts per audio file in CSV format', required=True)
    parser.add_argument('-e', '--erisk_codes_file', type=str, nargs=1,
                        help='path to Excel spreadsheet containing ERisk coded data', required=False)
    parser.add_argument('-w', '--word_embed_dir', type=str, nargs=1,
                        help='path to directory containing word embedding features', required=False)
    parser.add_argument('-a', '--acoustic_feature_dir', type=str, nargs=1,
                        help='path to directory containing baseline acoustic features', required=False)
    parser.add_argument('-o', '--output_file', type=str, nargs=1,
                        help='path of CSV file to save all features to', required=False)

    args = parser.parse_args()

    erisk_codes_file = None
    if args.erisk_codes_file is not None:
        erisk_codes_file = args.erisk_codes_file[0]
    transcript_file = args.transcript_file[0]
    acoustic_features_dir = None
    if args.acoustic_feature_dir is not None:
        acoustic_features_dir = args.acoustic_feature_dir[0]
    word_embed_dir = None
    if args.word_embed_dir is not None:
        word_embed_dir = args.word_embed_dir[0]
    output_file = None
    if args.output_file is not None:
        output_file = args.output_file[0]

    dl = DataLoader(transcript_file, acoustic_features_dir, erisk_codes_file, word_embed_dir, load_utterances_with_both_twins=False, load_both_speakers=False, merge_on='speaker')
    df = dl.process(pout=output_file)
    print(df)
