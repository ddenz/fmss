from data_loader import DataLoader, ALL_TARGETS
from imblearn.over_sampling import RandomOverSampler


if __name__ == '__main__':
    dl = DataLoader('../data/fmss_transcripts.csv', 'P', '../data/output_base_line/type-avec2013-seg-3600-hop-3600-avg', '../data/ERisk_coded_data_02Sep21.csv')
    df_data = dl.prepare_data()
    X = df_data.drop(ALL_TARGETS, axis=1)
    y = df_data['warme5_cat']
    ros = RandomOverSampler(random_state=7)
    X_r, y_r = ros.fit_resample(X, y)
    print(len(X), len(X_r))
    print(len(y), len(y_r))
    print(X_r)
