import argparse
import os
import pandas as pd

from glob import glob


def segment(pin):
    """
    Segment a transcript file into labelled speaker turns and timestamps
    :param pin: path of transcript text file
    :return: Pandas data frame containing all segments for the input transcript
    """
    filename = '_'.join(os.path.basename(pin).split('_')[0:2]) + '.csv'
    records = []
    lines = open(pin, 'r').read().split('\n')
    for i, line in enumerate(lines):
        if len(line) < 1:
            continue
        print(line)
        start, end, text = line.split('\t')
        speaker, text = text.split(':')
        # print('{}|{}|{}|{}|{}'.format(filename, speaker, i, start, end, text))
        records.append([filename, i, speaker, start, end, text])
    return pd.DataFrame(records, columns=['FILENAME', 'TURN', 'SPEAKER', 'START', 'END', 'TEXT'])


def process(pin):
    """
    Process file or batch process directory
    :param pin: input path (labelled Audacity file or directory containing labelled Audacity files)
    :return: Pandas data frame containing all segments for all transcripts in input path
    """
    df_data = pd.DataFrame()
    if os.path.isdir(pin):
        files = glob(pin + '/A*labels.txt')
        for f in files:
            df_data = df_data.append(segment(f))
        df_data.reset_index(drop=True, inplace=True)
    elif os.path.isfile(pin):
        df_data = segment(pout)
    return df_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMSS: covert transcripts in Audacity labelled format to CSV')
    parser.add_argument('-i', '--input_source', type=str, nargs=1,
                        help='path of file or directory containing transcripts in Audacity format', required=True)
    parser.add_argument('-o', '--output_file', type=str, nargs=1,
                        help='path of CSV file to save aggregated transcript data to', required=True)
    args = parser.parse_args()
    pin = args.input_source[0]
    pout = args.output_file[0]
    df_data = process(pin)
    df_data.to_csv(pout, sep=';', index=False)
    print('-- wrote all transcript data to {}'.format(pout))