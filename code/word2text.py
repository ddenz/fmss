import argparse
import docx
import os
import pandas as pd
import re
import sys

from glob import glob


def word2txt(pin):
    """
    Convert Word document to text
    :param pin: path of Word document
    :return: path of the text file
    """
    pout = os.path.splitext(pin)[0] + '.txt'
    doc = docx.Document(pin)
    text = '\n'.join([p.text for p in doc.paragraphs])
    with open(pout, 'w') as fout:
        print(text, file=fout)
    print('-- wrote text file: {}'.format(pout))
    return pout


def segment(pin):
    """
    Segment a transcript file into speaker turns and timestamps
    :param pin: path of transcript text file
    :return: Pandas data frame containing all segments for the input transcript
    """
    records = []
    text = open(pin, 'r').read()
    turns = re.sub('\n+', '\n', re.sub(']\n', '] ', text)).split('\n')
    filename = turns[0]
    # print(filename)
    for i, line in enumerate(turns[1:]):
        m = re.search('([IP]) +\[([^ ]+) - ([^\]]+)\] +(.+)', line)
        if m is not None:
            speaker = m.group(1)
            start = m.group(2)
            end = m.group(3)
            text = m.group(4)
            # print('{}|{}|{}|{}'.format(speaker, i, start, end, text))
            records.append([filename, i, speaker, start, end, text])
    return pd.DataFrame(records, columns=['FILENAME', 'TURN', 'SPEAKER', 'START', 'END', 'TEXT'])


def process(pin):
    """
    Process file or batch process directory
    :param pin: input path (Word file or directory containing Word files)
    :return: Pandas data frame containing all segments for all transcripts in input path
    """
    df_data = pd.DataFrame()
    if os.path.isdir(pin):
        files = glob(pin + '/A*.doc*')
        for f in files:
            pout = word2txt(f)
            df_data = df_data.append(segment(pout))
        df_data.reset_index(drop=True, inplace=True)
    elif os.path.isfile(pin):
        pout = word2txt(pin)
        df_data = segment(pout)
    return df_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMSS: covert transcripts in Word format to plain text')
    parser.add_argument('-i', '--input_source', type=str, nargs=1,
                        help='path of file or directory containing transcripts in Word format', required=True)
    parser.add_argument('-o', '--output_file', type=str, nargs=1,
                        help='path of CSV file to save aggregated transcript data to', required=True)
    args = parser.parse_args()
    pin = args.input_source[0]
    pout = args.output_file[0]
    df_data = process(pin)
    df_data.to_csv(pout, sep=';', index=False)
    print('-- wrote all transcript data to {}'.format(pout))