import os
import re
import sys

from glob import glob


def get_sec(time_str):
    m = re.search('([0-9][0-9]):([0-9][0-9]):([0-9]+\.[0-9]+)', time_str)
    if m is not None:
        h = 3600 * float(m.group(1))
        mins = 60 * float(m.group(2))
        s = float(m.group(3))
        return sum([h, mins, s])


def process(pin):
    if os.path.isdir(pin):
        print('-- processing transcript files (*.txt) in directory:', pin)
        inputs = glob(pin + '/*.txt')
    elif os.path.isfile(pin):
        print('-- processing transcript file:', pin)
        inputs = [pin]
    for t in inputs:
        pout = t.replace('.txt', '-labels.txt')
        fout = open(pout, 'w')
        text = open(t, 'r').read()
        text_l = re.sub('([0-9])\] *\n', u'\g<1>]@', text).split('\n')
        for line in text_l:
            m = re.search('^([IP]) +\[([^ ]+) - ([^\]]+)]@(.+)', line)
            if m is not None:
                speaker = 'int' if m.group(1) == 'I' else 'mum'
                start = format(float(get_sec(m.group(2))), '.6f')
                end = format(float(get_sec(m.group(3))), '.6f')
                text = m.group(4)
                print('{}\t{}\t{}:{}'.format(start, end, speaker, text), file=fout)
                print('{}\t{}\t{}:{}'.format(start, end, speaker, text))
        print('-- wrote label file:', pout)


if __name__ == '__main__':
    pin = sys.argv[1]
    process(pin)
