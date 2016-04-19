import os
from collections import defaultdict
from contextlib import contextmanager
from itertools import groupby, chain
import time
import re
import sys
from io import StringIO


_dotiming = 'TIMING' in os.environ
_timing = defaultdict(float)
_timing_stack = []


@contextmanager
def timing(name):
    if _dotiming:
        label = '>'.join(_timing_stack + [name])
        _timing[label]
        _timing_stack.append(name)
        tm = time.time()
    try:
        yield
    finally:
        if _dotiming:
            _timing[label] += time.time()-tm
            _timing_stack.pop(-1)


def print_timing():
    if _dotiming:
        groups = [sorted(group, key=lambda x: x[0])
                  for _, group
                  in groupby(_timing.items(), lambda x: x[0].split('>')[0])]
        groups.sort(key=lambda x: x[0][1], reverse=True)
        table = Table(align=['<', '<'])
        for group in groups:
            for row in group:
                table.add_row(re.sub(r'\w+>', 4*' ', row[0]),
                              '{:.4f}'.format(row[1]))
        print(table, file=sys.stderr)


class TableException(Exception):
    pass


def alignize(s, align, width):
    l = len(s)
    if l < width:
        if align == '<':
            s = s + (width-l)*' '
        elif align == '>':
            s = (width-l)*' ' + s
        elif align == '|':
            s = (-(l-width)//2)*' ' + s + ((width-l)//2)*' '
    return s


class Table:
    def __init__(self, **kwargs):
        self.rows = []
        self.set_format(**kwargs)

    def add_row(self, *row, free=False):
        self.rows.append((free, row))

    def set_format(self, sep=' ', align='>', indent=''):
        self.sep = sep
        self.align = align
        self.indent = indent

    def sort(self, key=lambda x: x[0]):
        self.rows.sort(key=lambda x: key(x[1]), reverse=True)

    def __str__(self):
        col_nums = [len(row) for free, row in self.rows if not free]
        if len(set(col_nums)) != 1:
            raise TableException('Unequal column lengths: {}'.format(col_nums))
        col_nums = col_nums[0]
        cell_widths = [[len(cell) for cell in row]
                       for free, row in self.rows if not free]
        col_widths = [max(col) for col in zip(*cell_widths)]
        seps = (col_nums-1)*[self.sep] if not isinstance(self.sep, list) else self.sep
        seps += ['\n']
        aligns = col_nums*[self.align] if not isinstance(self.align, list) else self.align
        f = StringIO()
        for free, row in self.rows:
            if free:
                f.write('{}\n'.format(row[0]))
            else:
                cells = (alignize(cell, align, width)
                         for cell, align, width
                         in zip(row, aligns, col_widths))
                f.write('{}{}'.format(self.indent,
                                      ''.join(chain.from_iterable(zip(cells, seps)))))
        return f.getvalue()[:-1]
