from functools import reduce
import tempfile
import os.path
import itertools
import subprocess as sp
import operator

import logging


def pause(s=None):
    eval(input("Press any key to continue ..." if s is None else s))

def vcmd(cmd, inp=None, shell=True):
    proc = sp.Popen(cmd, shell=shell, stdin=sp.PIPE,
                    stdout=sp.PIPE, stderr=sp.PIPE)
    out, err = proc.communicate(input=inp)
    return out.decode('utf-8'), err.decode('utf-8')


def vload(filename, mode='rb'):
    with open(filename, mode) as fh:
        import pickle
        sobj = pickle.load(fh)
    return sobj


def vsave(filename, sobj, mode='wb'):
    with open(filename, mode) as fh:
        import pickle
        pickle.dump(sobj, fh)


def vread(filename):
    with open(filename, 'r') as fh:
        return fh.read()

def vwrite(filename, contents, mode='w'):
    with open(filename, mode) as fh:
        fh.write(contents)

def iread(filename):
    """ return a generator """
    with open(filename, 'r') as fh:
        for line in fh:
            yield line


def strip_contents(lines, strip_c='#'):
    lines = (l.strip() for l in lines)
    lines = (l for l in lines if l)
    if strip_c:
        lines = (l for l in lines if not l.startswith(strip_c))
    return lines


def iread_strip(filename, strip_c='#'):
    """
    like iread but also strip out comments and empty line
    """
    return strip_contents(iread(filename), strip_c)


def vmul(l): return reduce(operator.mul, l, 1)

def getpath(f): return os.path.realpath(os.path.expanduser(f))

def file_basename(filename): return os.path.splitext(filename)[0]

# log utils
def getLogger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter("%(name)s:%(levelname)s:%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def getLogLevel(level):
    assert level in set(range(5))

    if level == 0:
        return logging.CRITICAL
    elif level == 1:
        return logging.ERROR
    elif level == 2:
        return logging.WARNING
    elif level == 3:
        return logging.INFO
    else:
        return logging.DEBUG
