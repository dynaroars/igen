from functools import total_ordering
from collections import OrderedDict
from multiprocessing import (Process, Queue, cpu_count)
import pdb

import helpers.vcommon
import settings

DBG = pdb.set_trace
mlog = helpers.vcommon.getLogger(__name__, settings.logger_level)


@total_ordering
class HDict(OrderedDict):
    """
    Hashable dictionary

    __eq__ and __lt__ + total_ordering is needed for __cmp__
    which is needed to compare or sort things


    >>> c = HDict([('f', frozenset(['0'])), ('g', frozenset(['0']))]) 
    >>> d = HDict([('f', frozenset(['0'])), ('g', frozenset(['1']))])
    >>> _ = {'c':c,'d':d}
    >>> _ = set([c,d])
    >>> sorted([c,d])
    [HDict([('f', frozenset(['0'])), ('g', frozenset(['0']))]), HDict([('f', frozenset(['0'])), ('g', frozenset(['1']))])]

    """

    def __eq__(self, other):
        return (other is self or
                (isinstance(other, HDict) and
                 self.hcontent.__eq__(other.hcontent)))

    def __lt__(self, other):
        return isinstance(other, HDict) and self.hcontent.__lt__(other.hcontent)

    @property
    def hcontent(self):
        try:
            return self._hcontent
        except AttributeError:
            self._hcontent = frozenset(self.items())
            return self._hcontent

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self.hcontent)
            return self._hash


def is_cov(cov): return (isinstance(cov, (set, frozenset)) and
                         all(isinstance(c, str) for c in cov))


def str_of_cov(cov):
    """
    >>> assert str_of_cov(set("L2 L1 L3".split())) == '(3) L1,L2,L3'
    """
    assert is_cov(cov), cov

    s = "({})".format(len(cov))
    if settings.show_cov:
        s = "{} {}".format(s, ','.join(sorted(cov)))
    return s


def is_setting(kv):
    k, v = kv
    return isinstance(k, str) and isinstance(v, str)


def str_of_setting(kv):
    """
    >>> print str_of_setting(('x','1'))
    x=1
    """
    k, v = kv
    assert is_setting((k, v)), (k, v)

    return '{}={}'.format(k, v)


def is_valset(vs): return (isinstance(vs, frozenset) and vs and
                           all(isinstance(v, str) for v in vs))


def str_of_valset(s):
    """
    >>> str_of_valset(frozenset(['1','2','3']))
    '1,2,3'
    """
    return ','.join(sorted(s))


def is_csetting(kvs):
    k, vs = kvs
    return isinstance(k, str) and is_valset(vs)


def str_of_csetting(kvs):
    """
    >>> print str_of_csetting(('x', frozenset(['1'])))
    x=1
    >>> print str_of_csetting(('x', frozenset(['3','1'])))
    x=1,3
    """
    assert is_csetting(kvs), kvs

    k, vs = kvs
    return '{}={}'.format(k, str_of_valset(vs))


def getWorkloads(tasks, maxProcessces, chunksiz):
    """
    >>> wls = Miscs.getWorkloads(range(12),7,1); wls
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]


    >>> wls = Miscs.getWorkloads(range(12),5,2); wls
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10, 11]]

    >>> wls = Miscs.getWorkloads(range(20),7,2); wls
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17], [18, 19]]

    >>> wls = Miscs.getWorkloads(range(20),20,2); wls
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]

    """
    assert len(tasks) >= 1, tasks
    assert maxProcessces >= 1, maxProcessces
    assert chunksiz >= 1, chunksiz

    # determine # of processes
    ntasks = len(tasks)
    nprocesses = int(round(ntasks/float(chunksiz)))
    if nprocesses > maxProcessces:
        nprocesses = maxProcessces

    # determine workloads
    cs = int(round(ntasks/float(nprocesses)))
    wloads = []
    for i in range(nprocesses):
        s = i*cs
        e = s+cs if i < nprocesses-1 else ntasks
        wl = tasks[s:e]
        if wl:  # could be 0, e.g., getWorkloads(range(12),7,1)
            wloads.append(wl)

    return wloads


def runMP(taskname, tasks, wprocess, chunksiz, doMP):
    """
    Run wprocess on tasks in parallel
    """
    if doMP:
        Q = Queue()
        wloads = getWorkloads(
            tasks, maxProcessces=cpu_count(), chunksiz=chunksiz)

        mlog.debug("workloads '{}' {}: {}"
                   .format(taskname, len(wloads), list(map(len, wloads))))

        workers = [Process(target=wprocess, args=(wl, Q)) for wl in wloads]

        for w in workers:
            w.start()
        wrs = []
        for _ in workers:
            wrs.extend(Q.get())
    else:
        wrs = wprocess(tasks, Q=None)

    return wrs
