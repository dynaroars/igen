import subprocess as sp
import cPickle as pickle
from datetime import datetime
def iread(filename):
    """ return a generator """
    with open(filename, 'r') as fh:
        for line in fh:
            yield line

def iread_strip(filename):
    """
    like iread but also strip out comments and empty line
    """
    lines = (l.strip() for l in iread(filename))
    lines = (l for l in lines if l and not l.startswith('#'))
    return lines


def vsave(filename,sobj,mode='wb'):
    with open(filename,mode) as fh:
        pickler = pickle.Pickler(fh,-1)
        pickler.dump(sobj)

def vload(filename,mode='rb'):
    with open(filename,mode) as fh:
        pickler = pickle.Unpickler(fh)
        sobj = pickler.load()
    return sobj

def vflatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def vcmd(cmd, inp=None, shell=True):
    proc = sp.Popen(cmd,shell=shell,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    return proc.communicate(input=inp)



def get_logger(logger_name,very_detail=True):
    if __debug__:
        assert is_str(logger_name) and logger_name, logger_name
        assert is_bool(very_detail), very_detail

    import logging

    logger = logging.getLogger(logger_name)
    ch = logging.StreamHandler()
    if very_detail:
        f = "%(asctime)s:%(filename)s:%(funcName)s:%(levelname)s: %(message)s"
        formatter = logging.Formatter(f,datefmt='%H:%M:%S')
    else:
        f = "%(levelname)s: %(message)s"
        formatter = logging.Formatter(f)

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


class VLog(object):
    """
    >>> logger = VLog('vu_logger')
    >>> assert logger.level  == VLog.INFO
    >>> logger.detail('detail msg')
    >>> logger.debug('debug msg')
    >>> logger.info('info msg')
    vu_logger:Info:info msg
    >>> logger.warn('warn msg')
    vu_logger:Warn:warn msg
    >>> logger.error('error msg')
    vu_logger:Error:error msg

    >>> logger.set_level(VLog.DETAIL)
    >>> logger.detail('detail msg')
    vu_logger:Detail:detail msg
    >>> f = lambda: 'detail msg'
    >>> logger.detail(f)
    vu_logger:Detail:detail msg
    >>> logger.debug('debug msg')
    vu_logger:Debug:debug msg
    >>> logger.info('info msg')
    vu_logger:Info:info msg
    >>> logger.warn('warn msg')
    vu_logger:Warn:warn msg
    >>> logger.error('error msg')
    vu_logger:Error:error msg
    """
    ERROR = 0
    WARN = 1
    INFO = 2
    DEBUG = 3
    DETAIL = 4
    
    PRINT_TIME = False

    level_d = {ERROR: 'Error',
               WARN: 'Warn',
               INFO: 'Info',
               DEBUG: 'Debug',
               DETAIL: 'Detail'}

    level_d_rev = dict((v,k) for (k,v) in level_d.iteritems())

    def __init__(self, name):
        self.name = name
        self.level = VLog.INFO

    def get_level(self): 
        return self._level

    def set_level(self, level):
        if __debug__:
            assert level in self.level_d, level
        self._level = level
    level = property(get_level,set_level)

    def print_s(self, s, level):
        if self.level < level:
            return
        else:
            if not isinstance(s,str):
                s = s()
                
            level_name =  self.level_d[level]
            if VLog.PRINT_TIME:
                print("{}:{}:{}:{}"
                      .format(get_cur_time(),self.name,level_name, s))
            else:
                print("{}:{}:{}"
                      .format(self.name,level_name, s)) 

    def detail(self, s): self.print_s(s, VLog.DETAIL)
    def debug(self, s): self.print_s(s, VLog.DEBUG)
    def info(self, s): self.print_s(s, VLog.INFO)
    def warn(self, s): self.print_s(s, VLog.WARN)
    def error(self, s): self.print_s(s, VLog.ERROR)


def get_cur_time(time_only=True):
    now = datetime.now()
    if time_only:
        return "{}:{}:{}".format(now.hour,now.minute,now.second)
    else:
        return str(now)    
