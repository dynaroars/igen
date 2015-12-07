from collections import OrderedDict
import vu_common as CM
import config_common as CC

logger = CM.VLog('CFG')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

class CFG(OrderedDict):
    """
    >>> lines = [\
    'LS',\
    'L0 LS',\
    'L1 L0',\
    'L2 L0',\
    'L3 L2',\
    'L4 L3 L2 L1',\
    'L5 L4']

    >>> c = CFG.mk_from_lines(lines)
    >>> print c
    CFG([('LS', []), ('L0', ['LS']), ('L1', ['L0']), ('L2', ['L0']), ('L3', ['L2']), ('L4', ['L3', 'L2', 'L1']), ('L5', ['L4'])])

    # >>> c.compute_paths()
    
    >>> print sorted(c.__sids__)
    ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'LS']

    >>> c.get_paths('L5')
    [['LS', 'L0', 'L2', 'L3', 'L4', 'L5'], ['LS', 'L0', 'L2', 'L4', 'L5'], ['LS', 'L0', 'L1', 'L4', 'L5']]
    
    >>> c.get_paths('L4')
    [['LS', 'L0', 'L2', 'L3', 'L4'], ['LS', 'L0', 'L2', 'L4'], ['LS', 'L0', 'L1', 'L4']]

    >>> c.get_paths('notexist')
    Traceback (most recent call last):
    ...
    AssertionError: invalid sid 'notexist'


    >>> c = CFG(OrderedDict([('LS', []), ('L0', ['LS']), \
    ('L1', ['L0']), ('L2', ['L0']), ('L3', ['L2']), ('L4', \
    ['L3', 'L2', 'L1']), ('L5', ['L4'])]))

    >>> print c
    CFG([('LS', []), ('L0', ['LS']), ('L1', ['L0']), ('L2', ['L0']), ('L3', ['L2']), ('L4', ['L3', 'L2', 'L1']), ('L5', ['L4'])])
    
    >>> d = CFG([['LS', []], ['L0', ['LS']], ['L1', ['L0']], ['L2', ['L0']], ['L3', ['L2']], ['L4', ['L3', 'L2', 'L1']], ['L5', ['L4']]])
    >>> assert c==d

    >>> lines = [\
    'LS',\
    'L0 LS',\
    'L4 L3',\
    'L4 L2    L1   L3',\
    'L1 L0',\
    'L2 L0',\
    'L2    L0',\
    'L3 L2',\
    'L5 L4']

    >>> c = CFG.mk_from_lines(lines)
    >>> print c
    CFG([('LS', []), ('L0', ['LS']), ('L4', ['L3', 'L2', 'L1']), ('L1', ['L0']), ('L2', ['L0']), ('L3', ['L2']), ('L5', ['L4'])])
    
    >>> print c.__str__(as_lines=True)
    LS
    L0 LS
    L4 L3 L2 L1
    L1 L0
    L2 L0
    L3 L2
    L5 L4
    """    
    def __init__(self,d):            
        OrderedDict.__init__(self, d)
        if __debug__:
            assert self, 'empty cfg'

        self.__paths__ = OrderedDict()
        self.__sids__ = set(sid for sid in self if not sid.startswith('fake'))

    def __str__(self, as_lines=False):
        if as_lines:
            lines = ['{} {}'.format(loc, CM.str_of_list(prevs, ' '))
                     for loc, prevs in self.iteritems()]
            lines = [line.strip() for line in lines]
            return CM.str_of_list(lines, '\n')
        else:
            return super(CFG, self).__str__()
        
    @staticmethod
    def mk_from_lines(lines):
        """
        Make a CFG from list of strings having the form 
        l prev1 prev2  where
        previ are's previous locs of l
        """
        rs = (l.split() for l in lines)
        prevs = OrderedDict()
        for l in rs:
            k = l[0]
            cs = l[1:]
            if k in prevs:
                cs = [c for c in cs if c not in prevs[k]]
                prevs[k].extend(cs)
            else:
                prevs[k]=cs

        return CFG(prevs)
    
        
    def get_paths(self, sid):
        assert isinstance(sid, str) and not sid.startswith('fake'), sid
        assert sid in self.__sids__, "invalid sid '{}'".format(sid)

        try:
            return self.__paths__[sid]
        except KeyError:
            pc = CFG._guess_paths_count(sid, self, frozenset())
            print 'guess npaths {}'.format(pc)
            CM.pause()
            paths = CFG._get_paths(sid, self, frozenset())
            paths = [[s for s in p if not s.startswith('fake')] for p in paths]
            print "guess", pc, "real", len(paths)
            if paths:
                self.__paths__[sid] = paths                
            else:
                logger.warn("sid {} has no paths".format(sid))
                self.__sids__.remove(sid)
            return paths
            
    
    @staticmethod
    def _get_paths(sid, preds_d, visited):
        """
        >>> CFG._get_paths(5,OrderedDict([(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]), frozenset())
        [[1, 2, 3, 4, 5]]

        >>> CFG._get_paths(3,OrderedDict([(3,[2]),(2,[1]),(1,[3])]), frozenset())
        []

        >>> CFG._get_paths(6,OrderedDict([(6,[5,7]),(7,[1]),(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]), frozenset())
        [[1, 2, 3, 4, 5, 6], [1, 7, 6]]
        """
        
        assert isinstance(sid,(str, int)), sid
        assert isinstance(preds_d, dict), preds_d
        assert isinstance(visited, frozenset), visited

        if sid in visited:
            return [[None]]

        if sid not in preds_d or not preds_d[sid]:
            return [[sid]]

        rs = []
        preds = preds_d[sid]
        for i,p in enumerate(preds):
            paths = CFG._get_paths(p, preds_d, frozenset(list(visited)+[sid]))

            for path in paths:
                if None not in path:
                    rs.append(path + [sid])

        return rs

    @staticmethod
    def _guess_paths_count(sid, preds_d, visited):
        """
        Quickly guess the # of paths by essentially counting the levels of the tree, 
        e.g., for n levels and each level has m preds then it's m^n.
        >>> CFG._guess_paths_count(0, OrderedDict([(0,[1,2]), (1,[3,4]),(3,[5]),(4,[]), (2,[6,7,8]), (6,[]), (7,[9, 10]), (9, []), (10, []), (8,[])]), frozenset())
        12

        Note: doesn't work very well on real apps (the greedy choice doesn't work in practice).  Can't think of any faster way to determine the 
        level of tree
        """
        
        assert isinstance(sid,(str, int)), sid
        assert isinstance(preds_d, dict), preds_d
        assert isinstance(visited, frozenset), visited

        if sid in visited:
            return 1

        if sid not in preds_d or not preds_d[sid]:
            return 1

        preds = preds_d[sid]
        #heuristic, only traverse the sid with largest preds
        lsid = max(preds, key=lambda s: len(preds_d[s]) if s in preds_d else 0)
        pc = CFG._guess_paths_count(lsid, preds_d, frozenset(list(visited)+[sid]))
            
        return len(preds) * pc
        
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
