from collections import OrderedDict
import vu_common as CM
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

    >>> print sorted(c.__sids__)
    ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'LS']

    >>> c.get_paths('L5')
    [['LS', 'L0', 'L2', 'L3', 'L4', 'L5'], ['LS', 'L0', 'L2', 'L4', 'L5'], ['LS', 'L0', 'L1', 'L4', 'L5']]
    
    >>> c.get_paths('L4')
    [['LS', 'L0', 'L2', 'L3', 'L4'], ['LS', 'L0', 'L2', 'L4'], ['LS', 'L0', 'L1', 'L4']]

    >>> c.get_paths('L4', 10)
    [['LS', 'L0', 'L2', 'L3', 'L4'], ['LS', 'L0', 'L2', 'L4'], ['LS', 'L0', 'L1', 'L4']]

    >>> c.get_paths('L4', 2)
    [['LS', 'L0', 'L2', 'L3', 'L4'], ['LS', 'L0', 'L2', 'L4']]

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
    
        
    def get_paths(self, sid, max_npaths=None):
        assert isinstance(sid, str), sid
        assert sid in self.__sids__, "invalid sid '{}'".format(sid)
        assert max_npaths is None or max_npaths > 0, max_npaths

        k = (sid, max_npaths)
        try:
            return self.__paths__[k]
        except KeyError:
            paths_generator = CFG._get_paths(sid, self, frozenset())
            if max_npaths:
                paths = []
                for path in paths_generator:
                    paths.append(path)
                    if len(paths) >=  max_npaths:
                        break

            else:
                paths = list(paths_generator)
                
            if paths:
                self.__paths__[k] = paths
            else:
                print("WARN: sid '{}' has no paths. "
                      "Removing sid from cfg".format(sid))
                self.__sids__.remove(sid)

            assert isinstance(paths, list), paths
            return paths
            
    @staticmethod
    def _get_paths(sid, preds_d, visited, acc=[]):
        """
        Get sid's paths using generator, which helps avoid too many paths

        >>> list(CFG._get_paths(5,OrderedDict([(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]), frozenset()))
        [[1, 2, 3, 4, 5]]

        >>> list(CFG._get_paths(3,OrderedDict([(3,[2]),(2,[1]),(1,[3])]), frozenset()))
        []

        >>> list(CFG._get_paths(6,OrderedDict([(6,[5,7]),(7,[1]),(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]), frozenset()))
        [[1, 2, 3, 4, 5, 6], [1, 7, 6]]

        """
        assert isinstance(sid,(str, int)), sid
        assert isinstance(preds_d, dict), preds_d
        assert isinstance(visited, frozenset), visited
        assert isinstance(acc, list), acc
        
        if sid not in preds_d or not preds_d[sid]: # base case
            yield [sid] + acc
            
        elif sid in visited:  #loop, will not use this path
            yield [None] + acc
            
        else: #recursive case
            for p in preds_d[sid]:
                paths = CFG._get_paths(
                    p, preds_d, frozenset(list(visited) + [sid]), [sid] + acc)

                for p_ in paths:
                    if None not in p_:
                        yield p_
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
