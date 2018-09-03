import os.path
import random
import z3
import z3util

import config_common as CC
from config_common import Configs_d #do not del, needed to read existing results

import settings
mlog = CC.getLogger(__name__, settings.logger_level)

# logger = CC.VLog(__name__)
# logger.level = CC.logger_level
# CC.VLog.PRINT_TIME = True

def compat(obj, cls):
    """
    For compatibility with old data
    """
    return isinstance(obj, cls) or obj.__class__.__name__ == cls.__name__

#Data Structures
class Dom(CC.Dom):
    """
    >>> dom = Dom([('x',frozenset(['1','2'])),\
    ('y',frozenset(['1'])),\
    ('z',frozenset(['0','1','2'])),\
    ('w',frozenset(['a','b','c']))\
    ])
    >>> assert dom.siz == len(dom.gen_configs_full()) == 18
    """

    def gen_configs_cex(self, sel_core, existing_configs, z3db):
        """
        >>> dom = Dom([('a', frozenset(['1', '0'])), \
        ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0', '2']))])
        >>> z3db = dom.z3db

        >>> c1 = Config([('a', '0'), ('b', '0'), ('c', '0')])
        >>> c2 = Config([('a', '0'), ('b', '0'), ('c', '1')])
        >>> c3 = Config([('a', '0'), ('b', '0'), ('c', '2')])

        >>> c4 = Config([('a', '0'), ('b', '1'), ('c', '0')])
        >>> c5 = Config([('a', '0'), ('b', '1'), ('c', '1')])
        >>> c6 = Config([('a', '0'), ('b', '1'), ('c', '2')])

        >>> c7 = Config([('a', '1'), ('b', '0'), ('c', '0')])
        >>> c8 = Config([('a', '1'), ('b', '0'), ('c', '1')])
        >>> c9 = Config([('a', '1'), ('b', '0'), ('c', '2')])

        >>> c10 = Config([('a', '1'), ('b', '1'), ('c', '0')])
        >>> c11 = Config([('a', '1'), ('b', '1'), ('c', '1')])
        >>> c12 = Config([('a', '1'), ('b', '1'), ('c', '2')])

        >>> configs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
        >>> nexpr = z3util.myOr([c.z3expr(z3db) for c in configs])
        >>> assert dom.gen_configs_exprs([None],[nexpr],k=1,config_cls=Config)[0] == c12

        >>> core = Core([('a',frozenset(['1']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> assert dom.gen_configs_exprs([None],[nexpr],k=1, config_cls=Config)[0] == c12

        >>> core = Core([('a',frozenset(['0']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> assert not dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)

        >>> core = Core([('c',frozenset(['0','1']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> assert not dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)

        >>> core = Core([('c',frozenset(['0','2']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> config = dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)[0]
        >>> print config
        a=1 b=1 c=2


        sel_core = (c_core,s_core)
        create counterexample configs by changing settings in c_core,
        but these configs must satisfy s_core
        x=0,y=1  =>  [x=0,y=0,z=rand;x=0,y=2,z=rand;x=1,y=1;z=rand]
        """
        
        assert isinstance(sel_core, SCore), sel_core
        assert isinstance(z3db, CC.Z3DB)
        
        configs = []
        c_core, s_core = sel_core

        #keep
        changes = []        
        if sel_core.keep and (len(self) - len(c_core)):
            changes.append(c_core)

        #change
        _new = lambda: Core((k, c_core[k]) for k in c_core)
        for k in c_core:
            vs = self[k] - c_core[k]
            for v in vs:
                new_core = _new()
                new_core[k] = frozenset([v])
                if s_core:
                    for sk,sv in s_core.iteritems():
                        assert sk not in new_core, sk
                        new_core[sk] = sv
                changes.append(new_core)

        e_configs = [c.z3expr(z3db) for c in existing_configs]
        for changed_core in changes:
            yexpr = changed_core.z3expr(z3db, z3util.myAnd)
            #yexpr = z3.simplify(z3.And(yexpr, constrains))
            nexpr = z3util.myOr(e_configs)
            #logger.debug('constraint: {}, changed_core: {}, yexpr: {} '.format(constrains, changed_core, yexpr))
            configs_ = self.gen_configs_exprs([yexpr],[nexpr],k=1, config_cls=Config)
            if not configs_:
                continue
            config=configs_[0]
            
            assert config.c_implies(changed_core)
            assert config not in existing_configs, \
                ("ERR: gen existing config {}".format(config))
                     
            configs.append(config)
            e_configs.append(config.z3expr(z3db))

        return configs

    @classmethod
    def get_dom(cls, dom_file):
        """
        Read domain info from a file. 
        Also *extract* default configs (.default*) 
        And *save*
        - kconstraint (.dimacs) if given
        - runscript (.run) if given 
        """
        assert os.path.isfile(dom_file), dom_file

        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs_ = []
            for parts in rs:
                var = parts[0]
                vals = frozenset(parts[1:])
                rs_.append((var, vals))
            return rs_

        #domain file   (*.dom)
        dom = get_lines(CC.iread_strip(dom_file))
        dom = cls(dom)
        
        #other files 
        dom_name = CC.file_basename(os.path.basename(dom_file)) #ex
        dom_dir = os.path.dirname(dom_file)
        fs = [os.path.join(dom_dir, f) for f in os.listdir(dom_dir) if dom_name in f]

        def _f(name, at_most_one):
            fs_ = [f for f in fs if name in f and os.path.isfile(f)]
            if not at_most_one:
                return fs_

            if fs_: 
                assert len(fs_) == 1, fs_
                return fs_[0]
            else:
                return None
            
            
        #potentially multiple default configs  (*.default*)
        configs = _f('.default', at_most_one=False)
        configs = [dict(get_lines(CC.iread_strip(f))) for f in configs
                   if os.path.isfile(f)]
        configs = [[(k, list(c[k])[0]) for k in dom] for c in configs]

        #1 constraint file  (*.dimacs*)
        dom.kconstraint_file = _f('.dimacs', at_most_one=True)
        if dom.kconstraint_file:
            mlog.debug("Using contraint from {}".format(dom.kconstraint_file))

        #1 runscript (*.run)
        dom.run_script = _f('.run', at_most_one=True)
        if dom.run_script:
            mlog.debug("Using run_script from {}".format(dom.run_script))

        return dom, configs 

    def get_kconstraint(self, z3db):
        assert isinstance(z3db, CC.Z3DB), z3db
        
        if self.kconstraint_file is None:
            return None

        lines = [l.strip() for l in CC.iread(self.kconstraint_file)]
        lines = [l for l in lines]
        
        symbols = {}
        clauses = []
        for l in lines:
            if l.startswith('c'):
                if l.startswith('c kconfig_variable'):
                    pair = l.split(' ')
                    sidx, symbol = pair[2], pair[3] #'2', 'y'
                    assert symbol in z3db, (z3db, symbol)
                    symbols[sidx] = symbol

            
            elif l.startswith("p"): #p cnf 3 4
                _, _, nvars, nclauses = l.split(" ")
                nsymbols = int(nvars)
                assert nsymbols == len(symbols)
                nclauses = int(nclauses)
                
            else:  #clause,  -1 2
                clause = l.split()  #-2 11 0
                clause = frozenset(clause[:-1])  #remove the last 0
                clauses.append(clause)

        clauses = frozenset(clauses)                
        assert len(clauses) == nclauses
        
        def _f(sidx):
            isNot = sidx.startswith('-')
            return isNot, sidx[1:] if isNot else sidx

        rs = []
        for clause in clauses:
            ss = [_f(sidx) for sidx in clause]  #-3
            ss = [z3db.get_eq_expr(symbols[sidx], '0' if isNot else '1')
                  for isNot, sidx in ss]
            rs.append(z3util.Or(ss))

        rs = z3util.And(rs)
        rs = z3.simplify(rs)
        return rs


class Config(CC.Config):
    """
    >>> c = Config([('a', '1'), ('b', '0'), ('c', '1')])
    >>> print c
    a=1 b=0 c=1

    >>> assert c.c_implies(Core()) and c.d_implies(Core())
    
    >>> core1 = Core([('a', frozenset(['0','1'])), ('b', frozenset(['0','1']))])    
    >>> assert c.c_implies(core1)
    >>> assert c.d_implies(core1)

    >>> core2 = Core([('a', frozenset(['0','1'])), ('x', frozenset(['0','1']))])    
    >>> assert not c.c_implies(core2)
    >>> assert c.d_implies(core2)

    >>> dom = Dom([('a',frozenset(['1','2'])),\
    ('b',frozenset(['0','1'])),\
    ('c',frozenset(['0','1','2']))])
    >>> c.z3expr(dom.z3db)
    And(a == 1, b == 0, c == 1)
    """

    def c_implies(self, core):
        """
        self => conj core
        x=0&y=1 => x=0,y=1
        not(x=0&z=1 => x=0,y=1)
        """
        assert isinstance(core, Core), core

        return (not core or
                all(k in self and self[k] in core[k] for k in core))

    def d_implies(self, core):
        """
        self => disj core
        """
        assert isinstance(core, Core),core

        return (not core or
                any(k in self and self[k] in core[k] for k in core))

    @classmethod
    def eval(cls, configs, get_cov_f, kconstraint, z3db):
        """
        Eval (e.g., get coverage) configurations using function get_cov_f
        Ret a list of configs and their results
        """
        assert (isinstance(configs, list) and
                all(isinstance(c, (cls, CC.Config)) for c in configs)
                and configs), configs
        assert callable(get_cov_f), get_cov_f
        assert kconstraint is None or z3.is_expr(kconstraint), kconstraint
        assert isinstance(z3db, CC.Z3DB), z3db

        def eval_get_cov(c):
            sids, outps = get_cov_f(c)
            rs = outps if CC.analyze_outps else sids
            if not rs:
                mlog.warn("'{}' produces nothing".format(c))
            return rs
        
        def eval_f(c):
            if kconstraint is not None:
                expr = c.z3expr(z3db)
                expr = z3.And(kconstraint, expr)
                if not z3util.is_sat(expr):  #invalid
                    #print "invalid"
                    rs = set(["invalid config"])
                    return rs
                
            return eval_get_cov(c)
                
        results = [(c, eval_f(c)) for c in set(configs)]
        return results
    

class Core(CC.HDict):
    """
    >>> print Core()
    true

    >>> c = Core([('x',frozenset(['2'])),('y',frozenset(['1'])),('z',frozenset(['0','1']))])
    >>> print c
    x=2 y=1 z=0,1

    >>> print ', '.join(map(CC.str_of_setting,c.settings))
    x=2, y=1, z=1, z=0

    >>> dom = Dom([('x',frozenset(['1','2'])),\
    ('y',frozenset(['1'])),\
    ('z',frozenset(['0','1','2'])),\
    ('w',frozenset(['a','b','c']))\
    ])

    >>> print c.neg(dom)
    x=1 z=2

    >>> c = Core([('x',frozenset(['2'])),('z',frozenset(['0','1'])),('w',frozenset(['a']))])
    >>> print c.neg(dom)
    x=1 z=2 w=b,c

    """
    def __init__(self, core=CC.HDict()):
        super(Core, self).__init__(core)
        
        assert all(CC.is_csetting(s) for s in self.iteritems()), self

    def __str__(self, delim=' '):
        if self:
            return delim.join(map(CC.str_of_csetting, self.iteritems()))
        else:
            return 'true'

    @property
    def settings(self):
        return [(k,v) for k,vs in self.iteritems() for v in vs]
        
    def neg(self, dom):
        try:
            return self._neg
        except AttributeError:
            assert compat(dom, Dom), dom
            ncore = ((k,dom[k] - self[k]) for k in self)
            self._neg = Core((k,vs) for k,vs in ncore if vs)
            return self._neg
        
    def z3expr(self, z3db, is_and):
        assert isinstance(z3db, CC.Z3DB)
        return z3db.expr_of_dict_dict(self, is_and)

    @staticmethod
    def maybe_core(c): return c is None or isinstance(c,Core)

class MCore(tuple):
    """
    Multiple (tuples) cores
    """
    def __init__(self,cores):
        tuple.__init__(self,cores)

        assert len(self) == 2 or len(self) == 4, self
        assert all(Core.maybe_core(c) for c in self), self

    @property
    def settings(self):
        core = (c for c in self if c)
        return set(s for c in core for s in c.iteritems())

    @property
    def values(self):
        core = (c for c in self if c)
        return set(s for c in core for s in c.itervalues())

    @property
    def sstren(self): return len(self.settings)
    
    @property
    def vstren(self): return sum(map(len, self.values))

class SCore(MCore):
    def __init__(self, (mc,sc)):
        """
        mc: main core that will generate cex's
        sc (if not None): sat core that is satisfied by all generated cex'
        """
        super(SCore, self).__init__((mc,sc))
        #additional assertion
        assert mc is None or isinstance(mc, Core) and mc, mc
        #sc is not None => ...
        assert not sc or all(k not in mc for k in sc), sc
        
        self.keep = False

    def set_keep(self):
        """
        keep: if true then generated cex's with diff settings than mc 
        and also those that have the settings of mc
        """
        self.keep = True
    
    @property
    def mc(self): return self[0]

    @property
    def sc(self): return self[1]

    def __str__(self):
        ss = []
        if self.mc:
            s = ""
            try:  #to support old format that doesn't have keep
                if self.keep:
                    s = "(keep)" 
            except AttributeError:
                mlog.warn("Old format, has no 'keep' in SCore")
                pass

            ss.append("mc{}: {}".format(s,self.mc))
                                            
                      
        if self.sc:
            ss.append("sc: {}".format(self.sc))
        return '; '.join(ss)        
        
    @classmethod
    def mk_default(cls): return cls((None, None))
        
class PNCore(MCore):
    """
    >>> pc = Core([('x',frozenset(['0','1'])),('y',frozenset(['1']))])
    >>> pd = None
    >>> nc = Core([('z',frozenset(['1']))])
    >>> nd = None
    >>> pncore = PNCore((pc,pd,nc,nd))
    >>> print pncore
    pc: x=0,1 y=1; nc: z=1

    >>> dom = Dom([('x',frozenset(['0','1','2'])),\
    ('y',frozenset(['0','1'])),\
    ('z',frozenset(['0','1'])),\
    ('w',frozenset(['0','1']))\
    ])
    >>> z3db = dom.z3db

    >>> print PNCore._get_str(pc,pd,dom,is_and=True)
    (x=0,1 & y=1)
    >>> print PNCore._get_expr(pc,pd,dom,z3db,is_and=True)
    And(Or(x == 1, x == 0), y == 1)

    >>> print PNCore._get_str(nd,nc,dom,is_and=False)
    z=0
    >>> print PNCore._get_expr(nd,nc,dom,z3db,is_and=False)
    z == 0
    >>> print pncore.z3expr(dom, z3db)
    And(And(Or(x == 1, x == 0), y == 1), z == 0)

    >>> pc = Core([])
    >>> pd = None
    >>> nc = None
    >>> nd = None
    >>> pncore = PNCore((pc,pd,nc,nd))

    >>> assert PNCore._get_str(pc,pd,dom,is_and=True) == 'true'
    >>> assert PNCore._get_str(nd,nc,dom,is_and=False) == 'true'
    >>> assert PNCore._get_expr(pc,pd,dom,z3db,is_and=True) is None
    >>> assert PNCore._get_expr(nd,nc,dom,z3db,is_and=True) is None
    >>> assert pncore.z3expr(dom, z3db) is None

    """

    def __init__(self,(pc,pd,nc,nd)):
        super(PNCore, self).__init__((pc,pd,nc,nd))

    @property
    def pc(self): return self[0]
    @property
    def pd(self): return self[1]
    @property
    def nc(self): return self[2]
    @property
    def nd(self): return self[3]

    @property
    def vtyp(self): return self._vtyp
    @vtyp.setter
    def vtyp(self, vt):
        assert isinstance(vt, str) and vt in 'conj disj mix'.split(), vt
            
        self._vtyp = vt
    
    @property
    def vstr(self): return self._vstr
    @vstr.setter
    def vstr(self,vs):
        assert isinstance(vs, str) and vs, vs
        
        self._vstr = vs

    @classmethod
    def mk_default(cls): return cls((None, None, None, None))

    def __str__(self):
        try:
            return "{} ({})".format(self.vstr, self.vtyp)
        except AttributeError:
            ss = ("{}: {}".format(s,c) for s,c in
                  zip('pc pd nc nd'.split(),self) if c is not None)
            return '; '.join(ss)

    def verify(self, configs, dom):
        assert self.pc is not None, self.pc #this never happens
        #nc is None => pd is None
        assert self.nc is not None or self.pd is None, (self.nc, self.nd)
        assert (all(isinstance(c, Config) for c in configs) and configs), configs
        assert isinstance(dom, Dom), dom

        pc, pd, nc, nd = self

        #traces => pc & neg(pd)
        assert not pc or all(c.c_implies(pc) for c in configs), pc

        if pd:
            pd_n = pd.neg(dom)
            if not all(c.d_implies(pd_n) for c in configs):
                mlog.debug('pd {} invalid'.format(pd))
                pd = None

        #neg traces => nc & neg(nd)
        #pos traces => neg(nc & neg(nd))
        #pos traces => nd | neg(nc) 
        if nc and not nd:
            nc_n = nc.neg(dom)
            if not all(c.d_implies(nc_n) for c in configs):
                mlog.debug('nc {} invalid'.format(nc))
                nc = None
        elif not nc and nd:
            if not all(c.c_implies(nd) for c in configs):
                mlog.debug('nd {} invalid'.format(nd))
                nd = None
        elif nc and nd:
            nc_n = nc.neg(dom)        
            if not all(c.c_implies(nd) or
                       c.d_implies(nc_n) for c in configs):
                mlog.debug('nc {} & nd {} invalid').format(nc,nd)
                nc = None
                nd = None

        return PNCore((pc, pd, nc, nd))
    
    @staticmethod
    def _get_expr(cc, cd, dom, z3db, is_and):
        assert Core.maybe_core(cc)
        assert Core.maybe_core(cd)
        assert isinstance(dom, Dom)
        assert isinstance(z3db, CC.Z3DB)
        
        k = (cc, cd, is_and)
        if k in z3db.cache: return z3db.cache[k]

        fs = []
        if cc:
            f = cc.z3expr(z3db, is_and=True)
            fs.append(f)
        if cd:
            cd_n = cd.neg(dom)
            f = cd_n.z3expr(z3db, is_and=False)
            fs.append(f)

        myf = z3util.myAnd if is_and else z3util.myOr
        expr = myf(fs)
        
        z3db.add(k, expr)
        return expr

    @staticmethod
    def _get_str(cc, cd, dom, is_and):
        and_delim = ' & '
        or_delim = ' | '

        def _str(core,delim):
            s = core.__str__(delim)
            if len(core) > 1:
                s = '({})'.format(s)
            return s
        
        ss = []
        if cc:
            s = _str(cc,and_delim)
            ss.append(s)
        if cd:
            cd_n = cd.neg(dom)
            s = _str(cd_n,or_delim)
            ss.append(s)

        if ss:
            delim = and_delim if is_and else or_delim
            return delim.join(sorted(ss))
        else:
            return 'true'

    @staticmethod
    def _get_expr_str(cc,cd,dom,z3db,is_and):
        expr = PNCore._get_expr(cc, cd, dom, z3db, is_and)
        vstr = PNCore._get_str(cc, cd, dom, is_and)
        return expr,vstr

    
    def simplify(self, dom, z3db, do_firsttime=True):
        """
        Compare between (pc,pd) and (nc,nd) and return the stronger one.
        This will set either (pc,pd) or (nc,nd) to (None,None)

        if do_firstime is False then don't do any checking,
        essentialy this option is used for compatibility purpose

        Assumption: all 4 cores are verified

        inv1 = pc & not(pd)
        inv2 = not(nc & not(nd)) = nd | not(nc)
        """
        assert isinstance(dom, Dom), dom
        assert isinstance(z3db, CC.Z3DB), z3db
        if __debug__:
            if do_firsttime:
                assert self.pc is not None, self.pc #this never could happen
                #nc is None => pd is None
                assert self.nc is not None or self.pd is None, (self.nc, self.nd)

        #pf = pc & neg(pd)
        #nf = neg(nc & neg(nd)) = nd | neg(nc)
        pc,pd,nc,nd = self

        #remove empty ones
        if not pc: pc = None
        if not pd: pd = None
        if not nc: nc = None
        if not nd: nd = None

        if pc is None and pd is None:
            expr,vstr = PNCore._get_expr_str(nd,nc,dom,z3db,is_and=False)
        elif nc is None and nd is None:
            expr,vstr = PNCore._get_expr_str(pc,pd,dom,z3db,is_and=True)
        else:
            pexpr,pvstr = PNCore._get_expr_str(pc,pd,dom,z3db,is_and=True)
            nexpr,nvstr = PNCore._get_expr_str(nd,nc,dom,z3db,is_and=False)
            
            assert pexpr is not None
            assert nexpr is not None
            
            if z3util.is_tautology(z3.Implies(pexpr, nexpr), z3db.solver):
                nc = None
                nd = None
                expr = pexpr
                vstr = pvstr
            elif z3util.is_tautology(z3.Implies(nexpr, pexpr), z3db.solver):
                pc = None
                pd = None
                expr = nexpr
                vstr = nvstr
            else:  #could occur when using incomplete traces
                mlog.warn("inconsistent ? {}\npf: {} ?? nf: {}"
                            .format(PNCore((pc,pd,nc,nd)),pexpr,nexpr))

                expr = z3util.myAnd([pexpr,nexpr])
                vstr = ','.join([pvstr,nvstr]) + '***'

        def _typ(s):
            #hackish way to get type
            if ' & ' in s and ' | ' in s:
                return 'mix'
            elif ' | ' in s:
                return 'disj'
            else:
                return 'conj'         
        
        core = PNCore((pc,pd,nc,nd))
        core.vstr = vstr
        core.vtyp = _typ(vstr)
        
        return core,expr

    def is_simplified(self):
        return ((self.pc is None and self.pd is None) or
                (self.nc is None and self.nd is None))

    def z3expr(self, dom, z3db):
        """
        Note: z3 expr "true" is represented (and returned) as None
        """
        assert isinstance(dom, Dom)
        assert isinstance(z3db, CC.Z3DB)        

        if self in z3db.cache:
            return z3db.cache[self]
        
        pc,pd,nc,nd = self
        if pc is None and pd is None:
            expr = PNCore._get_expr(nd, nc, dom, z3db, is_and=False)
        elif nc is None and nd is None:
            expr = PNCore._get_expr(pc, pd, dom, z3db, is_and=True)
        else:
            pexpr = PNCore._get_expr(pc, pd, dom, z3db, is_and=True)
            nexpr = PNCore._get_expr(nd, nc, dom, z3db, is_and=False)
            expr = z3util.myAnd([pexpr,nexpr])

        z3db.add(self, expr)
        return expr

class Cores_d(CC.CustDict):
    """
    rare case when diff c1 and c2 became equiv after simplification
    >>> dom = Dom([('a',frozenset(['0','1'])),('b',frozenset(['0','1']))])
    >>> z3db = dom.z3db

    c1 = a & b
    >>> pc = Core([('a',frozenset('1'))])
    >>> pd = Core([('b',frozenset('0'))])
    >>> nc = Core()
    >>> nd = Core()
    >>> c1 = PNCore((pc,pd,nc,nd))

    c2 = b & a 
    >>> pc = Core([('b',frozenset('1'))])
    >>> pd = Core([('a',frozenset('0'))])
    >>> nc = Core()
    >>> nd = Core()
    >>> c2 = PNCore((pc,pd,nc,nd))

    >>> cores_d = Cores_d()
    >>> cores_d['L1'] = c1
    >>> cores_d['L2'] = c2
    >>> print cores_d
    1. L1: pc: a=1; pd: b=0; nc: true; nd: true
    2. L2: pc: b=1; pd: a=0; nc: true; nd: true

    >>> mlog.level = CC.VLog.WARN
    >>> print cores_d.merge(dom, z3db)
    1. (2) pc: a=1; pd: b=0; nc: true; nd: true: (2) L1,L2

    >>> covs_d = CC.Covs_d()
    >>> config = Config([('a', '1'), ('b', '1')])
    >>> covs_d.add('L1',config)
    >>> covs_d.add('L2',config)

    >>> mlog.level = CC.VLog.WARN
    >>> cores_d = cores_d.analyze(dom, z3db, covs_d)
    >>> print cores_d.merge(dom, z3db, show_detail=False)
    1. (2) a=1 & b=1 (conj): (2) L1,L2

    >>> cores_d = cores_d.analyze(dom, z3db, covs_d=None)
    >>> print cores_d.merge(dom, z3db, show_detail=False)
    1. (2) a=1 & b=1 (conj): (2) L1,L2

    """
    def __setitem__(self,sid,pncore):
        assert isinstance(sid, str), sid
        assert isinstance(pncore, PNCore), pncore
        
        self.__dict__[sid]=pncore

    def __str__(self):
        return '\n'.join("{}. {}: {}"
                         .format(i+1,sid,self[sid])
                         for i,sid in enumerate(sorted(self)))

    def merge(self, dom, z3db, show_detail=False):
        assert isinstance(dom, Dom)
        assert isinstance(z3db, CC.Z3DB)

        mcores_d = Mcores_d()
        cache = {}
        for sid, core in self.iteritems():
            try:
                key = core.vstr 
            except AttributeError:
                key = core
                
            if key not in cache:
                cache[key] = core

            mcores_d.add(cache[key], sid)

        mcores_d = mcores_d.fix_duplicates(dom, z3db)
        
        if show_detail:
            mcores_d.show_results()
            
        return mcores_d

    def analyze(self, dom, z3db, covs_d):
        """
        Simplify cores. If covs_d then also check that cores are valid invs
        """
        assert isinstance(dom, Dom), dom
        assert isinstance(z3db, CC.Z3DB)
        if __debug__:
            if covs_d is not None:
                assert isinstance(covs_d, CC.Covs_d) and covs_d, covs_d
                assert len(self) == len(covs_d), (len(self), len(covs_d))

        if not self:
            return self
            
        def show_compare(sid,old_c,new_c):
            if old_c != new_c:
                mlog.debug("sid {}: {} ~~> {}".
                             format(sid,old_c,new_c))
        mlog.debug("analyze results for {} sids".format(len(self)))
        cores_d = Cores_d()

        if covs_d:
            mlog.debug("verify ...")
            cache = {}
            for sid,core in self.iteritems():
                configs = frozenset(covs_d[sid])
                key = (core, configs)
                if key not in cache:
                    core_ = core.verify(configs, dom)
                    cache[key]=core_
                    show_compare(sid,core,core_)
                else:
                    core_ = cache[key]

                cores_d[sid]=core
        else:
            cores_d = self

        mlog.debug("simplify ...")
        cache = {}
        for sid in cores_d:
            core = cores_d[sid]
            if core not in cache:
                core_,expr = core.simplify(
                    dom, z3db, do_firsttime=(covs_d is not None))
                cache[core]=core_
                show_compare(sid,core,core_)
            else:
                core_ = cache[core]
            cores_d[sid]=core_

        return cores_d
    
class Mcores_d(CC.CustDict):
    """
    A mapping from core -> {sids}
    """
    def add(self,core, sid):
        assert compat(core, PNCore),core
        assert isinstance(sid, str),str
        
        super(Mcores_d, self).add_set(core, sid)

    def __str__(self):
        mc = sorted(self.iteritems(),
                    key=lambda (core,cov): (core.sstren,core.vstren,len(cov)))
        ss = ("{}. ({}) {}: {}"
              .format(i+1,core.sstren,core,CC.str_of_cov(cov))
              for i,(core,cov) in enumerate(mc))
        return '\n'.join(ss)

    def fix_duplicates(self, dom, z3db):
        assert isinstance(dom, Dom)
        assert not isinstance(z3db, Dom)
        
        def find_dup(expr, d):
            for pc in d:
                expr_ = pc.z3expr(dom, z3db)
                if ((expr is None and expr_ is None) or 
                    (expr is not None and expr_ is not None and
                     z3util.is_tautology(expr == expr_, z3db.solver))):
                    return pc
                    
            return None #no dup

        uniqs = {}
        for pc in self:
            expr = pc.z3expr(dom, z3db)
            dup = find_dup(expr, uniqs)
            if dup:
                uniqs[dup].add(pc)
            else:
                uniqs[pc] = set()

        if len(uniqs) == len(self):  #no duplicates
            return self
        else:
            mlog.debug('merge {} dups'.format(len(self) - len(uniqs)))
            mc_d = Mcores_d()
            for pc in uniqs:
                for sid in self[pc]:
                    mc_d.add(pc, sid)

                for dup in uniqs[pc]:
                    for sid in self[dup]:
                        mc_d.add(pc, sid)
            return mc_d
                    
    @property
    def vtyps(self):
        try:
            return self._vtyps
        except AttributeError:
            d = {'conj' : 0, 'disj' : 0, 'mix' : 0}
            for core in self:
                vtyp = core.vtyp
                d[vtyp] += 1

            self._vtyps = (d['conj'], d['disj'], d['mix'])
            return self._vtyps
    
    @property
    def strens(self):
        """
        (strength,cores,sids)
        """
        strens = set(core.sstren for core in self)

        rs = []
        for stren in sorted(strens):
            cores = [c for c in self if c.sstren == stren]
            cov = set(sid for core in cores for sid in self[core])
            rs.append((stren,len(cores),len(cov)))
        return rs

    @property
    def strens_str(self): return self.str_of_strens(self.strens)

    def show_results(self):
        mlog.info("inferred results ({}):\n{}".format(len(self), self))
        mlog.debug("strens (stren, nresults, nsids): {}"
                     .format(self.strens_str))
        
    @classmethod
    def str_of_strens(cls, strens):
        return ', '.join("({}, {}, {})".format(siz, ncores, ncov)
                         for siz, ncores, ncov in strens)


class DTrace(object):
    """
    Object for saving information (for later analysis)
    """
    def __init__(self,citer,itime,xtime,
                 nconfigs,ncovs,ncores,
                 cconfigs_d,new_covs,new_cores,
                 sel_core,cores_d):

        self.citer = citer
        self.itime = itime
        self.xtime = xtime
        self.nconfigs = nconfigs
        self.ncovs = ncovs
        self.ncores = ncores
        self.cconfigs_d = cconfigs_d
        self.new_covs = new_covs
        self.new_cores = new_cores
        self.sel_core = sel_core
        self.cores_d = cores_d
        
    def show(self, dom, z3db):
        assert isinstance(dom, Dom)
        assert isinstance(z3db, CC.Z3DB)
        
        mlog.debug("ITER {}, ".format(self.citer) +
                    "{}s, ".format(self.itime) +
                    "{}s eval, ".format(self.xtime) +
                    "total: {} configs, {} covs, {} cores, "
                    .format(self.nconfigs,self.ncovs,self.ncores) +
                    "new: {} configs, {} covs, {} updated cores, "
                    .format(len(self.cconfigs_d),
                            len(self.new_covs),len(self.new_cores)) +
                    "{}".format("** progress **"
                                if self.new_covs or self.new_cores else ""))

        mlog.debug('select core: ({}) {}'.format(self.sel_core.sstren,
                                                   self.sel_core))
        mlog.debug('create {} configs'.format(len(self.cconfigs_d)))
        mlog.debug("\n"+str(self.cconfigs_d))
        
        mcores_d = self.cores_d.merge(dom, z3db)
        mlog.debug("infer {} interactions".format(len(mcores_d)))
        mlog.debug('\n{}'.format(mcores_d))
        
        mlog.debug("strens: {}".format(mcores_d.strens_str))

    @staticmethod
    def save_pre(seed,dom,tmpdir):
        CC.vsave(os.path.join(tmpdir,'pre'),(seed,dom))

    @staticmethod
    def save_post(pp_cores_d,itime_total,tmpdir):
        CC.vsave(os.path.join(tmpdir,'post'),(pp_cores_d,itime_total))

    @staticmethod
    def save_iter(cur_iter,dtrace,tmpdir):
        CC.vsave(os.path.join(tmpdir,'{}.tvn'.format(cur_iter)),dtrace)

    @staticmethod
    def load_pre(dir_):
        seed,dom = CC.vload(os.path.join(dir_,'pre'))
        return seed,dom

    @staticmethod
    def load_post(dir_):
        pp_cores_d,itime_total = CC.vload(os.path.join(dir_,'post'))
        return pp_cores_d,itime_total

    @staticmethod
    def load_iter(dir_,f):
        dtrace = CC.vload(os.path.join(dir_,f))
        return dtrace

    @staticmethod
    def str_of_summary(seed,iters,itime,xtime,nconfigs,ncovs,tmpdir):
        ss = ["Seed {}".format(seed),
              "Iters {}".format(iters),
              "Time ({}s, {}s)".format(itime,xtime),
              "Configs {}".format(nconfigs),
              "Covs {}".format(ncovs),
              "Tmpdir {}".format(tmpdir)]
        return "Summary: " + ', '.join(ss)    

    @classmethod
    def load_dir(cls, dir_):        
        seed,dom = cls.load_pre(dir_)
        dts = [cls.load_iter(dir_,f)
               for f in os.listdir(dir_) if f.endswith('.tvn')]
        try:
            pp_cores_d,itime_total = cls.load_post(dir_)
        except IOError:
            mlog.error("post info not avail")
            pp_cores_d, itime_total = None, None
        return seed, dom, dts, pp_cores_d, itime_total
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
