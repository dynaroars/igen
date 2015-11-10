"""
Utilities to read CFG file produced by gcc -fdump-tree-cfg-lineno

To test against lots of CFG's, try
for i in /home/tnguyen/iga_exps/benchmarks/coreutils/coreutils-8.24/obj-gcov/src/*.cfg; do echo $i; python ~/Dropbox/git/iga/src/gcc_cfg.py $i >> out ; done 
"""
import sys
import itertools
import vu_common as CM
from collections import OrderedDict

def splitlines(lines, _f):
    parts = []
    cur_part = []
    for l in lines:
        if _f(l):
            if cur_part:
                parts.append(cur_part)
                cur_part = []
        cur_part.append(l)

    parts.append(cur_part)    
    return parts

def parse_cfg_file(filename):
    """
    Parse a .cfg file produced by gcc -fdump-tree-vcg-lineno
    """
    lines = list(CM.iread_strip(filename,None))
    funs = splitlines(lines,lambda l: l.startswith(";; Function "))
    funs = [f for f in funs if f[0].startswith(";; Function ")]
    funs = [parse_function(f) for f in funs]
    funs_d = OrderedDict((fun_name, fun_d) for fun_name,fun_d in funs)

    #post processing
    #remove weird function
    for fun in funs_d:
        if '_GLOBAL__sub_I_65535_0' in fun:
            funs_d.pop(fun)
    
    #remove nondefined function, e.g., those defined in std lib
    for fun in funs_d:
        blocks = funs_d[fun]['blocks']
        for b in blocks:
            for b in blocks:
                stmts = blocks[b]
                for stmt in stmts:
                    blocks[b][stmt] = [f for f in blocks[b][stmt] if f in funs_d]

    #create fake blocks for blocks appear in succs but not in fun's
    for fun in funs_d:
        avail_blocks = funs_d[fun]['blocks']
        succ_blocks = funs_d[fun]['succs']
        succ_blocks = set(succ_blocks.keys() +
                          list(itertools.chain.from_iterable(
                              succ_blocks.values())))

        blocks = set([b for b in succ_blocks if b not in avail_blocks])
        for b in blocks:
            avail_blocks[b] = OrderedDict()

    #fix empty block, gives it a fake sid to fix the linkage problem,
    #when analyze things, just ignore these fake sid
    #note real code also contains empty blocks sometimes
    for fun in funs_d:
        fun_blocks = funs_d[fun]['blocks']
        for b in fun_blocks:
            if not fun_blocks[b]:
                fake_sid = "{}_b{}".format(fun, b)
                fake_sid = "fake_{}:{}".format(filename,fake_sid)
                fun_blocks[b][fake_sid] = []  #no functions
                #print ("create sid {}".format(fake_sid))

    return funs_d
    
def parse_function(lines):
    if __debug__:
        assert isinstance(lines,list), lines
        
    fun_name = [l for l in lines if l.startswith(";; Function")]
    assert len(fun_name) == 1
    fun_name = fun_name[0]
    fun_name = fun_name.split()[2]

    #parse successess   ;; 2 succs { 3 4 }
    succs = [l for l in lines if ";;" in l]
    succs = [parse_succs(l) for l in lines
             if all(s in l for s in "succs { }".split())]

    #parse blocks
    """
    foo (int x)
    {
    int D.2720;
    
    <bb 2>:
    if (x > 0)
    goto <bb 3>;
    else
    goto <bb 4>;
    
    <bb 3>:
    x = x + 1;
    D.2720 = x;
    goto <bb 5> (<L2>);
    
    <bb 4>:
    x = x + -1;
    D.2720 = x;
    
    <L2>:
    return D.2720;

    return 3 blocks bb2 bb3 bb4
    }
    """    
    blocks = [l for l in lines if ";;" not in l]
    blocks = splitlines(blocks,lambda l: l.startswith("<") and l.endswith(">:"))
    blocks = [b for b in blocks
              if b[0].startswith("<bb ") and b[0].endswith(">:")]
    blocks = [parse_block(b) for b in blocks]
    fun_d = {}
    fun_d['succs'] = OrderedDict(succs)
    fun_d['blocks'] = OrderedDict(blocks)

    return fun_name, fun_d


def parse_block(lines):
    """
    <bb 2>:
    [test_cfg.c : 6:6] if (x > 0)
    goto <bb 3>;
    else
    goto <bb 4>;
    {2: [6]}

    <bb 3>:
    [test_cfg.c : 7:5] x = x + 1;
    [test_cfg.c : 7:5] D.2720 = x;
    [test_cfg.c : 7 : 5] goto <bb 5> (<L2>);
    {3: [7]}

    <bb 4>:
    [test_cfg.c : 9:5] x = x + -1;
    [test_cfg.c : 9:5] D.2720 = x;
    {4: [9]}

    <bb 2>:
    [test_cfg.c : 17:20] D.2722 = argv + 8;
    [test_cfg.c : 17:11] D.2723 = [test_cfg.c : 17] *D.2722;
    [test_cfg.c : 17:7] a = atoi (D.2723);
    [test_cfg.c : 18:20] D.2724 = argv + 16;
    [test_cfg.c : 18:11] D.2725 = [test_cfg.c : 18] *D.2724;
    [test_cfg.c : 18:7] b = atoi (D.2725);
    [test_cfg.c : 19:7] c = foo (a);
    [test_cfg.c : 20:3] printf ([test_cfg.c : 20] "%d\n", c);
    [test_cfg.c : 22:3] D.2726 = 0;
    {2: {17, 18, (19,'foo'), 20}}

    <bb 3>:
    [test_cfg.c : 33:9] D.2739 = bar (b);
    [test_cfg.c : 33:7] c = foo (a, D.2739);
    """
    if __debug__:
        assert isinstance(lines,list), lines
    
    block_name = [l for l in lines if l.startswith('<bb') and l.endswith('>:')]
    assert len(block_name) == 1
    block_name = block_name[0]
    block_name = block_name.replace("<bb","").replace(">:","").strip()

    #parse line
    def is_sid(s):  #[../src/uname.c : 117:5] ....
        return (s.startswith('[') and ']' in s and
                s.count(':', 0,s.find(']')) == 2)
        
    lines = splitlines(lines, is_sid)
    lines = [l for l in lines if l[0].startswith("[") and "]" in l[0]]
    sids = [parse_sid(l) for l in lines]

    #merge, e.g., (f.c,5,None), (f.c,5,foo), (f.c,5,bar) => (f.c,5,[foo,bar])
    sids_d = OrderedDict()
    for sid,sfun in sids:
        if sid not in sids_d:
            sids_d[sid] = [] #functions

        if sfun is not None:
            assert sfun  #cannot be ''
            sids_d[sid].append(sfun)

    return block_name, sids_d
    

def parse_sid(lines):
    """
    [test_cfg.c : 25:8] if (c != 0)
    goto <bb 6>;
    else
    goto <bb 7>;
    
    (test_cfg, 25)
    """
    if __debug__:
        assert lines and all(s in lines[0] for s in '[ : ]'.split()), lines

    fun_name = [get_fun_name(l) for l in lines]
    fun_name = [fn for fn in fun_name if fn]
    if len(fun_name) == 0:
        fun_name = None
    else:
        assert len(fun_name) == 1, fun_name
        fun_name = fun_name[0]
    
    sid = lines[0] #'[test_cfg.c : 6:6] if (x > 0)'
    sid = sid.replace('[','')
    sid = sid.split(']')
    sid = sid[0].split(':')
    sid = ("{}:{}".format(sid[0].strip(),sid[1].strip()), fun_name)
    return sid

def get_fun_name(s):
    not_kw = ["if ", "goto ", "else ", "while ", "printf ", "sprintf ",
              " __assert_fail ",
              " __asm__" ,
              "__builtin_fwrite_unlocked "]

    if all(k in s for k in "( );".split()):
        if any(k in s for k in not_kw):
            return None
        s_orig = s
        #clean up special cases
        #[test_cfg.c : 29:11] D.2734 = [test_cfg.c : 29] *D.2733;        
        if s.startswith('['):  
            s = s[s.find(']')+1 :] #D.2734 = [test_cfg.c : 29] *D.2733;

        #error (0, D.5716, [blah.c : 70] " ... (now %s)", D.5714);
        if s.count('"') == 2:
            #error (0, D.5716, [blah.c : 70] , D.5714);            
            s = s[:s.find('"')] + s[s.rfind('"')+1:]

        if not all(k in s for k in "( );".split()):
            return None
        
        assert s.count(' (') == 1 and s.count(');') == 1, (s_orig,s)
        fun_name = s[:s.index('(')]
        assert fun_name.endswith(' ')
        fun_name = fun_name.split()[-1]
        return fun_name
    else:
        return None

def parse_succs(s):
    """
    >>> parse_succs(";; 2 succs { 3 4 }")
    ('2', ['3', '4'])
    >>> parse_succs(";; 5 succs { 1 }")
    ('5', ['1'])
    """
    s = s.replace(";; ","").replace("succs"," ").replace("{","").replace("}","")
    s = [s.strip() for s in s.split()]
    s = (s[0],s[1:])
    return s


fstelem = lambda elems: elems[0]
lstelem = lambda elems: elems[-1]

def stmt_of_line(cfg, fun, block, sid, do_fst):
    """
    Ret the fst (or last) stmt in the given line
    If the line is a function call, e.g., a=f(g(),h()), 
    then gcc breaks this into a list of functions [g,h,f]
    Thus the fst statement of that line is the fst statement of g.
    Similarly, the last statement of that line is the last statement of f.
    """
    funs = cfg[fun]['blocks'][block][sid]
    funs = [f for f in funs if f in cfg]
    if not funs:
        return sid
    else:
        f = fstelem(funs) if do_fst else lstelem(funs)
        return stmt_of_fun(cfg, f, do_fst)

def stmt_of_block(cfg, fun, bl, do_fst):
    lines = cfg[fun]['blocks'][bl].keys()
    assert lines
    line = (fstelem if do_fst else lstelem)(lines)
    return stmt_of_line(cfg, fun, bl, line, do_fst)

def stmt_of_fun(cfg, fun, do_fst):
    """
    Ret first statement in a function
    """
    blocks = cfg[fun]['blocks']
    sblocks = cfg[fun]['succs']
    assert blocks

    #first block in a function, i.e., not a succ of any block
    #could be empty due to loops while 1: x++, e.g., b succ {b1}, b1 succ {b}
    def _fst(blocks):
        blocks_ = [b for b in blocks
                   if all(b not in sblocks[b_] for b_ in sblocks)]
        if blocks_:
            assert len(blocks_) == 1, (blocks_, fun)
            return blocks_[0]
        else:
            b = min(blocks, key=lambda b: int(b)) #the smallest block id
            print("can't determine fst block in fun '{}', ".format(fun) + 
                  "choose one with smallest id {}".format(b))
            return b

    #last block in a function, one that has no successor
    #e.g., b succs {b'} where b' is not in blocks
    #_lst = lambda b: len(sblocks[b]) == 1 and sblocks[b][0] not in blocks
    def _lst(blocks):
        blocks_ = [b for b in blocks if b not in sblocks]
        assert len(blocks_) == 1, (blocks_, fun)
        return blocks_[0]
        

    b = (_fst if do_fst else _lst)(blocks)
    return stmt_of_block(cfg, fun, b, do_fst)
    
        
def compute_preds(cfg):
    preds = []
    
    #chain sids in block
    #block [sid1, .., sidn] means  sid_i is pred of sid_i+1
    for fun in cfg:
        blocks = cfg[fun]['blocks']
        for b in blocks:
            stmts = blocks[b].keys()
            assert stmts
            
            for s,s_ in zip(stmts[1:], stmts):
                #print s, s_, blocks[b][s_]
                funs = blocks[b][s_]
                if funs:
                    lstfun = lstelem(blocks[b][s_])
                    lst_st = stmt_of_fun(cfg, lstfun, do_fst=False)
                    fstfun = fstelem(blocks[b][s_])
                    fst_st = stmt_of_fun(cfg, fstfun, do_fst=True)
                    # print 'lstfun', lstfun, lst_st, s
                    # print 'fstfun', fstfun, fst_st, s
                    assert lst_st != s
                    preds.append((lst_st,s))
                    assert s != fst_st
                    preds.append((s_,fst_st))
                else:
                    assert s_ != s
                    preds.append((s_, s))


    #chain function calls
    #[f,g...] means lst(f) is pred of fst(g)
    for fun in cfg:
        blocks = cfg[fun]['blocks']
        for b in blocks:
            stmts = blocks[b]
            assert stmts

            for stmt in stmts:
                funs = blocks[b][stmt]
                if not funs:
                    continue

                for f,g in zip(funs, funs[1:]):
                    lst_f = stmt_of_fun(cfg, f, do_fst=False)
                    fst_g = stmt_of_fun(cfg, g, do_fst=True)
                    assert lst_f != fst_g , (fun, f, g)
                    preds.append((lst_f, fst_g))



    #chain function blocks
    #succs f: {g, h} means lst(b) is pred of fst(g), fst(h),
    for fun in cfg:
        blocks = cfg[fun]['blocks']
        sblocks = cfg[fun]['succs']
        for f in sblocks:
            assert f in blocks
            # print fun, f, sblocks[f]
            # CM.pause()
            for g in sblocks[f]:
                assert g in blocks
                lst_f = stmt_of_block(cfg, fun, f, do_fst=False)
                fst_g = stmt_of_block(cfg, fun, g, do_fst=True)
                
                #could happen with or, e.g., y = a || b;
                if lst_f != fst_g:
                    preds.append((lst_f, fst_g))

    preds_d = {}
    for f,g in preds:
        if g not in preds_d:
            preds_d[g] = set()
        preds_d[g].add(f)

    return preds_d


def write_preds(filename,preds_d):
    if __debug__:
        assert isinstance(filename,str), filename
        assert isinstance(preds_d,dict), preds_d
        
    strs = '\n'.join("{} {}".format(sid, ' '.join(map(str, preds)))
                     for sid,preds in sorted(preds_d.iteritems(),
                                             key=lambda (sid,preds): sid))
    filename_preds = "{}.preds".format(filename)
    print "write to '{}'".format(filename_preds)
    CM.vwrite(filename_preds, strs)
    
# if __name__ == '__main__':
#     # import doctest
#     # doctest.testmod()
    
#     #print sys.argv, len(sys.argv)

#     filename = sys.argv[1]
#     cfg = parse_cfg(filename)
#     #print cfg
#     preds_d = compute_preds(cfg)
    
    
        
        
    
    
