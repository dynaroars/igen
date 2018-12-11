import argparse
import tempfile
import os.path
from time import time

import vcommon as CM
import settings

def check_range(v, min_n=None, max_n=None):
    v = int(v)
    if min_n and v < min_n:
        raise argparse.ArgumentTypeError(
            "must be >= {} (inp: {})".format(min_n, v))
    if max_n and v > max_n:
        raise argparse.ArgumentTypeError(
            "must be <= {} (inpt: {})".format(max_n, v))
    return v

def get_sids(inp):
    """
    Parse statements from the inp string, 
    e.g., "f1.c:5 f2.c:15" or "path/to/file" that contains statements
    """
    assert inp is None or isinstance(inp, str), inp
    
    if not inp:
        return None
    
    inp = inp.strip()
    sids = frozenset(inp.split())
    if len(sids) == 1:
        sid_file = list(sids)[0]
        if os.path.isfile(sid_file):
            #parse sids from file
            lines = CM.iread_strip(sid_file)
            sids = []
            for l in lines:
                sids_ = [s.strip() for s in l.split(',')]
                sids.extend(sids_)
            sids = frozenset(sids)

    return sids if sids else None

def get_run_otter(prog, args, DS, ALG):
    sids = get_sids(args.sids)
    import get_cov_otter as Otter
    dom, get_cov_f, pathconds_d = Otter.prepare(prog, DS.Dom.get_dom)
    igen = ALG.IGen(dom, get_cov_f, sids)

    if sids:
        run_f = lambda seed, tdir: igen.go(seed=seed, tmpdir=tdir)
            
    elif args.cmp_rand:
        #TODO: test this 
        run_f = lambda seed,tdir,rand_n: igen.go_rand(
            rand_n=rand_n, seed=seed, tmpdir=tdir)
        
    elif args.do_full:
        run_f = lambda _,tdir: Otter.do_full(
            dom, pathconds_d, tmpdir=tdir,
            n=args.rand_n if args.rand_n else None)

    elif args.rand_n is None:  #default
        run_f = lambda seed,tdir: igen.go(seed=seed, tmpdir=tdir)
        
    else:
        run_f = lambda seed,tdir: igen.go_rand(
            rand_n=args.rand_n, seed=seed, tmpdir=tdir)

    return dom, get_cov_f, run_f

def get_run_default(prog, args, DS, ALG):
    sids = get_sids(args.sids)
    dom, default_configs, get_cov_f = get_cov_default(prog, sids, args, DS)
    if args.kconstraint_file:
        dom.kconstraint_file = args.kconstraint_file
    
    econfigs = [(c, None) for c in default_configs] if default_configs else []
    igen = ALG.IGen(dom, get_cov_f, sids)
    
    if sids:
        run_f = lambda seed, tdir: igen.go(
            seed=seed, econfigs=econfigs, tmpdir=tdir)

    elif args.cmp_rand:
        run_f = lambda seed, tdir, rand_n: igen.go_rand(
            rand_n=rand_n, seed=seed, econfigs=econfigs, tmpdir=tdir)
        
    elif args.do_full:
        run_f = lambda _,tdir: igen.go_full(tmpdir=tdir)
        
    elif args.rand_n is None:  #default
        run_f = lambda seed, tdir: igen.go(seed=seed, econfigs=econfigs, tmpdir=tdir)
        
    else:
        run_f = lambda seed, tdir: igen.go_rand(
            rand_n=args.rand_n, seed=seed, econfigs=econfigs, tmpdir=tdir)

    return dom, get_cov_f, run_f

def get_cov_default(prog, sids, args, DS):
    if args.dom_file:
        #general way to run prog using dom_file/runscript
        dom_file = CM.getpath(args.dom_file)
        dom, default_configs = DS.Dom.get_dom(dom_file)
        assert os.path.isfile(dom.run_script), dom.run_script
        
        import get_cov
        get_cov_f = lambda config: get_cov.runscript_get_cov(
            config, dom.run_script)
    else:
        import get_cov_coreutils as Coreutils

        dom, default_configs, get_cov_f = Coreutils.prepare(
            prog,
            DS.Dom.get_dom,
            settings.coreutils_main_dir,
            settings.coreutils_doms_dir,
            do_perl=args.do_perl)

    return dom, default_configs, get_cov_f

def get_run_f(prog, args, mlog):
    """
    Ret f that takes inputs seed, tmpdir 
    and call appropriate iGen function on those inputs
    """
    import config as DS
    import alg as ALG
    if prog in settings.otter_progs:
        dom, get_cov_f, run_f = get_run_otter(prog, args, DS, ALG)
    else:
        dom, get_cov_f, run_f = get_run_default(prog, args, DS, ALG)
    mlog.info("dom:\n{}".format(dom))

    return run_f, get_cov_f

if __name__ == "__main__":

    igen_file = CM.getpath(__file__)
    igen_name = os.path.basename(igen_file)
    igen_dir = os.path.dirname(igen_file)
        
    aparser = argparse.ArgumentParser("iGen (dynamic interaction generator)")
    ag = aparser.add_argument
    
    ag("inp", help="inp", nargs='?') 
    
    #0 Error #1 Warn #2 Info #3 Debug #4 Detail
    ag("--logger_level", "-logger_level",
                         "-log", "--log",
                         help="set logger info",
                         type=int, 
                         choices=range(5),
                         default = 4)    

    ag("--seed", "-seed",
                         type=float,
                         help="use this seed")

    ag("--rand_n", "-rand_n", 
                         type=lambda v:check_range(v, min_n=1),
                         help="rand_n is an integer")

    ag("--do_full", "-do_full",
                         help="use all possible configs",
                         action="store_true")

    ag("--noshow_cov", "-noshow_cov",
                         help="show coverage info",
                         action="store_true")

    ag("--analyze_outps", "-analyze_outps",
                         help="analyze outputs instead of coverage",
                         action="store_true")

    ag("--allow_known_errors", "-allow_known_errors",
       help="allows for potentially no coverage exec",
       action="store_true")
    
    ag("--benchmark", "-benchmark",
                         type=lambda v: check_range(v, min_n=1),
                         default=1,
                         help="benchmark program n times")

    ag("--dom_file", "-dom_file",
                         "--domain", "-domain",
                         help="file containing config domains",
                         action="store")

    ag("--run_script", "-run_script",
                         "--rscript", "-rscript",
                         help="script to obtain the program's coverage",
                         action="store")

    ag("--kconstraint_file", "-kconstraint_file",
                         "--kconstraint", "-kconstraint",
                         help="kconfig constraint file",
                         default=None,
                         action="store")

    ag("--do_perl", "-do_perl",
                         help="do coretutils written in Perl",
                         action="store_true")

    ag("--sids", "-sids",
                         help="find interactions for sids, e.g., -sids \"L1 L2\"",
                         action="store")
    
    #analysis options
    ag("--show_iters", "-show_iters",
                         help="for use with analysis, show stats of all iters",
                         action="store_true")

    ag("--minconfigs", "-minconfigs",
                         help=("for use with analysis, "
                               "compute a set of min configs"),
                         action="store",
                         nargs='?',
                         const='use_existing',
                         default=None,
                         type=str)

    ag("--influence", "-influence",
                         help="determine influential options/settings",
                         action="store_true")

    ag("--evolution", "-evolution",
                         help=("compute evolution progress using "
                               "v- and f- scores"),
                         action="store_true")

    ag("--precision", "-precision",
                         help="check if interactions are precise",
                         action="store_true")
    
    ag("--cmp_dir", "-cmp_dir",
                         help=("compare (-evolution or -precision) "
                               "to this dir"),
                         action="store",
                         default=None,
                         type=str)

    ag("--cmp_rand", "-cmp_rand",
                         help=("cmp results against rand configs "
                               "(req -evolution)"),
                         action="store",
                         nargs='?',
                         const='use_dom',
                         default=None,
                         type=str)

    ag("--nomp", "-nomp",
       action="store_true",
       help="don't use multiprocessing")
    
    args = aparser.parse_args()

    settings.doMP = not args.nomp
    
    if args.logger_level != settings.logger_level and 0 <= args.logger_level <= 4:
        settings.logger_level = args.logger_level
    settings.logger_level = CM.getLogLevel(settings.logger_level)
    mlog = CM.getLogger(__name__, settings.logger_level)
    
    if __debug__:
        mlog.warn("DEBUG MODE ON. Can be slow! (Use python -O ... for optimization)")
        
    if args.allow_known_errors: settings.allow_known_errors = True
    if args.noshow_cov: settings.show_cov = False
    if args.analyze_outps: settings.analyze_outps = True
        
    seed = round(time(), 2) if args.seed is None else float(args.seed)
    
    # two main modes:
    # 1. run iGen to find interactions and
    # 2. run Analysis to analyze iGen's generated files    
    analysis_f = None
    if args.inp and os.path.isdir(args.inp):
        from analysis import Analysis
        dirstat = Analysis.get_dir_stat(args.inp)
        if dirstat == Analysis.RUNDIR:
            analysis_f = Analysis.replay
        elif dirstat == Analysis.BENCHMARKDIR:
            analysis_f = Analysis.replay_dirs
                
    if not analysis_f: #run iGen
        prog = args.inp
        run_f, get_cov_f = get_run_f(prog, args, mlog)
        if not prog:
            prog = 'noname'

        prefix = "igen_{}_{}_{}_".format(
            args.benchmark, 'full' if args.do_full else 'normal', prog)
        tdir = tempfile.mkdtemp(dir=settings.tmp_dir, prefix=prefix)

        mlog.info("* benchmark '{}', {} runs, seed {}, results '{}'"
                   .format(prog, args.benchmark, seed, tdir))
        st = time()
        for i in range(args.benchmark):        
            st_ = time()
            seed_ = seed + i
            tdir_ = tempfile.mkdtemp(dir=tdir, prefix="run{}_".format(i))
            mlog.info("*run {}/{}".format(i+1, args.benchmark))
            _ = run_f(seed_, tdir_)  #start running
            mlog.info("*run {}, seed {}, time {}s, '{}'".format(
                i + 1, seed_, time() - st_, tdir_))

        print("** done {} runs, seed {}, time {}, results '{}'"
              .format(args.benchmark, seed, time() - st, tdir))
                            
    else: #run analysis
        do_minconfigs = args.minconfigs  
        if do_minconfigs and (do_minconfigs != 'use_existing' or args.dom_file):
            _, get_cov_f = get_run_f(do_minconfigs, args, mlog)
            do_minconfigs = get_cov_f

        cmp_rand = args.cmp_rand
        if cmp_rand:
            run_f, _ = get_run_f(cmp_rand, args, mlog)
            tdir = tempfile.mkdtemp(dir=settings.tmp_dir,
                                    prefix=cmp_rand + "igen_cmp_rand")
            cmp_rand = lambda tseed, rand_n: run_f(tseed, tdir, rand_n)

        cmp_dir = args.cmp_dir
        analysis_f(args.inp,
                   show_iters=args.show_iters,
                   do_minconfigs=do_minconfigs,
                   do_influence=args.influence,
                   do_evolution=args.evolution,
                   do_precision=args.precision,
                   cmp_rand=cmp_rand,
                   cmp_dir=cmp_dir)
