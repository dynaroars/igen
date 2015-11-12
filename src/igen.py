import os.path
import tempfile
from time import time
import vu_common as CM
import config_common as CC

def get_run_f(prog, args, mod):
    """
    Ret f that takes inputs seed, tmpdir 
    and call appropriate iGen function on those inputs
    """
    import get_cov_otter as Otter
    if prog in Otter.db:
        dom, get_cov_f, pathconds_d = Otter.prepare(prog, mod.Dom.get_dom)
        igen = mod.IGen(dom, get_cov_f, config_default=None)

        if args.cmp_rand:
            #TODO: test this 
            _f = lambda seed,tdir,rand_n: igen.go_rand(
                rand_n=rand_n, seed=seed, tmpdir=tdir)

        elif args.do_full:
            if args.rand_n:
                _f = lambda _,tdir: Otter.do_full(
                    dom, pathconds_d, tmpdir=tdir, n=args.rand_n)
            else:
                _f = lambda _,tdir: Otter.do_full(
                    dom, pathconds_d, tmpdir=tdir, n=None)
                
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed, tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(
                rand_n=args.rand_n, seed=seed, tmpdir=tdir)
    else:        
        if args.dom_file:
            #general way to run prog using dom_file/runscript
            dom, config_default = mod.Dom.get_dom(CM.getpath(args.dom_file))
            run_script = CM.getpath(args.run_script)
            
            import get_cov
            get_cov_f = lambda config: get_cov.runscript_get_cov(
                config, run_script)
        else:
            import igen_settings
            import get_cov_coreutils as Coreutils            
            dom, get_cov_f = Coreutils.prepare(
                prog,
                mod.Dom.get_dom,
                igen_settings.coreutils_main_dir,
                igen_settings.coreutils_doms_dir,
                do_perl=args.do_perl)
                    
            config_default = None  #no default config for these
            
        igen = mod.IGen(dom, get_cov_f, config_default=config_default)
        if args.cmp_rand:
            _f = lambda seed, tdir, rand_n: igen.go_rand(
                rand_n=rand_n, seed=seed, tmpdir=tdir)
            
        elif args.do_full:
            _f = lambda _,tdir: igen.go_full(tmpdir=tdir)
            
        elif args.rand_n is None:
            _f = lambda seed, tdir: igen.go(seed=seed, tmpdir=tdir)
            
        else:
            _f = lambda seed, tdir: igen.go_rand(
                rand_n=args.rand_n, seed=seed, tmpdir=tdir)
                
    return _f, get_cov_f

if __name__ == "__main__":
    
    igen_file = CM.getpath(__file__)
    igen_name = os.path.basename(igen_file)
    igen_dir = os.path.dirname(igen_file)

    import argparse
    def _check(v, min_n=None, max_n=None):
        v = int(v)
        if min_n and v < min_n:
            raise argparse.ArgumentTypeError(
                "must be >= {} (inp: {})".format(min_n, v))
        if max_n and v > max_n:
            raise argparse.ArgumentTypeError(
                "must be <= {} (inpt: {})".format(max_n, v))
        return v

    def _check_inps(args):
        t1 = args.run_script and args.dom_file
        t2 = args.run_script is None and args.dom_file is None
        #either t1 or t2
        if not (not t2 or args.inp):
            raise argparse.ArgumentTypeError("need some input")
        
        if not (t1 or t2):
            raise argparse.ArgumentTypeError(
                "req valid inp or use the options -run_script -dom together")
        #t2 => args.inp 
        
    aparser = argparse.ArgumentParser("iGen (dynamic interaction generator)")
    aparser.add_argument("inp", help="inp", nargs='?') 
    
    #0 Error #1 Warn #2 Info #3 Debug #4 Detail
    aparser.add_argument("--logger_level", "-logger_level",
                         help="set logger info",
                         type=int, 
                         choices=range(5),
                         default = 4)    
    
    aparser.add_argument("--seed", "-seed",
                         type=float,
                         help="use this seed")

    aparser.add_argument("--rand_n", "-rand_n", 
                         type=lambda v:_check(v,min_n=1),
                         help="rand_n is an integer")

    aparser.add_argument("--do_full", "-do_full",
                         help="use all possible configs",
                         action="store_true")

    aparser.add_argument("--noshow_cov", "-noshow_cov",
                         help="show coverage info",
                         action="store_true")

    aparser.add_argument("--analyze_outps", "-analyze_outps",
                         help="analyze outputs instead of coverage",
                         action="store_true")

    aparser.add_argument("--allows_known_errors", "-allows_known_errors",
                         help="allows for potentially no coverage exec",
                         action="store_true")
    
    aparser.add_argument("--benchmark", "-benchmark",
                         type=lambda v:_check(v,min_n=1),
                         default=1,
                         help="run benchmark program n times")

    aparser.add_argument("--dom_file", "-dom_file",
                         help="file containing config domains",
                         action="store")

    aparser.add_argument("--run_script", "-run_script",
                         help="a script running the subject program",
                         action="store")
    
    aparser.add_argument("--do_perl", "-do_perl",
                         help="do coretutils written in perl",
                         action="store_true")

    #replay options
    aparser.add_argument("--show_iters", "-show_iters",
                         help="for use with analysis, show stats of all iters",
                         action="store_true")

    aparser.add_argument("--do_min_configs", "-do_min_configs",
                         help=("for use with analysis, "
                               "compute a set of min configs"),
                         action="store",
                         nargs='?',
                         const='use_existing',
                         default=None,
                         type=str)

    aparser.add_argument("--cmp_rand", "-cmp_rand",
                         help=("for use with analysis, "
                               "cmp results against rand configs"),
                         action="store",
                         default=None,
                         type=str)
    
    aparser.add_argument("--cmp_gt", "-cmp_gt",
                         help=("for use with analysis, "
                               "cmp results against ground truth"),
                         action="store",
                         default=None,
                         type=str)

    args = aparser.parse_args()
    _check_inps(args)
    
    CC.logger_level = args.logger_level
    logger = CM.VLog(igen_name)
    logger.level = CC.logger_level
    if __debug__: logger.warn("DEBUG MODE ON. Can be slow !")
    
    seed = round(time(), 2) if args.seed is None else float(args.seed)
    
    if args.allows_known_errors:
        CC.allows_known_errors = True
    if args.noshow_cov:
        CC.show_cov = False
    if args.analyze_outps:
        CC.analyze_outps = True

    #import here so that settings in CC take effect
    #import config as IA
    import igen_alg as IA
    from igen_settings import tmp_dir
    from igen_analysis import Analysis

    # two main modes: 1. run iGen to find interactions and
    # 2. run Analysis to analyze iGen's generated files    
    analysis_f = None
    if args.inp and os.path.isdir(args.inp):
        is_run_dir = Analysis.is_run_dir(args.inp)
        if is_run_dir is not None:
            if is_run_dir:
                analysis_f = Analysis.replay
            else:
                analysis_f = Analysis.replay_dirs

    if analysis_f is None: #run iGen
        prog = args.inp
        _f, _ = get_run_f(prog, args, IA)

        prog_name = prog if prog else 'noname'
        prefix = "{}_{}_{}".format(
            args.benchmark,
            'full' if args.do_full else 'normal',
            prog_name)
        tdir = CC.mk_tmpdir(tmp_dir, "igen_" + prefix)

        logger.debug("* benchmark '{}',  {} runs, seed {}, results in '{}'"
                     .format(prog_name, args.benchmark, seed, tdir))
        st = time()
        for i in range(args.benchmark):        
            st_ = time()
            seed_ = seed + i
            tdir_ = tempfile.mkdtemp(dir=tdir, prefix="run{}_".format(i))
            logger.debug("*run {}/{}".format(i+1, args.benchmark))
            _ = _f(seed_, tdir_)
            logger.debug("*run {}, seed {}, time {}s, '{}'".format(
                i+1, seed_, time() - st_, tdir_))

        logger.debug("** done benchmark '{}' {} runs, seed {}, "
                     "time {}, results in '{}'"
                     .format(prog_name, args.benchmark,
                             seed, time() - st, tdir))

    else: #run analysis
        do_min_configs = args.do_min_configs  
        if do_min_configs and do_min_configs != 'use_existing':
            _,get_cov_f = get_run_f(do_min_configs, args, IA)
            do_min_configs = get_cov_f

        cmp_rand = args.cmp_rand
        if cmp_rand:
            _f,_ = get_run_f(cmp_rand,args, IA)
            tdir = CC.mk_tmpdir(tmp_dir, cmp_rand + "igen_cmp_rand")
            cmp_rand = lambda rand_n: _f(seed, tdir, rand_n)

        analysis_f(args.inp,
                   show_iters=args.show_iters,
                   do_min_configs=do_min_configs,
                   cmp_gt=args.cmp_gt,
                   cmp_rand=cmp_rand)
