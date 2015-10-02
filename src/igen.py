import os.path
import tempfile
from time import time
import vu_common as CM
import config_common as CC

def get_run_f(prog,args,mod):
    """
    Ret f that takes inputs seed,existing_results,tmpdir 
    and call appropriate iGen function on those inputs
    """
    import get_cov_otter as Otter
    if prog in Otter.db:
        dom,get_cov_f,pathconds_d=Otter.prepare(prog,mod.Dom.get_dom)
        igen = mod.IGen(dom,get_cov_f,config_default=None)

        if args.cmp_rand:
            _f = lambda seed,tdir,rand_n: igen.go_rand(
                rand_n=rand_n,seed=seed,tempdir=tdir)
        elif args.do_full:
            if args.rand_n:
                _f = lambda _,tdir: Otter.do_full(
                    dom,pathconds_d,tmpdir=tdir,n=args.rand_n)
            else:
                _f = lambda _,tdir: Otter.do_full(
                    dom,pathconds_d,tmpdir=tdir,n=None)
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(
                rand_n=args.rand_n,seed=seed,tmpdir=tdir)

    else:
        if args.dom_file:  #general way to run prog using a runscript
            dom,config_default = mod.Dom.get_dom(
                os.path.realpath(args.dom_file))
            run_script = os.path.realpath(args.run_script)
            
            import get_cov
            get_cov_f = lambda config: get_cov.runscript_get_cov(
                config,run_script)

        else:
            import igen_settings            
            import get_cov_example as Example
            import get_cov_coreutils as Coreutils
            
            if prog in Coreutils.db:
                dom,get_cov_f=Coreutils.prepare(
                    prog,
                    mod.Dom.get_dom,
                    igen_settings.coreutils_dir,
                    do_perl=args.do_perl)
            else:
                dom,get_cov_f=Example.prepare(
                    prog,mod.Dom.get_dom,igen_settings.examples_dir)
                    
            config_default = None  #no config default for these
            
        igen = mod.IGen(
            dom,get_cov_f,config_default=config_default)

        if args.cmp_rand:
            _f = lambda seed,tdir,rand_n: igen.go_rand(
                rand_n=rand_n,seed=seed,tmpdir=tdir)
        elif args.do_full:
            _f = lambda _,tdir: igen.go_full(tmpdir=tdir)
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(
                rand_n=args.rand_n,seed=seed,tmpdir=tdir)
                
    return _f,get_cov_f
        
def _tmpdir(tmp_dir,prog):
    import getpass
    d_prefix = "{}_bm_{}_".format(getpass.getuser(),prog)
    tdir = tempfile.mkdtemp(dir=tmp_dir,prefix=d_prefix)
    return tdir

if __name__ == "__main__":
    def _check(v,min_n=None,max_n=None):
        v = int(v)
        if min_n and v < min_n:
            raise argparse.ArgumentTypeError(
                "must be >= {} (inp: {})".format(min_n,v))
        if max_n and v > max_n:
            raise argparse.ArgumentTypeError(
                "must be <= {} (inpt: {})".format(max_n,v))
        return v
    
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument("inp", help="inp")
    
    aparser.add_argument("--debug","-debug",
                         help="set debug on (can be slow)",
                         action="store_true")
    
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
                         help="the domain file",
                         action="store")

    aparser.add_argument("--run_script", "-run_script",
                         help="a script running the subject program",
                         action="store")
    
    aparser.add_argument("--from_outfile", 
                         help="cov output to a file (DEPRECATED)",
                         action="store_true")
    
    aparser.add_argument("--do_perl", "-do_perl",
                         help="do coretutils written in perl",
                         action="store_true")

    #replay options
    aparser.add_argument("--replay", "-replay",
                         help="replay info from run dir",
                         action="store_true")

    aparser.add_argument("--replay_dirs", "-replay_dirs",
                         help="replay info from adir containing multiple run dirs",
                         action="store_true")

    aparser.add_argument("--show_iters", "-show_iters",
                         help="for use with replay, show stats of all iters",
                         action="store_true")

    aparser.add_argument("--do_min_configs", "-do_min_configs",
                         help=("for use with replay, "
                               "compute a set of min configs"),
                         action="store",
                         nargs='?',
                         const='use_existing',
                         default=None,
                         type=str)

    aparser.add_argument("--cmp_rand", "-cmp_rand",
                         help=("for use with replay, "
                               "cmp results against rand configs"),
                         action="store",
                         default=None,
                         type=str)
    
    aparser.add_argument("--cmp_gt", "-cmp_gt",
                         help=("for use with replay, "
                               "cmp results against ground truth"),
                         action="store",
                         default=None,
                         type=str)

    args = aparser.parse_args()
    CM.__vdebug__ = args.debug
    CC.logger_level = args.logger_level
    seed = round(time(),2) if args.seed is None else float(args.seed)
    
    if args.allows_known_errors:
        CC.allows_known_errors = True
    if args.noshow_cov:
        CC.show_cov = False
    if args.analyze_outps:
        CC.analyze_outps = True

    #import here so that settings in CC take effect
    import config as IC    
    from igen_settings import tmp_dir

    if args.replay or args.replay_dirs: #analyze results
        from igen_analysis import Analysis        
        analysis_f = (Analysis.replay if args.replay else
                      Analysis.replay_dirs)

        do_min_configs = args.do_min_configs  
        if do_min_configs and do_min_configs != 'use_existing':
            _,get_cov_f = get_run_f(do_min_configs,args,IC)
            do_min_configs = get_cov_f

        cmp_rand = args.cmp_rand
        if cmp_rand:
            _f,_ = get_run_f(cmp_rand,args,IC)
            tdir = _tmpdir(tmp_dir,
                           cmp_rand+"_cmp_rand")
            cmp_rand = lambda rand_n: _f(seed,tdir,rand_n)
            
        analysis_f(args.inp,show_iters=args.show_iters,
                   do_min_configs=do_min_configs,
                   cmp_gt=args.cmp_gt,
                   cmp_rand=cmp_rand)
            
    else: #run iGen
        prog = args.inp
        _f,_ = get_run_f(prog,args,IC)
        tdir = _tmpdir(tmp_dir,prog)
        
        print("* benchmark '{}',  {} runs, seed {}, results in '{}'"
              .format(prog,args.benchmark,seed,tdir))
        st = time()
        for i in range(args.benchmark):        
            st_ = time()
            seed_ = seed + i
            tdir_ = tempfile.mkdtemp(dir=tdir,prefix="run{}_".format(i))
            print("*run {}/{}".format(i+1,args.benchmark))
            _ = _f(seed_,tdir_)
            print("*run {}, seed {}, time {}s, '{}'".format(
                i+1,seed_,time()-st_,tdir_))

        print("** done benchmark '{}', {} runs, seed {}, time {}, results in '{}'"
              .format(prog,args.benchmark,seed,time()-st,tdir))


