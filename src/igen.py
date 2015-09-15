import os.path
import tempfile
from time import time
import vu_common as CM
import config

def runscript_get_cov(config, run_script):

    inputs = ' , '.join(['{} {}'.format(vname,vval) for
                         vname,vval in config.iteritems()])

    import get_cov
    cov = get_cov.run_runscript(run_script,inputs)
    return cov,[]

def get_run_f(args):

    import get_cov_otter as Otter
        
    if args.dom_file:  #general way to run program using a runscript
        dom,config_default = config.Dom.get_dom(
            os.path.realpath(args.dom_file))
        run_script = os.path.realpath(args.run_script)
        assert os.path.isfile(run_script)
        get_cov_f = lambda config: runscript_get_cov(config,run_script)
        igen = config.IGen(dom,get_cov_f,config_default=config_default)
        if args.rand_n:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    elif args.prog in Otter.db:
        dom,get_cov_f,pathconds_d=Otter.prepare(args.prog)
        igen = config.IGen(dom,get_cov_f,config_default=None)
        if args.do_full:
            if args.rand_n:
                _f = lambda _,tdir: Otter.do_full(
                    dom,pathconds_d,tmpdir=tdir,n=args.rand_n)
            else:
                _f = lambda _,tdir: Otter.do_full(dom,pathconds_d,tmpdir=tdir,n=None)
                                                  
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    else:
        import get_cov_motiv as Motiv
        import get_cov_coreutils as Coreutils
        
        if args.prog in Motiv.db:
            dom,get_cov_f=Motiv.prepare(args.prog)
        elif args.prog in Coreutils.db:
            dom,get_cov_f=Coreutils.prepare(args.prog,do_perl=args.do_perl)
        else:
            raise AssertionError("unrecognized prog '{}'".format(args.prog))

        igen = config.IGen(dom,get_cov_f,config_default=None)
        if args.do_full:
            _f = lambda _,tdir: igen.go_full(tmpdir=tdir)
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    return _f

if __name__ == "__main__":
    import argparse
    aparser = argparse.ArgumentParser()
    aparser.add_argument("prog", help="prog")
    
    aparser.add_argument("--debug",help="set debug on (can be slow)",
                         action="store_true")
    
    #0 Error #1 Warn #2 Info #3 Debug #4 Detail
    aparser.add_argument("--logger_level",
                         help="set logger info",
                         type=int, 
                         choices=range(5),
                         default = 4)    

    aparser.add_argument("--replay",
                         help="replay info from run dir",
                         action="store_true")

    aparser.add_argument("--replay_dirs",
                         help="replay info from adir containing multiple run dirs",
                         action="store_true")

    aparser.add_argument("--show_iters",
                         help="for use with replay, show stats of all iters",
                         action="store_true")

    aparser.add_argument("--do_min_configs",
                         help="for use with replay, compute a set of min configs",
                         action="store_true")

    aparser.add_argument("--seed",
                         type=float,
                         help="use this seed")

    aparser.add_argument("--rand_n",
                         type=int,
                         help="rand_n is an integer")

    aparser.add_argument("--do_full",
                         help="use all possible configs",
                         action="store_true")

    aparser.add_argument("--noshow_cov",
                         help="show coverage info",
                         action="store_true")

    aparser.add_argument("--analyze_outps",
                         help="analyze outputs instead of coverage",
                         action="store_true")

    aparser.add_argument("--allows_known_errors",
                         help="allows for potentially no coverage exec",
                         action="store_true")
    
    aparser.add_argument("--do_mixed_conj_disj",
                         help="do both conj and disj interactions",
                         action="store_true")

    aparser.add_argument("--benchmark",
                         type=int,
                         help="do benchmark")

    aparser.add_argument("--dom_file",
                         help="the domain file",
                         action="store")

    aparser.add_argument("--run_script",
                         help="a script running the subject program",
                         action="store")
    
    aparser.add_argument("--from_outfile",
                         help="cov output to a file (DEPRECATED)",
                         action="store_true")
    
    aparser.add_argument("--do_perl",
                         help="do coretutils written in perl",
                         action="store_true")
    
    args = aparser.parse_args()
    config.logger.level = args.logger_level
    CM.__vdebug__ = args.debug

    if args.allows_known_errors:
        config.allows_known_errors = True
    if args.noshow_cov:
        config.show_cov = False
    if args.analyze_outps:
        config.analyze_outps = True
        
    if args.replay or args.replay_dirs:
        import config_analysis as analysis
        analysis.logger.level = args.logger_level
        analysis_f =  (analysis.Analysis.replay if args.replay else
                       analysis.Analysis.replay_dirs)
        analysis_f(args.prog,show_iters=args.show_iters,
                   do_min_configs=args.do_min_configs)
            
        exit(0)

    nruns = args.benchmark if args.benchmark else 1
    _f = get_run_f(args)

    import getpass
    d_prefix = "{}_bm_{}_".format(getpass.getuser(),args.prog)

    tdir = tempfile.mkdtemp(dir='/var/tmp',prefix=d_prefix)
                            
    st = time()
    if args.seed is None:
        seed = round(time(),2)
    else:
        seed = float(args.seed)

    print("* benchmark '{}',  {} runs, seed {}, results in '{}'"
          .format(args.prog,nruns,seed,tdir))
    
    for i in range(nruns):        
        st_ = time()
        seed_ = seed + i
        tdir_ = tempfile.mkdtemp(dir=tdir,prefix="run{}_".format(i))
        print("*run {}/{}".format(i+1,nruns))
        _ = _f(seed_,tdir_)
        print("*run {}, seed {}, time {}s, '{}'".format(i+1,seed_,time()-st_,tdir_))

    print("** done benchmark '{}', {} runs, seed {}, time {}, results in '{}'"
          .format(args.prog,nruns,seed,time()-st,tdir))

    # print("\n***** STATISTICS *****\n")
    # config.Analysis.replay_dirs(tdir)
