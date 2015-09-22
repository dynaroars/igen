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

def get_run_f(prog,args):
    """
    Ret f that takes inputs seed,existing_results,tmpdir 
    and call appropriate iGen function on those inputs
    """
    import get_cov_otter as Otter
        
    if args.dom_file:  #general way to run program using a runscript
        dom,config_default = config.Dom.get_dom(
            os.path.realpath(args.dom_file))
        run_script = os.path.realpath(args.run_script)
        assert os.path.isfile(run_script)
        get_cov_f = lambda config: runscript_get_cov(config,run_script)
        igen = config.IGen(dom,get_cov_f,config_default=config_default)

        if args.rand_n:
            _f = lambda seed,_,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,_,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    elif prog in Otter.db:
        dom,get_cov_f,pathconds_d=Otter.prepare(prog)
        igen = config.IGen(dom,get_cov_f,config_default=None)
        if args.do_full:
            if args.rand_n:
                _f = lambda _,__,tdir: Otter.do_full(
                    dom,pathconds_d,tmpdir=tdir,n=args.rand_n)
            else:
                _f = lambda _,__,tdir: Otter.do_full(
                    dom,pathconds_d,tmpdir=tdir,n=None)
                                                  
        elif args.rand_n is None:
            _f = lambda seed,_,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,_,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    else:
        import get_cov_example as Example
        import get_cov_coreutils as Coreutils
        
        if prog in Example.db:
            dom,get_cov_f=Example.prepare(prog)
            
        elif prog in Coreutils.db:
            dom,get_cov_f=Coreutils.prepare(prog,do_perl=args.do_perl)
        else:
            raise AssertionError("unrecognized prog '{}'".format(prog))

        igen = config.IGen(dom,get_cov_f,config_default=None)
        if args.do_full:
            _f = lambda _,__,tdir: igen.go_full(tmpdir=tdir)
        # elif args.do_min_configs_from:
        #     _f = lambda seed,existing_results,tdir:\
        #          igen.go_minconfigs(seed=seed,
        #                             existing_results=existing_results,
        #                             tmpdir=tdir)
        elif args.rand_n is None:
            _f = lambda seed,_,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,_,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    return _f,get_cov_f

if __name__ == "__main__":

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
                         help="for use with replay, compute a set of min configs",
                         action="store",
                         nargs='?',
                         const='use_existing',
                         default=None,
                         type=str)
    
    aparser.add_argument("--seed", "-seed",
                         type=float,
                         help="use this seed")

    aparser.add_argument("--rand_n", "-rand_n", 
                         type=int,
                         help="rand_n is an integer")

    aparser.add_argument("--do_full", "-do_ful",
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
    
    aparser.add_argument("--benchmark", "-benchmark",
                         type=int,
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

    args = aparser.parse_args()
    config.logger.level = args.logger_level
    CM.__vdebug__ = args.debug

    if args.allows_known_errors:
        config.allows_known_errors = True
    if args.noshow_cov:
        config.show_cov = False
    if args.analyze_outps:
        config.analyze_outps = True

    def _tmpdir(prog):
        from igen_settings import tmp_dir    
        import getpass
        d_prefix = "{}_bm_{}_".format(getpass.getuser(),prog)
        tdir = tempfile.mkdtemp(dir=tmp_dir,prefix=d_prefix)
        return tdir
    
    if args.replay or args.replay_dirs: #analyze results
        import config_analysis as analysis
        analysis.logger.level = args.logger_level
        analysis_f = (analysis.Analysis.replay if args.replay else
                      analysis.Analysis.replay_dirs)

        do_min_configs = args.do_min_configs
        if do_min_configs and do_min_configs != 'use_existing':
            _,get_cov_f = get_run_f(do_min_configs,args)
            do_min_configs = get_cov_f
            
        analysis_f(args.inp,show_iters=args.show_iters,
                   do_min_configs=do_min_configs)
            
    else: #run iGen
        prog = args.inp
        _f,_ = get_run_f(prog,args)
        tdir = _tmpdir(prog)
        
        seed = round(time(),2) if args.seed is None else float(args.seed)
        print("* benchmark '{}',  {} runs, seed {}, results in '{}'"
              .format(prog,args.benchmark,seed,tdir))

        st = time()
        for i in range(args.benchmark):        
            st_ = time()
            seed_ = seed + i
            tdir_ = tempfile.mkdtemp(dir=tdir,prefix="run{}_".format(i))
            print("*run {}/{}".format(i+1,args.benchmark))
            _ = _f(seed_,None,tdir_)
            print("*run {}, seed {}, time {}s, '{}'".format(i+1,seed_,time()-st_,tdir_))

        print("** done benchmark '{}', {} runs, seed {}, time {}, results in '{}'"
              .format(prog,args.benchmark,seed,time()-st,tdir))

