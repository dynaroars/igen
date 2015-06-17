import os.path
import tempfile
from time import time

import vu_common as CM
import config
import config_coreutils

otter_d = {"vsftpd":None,
           "ngircd":None}

examples_d = {"ex_motiv1": "ex_motiv1",
              "ex_motiv1b": "ex_motiv1",
              "ex_motiv2" : "ex_motiv2",
              "ex_motiv2a" : "ex_motiv2",              
              "ex_motiv2b" : "ex_motiv2",
              "ex_motiv2c" : "ex_motiv2"}

def get_run_f(args):
    if args.prog in otter_d:
        dom,get_cov,pathconds_d=config.prepare_otter(args.prog)
        igen = config.IGen(dom,get_cov,config_default=None)
        if args.do_gt or args.do_full:
            if args.rand_n:
                _f = lambda _,tdir: config.do_gt(dom,pathconds_d,n=args.rand_n,tmpdir=tdir)
            else:
                _f = lambda _,tdir: config.do_gt(dom,pathconds_d,tmpdir=tdir)
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,seed=seed,tmpdir=tdir)
    else:
        if args.prog in examples_d:
            dom,get_cov=config.prepare_motiv(examples_d[args.prog],args.prog)
        elif args.prog in config_coreutils.coreutils_d:
            dom,get_cov=config_coreutils.prepare(args.prog)            
        else:
            raise AssertionError("unrecognized prog '{}'".format(args.prog))

        igen = config.IGen(dom,get_cov,config_default=None)
        if args.do_full:
            _f = lambda tdir: igen.go_full(tmpdir=tdir)
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,seed=seed,tmpdir=tdir)
        return _f

if __name__ == "__main__":
    """
    ./igen "ls" 
    ./igen "vsftpd" --do_gt
    ./igen "motiv2" --do_full
    
    """
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
                         default = 2)    

    aparser.add_argument("--replay",
                         help="replay info from run dir",
                         action="store_true")

    aparser.add_argument("--replay_dirs",
                         help="replay info from adir containing multiple run dirs",
                         action="store_true")
    
    aparser.add_argument("--seed",
                         type=int,
                         help="use this seed")

    aparser.add_argument("--rand_n",
                         type=int,
                         help="rand_n is an integer")

    aparser.add_argument("--do_full",
                         help="use all possible configs",
                         action="store_true")

    aparser.add_argument("--do_gt",
                         help="obtain ground truths",
                         action="store_true")

    aparser.add_argument("--show_cov",
                         help="show coverage info",
                         action="store_true")
    
    aparser.add_argument("--do_mixed_conj_disj",
                         help="do both conj and disj interactions",
                         action="store_true")

    aparser.add_argument("--benchmark",
                         type=int,
                         help="do benchmark")
    
    args = aparser.parse_args()
    prog = args.prog
    config.logger.level = args.logger_level
    CM.__vdebug__ = args.debug

    if args.replay:
        config.Analysis.replay(prog)
        exit(0)
        
    if args.replay_dirs:
        config.Analysis.replay_dirs(prog)
        exit(0)


    nruns = args.benchmark if args.benchmark else 1
    _f = get_run_f(args)        
    tdir = tempfile.mkdtemp(dir='/var/tmp',prefix="vu_bm")
    st = time()
    if args.seed is None:
        seed = round(time(),2)
    else:
        seed = float(args.seed)

    for i in range(nruns):        
        st_ = time()
        seed_ = seed + i
        tdir_ = tempfile.mkdtemp(dir=tdir,prefix="run{}_".format(i))
        print("*run {}/{}".format(i+1,nruns))
        _ = _f(seed_,tdir_)
        print("*run {}, seed {}, time {}s, '{}'".format(i+1,seed_,time()-st_,tdir_))

    print("** done with {} runs, seed {}, time {}s, results stored in '{}'"
          .format(nruns,seed,time()-st,tdir))
        

