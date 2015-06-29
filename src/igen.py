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
              "ex_motiv2c" : "ex_motiv2",
              'ex_simple_header': "ex_simple",
              'ex_simple_outp': "ex_simple_outp"
              }

def runscript_get_cov(config,run_script,outdir,from_outfile=False):
    inputs = ' , '.join(['{} {}'.format(vname,vval) for
                         vname,vval in config.iteritems()])
    outfile = os.path.join(outdir,"run_script_result.txt")
    cmd = "{} \"{}\" > {}".format(run_script,inputs,outfile)
    try:
        _,rs_err = CM.vcmd(cmd)
    except:
        print("runsript error: cmd '{}' failed".format(cmd))
        
    if from_outfile:
        cov_filename = list(set(CM.iread_strip(outfile)))
        assert len(cov_filename) == 1, cov_filename
        cov_filename = cov_filename[0]
        cov = set(CM.iread_strip(cov_filename))
        print "read {} covs from '{}'".format(len(cov),cov_filename)
    else:
        cov = set(CM.iread_strip(outfile))
        
    return cov,[]
    
def get_run_f(args):
    
    if args.dom_file:  #general way to run program
        dom,config_default = config.Dom.get_dom(os.path.realpath(args.dom_file))
        run_script = os.path.realpath(args.run_script)
        assert os.path.isfile(run_script)
        outdir=tempfile.mkdtemp(dir='/var/tmp',prefix="vu_outdir")
        get_cov = lambda config: runscript_get_cov(
            config,run_script,
            outdir=outdir,
            from_outfile=args.from_outfile)

        igen = config.IGen(dom,get_cov,config_default=config_default)        
        if args.rand_n:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    elif args.prog in otter_d:
        dom,get_cov,pathconds_d=config.prepare_otter(args.prog)
        igen = config.IGen(dom,get_cov,config_default=None)
        if args.do_gt or args.do_full:
            if args.rand_n:
                _f = lambda _,tdir: config.do_gt(dom,pathconds_d,
                                                 n=args.rand_n,tmpdir=tdir)
            else:
                _f = lambda _,tdir: config.do_gt(dom,pathconds_d)
        elif args.rand_n is None:
            _f = lambda seed,tdir: igen.go(seed=seed,tmpdir=tdir)
        else:
            _f = lambda seed,tdir: igen.go_rand(rand_n=args.rand_n,
                                                seed=seed,tmpdir=tdir)
    else:
        if args.prog in examples_d:
            dom,get_cov=config.prepare_motiv(examples_d[args.prog],args.prog)
        elif args.prog in config_coreutils.coreutils_d:
            dom,get_cov=config_coreutils.prepare(args.prog)
        else:
            raise AssertionError("unrecognized prog '{}'".format(args.prog))

        igen = config.IGen(dom,get_cov,config_default=None)
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
    
    aparser.add_argument("--seed",
                         type=float,
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
                        help="cov output to a file instead of stdout",
                        action="store_true")

    args = aparser.parse_args()
    prog = args.prog
    config.logger.level = args.logger_level
    CM.__vdebug__ = args.debug

    if args.allows_known_errors:
        config.allows_known_errors = True
    if args.noshow_cov:
        config.show_cov = False
    if args.analyze_outps:
        config.analyze_outps = True
        
    if args.replay:
        config.Analysis.replay(prog)
        exit(0)
    elif args.replay_dirs:
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
        

