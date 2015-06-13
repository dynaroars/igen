import config
import vu_common as CM

otter_d = {"vsftpd":None,
           "ngircd":None}
coreutils_d = {"ls":None,
               "uname":None,
               "date":None}
examples_d = {"ex_motiv1": "ex_motiv1",
              "ex_motiv1b": "ex_motiv1",
              "ex_motiv2" : "ex_motiv2",
              "ex_motiv2a" : "ex_motiv2",              
              "ex_motiv2b" : "ex_motiv2",
              "ex_motiv2c" : "ex_motiv2"}

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
    
    aparser.add_argument("--seed",
                         type=int,
                         help="use this seed")

    aparser.add_argument("--n",
                         type=int,
                         help="n is an integer")

    aparser.add_argument("--do_full",
                         help="use all possible configs",
                         action="store_true")

    #Options for running otter
    aparser.add_argument("--do_gt",
                         help="obtain ground truths",
                         action="store_true")

    args = aparser.parse_args()
    debug = args.debug
    prog = args.prog
    config.logger.level = args.logger_level
    seed = args.seed

    if args.replay:
        config.replay(args.prog)

    elif prog in otter_d:
        dom,get_cov,pathconds_d=config.prepare_otter(prog)
        if args.do_gt:
            if args.n:
                _ = config.do_gt(dom,pathconds_d,n=args.n)
            else:
                _ = config.do_gt(dom,pathconds_d)
        else:
            config.igen(dom,get_cov,seed=seed)
            
    else:
        if prog in examples_d:
            dom,get_cov=config.prepare_motiv(examples_d[prog],prog)            
        elif prog in coreutils_d:
            dom,get_cov=config.prepare_coreutils(prog)            
        else:
            raise AssertionError("unrecognized prog '{}'".format(prog))

        if args.do_full:
            _ = config.igen_full(dom,get_cov)

        elif args.n:
            _ = config.igen_rand(dom,get_cov,n=args.n,seed=seed)
        else:
            _ = config.igen(dom,get_cov,seed=seed)

