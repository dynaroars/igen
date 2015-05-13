import vu_common as CM


def my_get_cov(config, run_script,tmpdir,from_outfile=False):
    inputs = ' , '.join(['{} {}'.format(vname,vval) for vname,vval in config.iteritems()])
    outfile = os.path.join(tmpdir,"run_script_result.txt")
    cmd = "{} \"{}\" > {}".format(run_script,inputs,outfile)
    try:
        _,rs_err = CM.vcmd(cmd)
    except:
        print("cmd '{}' failed".format(cmd))
        
    if not from_outfile:
        cov = set(CM.iread_strip(outfile))
    else:
        cov_filename = list(set(CM.iread_strip(outfile)))
        assert len(cov_filename) == 1, cov_filename
        cov_filename = cov_filename[0]
        cov = set(CM.iread_strip(cov_filename))
        print "read {} covs from '{}'".format(len(cov),cov_filename)
    return cov

if __name__ == "__main__":
    import argparse

    import os.path
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--dom_file",
                        help="the domain file",
                        action="store")

    parser.add_argument("--run_script",
                        help="a script running the subject program",
                        action="store")

    parser.add_argument("--seed",
                        help="a seed to reproduce run",
                        action="store")

    parser.add_argument("--from_outfile",
                        help="cov output to a file instead of stdout",
                        action="store_true")

    parser.add_argument("--debug",
                        help="turn on debug option (make things slower)",
                        action="store_true")

    parser.add_argument("--print_cov",
                        help="print out coverage (for debugging)",
                        action="store_true")
    
    parser.add_argument("--verbose",
                        help="increase output verbosity",
                        default=2,
                        type=int,
                        action="store")
    
    aparse = parser.parse_args()

    #read the domain file
    dom_file = os.path.realpath(aparse.dom_file)
    run_script = os.path.realpath(aparse.run_script)
    assert os.path.isfile(run_script)

    import tempfile
    tmpdir = tempfile.mkdtemp(dir='/var/tmp',prefix='vu')

    
    #Run the algorithm
    import vconfig
    vconfig.logger.level = aparse.verbose
    vconfig.vdebug = aparse.debug
    vconfig.print_cov = aparse.print_cov
    
    get_cov = lambda config: my_get_cov(config,run_script,tmpdir,from_outfile=aparse.from_outfile)
    dom,config_default = vconfig.get_dom(dom_file)
    _ = vconfig.iterative_refine(dom=dom,
                                 get_cov=get_cov,
                                 seed=aparse.seed,
                                 tmpdir=None,
                                 pure_random_n=None,
                                 config_default=config_default)
