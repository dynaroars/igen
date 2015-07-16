#Motivation/Simple examples
import config as CF
import get_cov as GC

def prepare(dom_file,prog_name):
    if CM.__vdebug__:
        assert isinstance(dom_file,str),dom_file
        assert isinstance(prog_name,str),prog_name

    import platform
    dir_ = CF.getpath('~/Dropbox/git/config/benchmarks/examples')
    dom_file = CF.getpath(os.path.join(dir_,"{}.dom".format(dom_file)))
    dom,_ = Dom.get_dom(dom_file)
    logger.info("dom_file '{}': {}"
                .format(dom_file,dom))

    prog_exe = CF.getpath(os.path.join(dir_,"{}.{}.exe"
                                    .format(prog_name,platform.system())))
    logger.info("prog_exe: '{}'".format(prog_exe))

    gcno_file = getpath(os.path.join(dir_,"{}.gcno".format(prog_name)))
    get_cov_f = get_cov_gcov if os.path.isfile(gcno_file) else get_cov

    data = GC_D(var_names=dom.keys(),
                prog_name=prog_name,
                prog_exe=prog_exe,
                dir_=dir_,
                prog_dir = None,
                get_cov_f=get_cov_f)

    get_cov = lambda config: GC.get_cov_wrapper(config,data)
    return dom,get_cov

def get_cov(config,data):
    """
    Traces read from stdin
    """
    if CM.__vdebug__:
        assert isinstance(config,CF.Config),config
        assert isinstance(data,GC.GC_D),data
        
    tmpdir = '/var/tmp/'
    opts = ' '.join(config[vname] for vname in data.var_names)
    traces = os.path.join(tmpdir,'t.out')
    cmd = "{} {} > {}".format(data.prog_exe,opts,traces)
    outps = GC.run(cmd)
    sids = set(CM.iread_strip(traces))
    return sids,outps

def get_cov_gcov(config,args):
    """
    Traces ared from gcov info
    """
    if CM.__vdebug__:
        assert isinstance(config,CF.Config),config
        assert isinstance(data,GC.GC_D),data
    
    opts = ' '.join(config[vname] for vname in data.var_names)
    
    #cleanup
    cmd = "rm -rf *.gcov *.gcda"
    _ = run(cmd)
    #run testsuite
    cmd = "{} {}".format(data.prog_exe,opts)
    outps = GC.run(cmd)

    #read traces from gcov
    #/path/prog.Linux.exe -> prog
    cmd = "gcov {}".format(data.prog_name)
    _ = GC.run(cmd)
    gcov_dir = os.getcwd()
    sids = (GC.parse_gcov(os.path.join(gcov_dir,f))
            for f in os.listdir(gcov_dir) if f.endswith(".gcov"))
    sids = set(CM.iflatten(sids))
    return sids,outps


examples_d = {"ex_motiv1": "ex_motiv1",
              "ex_motiv1b": "ex_motiv1",
              "ex_motiv2" : "ex_motiv2",
              "ex_motiv2a" : "ex_motiv2",              
              "ex_motiv2b" : "ex_motiv2",
              "ex_motiv2c" : "ex_motiv2",
              "ex_motiv2d" : "ex_motiv2",              
              "ex_motiv4" : "ex_motiv4",
              "ex_motiv5" : "ex_motiv5",
              "ex_motiv6" : "ex_motiv6",
              "ex_motiv7" : "ex_motiv7",
              "ex_motiv8" : "ex_motiv8",
              "ex_motiv8b" : "ex_motiv8",              
              'ex_simple_header': "ex_simple",
              'ex_simple_outp': "ex_simple_outp"
              }

