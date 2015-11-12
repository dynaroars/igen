#Motivation/Simple examples
import os.path
import vu_common as CM

import config_common as CC
import get_cov as GC

logger = CM.VLog('example')
logger.level = CC.logger_level

def prepare(prog_name,get_dom_f,dir_):
    if __debug__:
        assert isinstance(prog_name,str),prog_name
        assert callable(get_dom_f),get_dom_f
        assert isinstance(dir_,str),dir_
        
    import platform
    dir_ = CM.getpath(dir_)
    dom_file = CM.getpath(os.path.join(dir_,'{}.dom'.format(prog_name)))
    dom,_ = get_dom_f(dom_file)
    logger.info("dom_file '{}': {}".format(dom_file,dom))
    prog_exe = CM.getpath(os.path.join(dir_,'{}.{}.exe'
                                    .format(prog_name,platform.system())))
    logger.info("prog_exe: '{}'".format(prog_exe))

    data = {'var_names':dom.keys(),
            'prog_name':prog_name,
            'prog_exe':prog_exe,
            'dir_':dir_,
            'get_cov_f':get_cov}

    get_cov_f = lambda config: GC.get_cov_wrapper(config,data)
    return dom,get_cov_f

def get_cov(config, data):
    """
    Traces read from stdin
    """
    if __debug__:
        assert isinstance(config, CC.Config),config
        GC.check_data(data)
        
    tmpdir = '/var/tmp/'
    opts = ' '.join(config[vname] for vname in data['var_names'])
    traces = os.path.join(tmpdir,'t.out')
    cmd = "{} {} > {}".format(data['prog_exe'], opts, traces)
    outps = GC.run(cmd)
    sids = set(CM.iread_strip(traces))
    return sids, outps

