#Motivation/Simple examples
from config import (Config,Dom)
import config as CF
import get_cov as GC
import vu_common as CM
import os.path

logger = CM.VLog('motiv')
logger.level = CF.logger.level

db = {"ex": "ex"}
examples_dir = '../examples'

def prepare(prog_name):
    if CM.__vdebug__:
        assert isinstance(prog_name,str),prog_name

    import platform
    dir_ = CM.getpath(examples_dir)
    dom_file = db[prog_name]
    dom_file = CM.getpath(os.path.join(dir_,"{}.dom".format(dom_file)))
    dom,_ = Dom.get_dom(dom_file)
    logger.info("dom_file '{}': {}".format(dom_file,dom))
    prog_exe = CM.getpath(os.path.join(dir_,"{}.{}.exe"
                                    .format(prog_name,platform.system())))
    logger.info("prog_exe: '{}'".format(prog_exe))

    gcno_file = CM.getpath(os.path.join(dir_,"{}.gcno".format(prog_name)))
    get_cov_f = get_cov_gcov if os.path.isfile(gcno_file) else get_cov

    data = {'var_names':dom.keys(),
            'prog_name':prog_name,
            'prog_exe':prog_exe,
            'dir_':dir_,
            'get_cov_f':get_cov_f}

    get_cov_f = lambda config: GC.get_cov_wrapper(config,data)
    return dom,get_cov_f

def get_cov(config,data):
    """
    Traces read from stdin
    """
    if CM.__vdebug__:
        assert isinstance(config,Config),config
        GC.check_data(data)
        
    tmpdir = '/var/tmp/'
    opts = ' '.join(config[vname] for vname in data['var_names'])
    traces = os.path.join(tmpdir,'t.out')
    cmd = "{} {} > {}".format(data['prog_exe'],opts,traces)
    outps = GC.run(cmd)
    sids = set(CM.iread_strip(traces))
    return sids,outps

