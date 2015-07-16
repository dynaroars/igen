import os.path
import vu_common as CM
import config as CF

# Real executions
def run_single(cmd):
    logger.detail(cmd)
    try:
        rs_outp,rs_err = CM.vcmd(cmd)
        if rs_outp:
            logger.detail("outp: {}".format(rs_outp))
        
        #IMPORTANT, command out the below allows
        #erroneous test runs, which can be helpful
        #to detect incorrect configs
        #assert len(rs_err) == 0, rs_err

        serious_errors = ["invalid",
                          "-c: line",
                          "/bin/sh"]

        known_errors = ["assuming not executed"]
        if rs_err:
            logger.detail("error: {}".format(rs_err))
            if any(serr in rs_err for serr in serious_errors): 
                raise AssertionError("Check this serious error !")

            if not allows_known_errors:
                if any(kerr in rs_err for kerr in known_errors):
                    raise AssertionError("Check this known error!")
                    
        return (rs_outp,rs_err)
    
    except Exception as e:
        raise AssertionError("cmd '{}' fails, raise error: {}".format(cmd,e))
    
def run(cmds,msg=''):
    "just exec command, does not return anything"
    assert cmds, cmds
    
    if not CM.is_iterable(cmds): cmds = [cmds]
    logger.detail('run {} cmds{}'
                  .format(len(cmds),' ({})'.format(msg) if msg else''))
    outp = tuple(run_single(cmd) for cmd in cmds)
    outp = hash(outp)
    return set([str(outp)])

from gcovparse import gcovparse
def parse_gcov(gcov_file):
    if CM.__vdebug__:
        assert os.path.isfile(gcov_file)

    gcov_obj = gcovparse(CM.vread(gcov_file))
    assert len(gcov_obj) == 1, gcov_obj
    gcov_obj = gcov_obj[0]
    sids = (d['line'] for d in gcov_obj['lines'] if d['hit'] > 0)
    sids = set("{}:{}".format(gcov_obj['file'],line) for line in sids)
    return sids


class GC_D(object):
    def __init__(var_names,
                 prog_name,
                 prog_exe,
                 prog_dir, #only used for coreutils
                 dir_,  #where execute prog_exe from
                 get_cov_f):


        assert os.path.isfile(prog_exe),prog_exe
        assert os.path.isdir(dir_)
        self.var_names = var_names
        self.prog_name = prog_name
        self.prog_exe = prog_exe
        self.prog_dir = prog_dir
        self.dir_ = dir_
        self.get_cov_f = get_cov_f        
        
def get_cov_wrapper(config,data):
    """
    If anything happens, return to current directory
    """
    if CM.__vdebug__:
        assert isinstance(data,GC_D),data 
        
    cur_dir = os.getcwd()
    try:
        os.chdir(data.dir_)
        rs = data.get_cov_f(config,data)
        os.chdir(cur_dir)
        return rs
    except:
        os.chdir(cur_dir)
        raise

        
