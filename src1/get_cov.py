import gcovparse
import os.path
import config_common as CC
import vcommon as CM
from time import time
import settings
mlog = CM.getLogger(__name__, settings.logger_level)

# Real executions


def run_single(cmd):
    mlog.debug(cmd)
    rs_err = "some error occured:"
    try:
        st = time()
        rs_outp, rs_err = CM.vcmd(cmd)
        if rs_outp:
            mlog.debug("outp: {}".format(rs_outp))
        print("etime {}".format(time()-st))
        # NOTE: comment out the below allows
        # erroneous test runs, which can be helpful
        # to detect incorrect configs
        #assert len(rs_err) == 0, rs_err

        serious_errors = ["invalid",
                          "-c: line",
                          "/bin/sh",
                          "assuming not executed"]

        known_errors = ["invalid file number in field spec"]
        if rs_err:
            mlog.debug("error: {}".format(rs_err))

            if settings.allow_known_errors:
                if (not any(kerr in rs_err for kerr in known_errors) and
                        any(serr in rs_err for serr in serious_errors)):
                    raise AssertionError("Check this serious error!")
            else:
                if any(kerr in rs_err for kerr in known_errors):
                    raise AssertionError("Check this known error!")
                if any(serr in rs_err for serr in serious_errors):
                    raise AssertionError("Check this serious error!")

        return (rs_outp, rs_err)

    except Exception as e:
        raise AssertionError("cmd '{}' fails, raise error: {}, {}"
                             .format(cmd, rs_err, e))


def runscript_get_cov(config, run_script):
    """
    Get cov from config (a dict with {var -> val} mapping)
    """
    assert os.path.isfile(run_script), run_script

    inputs = ' , '.join(['{} {}'.format(vname, vval) for
                         vname, vval in config.items()])

    cov = run_runscript(run_script, inputs)
    return cov, []


def run_runscript(run_script, arg):
    """
    Exec runscript on arg and return a single line representing the cov file
    E.g., ./runscript.sh "args"
    """

    cmd = "{} \"{}\"".format(run_script, arg)
    rs_outp, _ = run_single(cmd)
    cov_filename = [l for l in rs_outp.split('\n') if l]
    assert len(cov_filename) == 1, (cmd, rs_outp, cov_filename)
    cov_filename = cov_filename[0]
    cov = set(CM.iread_strip(cov_filename))
    mlog.debug("cmd {}, read {} covs from '{}'"
               .format(cmd, len(cov), cov_filename))
    return cov


def run(cmds, msg=''):
    "just exec command, does not return anything"
    assert cmds, cmds

    if not hasattr(cmds, "__iter__"):  # iterable
        cmds = [cmds]

    mlog.debug('run {} cmds{}'
               .format(len(cmds), ' ({})'.format(msg) if msg else''))
    outp = tuple(run_single(cmd) for cmd in cmds)
    outp = hash(outp)
    return set([str(outp)])


def parse_gcov(gcov_file):
    if __debug__:
        assert os.path.isfile(gcov_file)

    gcov_obj = gcovparse.gcovparse(CM.vread(gcov_file))
    assert len(gcov_obj) == 1, gcov_obj
    gcov_obj = gcov_obj[0]
    sids = (d['line'] for d in gcov_obj['lines'] if d['hit'] > 0)
    sids = set("{}:{}".format(gcov_obj['file'], line) for line in sids)
    return sids


def check_data(data):
    assert isinstance(data, dict)
    assert 'var_names' in data
    assert 'prog_name' in data
    assert 'prog_exe' in data
    assert 'dir_' in data  # where execute prog_exe from
    assert 'get_cov_f' in data


def get_cov_wrapper(config, data):
    """
    If anything happens, return to current directory
    """
    if __debug__:
        check_data(data)

    cur_dir = os.getcwd()
    try:
        os.chdir(data['dir_'])
        rs = data['get_cov_f'](config, data)
        os.chdir(cur_dir)
        return rs
    except:
        os.chdir(cur_dir)
        raise
