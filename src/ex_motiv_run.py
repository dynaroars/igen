import os.path
import vu_common as CM

def get_cov(config,args):
    varnames = args['varnames']
    prog = args['prog']
    #tmpdir = args['tmpdir']
    tmpdir = '/var/tmp/'
    prog = os.path.realpath(prog)
    
    config_d = dict(config)
    var_vals = tuple([str(config_d[vname]) for vname in varnames])
    traces = os.path.join(tmpdir,'t.out')
    # print prog
    # print var_vals
    # print traces
    cmd = "{} {} > {}".format(prog,' '.join(var_vals),traces)
    #print cmd
    try:
        _,rs_err = CM.vcmd(cmd)
        assert len(rs_err) == 0, rs_err
    except:
        print("cmd '{}' failed".format(cmd))

    traces = list(CM.iread_strip(traces))
    return traces

