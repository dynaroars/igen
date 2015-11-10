import os.path
import tempfile
from time import time
import vu_common as CM
import config_common as CC

    
def get_run_f(prog, args, mod):
    dom,config_default = mod.Dom.get_dom(
        os.path.realpath(args.dom_file))
    run_script = os.path.realpath(args.run_script)
    cfg = mod.CFG.mk_from_lines(CM.iread_strip(
        os.path.realpath(args.cfg_file)))
    import get_cov
    get_cov_f = lambda config: get_cov.runscript_get_cov(config,run_script)
        
    iga = mod.IGa(dom, cfg, get_cov_f)
    sids = set([args.sid]) if args.sid else iga.sids
    sids = [sid for sid in sids if 'fake' not in sid]
    _f = lambda seed, tdir: iga.go(seed=seed, sids=sids, tmpdir=tdir)
    return _f, get_cov_f

def _tmpdir(tmp_dir,prog):
    import getpass
    d_prefix = "{}_bm_{}_".format(getpass.getuser(),prog)
    tdir = tempfile.mkdtemp(dir=tmp_dir,prefix=d_prefix)
    return tdir

if __name__ == "__main__":
    import argparse
    aparser = argparse.ArgumentParser(
        "iGA (configuration generation using Gentic Algorithm)")
    aparser.add_argument("inp", help="inp", nargs='?')

    #0 Error #1 Warn #2 Info #3 Debug #4 Detail
    aparser.add_argument("--logger_level", "-logger_level",
                         help="set logger info",
                         type=int, 
                         choices=range(5),
                         default = 4)    
    
    aparser.add_argument("--seed", "-seed",
                         type=float,
                         help="use this seed")

    aparser.add_argument("--sid", "-sid", 
                         help="solve this specific sid",
                         action="store")
    
    aparser.add_argument("--random_only",
                         help="solve using pure random search",
                         default=False,
                         action="store_true")
    
    aparser.add_argument("--do_full", "-do_full",
                         help="use all possible configs",
                         action="store_true")

    aparser.add_argument("--noshow_cov",
                         help="show coverage info",
                         action="store_true")

    aparser.add_argument("--dom_file", "-dom_file",
                         help="file containing config domains",
                         action="store")

    aparser.add_argument("--cfg_file", "-cfg_file",
                         help="file containing precessors from cfg",
                         action="store")
    
    aparser.add_argument("--run_script", "-run_script",
                         help="a script running the subject program",
                         action="store")
    
    args = aparser.parse_args()
    CC.logger_level = args.logger_level
    seed = round(time(),2) if args.seed is None else float(args.seed)

    import iga_common as GO
    from iga_settings import tmp_dir
    prog = args.inp
    _f,_ = get_run_f(prog, args, GO)

    prog_name = prog if prog else 'noname'
    tdir = _tmpdir(tmp_dir, prog_name)
    _ = _f(seed,tdir)
