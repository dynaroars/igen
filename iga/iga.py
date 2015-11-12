from time import time
import vu_common as CM
import config_common as CC

def get_run_f(prog, args, mod):
    
    if args.dom_file:
        #general way to run prog using dom_file/runscript
        dom, config_default = mod.Dom.get_dom(CM.getpath(args.dom_file))
        run_script = CM.getpath(args.run_script)
        import get_cov
        get_cov_f = lambda config: get_cov.runscript_get_cov(config, run_script)
        cfg_file = args.cfg_file
    else:
        import iga_settings
        import get_cov_coreutils as Coreutils
        if prog in Coreutils.db:
            dom, get_cov_f = Coreutils.prepare(
                prog,
                mod.Dom.get_dom,
                iga_settings.coreutils_main_dir,
                iga_settings.coreutils_doms_dir,
                do_perl=False)

            import os.path
            cfg_file = os.path.join(iga_settings.coreutils_main_dir,
                                    'coreutils','obj-gcov', 'src',
                                    '{}.c.011t.cfg.preds'.format(prog))
            cfg_file = CM.getpath(cfg_file)

    cfg = mod.CFG.mk_from_lines(CM.iread_strip(CM.getpath(cfg_file)))
    iga = mod.IGa(dom, cfg, get_cov_f)
    sids = set([args.sid]) if args.sid else iga.sids
    sids = [sid for sid in sids if 'fake' not in sid]
    _f = lambda seed, tdir: iga.go(seed=seed, sids=sids, tmpdir=tdir)
    return _f, get_cov_f

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
    seed = round(time(), 2) if args.seed is None else float(args.seed)

    import iga_common as GO
    from iga_settings import tmp_dir
    prog = args.inp
    _f,_ = get_run_f(prog, args, GO)

    prog_name = prog if prog else 'noname'
    tdir = CC.mk_tmpdir(tmp_dir, "iga_" + prog_name)
    _ = _f(seed,tdir)
