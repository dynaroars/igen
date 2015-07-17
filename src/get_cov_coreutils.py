import abc 
import os.path
import vu_common as CM
import config as CF
import get_cov as CG
from collections import namedtuple

logger = CM.VLog('coreutils')
logger.level = CF.logger.level

TS_D = namedtupled('prog','opts','cdir','tdir')

def prepare(prog_name,do_perl):
    if CM.__vdebug__:
        assert isinstance(prog_name,str),prog_name
        assert isinstance(do_perl,bool), do_perl
        
    main_dir = CF.getpath('~/Dropbox/git/config/benchmarks/coreutils')
    dom_file = os.path.join(main_dir,"doms","{}.dom".format(prog_name))
    dom_file = CF.getpath(dom_file)
    dom,_ = CF.Dom.get_dom(dom_file)
    logger.info("dom_file '{}': {}".format(dom_file,dom))
    
    assert all(len(vs) >= 2 and "off" in vs 
               for vs in dom.itervalues()),"incorrect format"

    if do_perl:
        prog_dir = os.path.join(main_dir,'coreutils_perl')
        dir_ = None
    else:
        bdir = os.path.join(main_dir,'coreutils')
        prog_dir = os.path.join(bdir,'obj-gcov','src')
        dir_ = os.path.join(bdir,'src')
        assert os.path.isdir(dir_)
        
    prog_exe = os.path.join(prog_dir,prog_name)
    assert os.path.isfile(prog_exe),prog_exe
    logger.info("prog_exe: '{}'".format(prog_exe))

    data = GC.GC_D(dom.keys(),prog_name,prog_exe,
                   get_cov_perl if do_perl else get_cov_gcov,
                   prog_dir,
                   dir_,main_dir)
    get_cov_f = lambda config: CF.get_cov_wrapper(config,data)
    return dom,get_cov_f

def get_opts(config,ks):
    """
    >>> ks = "-x -y z + --o".split()
    >>> config = dict(zip(ks,'on off something "%M" "date"'.split()))
    >>> assert get_opts(config,ks) == '-x z something +"%M" --o="date"'
    """
    opts = []
    for k in ks:
        if config[k] == "off":
            continue
        elif config[k] == "on":
            opts.append(k)
        elif k == "+": #nospace
            opts.append("{}{}".format(k,config[k]))
        elif k.startswith("--"):  # --option=value
            opts.append("{}={}".format(k,config[k]))
        else: #k v
            opts.append("{} {}".format(k,config[k]))

    return ' '.join(opts)


def get_cov_perl(config,args):
    if __CM.vdebug__:
        assert isinstance(data,GC_D), data
    
    #clean up ?
    
    #run testsuite
    ts_data = TS_D(prog=data.prog_exe,
                   opts=get_opts(config,data.var_names) ,
                   cdir=os.path.join(data.main_dir,'testfiles','common'),
                   tdir=os.path.join(data.main_dir,'testfiles',data.prog_name))
    ts = coreutils_d[data.prog_name](ts_data)
    ts = ["perl -MDevel::Cover {}".format(t) for t in ts]
    outps = ts.run()

    #read traces
    parse_script = 'perl parse_script'
    sids = CF.run("{} {}".format(parse_script,os.getcwd()))
    sids = set(sids)
    if not sids:
        logger.warn("config {} has NO cov".format(config))

    return sids,outps

def get_cov_gcov(config,data):
    if __CM.vdebug__:
        assert isinstance(data,GC_D), data
        
    #cleanup
    cmd = "rm -rf {}/*.gcov {}/*.gcda".format(data.dir_,data.prog_dir)
    _ = CF.run(cmd,'cleanup')
    
    #run testsuite
    ts_data = TS_D(prog=data.prog_exe,
                   opts=get_opts(config,data.var_names) ,
                   cdir=os.path.join(data.main_dir,'testfiles','common'),
                   tdir=os.path.join(data.main_dir,'testfiles',data.prog_name))
    ts = coreutils_d[data.prog_name](ts_data)
    outps = ts.run()

    #read traces from gcov
    #/path/prog.Linux.exe -> prog
    src_dir = os.path.join(data.dir_,'src')
    cmd = "gcov {} -o {}".format(data.prog_name,data.prog_dir)
    _ = CF.run(cmd,'gcov')
    
    gcov_dir = os.getcwd()
    sids = (CF.parse_gcov(os.path.join(gcov_dir,f))
            for f in os.listdir(gcov_dir) if f.endswith(".gcov"))
    sids = set(CM.iflatten(sids))
    if not sids:
        logger.warn("config {} has NO cov".format(config))
        
    return sids,outps

class TestSuite_COREUTILS(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,data):
        if CM.__vdebug__:
            assert isinstance(data,TS_D), data

        self.prog = data.prog
        self.opts = data.opts
        self.cdir = data.cdir
        self.tdir = data.tdir

    @abc.abstractmethod
    def get_cmds(self): pass

    def run(self):
        cmds = self.get_cmds()
        outps = CF.run(cmds,'run testsuite')
        return outps
    
    @property
    def cmd_default(self): return "{} {}".format(self.prog,self.opts)

    @property
    def cmd_notexist(self): return "{} {} ~notexistfile".format(self.prog,self.opts)
    
class TS_uname(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append(self.cmd_default)
        cmds.append(self.cmd_notexist)
        cmds.append("{} {} a".format(self.prog,self.opts))
        return cmds
    
class TS_id(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append(self.cmd_default)
        cmds.append(self.cmd_notexist)        
        cmds.append("{} {} root".format(self.prog,self.opts)) #id user
        cmds.append("{} {} {}".format(self.prog,self.opts,self.tdir)) #id nonexisting user
        return cmds

class TS_cat(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append(self.cmd_default)
        cmds.append(self.cmd_notexist)
        cmds.append("{} {} {}/file1".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/file3".format(self.prog,self.opts,self.tdir))        
        cmds.append("{} {} < {}/file2".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/binary.dat".format(self.prog,self.opts,self.cdir))
        cmds.append("{} {} {}/small.jpg".format(self.prog,self.opts,self.cdir))
        
        cmds.append("{} {} {}/*".format(self.prog,self.opts,self.tdir)) #cat /path/*
        cmds.append("{} {} {}/file1 {}/file2".format(self.prog,self.opts,self.tdir,self.tdir))
        cmds.append("ls /boot | {} {}".format(self.prog,self.opts)) #stdin
        #cat f1 - f2
        cmds.append("ls /boot | {} {} {}/file1 - {}/file2".format(self.prog,self.opts,self.tdir,self.tdir)) 
        cmds.append("{} /boot".format(self.prog,self.opts,self.tdir))
        return cmds

class TS_cp(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append("{} {} {}/file1 {}".format(prog,opts,self.tdir,self.tdir))  #cp f .
        #cp f1 f2
        cmds.append("{} {} {}/files1 {}/files2; rm -rf {}/files2".format()) 
        cmds.append("{} {} {}/small.jpg {}/small_t.jpg; rm -rf {}/small_t.jpg"
                    .format(self.prog,self.opts,self.cdir,self.tdir,self.tdir))
        cmds.append("rm -rf {}/tdir/*; {} {} {}/* {}/tdir"
                    .format(tdir,prog,opts,cdir,self.tdir))
        cmds.append("rm -rf {}/tdir/*; {} {} {}/binary.dat {}/file1 {}/tdir"
                    .format(tdir,prog,opts,cdir,self.tdir,self.tdir))
        #recursive copy
        cmds.append("rm -rf {}/tdir/*; ".format(tdir) +
                    "{} {} {}/dir2levels/ {}/tdir".format(prog,opts,self.tdir,self.tdir))
        return cmds

class TS_mv(TestSuite_COREUTILS):
    """
    done
    covs: 151
    (0,1,51), (1,9,36), (2,3,53), (3,3,10), (4,1,1)  
    (0,1,51), (1,9,36), (2,3,53), (3,4,11)
    """
    def get_cmds(self):
        cmds = []
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a ;".format(self.tdir) +
                    "{} {} {}/a {}/b".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a ;".format(self.tdir) +
                    "{} {} ~/b {}/a".format(self.prog,self.opts,self.tdir))
        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a ;".format(self.tdir) +
                    "{} {} {}/a {}/a".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b;".format(self.tdir,self.tdir) +
                    "{} {} {}/a {}/b".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/a {}/b {}/d"
                    .format(self.prog,self.opts,self.tdir,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/d {}/d"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/d {}/e"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/d/ {}/e"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/* {}/d"
                    .format(self.prog,self.opts,self.tdir,self.tdir))
        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "mkdir {}/d ; cd {}/d ; touch a b; "
                    .format(self.tdir,self.tdir) +
                    "{} {} * .."
                    .format(self.prog,self.opts))
        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b;".format(self.tdir,self.tdir) +
                    "{} {} --suffix='_vv' {}/a {}/b "
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b {}/c {}/d;".format(self.tdir,self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/a {}/b ;".format(self.prog,self.opts,self.tdir,self.tdir) +
                    "{} {} {}/c {}/d".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "{} {} /bin/echo ."
                    .format(self.prog,self.opts))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ;"
                    .format(self.tdir,self.tdir) +
                    "{} {} {}/a {}/b /usr/bin/"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        return cmds


class TS_ln(TestSuite_COREUTILS):
    """
    done
    cov 182
    (0,1,13), (1,9,42), (2,3,41), (3,3,8), (4,4,63), (5,1,3), (6,6,9), (7,2,3)
    """
    def get_cmds(self):
        cmds = []
        cmds = []
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a ;".format(self.tdir) +
                    "{} {} {}/a {}/b".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a ;".format(self.tdir) +
                    "{} {} ~/b {}/a".format(self.prog,self.opts,self.tdir))
        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a ;".format(self.tdir) +
                    "{} {} {}/a {}/a".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b;".format(self.tdir,self.tdir) +
                    "{} {} {}/a {}/b".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/a {}/b {}/d"
                    .format(self.prog,self.opts,self.tdir,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/d {}/d"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/d {}/e"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/d/ {}/e"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ; mkdir {}/d ;"
                    .format(self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/* {}/d"
                    .format(self.prog,self.opts,self.tdir,self.tdir))
        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "mkdir {}/d ; cd {}/d ; touch a b; "
                    .format(self.tdir,self.tdir) +
                    "{} {} * .."
                    .format(self.prog,self.opts))
        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b;".format(self.tdir,self.tdir) +
                    "{} {} --suffix='_vv' {}/a {}/b "
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        
        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b {}/c {}/d;".format(self.tdir,self.tdir,self.tdir,self.tdir) +
                    "{} {} {}/a {}/b ;".format(self.prog,self.opts,self.tdir,self.tdir) +
                    "{} {} {}/c {}/d".format(self.prog,self.opts,self.tdir,self.tdir))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "{} {} /bin/echo ."
                    .format(self.prog,self.opts))

        cmds.append("rm -rf {}/* ;".format(self.tdir) +
                    "touch {}/a {}/b ;"
                    .format(self.tdir,self.tdir) +
                    "{} {} {}/a {}/b /usr/bin/"
                    .format(self.prog,self.opts,self.tdir,self.tdir))

        #todo , see option -L in manpage

        return cmds
    
class TS_touch(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append("rm -rf {}/* ;".format(tdir) +
                    "{} {} {}/a {}/b".format(prog,exe,self.tdir))

        cmds.append("rm -rf {}/* ;".format(tdir) +
                    "{} {} {}/a {}/b;".format(prog,exe,self.tdir)+
                    "{} {} {}/*".format(prog,exe,self.tdir))

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a -d 'next Thursday'".format(prog,exe,self.tdir))

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a -r {}/binary.dat".format(prog,exe,cdir,self.tdir))

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a -t 15151730.55")

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a/".format(prog,exe,self.tdir))

        cmds.append("rm -rf {}/tdi;r".format(self.tdir) +
                    "{} {} {}/tdir;".format(self.prog,self.opts,self.tdir) +
                    "{} {} {}/tdir -d '2012-10-19 12:12:12.000000000 +0530'"
                    .format(self.prog,self.opts,self.tdir))
        
        cmds.append("{} {} /usr/bin")
        return cmds


class TS_who(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmd.append("{} {}")
        cmd.append("{} {} 'am i'")        
        cmd.append("{} {} {}/junkfood_ringding_wtmp".format(prog,opts,self.tdir))
        cmd.append("{} {} {}/notexist".format(prog,opts,self.tdir))        
        
        return cmds


    
class TS_pr(TestSuite_COREUTILS):
    """
    ls -a | pr -n -h "Files in $(pwd)" > directory.txt
    pr -l 40 -h Using Hypertext comms_paper
    pr -n2
    http://docstore.mik.ua/orelly/unix3/upt/ch21_15.htm
    """
    def get_cmds(self):
        cmds = []
        return cmds


class TS_rm(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        return cmds


class TS_du(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        return cmds



# class TS_cp(TestSuite_COREUTILS):
#     def get_cmds(self):
#         cmds = []
#         return cmds


# class TS_cp(TestSuite_COREUTILS):
#     def get_cmds(self):
#         cmds = []
#         return cmds
    

class TS_date(TestSuite_COREUTILS):
    def get_cmds(self):

        date_file = os.path.join(self.tdir,'datefile')
        assert os.path.isfile(date_file),date_file

        cmds = []
        cmds.append("{} {}".format(self.prog,self.opts))  #date -options
        cmds.append("{} {} -d '2004-02-29 16:21:42'".format(self.prog,self.opts))
        cmds.append("{} {} -d 'next  Thursday'".format(self.prog,self.opts))
        cmds.append("{} {} -d 'Sun, 29 Feb 2004 16:21:42 -0800'"
                    .format(self.prog,self.opts))
        cmds.append("{} {} -d '@2147483647'".format(self.prog,self.opts))
        cmds.append("{} {} -d 'TZ=\"America/Los_Angeles\" 09:00 next Fri'"
                    .format(self.prog,self.opts))
        cmds.append("TZ='America/Los_Angeles' {} {}".format(self.prog,self.opts))

        cmds.append("{} {} -r /bin/date".format(self.prog,self.opts)) #date -otions /bin/date
        cmds.append("{} {} -r ~".format(self.prog,self.opts)) #date -otions /bin/date

        cmds.append("{} {} -f {}".format(self.prog,self.opts,date_file)) #date -f filename
        cmds.append("{} {} -f notexist".format(self.prog,self.opts)) #date -f filename    
        return cmds

class TS_sort(TestSuite_COREUTILS):
    def get_cmds(self):
        """
        Examples:
        http://www.theunixschool.com/2012/08/linux-sort-command-examples.html
        """
        cmds = []
        cmds.append(self.cmd_default)
        cmds.append(self.cmd_notexist)        
        cmds.append("ls -s /usr/lib/* | {} {}".format(self.prog,self.opts))
        cmds.append("ls -s /boot | {} {}".format(self.prog,self.opts))
        cmds.append("ls /boot/* | {} {}".format(self.prog,self.opts)) 
        cmds.append("{} {} {}/file1".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/file3".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/file1 {}/file2 {}/file3"  
                    .format(self.prog,self.opts,self.tdir,self.tdir,self.tdir)) #multiple fiels
        cmds.append("find {} -name 'file*' -print0 | {} {} --files0-from=-"
                    .format(self.tdir,self.prog,self.opts))
        return cmds
        
class TS_ls(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        #probably don't want these 2, kind useless
        # cmds.append(self.cmd_default)
        # cmds.append(self.cmd_notexist)

        cmds.append("{} {} {}/a".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/a_ln".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}".format(self.prog,self.opts,self.tdir))

        cmds.append("{} {} {}/noexist_ln".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/d1/.hidden1".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/d1/d1d1/b".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/d2/a.sh".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/d2/dirA".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/d2/dirA/*".format(self.prog,self.opts,self.tdir))

        cmds.append("{} {} {}/d2_ln/a.sh".format(self.prog,self.opts,self.tdir))
        cmds.append("{} {} {}/d2_ln/.hidden_dir".format(self.prog,self.opts,self.tdir))

        cmds.append("{} {} /boot".format(self.prog,self.opts))
        cmds.append("{} {} /boot/*".format(self.prog,self.opts))
        cmds.append("{} {} /bin/ls".format(self.prog,self.opts))
        cmds.append("{} {} .".format(self.prog,self.opts))
        cmds.append("{} {} ..".format(self.prog,self.opts))
        return cmds

    
 # #test 1
 #    cmds = []    
 #    cmds.append("touch a")
 #    cmds.append("ln {} a b".format(opts_str))

 #    #test 2
 #    cmds = []        
 #    cmds.append("mkdir d")
 #    cmds.append("ln {} d e".format(opts_str))
 #    myexec(test_dir,cmds)
    
 #    #from coreutils tests
 #    #test 3  # d/f -> ../f
 #    cmds = []        
 #    cmds.append("mkdir d")
 #    cmds.append("ln {} --target_dir=d ../f".format(opts_str))
 #    myexec(test_dir,cmds)
    
 #    #test 4
 #    # Check that a target directory of '.' is supported
 #    # and that indirectly specifying the same target and link name
 #    # through that is detected.
 #    cmds = []        
 #    cmds.append("ln {} . b".format(opts_str))
 #    cmd.append("echo foo > a")      
 #    cmds.append("ln {} a b > err 2>&1)".format(opts_str))
 #    myexec(test_dir,cmds)


coreutils_d = {'date': TS_date,
               'ls': TS_ls,
               'ln': TS_ln,
               'sort': TS_sort,
               'id': TS_id,
               'mv': TS_mv,
               'ln': TS_ln,  
               'uname': TS_uname,
               'cat':TS_cat}


# def testsuite_md5sum(args):
#     """
#     """
#     if CM.__vdebug__:
#         assert isinstance(args,dict),args
#         assert 'opts' in args
#         assert 'prog_exe' in args
#         assert 'testfiles_dir' in args        

#     prog_exe = args['prog_exe']
#     opts = args['opts']
#     tdir = args['testfiles_dir']

#     cmds = []
#     cmds.append("{} {} {}/file1".format(self.prog,self.opts,self.tdir))
#     cmds.append("{} {} {}/file2".format(self.prog,self.opts,self.tdir))
#     cmds.append("{} {} {}/binary.dat".format(self.prog,self.opts,self.tdir))    
#     cmds.append("{} {} {}/file1 {}/binary.dat".format(self.prog,self.opts,self.tdir,self.tdir))
#     cmds.append("{} {} {}/*".format(self.prog,self.opts,self.tdir))
#     cmds.append("cat {}/file1 | {} {} -".format(tdir,self.prog,self.opts))    
#     cmds.append("{} {} {}/results.md5".format(self.prog,self.opts,self.tdir))
#     cmds.append("{} {} {}/results_failed.md5".format(self.prog,self.opts,self.tdir))
    
#     CF.void_run(cmds)
    



# def get_cov_common(args):
#     """
#     >>> args = {'prog_dir': '/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/coreutils/obj-gcov/src', 'var_names': ['-a', '-s', '-n', '-r', '-v', '-m', '-p', '-i', '-o', '--help', '--version'], 'dir_': '/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/coreutils/src', 'prog_exe': '/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/coreutils/obj-gcov/src/uname', 'prog_name': 'uname'}
#     >>> config = HDict([('-a', 'on'), ('-s', 'off'), ('-n', 'off'), ('-r', 'off'), ('-v', 'off'), ('-m', 'off'), ('-p', 'off'), ('-i', 'off'), ('-o', 'off'), ('--help', 'off'), ('--version', 'off')])

#     #>>> sids = get_cov_coreutils(config,args);  len(sids) == 
#     """
#     if CM.__vdebug__:
#         assert isinstance(args,dict), args
#         assert 'main_dir' in args
#         assert 'dir_' in args
#         assert 'prog_dir' in args
#         assert 'prog_name' in args
#         assert 'prog_exe' in args
#         assert 'var_names' in args

