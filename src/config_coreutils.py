import abc 
import os.path
import vu_common as CM
import config as CF

logger = CM.VLog('coreutils')
logger.level = CF.logger.level

#Coreutils
def prepare(prog_name):
    if CM.__vdebug__:
        assert isinstance(prog_name,str),prog_name

    main_dir = CF.getpath('~/Dropbox/git/config/benchmarks/coreutils')
    dom_file = os.path.join(main_dir,"doms","{}.dom".format(prog_name))
    dom_file = CF.getpath(dom_file)
    assert os.path.isfile(dom_file),dom_file

    dom,_ = CF.Dom.get_dom(dom_file)
    logger.info("dom_file '{}': {}".format(dom_file,dom))

    bdir = os.path.join(main_dir,'coreutils')
    prog_dir = os.path.join(bdir,'obj-gcov','src')
    prog_exe = os.path.join(prog_dir,prog_name)
    assert os.path.isfile(prog_exe),prog_exe
    logger.info("prog_exe: '{}'".format(prog_exe))

    dir_ = os.path.join(bdir,'src')
    assert os.path.isdir(dir_)
    args = {'var_names':dom.keys(),
            'prog_name': prog_name,
            'prog_exe': prog_exe,
            'get_cov': get_cov,
            'prog_dir':prog_dir,
            'dir_': dir_,
            'main_dir':main_dir}
    get_cov_f = lambda config: CF.get_cov_wrapper(config,args)    
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

def get_cov(config,args):
    """
    >>> args = {'prog_dir': '/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/coreutils/obj-gcov/src', 'var_names': ['-a', '-s', '-n', '-r', '-v', '-m', '-p', '-i', '-o', '--help', '--version'], 'dir_': '/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/coreutils/src', 'prog_exe': '/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/coreutils/obj-gcov/src/uname', 'prog_name': 'uname'}
    >>> config = HDict([('-a', 'on'), ('-s', 'off'), ('-n', 'off'), ('-r', 'off'), ('-v', 'off'), ('-m', 'off'), ('-p', 'off'), ('-i', 'off'), ('-o', 'off'), ('--help', 'off'), ('--version', 'off')])

    #>>> sids = get_cov_coreutils(config,args);  len(sids) == 
    """
    if CM.__vdebug__:
        assert isinstance(args,dict), args
        assert 'main_dir' in args
        assert 'dir_' in args
        assert 'prog_dir' in args
        assert 'prog_name' in args
        assert 'prog_exe' in args
        assert 'var_names' in args


    dir_ = args['dir_']
    main_dir = args['main_dir']    
    prog_dir = args['prog_dir']
    prog_name = args['prog_name']
    prog_exe = args['prog_exe']
    var_names = args['var_names']
    opts = get_opts(config,var_names)

    #cleanup
    cmd = "rm -rf {}/*.gcov {}/*.gcda".format(dir_,prog_dir)
    CF.void_run(cmd)
    #CM.pause()
    
    #run testsuite
    cdir = os.path.join(main_dir,'testfiles','common')    
    tdir = os.path.join(main_dir,'testfiles',prog_name)
    margs = {'prog':prog_exe, 'opts':opts,'cdir':cdir,'tdir':tdir}
    ts = coreutils_d[prog_name](margs)
    ts.run()

    #read traces from gcov
    #/path/prog.Linux.exe -> prog
    src_dir = os.path.join(dir_,'src')
    cmd = "gcov {} -o {}".format(prog_name,prog_dir)
    CF.void_run(cmd)
    
    gcov_dir = os.getcwd()
    sids = (CF.parse_gcov(os.path.join(gcov_dir,f))
            for f in os.listdir(gcov_dir) if f.endswith(".gcov"))
    sids = set(CM.iflatten(sids))
    if not sids:
        logger.warn("config {} has NO cov".format(config))
        
    return sids

class TestSuite_COREUTILS(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self,args):
        if CM.__vdebug__:
            assert isinstance(args,dict),args
            assert 'prog' in args            
            assert 'opts' in args
            assert 'cdir' in args and os.path.isdir(args['cdir']),args['cdir']            
            assert 'tdir' in args and os.path.isdir(args['tdir']),args['tdir']

        self.prog = args['prog']
        self.opts = args['opts']
        self.cdir = args['tdir']        
        self.tdir = args['tdir']

    @abc.abstractmethod
    def get_cmds(self): pass

    def run(self):
        cmds = self.get_cmds()
        assert cmds
        CF.void_run(cmds)

    @property
    def cmds_default(self):
        cmds = []
        cmds.append("{} {}".format(self.prog,self.opts))
        return cmds


class TS_uname(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = self.cmds_default
        cmds.append("{} {} a".format(self.prog,self.opts))
        return cmds
    
class TS_id(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = self.cmds_default
        cmds.append("{} {} root".format(self.prog,self.opts)) #id user
        cmds.append("{} {} {}".format(self.prog,self.opts,self.tdir)) #id nonexisting user
        return cmds

class TS_cat(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append("{} {} {}/file1".format(prog_exe,opts,tdir))
        cmds.append("{} {} {}/file3".format(prog_exe,opts,tdir))        
        cmds.append("{} {} < {}/file2".format(prog_exe,opts,tdir))
        cmds.append("{} {} {}/binary.dat".format(prog_exe,opts,cdir))
        cmds.append("{} {} {}/small.jpg".format(prog_exe,opts,cdir))        
        
        cmds.append("{} {} {}/*".format(prog_exe,opts,tdir)) #cat /path/*
        cmds.append("{} {} {}/file1 {}/file2".format(prog_exe,opts,tdir,tdir))
        cmds.append("ls /usr/bin | {} {}".format(prog_exe,opts)) #stdin
        #cat f1 - f2
        cmds.append("ls /usr/bin | {} {} {}/file1 - {}/file2".format(prog_exe,opts,tdir,tdir)) 

        cmds.append("{} nonexist".format(prog_exe,opts,tdir))
        cmds.append("{} /usr/bin".format(prog_exe,opts,tdir))
        return cmds

class TS_cp(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append("{} {} {}/file1 {}".format(prog,opts,tdir,tdir))  #cp f .
        #cp f1 f2
        cmds.append("{} {} {}/files1 {}/files2; rm -rf {}/files2".format()) 
        cmds.append("{} {} {}/small.jpg {}/small_t.jpg; rm -rf {}/small_t.jpg"
                    .format(self.prog,self.opts,self.cdir,self.tdir,self.tdir))
        cmds.append("rm -rf {}/tdir/*; {} {} {}/* {}/tdir"
                    .format(tdir,prog,opts,cdir,tdir))
        cmds.append("rm -rf {}/tdir/*; {} {} {}/binary.dat {}/file1 {}/tdir"
                    .format(tdir,prog,opts,cdir,tdir,tdir))
        #recursive copy
        cmds.append("rm -rf {}/tdir/*; ".format(tdir) +
                    "{} {} {}/dir2levels/ {}/tdir".format(prog,opts,tdir,tdir))
        return cmds

class TS_mv(TestSuite_COREUTILS):
    """
    tdir: a b c d e f->b dir1 dir2/a,b,z dir3/c,d dir4
    cd tdir; mv a a_cp
    cd tdir; mv b dir1
    cd tdir; mv c dir1/
    cd tdir; mv e dir2/z
    cd tdir; mv d e
    cd tdir; mv dir2/z ..
    cd tdir; mv dir2/* dir3;
    cd tdir; mv dir3 dir4
    """
    def get_cmds(self):
        cmds =[]
        cmds.append("rm -rf {}/*;".format(tdir) +
                    "cd {}; mkdir d1 d2 d3 d4; touch a b c d e d2/a d2/b d2/z d3/c d3/d ; ln -sf e f".format(prog.tdir))
        cmds.append("cd {}".format(tdir) +
                    "{} {} a a_cp".format(self.prog,self.opts))
        cmds.append("{} {} {}/a_cp {}/a".format(self.prog,self.opts))
        cmds.append("{} {} {}/a {}/d".format(self.prog,self.opts))
        cmds.append("{} {} {}/a {}/d".format(self.prog,self.opts))        
        
        return cmds
class TS_touch(TestSuite_COREUTILS):
    def get_cmds(self):
        cmds = []
        cmds.append("rm -rf {}/* ;".format(tdir) +
                    "{} {} {}/a {}/b".format(prog,exe,tdir))

        cmds.append("rm -rf {}/* ;".format(tdir) +
                    "{} {} {}/a {}/b;".format(prog,exe,tdir)+
                    "{} {} {}/*".format(prog,exe,tdir))

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a -d 'next Thursday'".format(prog,exe,tdir))

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a -r {}/binary.dat".format(prog,exe,cdir,tdir))

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a -t 15151730.55")

        cmds.append("rm -rf {}/*;".format(tdir) +
                    "{} {} {}/a/".format(prog,exe,tdir))

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
        cmd.append("{} {} {}/junkfood_ringding_wtmp".format(prog,opts,tdir))
        cmd.append("{} {} {}/notexist".format(prog,opts,tdir))        
        
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

def testsuite_md5sum(args):
    """
    """
    if CM.__vdebug__:
        assert isinstance(args,dict),args
        assert 'opts' in args
        assert 'prog_exe' in args
        assert 'testfiles_dir' in args        

    prog_exe = args['prog_exe']
    opts = args['opts']
    tdir = args['testfiles_dir']

    cmds = []
    cmds.append("{} {} {}/file1".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/file2".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/binary.dat".format(prog_exe,opts,tdir))    
    cmds.append("{} {} {}/file1 {}/binary.dat".format(prog_exe,opts,tdir,tdir))
    cmds.append("{} {} {}/*".format(prog_exe,opts,tdir))
    cmds.append("cat {}/file1 | {} {} -".format(tdir,prog_exe,opts))    
    cmds.append("{} {} {}/results.md5".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/results_failed.md5".format(prog_exe,opts,tdir))
    
    CF.void_run(cmds)
    
def testsuite_ls(args):
    """
    testsuite_ls({'opts':'-d','prog_exe':'/bin/ls', 'testfiles_dir':'/home/tnguyen/Dropbox/git/config/benchmarks/coreutils/testfiles/ls'})
    """
    if CM.__vdebug__:
        assert isinstance(args,dict),args
        assert 'opts' in args
        assert 'prog_exe' in args
        assert 'testfiles_dir' in args        

    prog_exe = args['prog_exe']
    opts = args['opts']
    testfiles_dir = args['testfiles_dir']
    tdir = os.path.join(args['testfiles_dir'],"edir")
    cmds = []
    cmds.append("{} {} {}/a_ln".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/noexist_ln".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/d1/.hidden1".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/d1/d1d1/b".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/d2/a.sh".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/d2/dirA".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/d2/dirA/*".format(prog_exe,opts,tdir))
    
    cmds.append("{} {} {}/d2_ln/a.sh".format(prog_exe,opts,tdir))
    cmds.append("{} {} {}/d2_ln/.hidden_dir".format(prog_exe,opts,tdir))

    cmds.append("{} {} /boot".format(self.prog,self.opts))
    cmds.append("{} {} /boot/*".format(self.prog,self.opts))
    cmds.append("{} {} /bin/ls".format(self.prog,self.opts))
    assert os.path.isdir(getpath("~/ls_test")), "create ~/ls_test"
    cmds.append("{} {} ~/ls_test".format(self.prog,self.opts))
    cmds.append("{} {} .".format(self.prog,self.opts))
    cmds.append("{} {} ..".format(self.prog,self.opts))
    CF.void_run(cmds)


def testsuite_sort(args):
    """
    Examples for tests: http://www.theunixschool.com/2012/08/linux-sort-command-examples.html
    """
    if CM.__vdebug__:
        assert isinstance(args,dict),args
        assert 'opts' in args
        assert 'prog_exe' in args
        assert 'testfiles_dir' in args        

    prog_exe = args['prog_exe']
    opts = args['opts']
    tdir = args['testfiles_dir']

    cmds = []
    #cmds.append("wc -l /usr/bin/* | {} {}".format(self.prog,self.opts)) #wc -l . | sort
    cmds.append("{} {} {}/file1".format(prog_exe,opts,tdir))  #sort file1
    cmds.append("{} {} {}/file3".format(prog_exe,opts,tdir))  #sort file1
    cmds.append("{} {} {}/file1 {}/file2 {}/file3"  
                .format(prog_exe,opts,tdir,tdir,tdir)) #multiple fiels

    #some error with this command, cannot read
    # cmds.append("{} {} --files0-from={}/allfiles"
    #             .format(prog_exe,opts,tdir)) #multiple fiels
    
    CF.void_run(cmds)


    
def testsuite_ln(args):
    if CM.__vdebug__:
        assert isinstance(args,dict),args
        assert 'opts' in args
        assert 'prog_exe' in args

        assert 'testfiles_dir' in args        


    prog_exe = args['prog_exe']
    opts = args['opts']
    testfiles_dir = args['testfiles_dir']

    cmds = []
 #test 1
    cmds = []    
    cmds.append("touch a")
    cmds.append("ln {} a b".format(opts_str))

    #test 2
    cmds = []        
    cmds.append("mkdir d")
    cmds.append("ln {} d e".format(opts_str))
    myexec(test_dir,cmds)
    
    #from coreutils tests
    #test 3  # d/f -> ../f
    cmds = []        
    cmds.append("mkdir d")
    cmds.append("ln {} --target_dir=d ../f".format(opts_str))
    myexec(test_dir,cmds)
    
    #test 4
    # Check that a target directory of '.' is supported
    # and that indirectly specifying the same target and link name
    # through that is detected.
    cmds = []        
    cmds.append("ln {} . b".format(opts_str))
    cmd.append("echo foo > a")      
    cmds.append("ln {} a b > err 2>&1)".format(opts_str))
    myexec(test_dir,cmds)


coreutils_d = {'date': TS_date,
               'ls': testsuite_ls,
               'ln': testsuite_ln,
               'sort': testsuite_sort,
               'id': TS_id,
               'uname': TS_uname}
        

