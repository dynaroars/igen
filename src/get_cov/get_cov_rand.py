#!/usr/bin/env python
import os.path
import subprocess as sp
import random
from collections import OrderedDict
            
#read in domain
#domain has the following format
"""
configname1   val1 val2
configname2   val1 val 2
...

for example
x  0 1 2
y  True False
"""


def iread(filename):
    """ return a generator """
    with open(filename, 'r') as fh:
        for line in fh:
            yield line

def iread_strip(filename):
    """
    like iread but also strip out comments and empty line
    """
    lines = (l.strip() for l in iread(filename))
    lines = (l for l in lines if l and not l.startswith('#'))
    return lines


def vcmd(cmd, inp=None, shell=True):
    proc = sp.Popen(cmd,shell=shell,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    return proc.communicate(input=inp)

####

def get_dom(dom_file):
    dom_lines = iread_strip(dom_file)    
    dom = OrderedDict()
    for line in dom_lines:
        parts = line.split()
        varname = parts[0]
        varvals = parts[1:]
        dom[varname] = varvals
    return dom


def gen_rand_config(dom):
    config = []
    for k,vs in dom.iteritems():
        config.append((k,random.choice(vs)))
    return OrderedDict(config)

def get_cov(config, run_script):

    inputs = ' , '.join(['{} {}'.format(vname,vval) for vname,vval in config.iteritems()])
    outfile = "/var/tmp/run_script_result.txt"
    cmd = "{} \"{}\" > {}".format(run_script,inputs,outfile)
    print cmd
    try:
        _,rs_err = vcmd(cmd)
    except:
        print("cmd '{}' failed".format(cmd))

    cov = set(iread_strip(outfile))
    return cov
    


if __name__ == "__main__":
    """
    python get_cov dom_file run_script
    """
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dom_file",
                        help="the domain file",
                        action="store")

    parser.add_argument("--run_script",
                        help="a script that run the main program",
                        action="store")
    args = parser.parse_args()

    #read the domain file
    dom_file = os.path.abspath(args.dom_file)
    run_script = os.path.abspath(args.run_script)
    print dom_file
    print run_script
    
    dom = get_dom(dom_file)
    print "domain contents: ", dom
    
    #create some random configs to test
    for _ in range(5):
        config = gen_rand_config(dom)
        #get the coverage of these random configs
        print "random config: {}".format(config)
        cov = get_cov(config,run_script)
        print "coverage ({}): {}".format(len(cov),','.join(cov))
    
