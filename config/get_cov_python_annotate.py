#read coverage from files
import config_common as CC

def parseFile(filename, covered, uncovered):
    for i,l in enumerate(CC.iread(filename)):
        lid = "{}:{}".format(filename,i)
        if l.startswith(">"):
            covered.add(lid)
        elif l.startswith("!"):
            uncovered.add(lid)

def parseFiles(filenames, covered, uncovered):
    for filename in filenames:
        parseFile(filename, covered, uncovered)

def parseDir(dirname, covered, uncovered):
    import os
    filenames = [f for f in os.listdir(dirname) if f.endswith(',cover')]
    filenames = sorted(filenames)
    filenames = [os.path.join(dirname,f) for f in filenames]
    parseFiles(filenames, covered, uncovered)
    
def parse(dirname):
    covered = set()
    uncovered = set()
    parseDir(dirname, covered, uncovered)
    return covered, uncovered

def cleanup(lines, dirname):
    """
    clean up resulting format
    E.g., dir/file,cover:n  ->  file:n
    """
    #remov "/var/tmp/me_prog_cov"
    lines = set(l.replace(dirname + '/', '') for l in lines)
    #remove ",cover"
    lines = set(l.replace(",cover", '') for l in lines)
    return lines
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dirname", help="dir containing coverage")
    args = parser.parse_args()

    covered, uncovered = parse(args.dirname)
    print '{} covered lines:'.format(len(covered))
    print '\n'.join(sorted(covered))

    print '{} uncovered lines:'.format(len(uncovered))
    print '\n'.join(sorted(uncovered))
    
