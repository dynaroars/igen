import os.path
import z3

def read(filename):
    with open(filename, 'r') as fh:
        for line in fh:
            yield line

class VarContents:
    def __init__(self, name, typ, defval):
        assert isinstance(name, str) and name, name
        assert (isinstance(typ, bool) and
                typ in set(["bool", "nonbool"])), typ
        assert (defval is None or
                (isinstance(defval, str) and defval)), defvalt
        
        self.name = name
        self.typ = typ
        self.defval = defval

    @property
    def z3var(self):
        if self.typ == "bool":
            return z3.Bool(self.name)
        else:
            if '"' in self.defval:
                # enumerative type
                pass
            else:
                pass


def getZ3Var(k, d, g):
    try:
        return d[k]
    except KeyError:
        v = k[1:] if k.startswith('-') else k

        #print k, v, g
        v_ = g[v][0] if g else 'v' + v
        v_ = z3.Bool(v_)

        d[v] = v_
        d['-' + v] = z3.Not(v_)
        return d[k]
            
def convert(lines, doReadVarname=False):
    pIdx = None
    nvars = None
    nclauses = None

    varsD = {}
    z3varsD = {}
    clauses = []
    inVarDecls = False

    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l]


    #var info
    for l in lines:
        if l.startswith('c kconfig_variable'):
            pass
    

    fromkconstraints = False
    for i, l in enumerate(lines):
        if l.startswith('c'):
            if 'fromkconstraints' in l:
                fromkconstraints = True
                continue

            if not fromkconstraints:
                continue
            
            ps = l.split()
            assert len(ps) >= 4, ps

            c, nameId, name, typ = ps[:4]
            defv = None
            if len(ps) > 4:
                assert typ == 'nonbool', typ
                defv = ' '.join(ps[4:])
            
            assert nameId not in varsD
            varsD[nameId] = (name, typ, typ, defv)
            print nameId, varsD[nameId]
            
        elif l.startswith("p"): #p cnf 3 4
            inVarDecls = False
            
            pIdx = i
            _, _, nvars, nclauses = l.split(" ")
            nvars = int(nvars)
            nclauses = int(nclauses)

        else:  #clause,  -1 2
            inVarDecls = False
            
            assert i > pIdx, (i, pIdx)
            vs = l.split()  #-86 -880 0
            vs = vs[:-1]  #remove the last 0
            clause = [getZ3Var(v, z3varsD, varsD) for v in vs]
            clauses.append(clause)
            

    if len(clauses) != nclauses:
        wmsg = ["W: # collected clauses {}".format(len(clauses)),
                "is not similar to",
                "# of specified clause {}".format(nclauses)]
        print ' '.join(wmsg)

    z3clauses = [z3.Or(clause) if clause else clause for clause in clauses]
    f = z3.And(z3clauses) if z3clauses else z3clauses
    return f

if __name__ == "__main__":
    import argparse
    aparser = argparse.ArgumentParser("Dimacs to Z3 converter")
    aparser.add_argument("dimacsfile",
                         help="dimacs file ",
                         action="store")
    
    
    import os.path
    args = aparser.parse_args()

    f = convert(read(args.dimacsfile))
    print f
