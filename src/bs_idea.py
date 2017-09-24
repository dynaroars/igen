_newX = lambda x: Core((k, x[k]) for k in x)
middleOfInt=(len(c_core)+1)/2
optionCounter=0
new_cores=[[_new()],[_new()]]
for k in c_core:
    indx=optionCounter/middleOfInt
    vs = self[k] - c_core[k]
    if len(vs)==1:
        new_cores[indx][0][k] = frozenset(next(iter(vs)))
    else:
        for v in vs:
            nc=_newX(new_cores[indx][0])
            nc[k]=frozenset([v])
            if s_core:
                for sk,sv in s_core.iteritems():
                    assert sk not in nc, sk
                    nc[sk] = sv
            new_cores[indx].append(nc)
        new_cores[indx][0]=new_cores[indx][-1]
        new_cores[indx]=new_cores[indx][:-1]
        
    optionCounter += 1
print new_cores[0]
print new_cores[1]

changes.extend(new_cores[0])
changes.extend(new_cores[1])