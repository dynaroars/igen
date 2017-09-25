chunk_size=(len(c_core)+1)/(2**dd_level)
iter_counter+=1
if chunk_size==0:
    chunk_size=1
optionCounter=0
new_cores=[]
for x in xrange(0,2**dd_level):
    new_cores.append([_newX(c_core)])
print "---new_cores-brefore---"
print new_cores
for k in c_core:
    indx=optionCounter/chunk_size
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
        new_cores[indx][0]=new_cores[indx].pop()
        
    optionCounter += 1

print "---new_cores-after---"
print new_cores
changes.extend(new_cores[0])
changes.extend(new_cores[1])
print "---changes---"
print changes
changes_map[sel_core]=changes