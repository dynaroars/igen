#!/usr/bin/env python

import re, z3
import shuntingyard as sy

FALSE = z3.BoolVal(False)
TRUE = z3.BoolVal(True)

with open('kconfig_constraints.txt') as f:
    lines = f.read().replace('&&','&').replace('||','|').replace('!','! ').replace('(','( ').replace(')',' )').replace('  ',' ').splitlines()

suffix="is a root"
constraints=[]
for line in lines:
	if line.endswith(suffix):
		continue
	prefix_form=sy.infix_to_prefix(line)
	constraint=[]
	for item in prefix_form:
		if item is "!":
			constraint.append(constraint.pop().children()[0]==FALSE)
		elif item is "|":
			f=z3.Or(constraint.pop(), constraint.pop())
			constraint.append(f)
		elif item is "&":
			f=z3.And(constraint.pop(),constraint.pop())
			constraint.append(f)
		else:
			constraint.append(z3.Bool(item)==TRUE)
	constraints.append(constraint[0])


f = z3.And(constraints)
print z3.solve(f)