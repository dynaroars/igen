import os.path
import random

import z3
import z3util
import config

dom = Dom([('a', frozenset(['1', '0'])), ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0', '2']))])
z3db = dom.z3db

c1 = Config([('a', '0'), ('b', '0'), ('c', '0')])
c2 = Config([('a', '0'), ('b', '0'), ('c', '1')])
c3 = Config([('a', '0'), ('b', '0'), ('c', '2')])

c4 = Config([('a', '0'), ('b', '1'), ('c', '0')])
c5 = Config([('a', '0'), ('b', '1'), ('c', '1')])
c6 = Config([('a', '0'), ('b', '1'), ('c', '2')])

c7 = Config([('a', '1'), ('b', '0'), ('c', '0')])
c8 = Config([('a', '1'), ('b', '0'), ('c', '1')])
c9 = Config([('a', '1'), ('b', '0'), ('c', '2')])

c10 = Config([('a', '1'), ('b', '1'), ('c', '0')])
c11 = Config([('a', '1'), ('b', '1'), ('c', '1')])
c12 = Config([('a', '1'), ('b', '1'), ('c', '2')])

configs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
nexpr = z3util.myOr([c.z3expr(z3db) for c in configs])
assert dom.gen_configs_exprs([None],[nexpr],k=1,config_cls=Config)[0] == c12

core = Core([('a',frozenset(['1']))])
core_expr = core.z3expr(z3db,z3util.myAnd)
assert dom.gen_configs_exprs([None],[nexpr],k=1, config_cls=Config)[0] == c12

core = Core([('a',frozenset(['0']))])
core_expr = core.z3expr(z3db,z3util.myAnd)
assert not dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)

core = Core([('c',frozenset(['0','1']))])
core_expr = core.z3expr(z3db,z3util.myAnd)
assert not dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)

core = Core([('c',frozenset(['0','2']))])
core_expr = core.z3expr(z3db,z3util.myAnd)
config = dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)[0]
print config