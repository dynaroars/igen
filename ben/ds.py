#from sklearn.tree import DecisionTreeClassifier
#import sklearn.tree
import itertools
import os
import os.path
from pathlib import Path
import random


import vcommon as CM

import z3
import z3util

import data.dom

import settings
mlog = CM.getLogger(__name__, settings.logger_level)


# Data Structures

class DTrace(object):
    """
    Object for saving information (for later analysis)
    """

    def __init__(self, citer, itime, xtime,
                 nconfigs, ncovs, ncores,
                 cconfigs_d, new_covs, new_cores,
                 sel_core, cores_d):

        self.citer = citer
        self.itime = itime
        self.xtime = xtime
        self.nconfigs = nconfigs
        self.ncovs = ncovs
        self.ncores = ncores
        self.cconfigs_d = cconfigs_d
        self.new_covs = new_covs
        self.new_cores = new_cores
        self.sel_core = sel_core
        self.cores_d = cores_d

    def show(self, dom, z3db):
        assert isinstance(dom, data.dom.Dom), dom
        assert isinstance(z3db, data.dom.Z3DB), z3db

        mlog.info("ITER {}, ".format(self.citer) +
                  "{}s, ".format(self.itime) +
                  "{}s eval, ".format(self.xtime) +
                  "total: {} configs, {} covs, {} cores, "
                  .format(self.nconfigs, self.ncovs, self.ncores) +
                  "new: {} configs, {} covs, {} updated cores, "
                  .format(len(self.cconfigs_d),
                          len(self.new_covs), len(self.new_cores)) +
                  "{}".format("** progress **"
                              if self.new_covs or self.new_cores else ""))

        mlog.info('select core: ({}) {}'.format(self.sel_core.sstren,
                                                self.sel_core))
        mlog.info('create {} configs'.format(len(self.cconfigs_d)))
        mlog.debug("\n"+str(self.cconfigs_d))

        mcores_d = self.cores_d.merge(dom, z3db)
        mlog.info("infer {} interactions".format(len(mcores_d)))
        mlog.debug('\n{}'.format(mcores_d))

        mlog.info("strens: {}".format(mcores_d.strens_str))

    @staticmethod
    def save_pre(seed, dom, tmpdir):
        CM.vsave(tmpdir / 'pre', (seed, dom))

    @staticmethod
    def save_post(pp_cores_d, itime_total, tmpdir):
        CM.vsave(os.path.join(tmpdir, 'post'), (pp_cores_d, itime_total))

    @staticmethod
    def save_iter(cur_iter, dtrace, tmpdir):
        CM.vsave(os.path.join(tmpdir, '{}.tvn'.format(cur_iter)), dtrace)

    @staticmethod
    def load_pre(dir_):
        seed, dom = CM.vload(os.path.join(dir_, 'pre'))
        return seed, dom

    @staticmethod
    def load_post(dir_):
        pp_cores_d, itime_total = CM.vload(os.path.join(dir_, 'post'))
        return pp_cores_d, itime_total

    @staticmethod
    def load_iter(dir_, f):
        dtrace = CM.vload(os.path.join(dir_, f))
        return dtrace

    @staticmethod
    def str_of_summary(seed, iters, itime, xtime, nconfigs, ncovs, tmpdir):
        ss = ["Seed {}".format(seed),
              "Iters {}".format(iters),
              "Time ({}s, {}s)".format(itime, xtime),
              "Configs {}".format(nconfigs),
              "Covs {}".format(ncovs),
              "Tmpdir {}".format(tmpdir)]
        return "Summary: " + ', '.join(ss)

    @classmethod
    def load_dir(cls, dir_):
        seed, dom = cls.load_pre(dir_)
        dts = [cls.load_iter(dir_, f)
               for f in os.listdir(dir_) if f.endswith('.tvn')]
        try:
            pp_cores_d, itime_total = cls.load_post(dir_)
        except IOError:
            mlog.error("post info not avail")
            pp_cores_d, itime_total = None, None
        return seed, dom, dts, pp_cores_d, itime_total


class Z3(object):
    @staticmethod
    def is_unsat(exprs, print_unsat_core=False):
        assert all(z3.is_expr(e) for e in exprs), exprs

        s = z3.Solver()
        expr = z3.And(exprs)
        s.add(expr)
        stat = s.check()
        isunsat = not (stat == z3.sat)

        if stat == z3.unsat and print_unsat_core:
            ps = [z3.Bool('p{}'.format(i)) for i in range(len(exprs))]
            exprs = [z3.Implies(p, e) for p, e in zip(ps, exprs)]
            expr = z3.And(exprs)
            s.add(expr)
            stat = s.check(ps)
            assert stat == z3.unsat
            unsat_ps = s.unsat_core()
            unsat_idxs = [str(p)[1:] for p in unsat_ps]
            # print unsat_idxs
            # print [exprs[int(idx)] for idx in unsat_idxs]

        return isunsat


# class DecisionTree:
#     def __init__(self, tree, myvars):
#         self.tree = tree
#         self.myvars = myvars

#     def __str__(self):
#         ss = []
#         self.print_tree(root=0, depth=1, ss=ss)
#         return '\n'.join(ss)

#     @property
#     def node_count(self):
#         return self.tree.tree_.node_count

#     @property
#     def max_depth(self):
#         try:
#             return self._max_depth
#         except AttributeError:
#             pos_paths, neg_paths = self.path_conds
#             paths = pos_paths + neg_paths
#             self._max_depth = max(map(len, paths))
#             return self._max_depth

#     @property
#     def path_conds(self):
#         """
#         [0, 0, 1, 1, 0] -> ('<', 6, 2.5)
#         [0, 0, 1, 1, 0] -> [('>=', 6, 2.5), ('<', 5, 0.5)]
#         [1, 1, 0, 0, 0] -> [('>=', 6, 2.5), ('>=', 5, 0.5)]
#         """
#         try:
#             return self._path_conds
#         except AttributeError:
#             self._path_conds = {}
#             paths = self.collect_paths(root=0)
#             for path in paths:
#                 assert path
#                 path_, covs = path[:-1], tuple(path[-1])
#                 self._path_conds[covs] = path_

#             return self._path_conds

#     @property
#     def str_of_path_conds(self):
#         """
#         pretty print
#         """

#         for covs in self.path_conds:
#             [cov for cov in covs if cov]

#     def get_path_conds_z3(self, z3db):
#         """
#         Return path constraints for truth paths
#         """
#         try:
#             return self._path_conds_z3
#         except AttributeError:
#             self._path_conds_z3 = {k: self._get_conds(path, z3db)
#                                    for k, path in self.path_conds.items()}
#             return self._path_conds_z3

#     @classmethod
#     def _get_conds(cls, path, z3db):
#         def to_z3(t):
#             """
#             t = ('<', 1, 0.5)
#             """
#             assert isinstance(t, tuple) and len(t) == 3
#             myop, myvar, myval = t
#             myvar = z3db.myvars[myvar]
#             myval = z3.RealVal(myval)
#             if myop == '<':
#                 return myvar < myval
#             else:
#                 return myvar >= myval

#         return [to_z3(t) for t in path]

#     @classmethod
#     def learn(cls, X, y):
#         dt = DecisionTreeClassifier(random_state=42)
#         dt.fit(X, y)
#         return dt

#     def score(self, X, y):
#         y_ = self.tree.predict(X)
#         n_row = len(y_)
#         n_col = len(y_[0])
#         assert n_row == len(y)
#         assert n_col == len(y[0])

#         n_mismatch = 0
#         for i in range(n_row):
#             for j in range(n_col):
#                 if int(y_[i][j]) != y[i][j]:
#                     n_mismatch += 1

#         myscore = n_mismatch/(n_row * n_col)
#         return myscore

#     def print_tree(self, root=0, depth=1, ss=[]):
#         tree = self.tree.tree_
#         indent = '    '*depth
#         # ss.append("{} # node {}: impurity = {}".format(
#         #     indent, root, tree.impurity[root]))

#         left_child = tree.children_left[root]
#         right_child = tree.children_right[root]

#         if left_child == sklearn.tree._tree.TREE_LEAF:
#             ss.append("{} return {} # (node {}, impurity {})".format(
#                 indent, tree.value[root], root, tree.impurity[root]))
#         else:
#             ss.append("{} if {} < {}: # (node {})".format(
#                 indent, self.myvars[tree.feature[root]],
#                 tree.threshold[root], root))
#             self.print_tree(root=left_child, depth=depth+1, ss=ss)

#             ss.append("{} else".format(indent))
#             self.print_tree(root=right_child, depth=depth+1, ss=ss)

#     @classmethod
#     def get_truth(self, values):
#         """
#         values =
#         [[3., 0.],
#         [0., 3.],
#         [0., 3.],
#         [3., 0.]]
#         returns
#         [0, 1, 1, 0]
#         """
#         assert len(values) and len(values[0]) == 2, values

#         vs = []
#         for v in values:
#             num_neg, num_pos = v
#             assert ((num_pos == 0 and num_neg > 0) or
#                     (num_pos > 0 and num_neg == 0)), (num_pos, num_neg)

#             if num_pos == 0:
#                 vs.append(0)
#             else:
#                 assert num_neg == 0 and num_pos > 0
#                 vs.append(1)

#         return vs

#     def collect_paths(self, root):
#         tree = self.tree.tree_

#         left_child = tree.children_left[root]
#         right_child = tree.children_right[root]

#         if left_child == sklearn.tree._tree.TREE_LEAF:
#             rv = self.get_truth(tree.value[root])
#             return [[rv]]

#         myvar = tree.feature[root]
#         myval = tree.threshold[root]
#         node = ('<', myvar, myval)

#         left_paths = self.collect_paths(left_child)
#         left_paths = [[node] + path for path in left_paths]

#         node = ('>=', myvar, myval)
#         right_paths = self.collect_paths(right_child)
#         right_paths = [[node] + path for path in right_paths]

#         paths = left_paths + right_paths
#         return paths


# class DecisionTrees_d(dict):
#     """
#     {cov -> dt}
#     """
#     pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
