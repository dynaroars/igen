#!/usr/bin/env python3

import z3
import vcommon as CM
import argparse
import itertools
import tempfile
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
import os.path
import random
from time import time
from collections import OrderedDict
import pdb
trace = pdb.set_trace


mlog = None


def str_of_list(l, delim=','):
    return ','.join(map(str, l))


class Settings:
    logger_level = 3
    tmpdir = "/var/tmp"


class Exec:
    def __init__(self, myvars, run_script):
        assert os.path.isfile(run_script), run_script
        self.run_script = run_script
        self.myvars = myvars

    def run(self, config):
        """
        Get cov from config.
        Exec runscript return a single line representing exec results
        E.g., ./runscript.sh "inputs"

        """
        inputs = ' , '.join('{} {}'.format(vname, vval) for
                            vname, vval in zip(self.myvars, config))

        cmd = "{} \"{}\"".format(self.run_script, inputs)

        # mlog.debug(cmd)
        rs_err = "some error occured:"
        try:
            rs_outp, rs_err = CM.vcmd(cmd)
            # mlog.debug(rs_outp)

            # NOTE: comment out the below allows
            # erroneous test runs, which can be helpful
            # to detect incorrect configs
            assert len(rs_err) == 0, rs_err

            serious_errors = ["invalid",
                              "-c: line",
                              "/bin/sh",
                              "assuming not executed"]

            known_errors = ["invalid file number in field spec"]
            if rs_err:
                mlog.debug("error: {}".format(rs_err))
                if any(kerr in rs_err for kerr in known_errors):
                    raise AssertionError("Check this known error!")
                if any(serr in rs_err for serr in serious_errors):
                    raise AssertionError("Check this serious error!")

            # cov_filename = [l for l in rs_outp.split('\n') if l]
            # assert len(cov_filename) == 1, (cmd, rs_outp, cov_filename)
            # cov_filename = cov_filename[0]
            # cov = list(set(CM.iread_strip(cov_filename)))

            cov = set(x.strip() for x in rs_outp.split(','))
            cov = list(x for x in cov if x)
            mlog.debug("cmd {}, read {} covs: {}".format(
                cmd, len(cov), ','.join(cov)))

            return cov

        except Exception as e:
            raise AssertionError("cmd '{}' fails, raise error: {}, {}"
                                 .format(cmd, rs_err, e))


class Dom(OrderedDict):
    @property
    def myvars(self):
        try:
            return self._myvars
        except AttributeError:
            self._myvars = list(self.keys())
            return self._myvars

    @property
    def real_vals(self):
        try:
            return self._vals
        except AttributeError:
            self._vals = list(self.values())
            return self._vals

    @property
    def idx_vals(self):
        try:
            return self._idx_vals
        except AttributeError:
            self._idx_vals = [list(range(len(v))) for v in self.real_vals]
            return self._idx_vals

    def __str__(self):
        ss = []
        ss.append("dom file: '{}'".format(self.dom_file))
        ss.append("{} vars and {} pos configs".format(len(self), self.siz))
        ss.append('\n'.join("{}. {}: ({}) {}".format(
            i+1, k, len(vs), str_of_list(vs))
            for i, (k, vs) in enumerate(self.items())))
        ss.append("run script: '{}'".format(self.run_script))
        return '\n'.join(ss)

    @property
    def siz(self):
        return CM.vmul(len(vs) for vs in self.real_vals)

    def gen_idx_configs_tcover1(self):
        """
        Return a set of tcover array of stren 1
        """
        unused = dict(zip(self.myvars, map(list, self.idx_vals)))

        def mk():
            config = []
            for k, vs in zip(self.myvars, self.idx_vals):
                if k in unused:
                    v = random.choice(unused[k])
                    unused[k].remove(v)
                    if not unused[k]:
                        unused.pop(k)
                else:
                    v = random.choice(vs)

                config.append(v)
            return tuple(config)

        configs = []
        while unused:
            configs.append(mk())
        return configs

    def gen_idx_configs_full(self):  # TODO kconfig_contraint
        configs = [tuple(c) for c in itertools.product(*self.idx_vals)]
        return configs

    def to_real_config(self, idx_config):
        return [self.real_vals[i][j] for i, j in enumerate(idx_config)]

    @classmethod
    def get_dom(cls, dom_dir: str):
        def check(f, must_exist):
            if os.path.isfile(f):
                return f
            else:
                assert not must_exist, "'{}' does not exist".format(f)
                return None

        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs_ = []
            for parts in rs:
                var = parts[0]
                vals = parts[1:]
                rs_.append((var, vals))
            return rs_

        # domain file
        dom_file = check(os.path.join(dom_dir, 'dom'), must_exist=True)
        dom = get_lines(CM.iread_strip(dom_file))
        dom = cls(dom)
        dom.dom_file = dom_file
        dom.run_script = check(
            os.path.join(dom_dir, 'run'), must_exist=True)
        return dom


class Z3DB(dict):

    def __init__(self, dom):
        self.myvars = [z3.Real(v) for v in dom.myvars]
        self.idx_vals = [[z3.RealVal(v) for v in vs] for vs in dom.idx_vals]
        self.var_idx_d = {v: i for i, v in
                          enumerate(self.myvars)}  # var -> idx

        self.facts = z3.And([z3.Or([x == y for y in self.idx_vals[i]])
                             for i, x in enumerate(self.myvars)])

        self.cache = {}

    def convert_config(self, idx_config):
        if idx_config in self.cache:
            return self.cache[idx_config]

        assert isinstance(idx_config, tuple), type(idx_config)
        assert len(idx_config) == len(self.myvars)
        f = z3.And([x == y for x, y in zip(self.myvars, idx_config)])
        self.cache[idx_config] = f

        return f

    def get_config_from_model(self, model):
        print(model)
        config = [None] * len(self.myvars)
        for x in model:
            idx = self.var_idx_d[x()]
            config[idx] = int(str(model[x]))
        return tuple(config)

    # Z3 utils

    @classmethod
    def get_models(cls, f, k):
        assert k >= 1
        s = z3.Solver()
        s.add(f)

        models = []
        i = 0

        is_sat = s.check() == z3.sat
        while is_sat and i < k:
            i = i + 1
            m = s.model()

            if not m:  # if m == []
                break

            models.append(m)
            block = z3.Not(z3.And([v() == m[v] for v in m]))
            s.add(block)
            is_sat = s.check() == z3.sat

        if models:
            return models
        else:
            if is_sat == z3.unknown:
                return None
            else:
                assert s.check() == z3.unsat and i == 0
                return False  # f is unsat


class Data:
    cov_str = 'COV'

    def __init__(self):
        """
        len(configs) == len(covs)
        len(configs[0]) == len(myvars)
        """
        self.configs = []
        self.covss = []
        self.configs_cache = set()

    @classmethod
    def _str_configs(cls, configs):
        return '\n'.join(str_of_list(c) for c in configs)

    def __str__(self):
        ss = []
        ss.extend("{} : {}".format(self._str(config), self._str(covs))
                  for config, covs in zip(self.configs, self.covss))
        return '\n'.join(ss)

    def get_idxs(self, cov, start_idx=0):
        pos_idxs = []
        neg_idxs = []
        for idx in range(start_idx, len(self.configs)):
            if cov in self.covss[idx]:
                pos_idxs.append(idx)
            else:
                neg_idxs.append(idx)

        return tuple(pos_idxs), tuple(neg_idxs)

    def get_configs(self, idxs):
        return [self.configs[idx] for idx in idxs]

    def append(self, config, covs):
        assert config not in self.configs_cache

        self.configs.append(config)
        self.covss.append(covs)
        self.configs_cache.add(config)


class DecisionTree:
    test_size = .0  # percent

    def __init__(self, tree, myvars):
        self.tree = tree
        self.myvars = myvars

    def __str__(self):
        ss = []
        self.get_str(root=0, depth=1, ss=ss)
        return '\n'.join(ss)

    @property
    def node_count(self):
        return self.tree.tree_.node_count

    @property
    def max_depth(self):
        try:
            return self._max_depth
        except AttributeError:
            pos_paths, neg_paths = self.path_conds
            paths = pos_paths + neg_paths
            self._max_depth = max(map(len, paths))
            return self._max_depth

    @property
    def path_conds(self):
        try:
            return self._path_conds
        except AttributeError:
            self._path_conds = {}
            paths = self.collect_paths(root=0)
            pos_paths, neg_paths = [], []
            for path in paths:
                assert path
                path_, truth = path[:-1], path[-1]
                assert truth is True or truth is False, truth
                (pos_paths if truth else neg_paths).append(path_)

            self._path_conds = pos_paths, neg_paths
            return self._path_conds

    def get_path_conds_z3(self, z3db):
        """
        Return path constraints for truth paths
        """
        try:
            return self._path_conds_z3
        except AttributeError:
            pos_paths, neg_paths = self.path_conds
            pos_conds = [self._get_conds(path, z3db) for path in pos_paths]
            neg_conds = [self._get_conds(path, z3db) for path in neg_paths]
            self._path_conds_z3 = pos_conds, neg_conds
        return self._path_conds_z3

    @classmethod
    def _get_conds(cls, path, z3db):
        def to_z3(t):
            """
            t = ('<', 1, 0.5)
            """
            assert isinstance(t, tuple) and len(t) == 3
            myop, myvar, myval = t
            myvar = z3db.myvars[myvar]
            myval = z3.RealVal(myval)
            if myop == '<':
                return myvar < myval
            else:
                return myvar >= myval

        return [to_z3(t) for t in path]

    @classmethod
    def learn(cls, pos_configs, neg_configs):
        X = pos_configs + neg_configs
        y = [1]*len(pos_configs) + [0]*len(neg_configs)
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        return dt

    def score(self, X, y):
        return self.tree.score(X, y)

    def get_str(self, root=0, depth=1, ss=[]):
        tree = self.tree.tree_
        indent = '    '*depth
        ss.append("{} # node %s: impurity = {}".format(
            indent, root, tree.impurity[root]))

        left_child = tree.children_left[root]
        right_child = tree.children_right[root]

        if left_child == sklearn.tree._tree.TREE_LEAF:
            ss.append("{} return {} # (node {})".format(
                indent, tree.value[root], root))
        else:
            ss.append("{} if {} < {}: # (node {})".format(
                indent, self.myvars[tree.feature[root]],
                tree.threshold[root], root))
            self.get_str(root=left_child, depth=depth+1, ss=ss)

            ss.append("{} else".format(indent))
            self.get_str(root=right_child, depth=depth+1, ss=ss)

    @classmethod
    def get_truth(self, value):
        # value = [[3. 0.]]
        assert len(value) == 1 and len(value[0]) == 2, value

        num_neg, num_pos = value[0]
        assert ((num_pos == 0 and num_neg > 0) or
                (num_pos > 0 and num_neg == 0)), (num_pos, num_neg)

        if num_pos == 0:
            return False
        else:
            assert num_neg == 0 and num_pos > 0
            return True

    def collect_paths(self, root):
        tree = self.tree.tree_

        left_child = tree.children_left[root]
        right_child = tree.children_right[root]

        if left_child == sklearn.tree._tree.TREE_LEAF:
            rv = self.get_truth(tree.value[root])
            return [[rv]]

        myvar = tree.feature[root]
        myval = tree.threshold[root]
        node = ('<', myvar, myval)

        left_paths = self.collect_paths(left_child)
        left_paths = [[node] + path for path in left_paths]

        node = ('>=', myvar, myval)
        right_paths = self.collect_paths(right_child)
        right_paths = [[node] + path for path in right_paths]

        paths = left_paths + right_paths
        return paths


class MBuild:
    def __init__(self, dom, get_cov):
        assert isinstance(dom, Dom), dom
        assert callable(get_cov), get_cov

        self.dom = dom
        self.get_cov = get_cov
        self.data = Data()
        self.z3db = Z3DB(dom)

        self.idxs_dt_cache = {}  # data idxs -> decision tree
        self.cov_dts_cache = {}  # cov -> decision tree  (#main results)

    def go(self, seed, tmpdir):
        random.seed(seed)
        mlog.info("seed: {}, tmpdir: {}".format(seed, tmpdir))

        idx_configs = self.gen_idx_configs_init()
        econfigs = self.eval_configs(idx_configs)
        progress = self.learn(econfigs)
        cur_iter = 0
        while progress:
            cur_iter += 1

            # TODO:  need a while loop here to keep on selecting
            select_cov, select_dt = self.select_dt()
            mlog.info("*** iter {}, select dt of '{}', "
                      "configs {}, covs {}".format(
                          cur_iter, select_cov,
                          len(self.data.configs),
                          len(self.cov_dts_cache)))

            idx_configs = self.gen_idx_configs_cex(select_dt)
            econfigs = self.eval_configs(idx_configs)
            progress = self.learn(econfigs)

        # print results
        for cov in sorted(set(itertools.chain(*self.data.covss))):
            if cov not in self.cov_dts_cache:
                cond = True
            else:
                dt = self.cov_dts_cache[cov]
                cond = dt.get_path_conds_z3(self.z3db)

            mlog.info("{}: {}".format(cov, cond))

    def select_dt(self):
        cov = max(self.cov_dts_cache,
                  key=lambda cov:
                  self.cov_dts_cache[cov].node_count)

        dt = self.cov_dts_cache[cov]
        return cov, dt

    def gen_idx_configs_init(self):
        configs = self.dom.gen_idx_configs_tcover1()
        # configs = self.dom.gen_idx_configs_full()
        return configs

    def gen_idx_configs_cex(self, dt):
        """
        Create new configs to break given decision tree dt
        """
        pos_conds, neg_conds = dt.get_path_conds_z3(self.z3db)
        conds = pos_conds + neg_conds

        configs = set()
        new_configs = []
        existing_configs = [self.z3db.convert_config(c)
                            for c in self.data.configs]

        for cond in conds:
            for i in range(len(cond)):
                not_existing_configs = z3.Not(z3.Or(existing_configs))
                cond_ = [c for c in cond]
                cond_[i] = z3.Not(cond[i])
                f = z3.And(self.z3db.facts,
                           not_existing_configs, z3.And(*cond_))

                models = self.z3db.get_models(f, k=1)
                if not models:
                    continue

                new_configs = [
                    self.z3db.get_config_from_model(m) for m in models]

                if not new_configs:
                    continue

                for c in new_configs:
                    assert c not in configs, c
                    configs.add(c)

                new_configs = [self.z3db.convert_config(c)
                               for c in new_configs]
                existing_configs = existing_configs + new_configs

        return configs

    def eval_configs(self, idx_configs):
        def eval_config(c):
            sids = self.get_cov(c)
            return sids

        results = []
        for c in idx_configs:
            results.append((c, eval_config(self.dom.to_real_config(c))))
        return results

    def learn(self, configs_covss):
        """
        Learn decision trees for each cov
        Return {cov: dt}

        configs_covs = [(config, covs)]
        """
        progress = False
        start_idx = len(self.data.configs)
        new_covs, update_covs = set(), set()

        for config, covs in configs_covss:
            assert config not in self.data.configs_cache

            # update data
            self.data.append(config, covs)

            for cov in covs:
                if cov not in self.cov_dts_cache:
                    new_covs.add(cov)
                else:
                    update_covs.add(cov)

        # learn dt for new cov or old ones which didn't have a dt
        for cov in sorted(new_covs):
            dt = self.learn_cov(cov)
            if dt:
                self.cov_dts_cache[cov] = dt
                if not progress:
                    progress = True

        # check and refine dt for old covs with new configs
        for cov in sorted(update_covs):

            assert cov in self.cov_dts_cache
            dt = self.cov_dts_cache[cov]
            assert dt
            pos_idxs, neg_idxs = self.data.get_idxs(cov, start_idx)
            pos_configs = self.data.get_configs(pos_idxs)
            neg_configs = self.data.get_configs(neg_idxs)
            X = pos_configs + neg_configs
            y = [1]*len(pos_configs) + [0]*len(neg_configs)
            score = dt.score(X, y)
            print("refining {}, old dt has prediction score {}".format(cov, score))
            print('pos')
            print(Data._str_configs(pos_configs))
            print('neg')
            print(Data._str_configs(neg_configs))
            if score != 1.0:
                dt = self.learn_cov(cov)
                self.cov_dts_cache[cov] = dt
                if dt and not progress:
                    progress = True

        return progress

    def learn_cov(self, cov):
        """
        Learn classifier for cov
        """
        pos_idxs, neg_idxs = self.data.get_idxs(cov)
        key = (pos_idxs, neg_idxs)
        if key in self.idxs_dt_cache:
            mlog.debug(
                "SKIP: already compute classifier for {}".format(key))
            return self.idxs_dt_cache[key]

        pos_configs = self.data.get_configs(pos_idxs)
        neg_configs = self.data.get_configs(neg_idxs)

        mlog.debug("{}: {} pos configs, {} neg configs".format(
            cov, len(pos_configs), len(neg_configs)))
        if not pos_configs or not neg_configs:
            mlog.warning("cannot create classifier!")
            dt = None
        else:
            dt = DecisionTree.learn(pos_configs, neg_configs)
            dt = DecisionTree(dt, self.myvars)

            mlog.debug("dt for '{}': {} nodes, max depth {}: {}".format(
                cov, dt.node_count, dt.max_depth,
                dt.get_path_conds_z3(self.z3db)))
            print(dt)
            print('pos')
            print(Data._str_configs(pos_configs))
            print('neg')
            print(Data._str_configs(neg_configs))

        self.idxs_dt_cache[key] = dt
        return dt


if __name__ == "__main__":

    aparser = argparse.ArgumentParser(
        "mbuild: analyze build systems dynamically")
    ag = aparser.add_argument

    ag("inp", help="inp")

    ag("--logger_level", "-logger_level", "-log", "--log",
       help="set logger info", type=int, choices=range(5), default=4)

    ag("--seed", "-seed",
       type=float,
       help="use this seed")

    args = aparser.parse_args()

    if (args.logger_level != Settings.logger_level and
            0 <= args.logger_level <= 4):
        Settings.logger_level = args.logger_level
    Settings.logger_level = CM.getLogLevel(Settings.logger_level)
    mlog = CM.getLogger(__name__, Settings.logger_level)

    seed = round(time(), 2) if args.seed is None else float(args.seed)

    dom = Dom.get_dom(args.inp)
    mlog.info(dom)
    exec = Exec(dom.keys(), dom.run_script)

    def run_f(config): return exec.run(config)

    mbuild = MBuild(dom, run_f)
    mbuild.go(seed, tmpdir=Settings.tmpdir)


# 9 {(0, 0, 1, 1, 0, 0, 0): [('>=', 4, 0.5), ('<', 5, 0.5), ('>=', 2, 0.5), ('<', 3, 0.5)], (0, 0, 1, 1, 0, 1, 0): [('>=', 4, 0.5), ('<', 5, 0.5), ('>=', 2, 0.5), ('>=', 3, 0.5), ('<', 1, 0.5), ('<', 0, 0.5)], (0, 0, 1, 1, 0, 1, 1): [('>=', 4, 0.5), ('<', 5, 0.5), ('>=', 2, 0.5), ('>=', 3, 0.5), ('>=', 1, 0.5)], (1, 1, 0, 0, 0, 0, 0): [('>=', 4, 0.5), ('>=', 5, 0.5), ('>=', 6, 2.5), ('>=', 2, 0.5), ('<', 3, 0.5)], (1, 1, 0, 0, 0, 1, 0): [('>=', 4, 0.5), ('>=', 5, 0.5), ('>=', 6, 2.5), ('>=', 2, 0.5), ('>=', 3, 0.5), ('<', 1, 0.5), ('<', 0, 0.5)], (1, 1, 0, 0, 0, 1, 1): [('>=', 4, 0.5), ('>=', 5, 0.5), ('>=', 6, 2.5), ('>=', 2, 0.5), ('>=', 3, 0.5), ('>=', 1, 0.5)], (1, 0, 0, 0, 0, 0, 0): [('>=', 4, 0.5), ('>=', 5, 0.5), ('<', 6, 2.5), ('>=', 6, 0.5), ('>=', 2, 0.5), ('<', 3, 0.5)], (1, 0, 0, 0, 0, 1, 0): [('>=', 4, 0.5), ('>=', 5, 0.5), ('<', 6, 2.5), ('>=', 6, 0.5), ('>=', 2, 0.5), ('>=', 3, 0.5), ('<', 1, 0.5), ('<', 0, 0.5)], (1, 0, 0, 0, 0, 1, 1): [('>=', 4, 0.5), ('>=', 5, 0.5), ('<', 6, 2.5), ('>=', 6, 0.5), ('>=', 2, 0.5), ('>=', 3, 0.5), ('>=', 1, 0.5)]}


# Ground truths for ex
#   iGen results
  # 1. (0) true (conj): (1) L3
  # 2. (2) (u=1 & v=1) (conj): (1) L4
  # 3. (2) (x=1 & y=1) (conj): (1) L0
  # 4. (2) (x=0 | y=0) (disj): (2) L2,L2a
  # 5. (3) (x=1 & y=1 & z=0,3,4) (conj): (1) L1
  # 6. (4) (s=1 | t=1) & (u=1 & v=1) (mix): (1) L5
# __main__:INFO:*** iter 1, select dt of 'L1', configs 320, covs 6
# __main__:INFO:L0: ([[1/2 <= y, 1/2 <= x]], [[1/2 > y], [1/2 <= y, 1/2 > x]])
# __main__:INFO:L1: ([[1/2 <= y, 1/2 <= x, 5/2 > z, 1/2 > z], [1/2 <= y, 1/2 <= x, 5/2 <= z]], [[1/2 > y], [1/2 <= y, 1/2 > x], [1/2 <= y, 1/2 <= x, 5/2 > z, 1/2 <= z]])
# __main__:INFO:L2: ([[1/2 > y], [1/2 <= y, 1/2 > x]], [[1/2 <= y, 1/2 <= x]])
# __main__:INFO:L2a: ([[1/2 > y], [1/2 <= y, 1/2 > x]], [[1/2 <= y, 1/2 <= x]])
# __main__:INFO:L3: True
# __main__:INFO:L4: ([[1/2 <= v, 1/2 <= u]], [[1/2 > v], [1/2 <= v, 1/2 > u]])
# __main__:INFO:L5: ([[1/2 <= v, 1/2 <= u, 1/2 > s, 1/2 <= t], [1/2 <= v, 1/2 <= u, 1/2 <= s]], [[1/2 > v], [1/2 <= v, 1/2 > u], [1/2 <= v, 1/2 <= u, 1/2 > s, 1/2 > t]])

# L0: [y and x] ,   [!y or (y and !x)]
# L1: [y and x and z=0,  y and x and z = 3,4],  [Or(!y, And(y and !x), And(y,x,z=1,2)]
# ..
