#!/usr/bin/env python3

import pdb
from collections import OrderedDict
from time import time
import random
import os.path
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree
import tempfile
import itertools
import argparse
import vcommon as CM
import z3


# from sklearn.model_selection import train_test_split
DBG = pdb.set_trace
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

    def __init__(self, dom):
        """
        len(configs) == len(covs)
        len(configs[0]) == len(myvars)
        """
        self.configs = []   # [tuple]
        self.covss = []     # [list]    [[0,1,0]]
        self.configs_cache = set()

    @classmethod
    def _str_configs(cls, configs):
        return '\n'.join(str_of_list(c) for c in configs)

    def str_of_data(self, myvars, covs_d):
        ss = ["{}: {}".format(str_of_list(myvars),
                              str_of_list(covs_d))]
        ss.extend("{} : {}".format(str_of_list(config), str_of_list(covs))
                  for config, covs in zip(self.configs, self.covss))
        return '\n'.join(ss)

    def extend(self, configs_covss, covs_d):

        # check if any new cov, if so need to update covss
        new_covs = set()
        for config, covs in configs_covss:
            assert config not in self.configs_cache
            for cov in covs:
                if cov not in covs_d:
                    new_covs.add(cov)

        if new_covs:
            new_covs = list(sorted(new_covs))
            for cov in new_covs:
                covs_d[cov] = len(covs_d)

            for i in range(len(self.covss)):
                self.covss[i] = self.covss[i] + [0]*len(new_covs)

        for config, covs in configs_covss:
            self.configs.append(config)
            self.configs_cache.add(config)

            covs_ = [0] * len(covs_d)
            for cov in covs:
                covs_[covs_d[cov]] = 1
            self.covss.append(covs_)

        return new_covs


class DecisionTree:
    test_size = .0  # percent

    def __init__(self, tree, myvars):
        self.tree = tree
        self.myvars = myvars

    def __str__(self):
        ss = []
        self.print_tree(root=0, depth=1, ss=ss)
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
        """
        [0, 0, 1, 1, 0] -> ('<', 6, 2.5)
        [0, 0, 1, 1, 0] -> [('>=', 6, 2.5), ('<', 5, 0.5)]
        [1, 1, 0, 0, 0] -> [('>=', 6, 2.5), ('>=', 5, 0.5)]
        """
        try:
            return self._path_conds
        except AttributeError:
            self._path_conds = {}
            paths = self.collect_paths(root=0)
            for path in paths:
                assert path
                path_, covs = path[:-1], tuple(path[-1])
                self._path_conds[covs] = path_

            return self._path_conds

    @property
    def str_of_path_conds(self):
        """
        pretty print
        """

        for covs in self.path_conds:
            [cov for cov in covs if cov]

    def get_path_conds_z3(self, z3db):
        """
        Return path constraints for truth paths
        """
        try:
            return self._path_conds_z3
        except AttributeError:
            self._path_conds_z3 = {k: self._get_conds(path, z3db)
                                   for k, path in self.path_conds.items()}
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
    def learn(cls, X, y):
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        return dt

    def score(self, X, y):
        y_ = self.tree.predict(X)
        n_row = len(y_)
        n_col = len(y_[0])
        assert n_row == len(y)
        assert n_col == len(y[0])

        n_mismatch = 0
        for i in range(n_row):
            for j in range(n_col):
                if int(y_[i][j]) != y[i][j]:
                    n_mismatch += 1

        myscore = n_mismatch/(n_row * n_col)
        return myscore

    def print_tree(self, root=0, depth=1, ss=[]):
        tree = self.tree.tree_
        indent = '    '*depth
        # ss.append("{} # node {}: impurity = {}".format(
        #     indent, root, tree.impurity[root]))

        left_child = tree.children_left[root]
        right_child = tree.children_right[root]

        if left_child == sklearn.tree._tree.TREE_LEAF:
            ss.append("{} return {} # (node {}, impurity {})".format(
                indent, tree.value[root], root, tree.impurity[root]))
        else:
            ss.append("{} if {} < {}: # (node {})".format(
                indent, self.myvars[tree.feature[root]],
                tree.threshold[root], root))
            self.print_tree(root=left_child, depth=depth+1, ss=ss)

            ss.append("{} else".format(indent))
            self.print_tree(root=right_child, depth=depth+1, ss=ss)

    @classmethod
    def get_truth(self, values):
        """
        values =
        [[3., 0.],
        [0., 3.],
        [0., 3.],
        [3., 0.]]
        returns
        [0, 1, 1, 0]
        """
        assert len(values) and len(values[0]) == 2, values

        vs = []
        for v in values:
            num_neg, num_pos = v
            assert ((num_pos == 0 and num_neg > 0) or
                    (num_pos > 0 and num_neg == 0)), (num_pos, num_neg)

            if num_pos == 0:
                vs.append(0)
            else:
                assert num_neg == 0 and num_pos > 0
                vs.append(1)

        return vs

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
        self.data = Data(dom)
        self.myvars = self.dom.myvars
        self.z3db = Z3DB(dom)
        self.dt = None

        self.covs_d = {}  # L0 -> 0,  L1 -> 1

        # self.idxs_dt_cache = {}  # data idxs -> decision tree
        # self.cov_dts_cache = {}  # cov -> decision tree  (#main results)

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
            # select_cov, select_dt = self.select_dt()
            mlog.info("*** iter {}, "
                      "configs {}, covs {}".format(
                          cur_iter,
                          len(self.data.configs),
                          len(self.covs_d)))

            idx_configs = self.gen_idx_configs_cex()
            econfigs = self.eval_configs(idx_configs)
            progress = self.learn(econfigs)

        # print results
        # for cov in sorted(set(itertools.chain(*self.data.covss))):
        #     if cov not in self.cov_dts_cache:
        #         cond = True
        #     else:
        #         dt = self.cov_dts_cache[cov]
        #         cond = dt.get_path_conds_z3(self.z3db)

        #     mlog.info("{}: {}".format(cov, cond))

    def select_dt(self):
        cov = max(self.cov_dts_cache,
                  key=lambda cov:
                  self.cov_dts_cache[cov].node_count)

        dt = self.cov_dts_cache[cov]
        return cov, dt

    def gen_idx_configs_init(self):
        #configs = self.dom.gen_idx_configs_tcover1()
        configs = self.dom.gen_idx_configs_full()
        return configs

    def gen_idx_configs_cex(self):
        """
        Create new configs to break given decision tree dt
        """
        dt = self.dt
        conds = dt.get_path_conds_z3(self.z3db).values()
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

        ss = []
        for config, covs in results:
            ss.append(','.join(map(str, list(config) + covs)))
        print('\n'.join(ss))
        return results

    def learn(self, configs_covss):
        """
        Learn decision trees for each cov
        Return {cov: dt}

        configs_covs = [(config, covs)]
        """
        progress = False
        start_idx = len(self.data.configs)
        new_covs = self.data.extend(configs_covss, self.covs_d)
        print("data size {} (old {})".format(
            len(self.data.configs), start_idx))

        if not self.data or new_covs:
            # build entire tree
            dt = DecisionTree.learn(self.data.configs, self.data.covss)
            self.dt = DecisionTree(dt, self.dom.myvars)
            print(self.dt)
            print(self.covs_d,
                  len(self.dt.path_conds),
                  self.dt.path_conds)
            progress = True
        else:
            # predict and test over new set of data only.
            X = self.data.configs[start_idx:]
            y = self.data.covss[start_idx:]
            score = self.dt.score(X, y)
            if score != 1.0:
                print("score is {}".format(score))
                dt = DecisionTree.learn(
                    self.data.configs, self.data.covss)
                self.dt = DecisionTree(dt, self.dom.myvars)
                print(len(self.dt.path_conds), self.dt.path_conds)
                progress = True

        return progress

        # new_covs, update_covs = set(), set()

        # for config, covs in configs_covss:
        #     assert config not in self.data.configs_cache

        #     # update data
        #     self.data.append(config, covs)

        #     for cov in covs:
        #         if cov not in self.cov_dts_cache:
        #             new_covs.add(cov)
        #         else:
        #             update_covs.add(cov)

        # # learn dt for new cov or old ones which didn't have a dt
        # for cov in sorted(new_covs):
        #     dt = self.learn_cov(cov)
        #     if dt:
        #         self.cov_dts_cache[cov] = dt
        #         if not progress:
        #             progress = True

        # # check and refine dt for old covs with new configs
        # for cov in sorted(update_covs):

        #     assert cov in self.cov_dts_cache
        #     dt = self.cov_dts_cache[cov]
        #     assert dt
        #     pos_idxs, neg_idxs = self.data.get_idxs(cov, start_idx)
        #     pos_configs = self.data.get_configs(pos_idxs)
        #     neg_configs = self.data.get_configs(neg_idxs)
        #     X = pos_configs + neg_configs
        #     y = [1]*len(pos_configs) + [0]*len(neg_configs)
        #     score = dt.score(X, y)
        #     print("refining {}, old dt has prediction score {}".format(cov, score))
        #     print('pos')
        #     print(Data._str_configs(pos_configs))
        #     print('neg')
        #     print(Data._str_configs(neg_configs))
        #     if score != 1.0:
        #         dt = self.learn_cov(cov)
        #         self.cov_dts_cache[cov] = dt
        #         if dt and not progress:
        #             progress = True

        return progress

    # def learn_cov(self, cov):
    #     """
    #     Learn classifier for cov
    #     """
    #     pos_idxs, neg_idxs = self.data.get_idxs(cov)
    #     key = (pos_idxs, neg_idxs)
    #     if key in self.idxs_dt_cache:
    #         mlog.debug(
    #             "SKIP: already compute classifier for {}".format(key))
    #         return self.idxs_dt_cache[key]

    #     pos_configs = self.data.get_configs(pos_idxs)
    #     neg_configs = self.data.get_configs(neg_idxs)

    #     mlog.debug("{}: {} pos configs, {} neg configs".format(
    #         cov, len(pos_configs), len(neg_configs)))
    #     if not pos_configs or not neg_configs:
    #         mlog.warning("cannot create classifier!")
    #         dt = None
    #     else:
    #         dt = DecisionTree.learn(pos_configs, neg_configs)
    #         dt = DecisionTree(dt, self.myvars)

    #         mlog.debug("dt for '{}': {} nodes, max depth {}: {}".format(
    #             cov, dt.node_count, dt.max_depth,
    #             dt.get_path_conds_z3(self.z3db)))
    #         print(dt)
    #         print('pos')
    #         print(Data._str_configs(pos_configs))
    #         print('neg')
    #         print(Data._str_configs(neg_configs))

    #     self.idxs_dt_cache[key] = dt
    #     return dt


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
