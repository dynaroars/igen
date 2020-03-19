//
// Created by kh on 3/14/20.
//

#include "Algo.h"

#include <igen/Context.h>
#include <igen/Domain.h>
#include <igen/Config.h>
#include <igen/ProgramRunner.h>
#include <igen/CoverageStore.h>

#include <klib/print_stl.h>
#include <klib/vecutils.h>
#include <igen/c50/CTree.h>
#include <fstream>

#include <boost/algorithm/string.hpp>
#include <boost/container/flat_map.hpp>

namespace igen {


class Analyzer : public Object {
public:
    explicit Analyzer(PMutContext ctx) : Object(move(ctx)) {}

    map<std::pair<int, int>, int> map_count_cex;

    expr to_expr(const z3::model &m) {
        z3::expr_vector vecExpr(m.ctx());
        int nvars = m.num_consts();
        vecExpr.resize(nvars);
        for (int i = 0; i < nvars; ++i) {
            const auto &de = m.get_const_decl(i);
            z3::expr eqExpr = (de() == m.get_const_interp(de));
            vecExpr.set(i, eqExpr);
        }
        return z3::mk_and(vecExpr);
    }

    int count_cex(const expr &a, const expr &b, int lim = 1000) {
        auto it = map_count_cex.find({a.id(), b.id()});
        if (it != map_count_cex.end()) return it->second;
        int &ncex = map_count_cex[{a.id(), b.id()}];

        expr bi = (a != b);
        auto solver = ctx()->zscope();
        solver->add(a != b);

        while (ncex < lim) {
            z3::check_result checkres = solver->check();
            if (checkres == z3::unsat) {
                break;
            } else if (checkres == z3::unknown) {
                LOG(WARNING, "Z3 solver returns unknown:\n") << *solver;
                ncex = lim;
                break;
            }
            CHECK_EQ(checkres, z3::sat);
            ncex++;
            z3::model m = solver->get_model();
            expr mexpr = to_expr(m);
            solver->add(!mexpr);
        }
        return ncex;
    }

    map<str, expr> read_file(const str &path) {
        map<str, expr> res;
        std::ifstream f(path);
        CHECKF(!f.fail(), "Error reading file: {}", path);
        str line;
        bool read_loc = true;
        z3::sort_vector empty_sort_vector(zctx());
        vec<str> locs;
        str sexpr = "(assert ";
        set<unsigned> sexprid;
        while (getline(f, line)) {
            boost::algorithm::trim(line);
            if (line.empty() || line[0] == '#') continue;
            if (read_loc && line[0] == '-') {
                CHECKF(!locs.empty(), "Empty location set ({})", path);
                read_loc = false;
                continue;
            } else if (line[0] == '=') {
                if (locs.empty()) continue;
                sexpr += ')';
                z3::expr_vector evec = ctx()->zctx().parse_string(
                        sexpr.c_str(), empty_sort_vector, dom()->func_decl_vector());
                CHECK_EQ(evec.size(), 1);
                expr e = evec[0];
                CHECKF(sexprid.insert(e.id()).second, "Duplicated expression ({})", path);
                //LOG(INFO, "EXPR: ") << e;
                for (const str &s : locs) {
                    CHECKF(!res.contains(s), "Duplicated location ({}): {}", path, s);
                    res.emplace(s, e);
                }
                read_loc = true, locs.clear(), sexpr = "(assert ";
                continue;
            }
            if (read_loc) {
                for (char &c : line) if (c == ',') c = ' ';
                std::stringstream ss(line);
                str tok;
                while (ss >> tok) locs.emplace_back(move(tok));
            } else {
                sexpr += line, sexpr += '\n';
            }
        }
        return res;
    }

    // compare two output
    void run_analyzer_0() {
        auto finp = get_inp();
        CHECK_EQ(finp.size(), 2) << "Need two input files to compare";
        auto ma = read_file(finp.at(0));
        auto mb = read_file(finp.at(1));
        set<unsigned> sdiff, smissing, slocsa, slocsb, sprintdiff;
        int cntdiff = 0, cntmissing = 0;
        for (const auto &p : ma) {
            slocsa.insert(p.second.id());
            auto it = mb.find(p.first);
            if (it == mb.end()) {
                LOG(INFO, "{} not found in B", p.first);
                smissing.insert(p.second.id()), cntmissing++;
                continue;
            }
            int num_cex = count_cex(p.second, it->second);
            if (num_cex == 0) {
                //VLOG(0, "{} ok", p.first);
            } else {
                LOG(INFO, "{} diff (cex = {}) ({})", p.first, num_cex, p.second.id());
                if (sprintdiff.insert(p.second.id()).second)
                    GVLOG(0) << "\nA: " << p.second << "\nB: " << it->second;
                sdiff.insert(p.second.id()), cntdiff++;
            }
        }
        for (const auto &p : mb) {
            slocsb.insert(p.second.id());
            if (!ma.contains(p.first)) {
                LOG(INFO, "{} not found in A", p.first);
                smissing.insert(p.second.id()), cntmissing++;
            }
        }
        LOG(INFO, "{:=^80}", "  FINAL RESULT  ");
        LOG(INFO, "Total: diff {:>4}, miss {:>4}, locs A {:>4}, B {:>4}", cntdiff, cntmissing, ma.size(), mb.size());
        LOG(INFO, "Uniq : diff {:>4}, miss {:>4}, locs A {:>4}, B {:>4}", sdiff.size(), smissing.size(),
            slocsa.size(), slocsb.size());
    }


    vec<str> get_inp() {
        str inp = ctx()->get_option_as<str>("input");
        boost::algorithm::replace_all(inp, ",", " ");
        std::stringstream ss(inp);
        str tok;
        vec<str> res;
        while (ss >> tok) res.push_back(tok);
        return res;
    }

    void run_alg() {
        switch (ctx()->get_option_as<int>("alg-version")) {
            case 0:
                return run_analyzer_0();
            default:
                CHECK(0);
        }
    }
};


int run_analyzer(const boost::program_options::variables_map &vm) {
    PMutContext ctx = new Context();
    for (const auto &kv : vm) {
        ctx->set_option(kv.first, kv.second.value());
    }
    ctx->init();
    //ctx->program_runner()->init();
    {
        ptr<Analyzer> ite_alg = new Analyzer(ctx);
        ite_alg->run_alg();
    }
    ctx->cleanup();
    return 0;
}


}