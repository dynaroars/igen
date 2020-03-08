//
// Created by KH on 3/7/2020.
//

#include "CNode.h"
#include "CTree.h"

#include <klib/math.h>
#include <boost/scope_exit.hpp>

namespace igen {


CNode::CNode(CTree *tree, CNode *parent, std::array<boost::sub_range<vec<PConfig>>, 2> configs) :
        tree(tree), parent(parent), configs_(std::move(configs)) {
    depth_ = (parent ? parent->depth() + 1 : 0);
}

PDomain CNode::dom() const {
    return tree->ctx()->dom();
}

void CNode::calc_freq() {
    for (int hit = 0; hit <= 1; ++hit) {
        freq[hit] = dom()->create_vec_vars_values<int>();
        for (const auto &c : configs_[hit]) {
            const auto &vals = c->values();
            for (int var_id = 0; var_id < dom()->n_vars(); ++var_id) {
                freq[hit][var_id][vals[var_id]]++;
            }
        }
    }
}

void CNode::calc_inf_gain() {
    bestvar = -1;
    double log2ntotal = log2(n_total());

    // Calc inf
    inf = dom()->create_vec_vars<double>();
    for (int var_id = 0; var_id < dom()->n_vars(); ++var_id) {
        double sum = 0;
        for (int val = 0; val < dom()->n_values(var_id); ++val) {
            int n = freq[0][var_id][val] + freq[1][var_id][val];
            sum += n * log2(n);
        }
        inf[var_id] = log2ntotal - sum / n_total();
    }

    // Calc base_info
    double base_info = 0;
    for (int hit = 0; hit <= 1; ++hit) {
        int n = int(configs_[hit].size());
        base_info += n * log2(n);
    }
    base_info = log2ntotal - base_info / n_total();

    // Calc gain
    double total_gain = 0;
    gain = dom()->create_vec_vars<double>();
    for (int var_id = 0; var_id < dom()->n_vars(); ++var_id) {
        for (int val = 0; val < dom()->n_values(var_id); ++val) {
            double sum = 0;
            int total = 0;
            for (int hit = 0; hit <= 1; ++hit) {
                int n = freq[hit][var_id][val];
                sum += n * log2(n);
                total += n;
            }
            sum = total * log2(total) - sum;
            gain[var_id] += sum;
        }
        gain[var_id] = base_info - gain[var_id] / n_total();
        total_gain += gain[var_id];
    }

    // Calc avgain & mdl
    avgain = total_gain / dom()->n_vars();
}

bool CNode::evaluate_split() {
    BOOST_SCOPE_EXIT(this_) { this_->clear_all_tmp_data(); }
    BOOST_SCOPE_EXIT_END
    if (is_leaf())
        return false;

    calc_freq();
    calc_inf_gain();
    VLOG_BLOCK(30, print_tmp_state(log << "\n<" << depth() << ">: " << n_total() << " cases\n"));

    return split_by != nullptr;
}

void CNode::clear_all_tmp_data() {
    freq[0].clear(), freq[1].clear();
    inf.clear(), gain.clear();
}

std::ostream &CNode::print_tmp_state(std::ostream &output, const str &indent) const {
    for (const auto &var : dom()->vars()) {
        output << indent << "Var " << var->name() << ": \n";
        fmt::print(output, "{}{}[{:>4}{:>8}{:>8}]\n", indent, indent, "Val", "MISS", "HIT");
        for (int i = 0; i < var->n_values(); ++i) {
            fmt::print(output, "{}{}[{:>4}{:>8}{:>8}]\n", indent, indent,
                       i, freq[0][var->id()][i], freq[1][var->id()][i]);
        }
        fmt::print(output, "{}{}inf {:.3f}, gain {:.3f}, ratio {:.3f}\n", indent, indent,
                   inf[var->id()], gain[var->id()], gain[var->id()] / inf[var->id()]);
    }
    fmt::print(output, "{}{}av gain={:.3f}, best var ", indent, indent, avgain);
    if (bestvar == -1) {
        output << "N/A\n";
    } else {
        fmt::print(output, "{}: inf {:.3f}, gain {:.3f}, ratio {:.3f}\n",
                   dom()->name(bestvar), inf[bestvar], gain[bestvar], gain[bestvar] / inf[bestvar]);
    }
    return output;
}

}