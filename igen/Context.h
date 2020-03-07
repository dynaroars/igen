//
// Created by KH on 3/5/2020.
//

#ifndef IGEN4_CONTEXT_H
#define IGEN4_CONTEXT_H

#include <klib/common.h>
#include <boost/container/flat_map.hpp>
#include <boost/any.hpp>
#include <z3++.h>

#include "Z3Scope.h"

namespace igen {

class Domain;

void intrusive_ptr_release(Domain *);

class ProgramRunner;

void intrusive_ptr_release(ProgramRunner *);

class CoverageStore;

void intrusive_ptr_release(CoverageStore *);

class Context : public intrusive_ref_base_st<Context> {
public:
    Context();

    virtual ~Context();

    void set_option(const str &key, boost::any val);

    bool has_option(const str &key) const;

    boost::any get_option(const str &key) const;

    template<class T>
    T get_option_as(const str &key) const { return boost::any_cast<T>(get_option(key)); }

    void init();

    void cleanup();

    ptr<const Domain> dom() const;

    const ptr<Domain> &dom();

    ptr<const ProgramRunner> program_runner() const;

    const ptr<ProgramRunner> &program_runner();

    ptr<const CoverageStore> cov() const;

    const ptr<CoverageStore> &cov();

private:
    friend class Object;

    map<str, boost::any> options;

    z3::context z3ctx;
    z3::solver z3solver;
    z3::expr z3true;
    z3::expr z3false;

    ptr<Domain> dom_;
    ptr<ProgramRunner> program_runner_;
    ptr<CoverageStore> coverage_store_;
};

using PContext = ptr<const Context>;
using PMutContext = ptr<Context>;

class Object : public intrusive_ref_base_st<Object> {
public:
    PContext ctx() const { return ctx_; }

    PMutContext ctx() { return ctx_; }

    PMutContext ctx_mut() const { return ctx_; }

    z3::context &zctx() { return ctx()->z3ctx; }

    z3::solver &zsolver() { return ctx()->z3solver; }

    Z3Scope zscope() { return Z3Scope(&ctx()->z3solver); }

    const z3::expr &ztrue() const { return ctx()->z3true; }

    const z3::expr &zfalse() const { return ctx()->z3false; }

    ptr<const Domain> dom() const;

    ptr<const CoverageStore> cov() const;

    ptr<CoverageStore> cov();

    virtual ~Object() = default;

protected:
    explicit Object(PMutContext ctx) : ctx_(move(ctx)) {};

private:
    PMutContext ctx_;
};

using PObject = ptr<const Object>;
using PMutObject = ptr<Object>;

}

#endif //IGEN4_CONTEXT_H
