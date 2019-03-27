open Cil
module E = Errormsg
module L = List
module P = Printf
module Ih = Inthash
module H = Hashtbl
  
let write_file_str (filename:string) (content:string): unit = 
  let fout = open_out filename in
  P.fprintf fout "%s" content; 
  close_out fout;
  E.log "write_file_str: '%s'\n" filename

let write_src ?(use_stdout:bool=false)
    (filename:string) (ast:file): unit = 
  let df oc =  dumpFile defaultCilPrinter oc filename ast in
  if use_stdout then df stdout else (
    let fout = open_out filename in
    df fout;
    close_out fout;
    E.log "write_src: '%s'\n" filename
  )

let copy_obj (x : 'a) = 
  let s = Marshal.to_string x [] in (Marshal.from_string s 0 : 'a)

let string_of_stmt (s:stmt) = Pretty.sprint ~width:80 (dn_stmt () s)
let string_of_list ?(delim:string = ", ") =  String.concat delim
let string_of_ints ?(delim:string = ", ") (l:int list): string = 
  string_of_list ~delim (L.map string_of_int l)
type sid_t = int
let sids_of_stmts = L.map (fun stmt -> stmt.sid)
let exp_of_vi (vi:varinfo): exp = Lval (var vi)

let mk_vi ?(ftype=TVoid []) fname: varinfo =
  makeVarinfo true fname ftype

let mk_call ?(ftype=TVoid []) ?(av=None) fname args : instr = 
  let f = var(mk_vi ~ftype:ftype fname) in
  Call(av, Lval f, args, !currentLoc)

let stderr_vi = mk_vi ~ftype:(TPtr(TVoid [], [])) "_coverage_fout"

  
(*
  walks over AST and preceeds each stmt with a printf that writes out its sid
  create a stmt consisting of 2 Call instructions
  fprintf "_coverage_fout, sid"; 
  fflush();
*)

class coverageVisitor = object(self)
  inherit nopCilVisitor

  method private create_fprintf_stmt (sid : sid_t) :stmt = 
    let str = P.sprintf "%d\n" sid in
    let stderr = exp_of_vi stderr_vi in
    let instr1 = mk_call "fprintf" [stderr; Const (CStr(str))] in 
    let instr2 = mk_call "fflush" [stderr] in
    mkStmt (Instr([instr1; instr2]))


  method vblock b = 
    let action (b: block) :block= 
      let insert_printf (s: stmt): stmt list = 
	[self#create_fprintf_stmt s.sid; s]
      in
      let stmts = L.map insert_printf b.bstmts in 
      {b with bstmts = L.flatten stmts}
    in
    ChangeDoChildrenPost(b, action)
end


class printCFGVisitor cfg_contents = object(self)
  inherit nopCilVisitor
  method vstmt s =
    let neg_id_in_stmts stmts = L.mem (-1) (sids_of_stmts stmts) in
    
    E.log "\n*** sid %d *** \n%a\n" s.sid dn_stmt s;

    let preds = string_of_ints  ~delim:" " (sids_of_stmts s.preds) in
    let succs = string_of_ints ~delim:" " (sids_of_stmts s.succs) in
    E.log "preds: [%s]\nsuccs: [%s]\n*****\n" preds succs;

    if neg_id_in_stmts s.preds || neg_id_in_stmts s.succs then
      (E.s (E.error "Problem: -1 in list"));
    
    cfg_contents := !cfg_contents ^ (P.sprintf "%d %s\n" s.sid preds);
    DoChildren
end


let () = begin

  let inp = ref "" in
  let seed = ref 0. in 
  let sid = ref 0 in 

  let argDescr = [
    "--seed", Arg.Set_float seed, "use this seed";
    "--sid", Arg.Set_int sid, "solve for this sid";
  ] in

  
  let handleArg s = 
    if !inp = "" then inp := s
    else raise (Arg.Bad "too many input files")
  in

  Arg.parse (Arg.align argDescr) handleArg 
    "prog C file";

  P.printf "inp: %s\n" !inp;
  
  initCIL();
  let ast = Frontc.parse !inp () in
  Cfg.computeFileCFG ast;
  write_src (!inp ^ ".cil.c") ast; (*save orig file after computing ast*)

  let inp = Filename.chop_extension (!inp) in    
  let cfg_contents = ref "" in
  ignore(visitCilFileSameGlobals (new printCFGVisitor cfg_contents) ast);
  E.log "%s" !cfg_contents;
  write_file_str (inp ^ ".cfg") !cfg_contents;

  let stmts:stmt list = Cfg.allStmts ast in
  E.log "all stmts in file (%d): [%s]\n" (L.length stmts)
    (string_of_ints ~delim:" " (sids_of_stmts stmts));
    

  (*Create coverage info by printing stmt id*)
  let cov_ast = copy_obj ast in 
  ignore(visitCilFileSameGlobals (new coverageVisitor) cov_ast);
  (*Add to global
    _coverage_fout = fopen("file.path", "ab");
  *)
  let new_global = GVarDecl(stderr_vi, !currentLoc) in
  cov_ast.globals <- new_global :: cov_ast.globals;

  let lhs = var(stderr_vi) in
  let arg1 = Const(CStr(inp ^ ".path")) in
  let arg2 = Const(CStr("ab")) in
  let instr = mk_call ~av:(Some lhs) "fopen" [arg1; arg2] in
  let new_s = mkStmt (Instr[instr]) in

  let fd = getGlobInit cov_ast in
  fd.sbody.bstmts <- new_s :: fd.sbody.bstmts;
  
  write_src (inp ^ ".cov.c") cov_ast
    
end


