UNAME := $(shell uname)
#OCAML_OPTIONS = -I /usr/local/lib/ocaml/3.12.1/cil/
ifeq ($(UNAME), Linux)
OCAML_OPTIONS = -I /home/tnguyen/Src/Devel/CIL/cil-1.3.7/obj/x86_LINUX
else
OCAML_OPTIONS = -I /usr/local/lib/ocaml/site-lib/cil/
endif


OCAMLC =    ocamlfind  ocamlc   $(OCAML_OPTIONS)
OCAMLOPT =  ocamlfind ocamlopt $(OCAML_OPTIONS)
OCAMLDEP =  ocamldep $(OCAML_OPTIONS)
OCAMLLEX =  ocamllex 

all: cdg

# %.cmo: %.ml 
# 	@if [ -f $*.mli -a ! -f $*.cmi ] ; then $(OCAMLC) -c -g $*.mli ; fi 
# 	$(OCAMLC) -c -g $*.ml
# 	@$(OCAMLDEP) $*.ml > $*.d 

%.cmx: %.ml 
	@if [ -f $*.mli -a ! -f $*.cmi ] ; then $(OCAMLC) -c -g $*.mli ; fi 
	$(OCAMLOPT) -c $*.ml
	@$(OCAMLDEP) $*.ml > $*.d 

%.cmi: %.mli
	$(OCAMLC) -c -g $*.mli

# %.ml: %.mll
# 	$(OCAMLLEX) $*.mll

# NOTE: Module order is important!  OCaml module dependencies cannot
# be cyclic, and the order presented must respect the dependency order.

CDG_MODULES = \
	cdg.cmo \

cdg: $(CDG_MODULES:.cmo=.cmx) 
	$(OCAMLOPT) -o $@ unix.cmxa str.cmxa nums.cmxa cil.cmxa  $^


# dependencies
ALL_MODULES =  $(MAIN_MODULES) 

-include $(ALL_MODULES:.cmo=.d)

clean:
	rm -f *.cmo *.cmi *.d *.cmx *.dx *.o cdg
