```
#!shell

sage: %attach gcc_cfg.py
sage: cfg2preds('/home/tnguyen/igen_exps/coreutils/coreutils-8.23/obj-gcov/src/join.c.011t.cfg')
write to '/home/tnguyen/igen_exps/coreutils/coreutils-8.23/obj-gcov/src/join.c.011t.cfg.preds'

cd examples/iga
python -O ../../iga/iga.py -dom_file ex5.dom -cfg ex5.c.011t.cfg.preds -run_script run_script5 -seed 0 -logger_level 2


cd iga

python -O iga.py uname -seed 0 -sid ../src/uname.c:277 -logger_level 3
```


