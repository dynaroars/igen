# Reproduce Results
This section shows how to reproduce the results reported in the FSE'16 paper *iGen: Dynamic Interaction Inference for Configurable Software*. The instructions below have been tested on a Debian machine.

These results were obtained over 21 iGen's runs per program, which can be downloaded from https://drive.google.com/folderview?id=0By7WC-3FhI4sQTJ1SWlYVGpsdWM&usp=sharing  (then go to the *data* directory)

Here is an example to reproduce the results for the GNU utils `uname` command:

1. Obtain `uname` data
    * igen: data/uname_cegir_data.tar.gz
    * full: data/uname_full_data.tar.gz
    
   ```
   $ cd data_dir
   $ wget ...
   $ untar ...

   ```

2. We can do a general analysis on these iterative runs (Tables 2 and 5)
```
$ python -O $IGEN/src/igen.py uname_cegir_data
 ...
10:29:48:analysis:Info:iter 21.0 (1.5), ints 25.0 (0.0), time 15.3179090023 (1.39020037651), xtime 8.89465141296 (0.958407521248), configs 95.0 (5.0), covs 87.0 (0.5), nminconfigs 0.0 (0.0), nmincovs 0.0 (0.0)
10:29:48:analysis:Info:Int types: conjs 16.0 (0.5), disjs 2.0 (0.5), mixed 7.0 (1.0)
10:29:48:analysis:Info:Int strens: (0, 1.0 (0.0), 10.0 (0.5)), (1, 10.0 (0.0), 32.0 (0.0)), (2, 3.0 (0.0), 32.0 (3.5)), (4, 9.0 (0.5), 11.0 (0.5)), (8, 0.0 (0.5), 0.0 (3.0)), (11, 2.0 (1.0), 2.0 (1.0))
```

    For these 21 runs for `uname`, we obtain 
    * General: config `95 (5)`, cov `87 (0)`
    * Time: search 9 (1), total `15 (1)`
    * Interactions
       - Types: conj `16 (1)`, disj `2 (1)`, mixed `7.0 (1.0)`
       - Strengths: `(0, 1, 10), (1, 10, 32), (2, 3, 32), (4, 9, 11), (11, 2, 2)`
		  
3. Compare iterative to exhaustive runs (Table 3 and Figure 3)

```
$ python -O $IGEN/src/igen.py uname_cegir_data -cmp_dir ~/igen_exps/data_dirs/uname_full_data/run0__9N51U/ --evolution
 ...
18:19:30:alg_miscs:Info:fscores (iter, fscore, configs): (1, 0.29012673150604196, 2) -> (2, 0.8853353543008714, 13) -> (3, 0.9171655930276619, 23) -> (4, 0.9348659003831417, 32) -> (5, 0.9424368734713563, 40) -> (6, 0.9486169003410384, 47) -> (7, 0.8961898892933374, 53) -> (8, 0.9010430182843976, 58) -> (9, 0.90748204541308, 63) -> (10, 0.919842099152444, 68) -> (11, 0.919842099152444, 71) -> (12, 0.919842099152444, 73) -> (13, 0.9260536398467433, 80) -> (14, 0.9268746579091405, 86) -> (15, 0.9279693486590037, 89) -> (16, 0.9279693486590037, 91) -> (17, 0.9279693486590037, 93) -> (18, 0.9279693486590037, 95) -> (19, 0.929501915708812, 97) -> (20, 0.929501915708812, 99) -> (21, 0.929501915708812, 101) -> (22, 0.929501915708812, 103) -> (23, 0.929501915708812, 105) -> (24, 0.929501915708812, 106)
```

   The `--evolution` flag compares iGen's iterative results with a particular run (in this case the exhaustive run), whose data is stored in dir `~/igen_exps/data_dirs/uname_full_data/run0__9N51U/`.
   The _fscore_ result `0.93` indicates iGen's interative and exhaustive runs produce very similar results.
   
4. Compare random to exhaustive runs (Table 4)
```
$ python -O $IGEN/src/igen.py -log 2 -evolution -cmp_rand uname ~/igen_exps/data_dirs/uname_full_data/run0__9N51U/
...
23:1:52:analysis:Info:rand: configs 90 cov 87 vscore 3993 fscore 0.0919540229885
23:1:52:analysis:Info:cegir: configs 90 cov 87 vscore 3749 fscore 0.952490421456
```




### GNU Utils ###
1. uname
 * igen: https://drive.google.com/uc?export=download&id=0By7WC-3FhI4sczRaTjJ3Q1I3c0k
 * full: https://drive.google.com/uc?export=download&id=0By7WC-3FhI4sa3NobFZPakNIcTg



## Running Experiments ##

This section shows how to setup and run the programs used in the FSE'16 paper *iGen: Dynamic Interaction Inference for Configurable Software*. The instructions below have been tested on a Debian machine.

For demonstration, we will setup these experiments in the  directory `~\igen_exps`.

## GNU Coreutils

### Setup

We use `gcc` and `gcov` to obtain coverage information for `coreutil` commands. First, download [`coreutils-8-23`](http://ftp.gnu.org/gnu/coreutils/coreutils-8.23.tar.xz). Then
```
$ cd ~/igen_exps/mycoreutils; tar xf /PATH/TO/coreutils-8.23.tar.xz; ln -sf coreutils-8.23 coreutils; cd coreutils; tar xf $IGEN/scripts/coreutils_testfiles.tar.gz
```
Next compile these programs
```
#!shell
$ mkdir obj-gcov/; cd obj-gcov
$ ../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage" #make sure no error
# if want to get cfg then use `../configure --disable-nls CFLAGS="-g -fprofile-arcs -ftest-coverage -fdump-tree-cfg-lineno"`
$ make  #make sure  no error
$ cd src  #obj-gcov/src dir contains binary programs
$ rm -rf *.gcda

# Test to see that it works
$ ./uname #this creates gcov data file uname.gcda -- *make sure that it does*, if not then it doesn't work !
$ cd ../../src  (src dir containing src)
$ gcov uname.c -o ../obj-gcov/src  #reads the echo.gcda in obj-gcov and generates human readable format
File '../src/uname.c'
Lines executed:37.50% of 88
Creating 'uname.c.gcov'

File '../src/system.h'
Lines executed:0.00% of 10
Creating 'system.h.gcov'

#you should see some lines like above saying "lines executed ... X% of Y
#iGen uses the generated uname.c.gcov and system.h.gcov files for coverage.    
```

Finally, edit `$IGEN/src/igen_settings.py` so that `coreutils_dir` points to `~/igen_exps/mycoreutils` directory. 
*NOTE*: if you move the above directories (e.g., `mycoreutils`) to different locations,  it's best to recompile everything again.

### Run
If everything is done correctly, we can now run iGen for `coreutils` commands (supported commands include `uname, cat, cp, date, hostname, id, join, ln, ls, mv, sort`)

```
# Generate interactions for the `uname` command 
# notice that iGen already contains the necessary runscripts and dom files for `coreutils`
$ python -O $IGEN/src/igen.py uname
...
```

## Perl Power Tools

We also experiment with the [Perl Power Tools implementation](http://search.cpan.org/dist/ppt) (ppt) of several `coreutils` programs. 

### Setup
First, download [`ppt 0.14`](http://search.cpan.org/CPAN/authors/id/C/CW/CWEST/ppt-0.14.tar.gz), then 

```
$ cd ~/igen_exps; mkdir ppt; cd ppt; mkdir archive; tar xf PATH/TO/ppt-0.14.tar.gz; cd ppt-0.14
```


To obtain coverage, we use `Devel::Cover` and `HTML::TableExtract` modules, which can be installed on a Debian-based system using
```
$sudo apt-get install libdevel-cover-perl libhtml-tableextract-perl 
```

Now, do some tests to make sure everything works

```
#!shell
$ perl -MDevel::Cover bin/date
Devel::Cover 1.21: Collecting coverage data for branch, condition, statement, subroutine and time.
...
Devel::Cover: Writing coverage database to ...
----------------------------------- ------ ------ ------ ------ ------ ------
File                                  stmt   bran   cond    sub   time  total
----------------------------------- ------ ------ ------ ------ ------ ------
bin/date                              78.7   50.0   33.3   88.8  100.0   65.5
Total                                 78.7   50.0   33.3   88.8  100.0   65.5
----------------------------------- ------ ------ ------ ------ ------ ------

```

If you see a coverage report similar to above one, `ppt` and `Devel::Cover` work fine.

Finally, edit `$IGEN/scripts/pptCoverageHelper.pl` so that `$SUT_DIR` points to the `ppt` directory. Then, test `pptCoverageHelper.pl` script as follows:

```
#!shell
$ cd $IGEN/scripts; ./pptCoverageHelper.pl "@@@date " 
```
It will output the name of coverage file if everything works fine.

Finally, domain files for PPT programs are located under `$IGEN/benchmarks/doms` and igen already know about them. There is no need to change these domain files.

### Run
Now to run igen on `PPT` programs, you need to use `-do_perl` option. For example `PPT uname` can be run as follows:

```
$ python -O $IGEN/src/igen.py uname -do_perl
```

## Ack and Cloc ##

## Setup
[`Ack`](http://search.cpan.org/dist/ack/ack) is a grep-like search tool for developers and [`cloc`](http://cloc.sourceforge.net/) is line of code counter. They both are written in `Perl`. To get their coverage, use the instructions above for `Powertools` to install `Devel::Cover`.

## Run
To run `iGen` on either of them you need to use `-run_script` and `-dom_file` options (as in httpd).
Run scripts for them are located under scripts directory, and the domain files are under benchmarks/doms directory.

For `ack`:
```
#!shell
$ python -O $IGEN/igen.py "ack" -run_script /path/to/run_script -dom_file /path/to/domain_file
```

And for `cloc`:
```
$ python -O $IGEN/igen.py "cloc" -run_script /path/to/run_script -dom_file /path/to/domain_file
```
Todo: replace above /path/to/ with real examples

## Ocaml Apps

## Haskell Apps

## Python Apps


## VSFTPD and NGIRCD

## Apache HTTPD
### Setup
Apache Httpd is a webserver. In our experiments we used [`httpd 2.2.29`](http://archive.apache.org/dist/httpd/). 

### Run
To run igen for httpd, `-run_script` and `-dom_file` options are needed to be used as follows:

```
#!shell
$ python -O $IGEN/src/igen.py "httpd" -run_script /path/to/run_script -dom_file /path/to/domain_file
```

`run_script` for httpd is under `scripts` directory: `run_httpd.pl` and domain file is under` benchmarks/doms/dom_httpd` directory: `config_space_model.dom`.

In `run_httpd.pl`, the variable `$SUT_DIR` should be updated accordingly.

