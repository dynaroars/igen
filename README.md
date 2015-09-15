# README #


iGen (Interaction Generator) is a dynamic analysis tool that discovers interactions among configurable options from traces. iGen employs a stochastic search method to iteratively search for a small and "good" set of configurations to find desired interactions.  Igen performs trace partitioning on both positive and negative traces to generate expressive interactions, e.g., combination of both conjunctive and disjunctive formulae. Preliminary results show that Intgen is highly efficient and effective on a set of benchmarks consisting of highly-configurable software, e.g., apache httpd, mysql.

## Setup ##

The source code of iGen is released under the BSD license and can be downloaded using the command "hg clone https://nguyenthanhvuh@bitbucket.org/nguyenthanhvuh/config/"

iGen uses Python. Some operations, e.g., verifying candidate interactions, require a recent SMT solver.  

iGen has been tested using the following setup:

* Debian Linux 7 (Wheezy)
* Python 2.7.x
* Microsoft Z3 SMT solver 4.x


First, setup Z3 using its own build instruction (make sure to do "make install" at the end).
Now you should be able to do something like "import z3" in a python interpreter
Finally, setup the SAGE environment as follows.

### Experimentations ###
Download the source code and read the README file for information to reproduce the experimental results.
