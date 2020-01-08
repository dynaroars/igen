To setup a Kbuild directory do the following.  The example is from `kbuild_example/`.

1. Setup your domain.txt file

        A y,n
        B y,n
        C y,n
        E y,n
        F y,n
        G y,n

2. Generate a `.kmax` file

        /path/to/kmax/kconfig/check_dep --dimacs > kconfig.kmax
        
    This is used for the compare script
    
3. Create a run script

        #!/bin/bash

        cd $(dirname ${0})
        python3 ../../utils/run.py ./ "$1" "$2"


## Usage Example

This is a valid config that will yield a list of the .o files created during the build:

    ./run.sh "A,B,C,E,F,G" "y,y,n,y,n,y"
    
This is an invalid config that will yield nothing:

    ./run.sh "A,B,C,E,F,G" "y,y,n,y,y,n"

You can use the general-purpose run-script like this:

    python3 ../utils/run.py kbuild_test/ "A,B,C,E,F,G" "y,n,n,y,y,n"

It assumes you have a kconfig.kmax file in the directory.
