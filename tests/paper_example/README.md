Try out the various configurations:

    make clean; CONFIG_B=  CONFIG_A=
    make clean; CONFIG_B=  CONFIG_A=y
    make clean; CONFIG_B=y CONFIG_A=
    make clean; CONFIG_B=y CONFIG_A=y


Try collecting created files

    make clean; ../../utils/collect.sh make CONFIG_B=y CONFIG_A=y
