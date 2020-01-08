#!/bin/bash

# strace collects system calls.  -f traces any child processes.
strace -f $@ |& grep openat | grep O_CREAT | grep -o "\".*\"" | tr -d "\""
# TODO: need to properly parse strace output, e.g., files with double-quotes in the names
