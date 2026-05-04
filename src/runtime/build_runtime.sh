#!/bin/bash
cd "$(dirname "$0")"
gcc -shared -fPIC -o libfortran_runtime.so fortran_runtime.c -lpthread
echo "Runtime library built: libfortran_runtime.so"
