#!/bin/csh

echo "Setting environment variables for matcaffe on muir..."
setenv LD_LIBRARY_PATH /opt/intel/mkl/lib/intel64
setenv LD_PRELOAD /usr/lib/x86_64-linux-gnu/libstdc++.so.6

