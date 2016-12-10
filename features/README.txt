1. Set environment variables by:

source setenv_matcaffe.sh

2. Run script: (singleCompThread is optional, but probably won't make
a difference since most of the work is in caffe)

matlab -nodisplay -singleCompThread -r "run_batch(); exit(0);"
