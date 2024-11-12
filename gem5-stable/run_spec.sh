./build/ARM/gem5.fast configs/example/arm/starter_se_spec.py --cpu=o3 --maxinsts=20'000'000\
    "/worksapce/TAO/workload/cpu2006/spec2006/benchspec/CPU2006/400.perlbench/build/build_base_gcc43-64bit.0000/perlbench
    /worksapce/TAO/workload/cpu2006/spec2006/benchspec/CPU2006/401.bzip2/data/test/input/control"


export M5_PATH=/worksapce/TAO/m5_binaries
./build/ARM/gem5.fast -d m5out_fs \
    configs/example/arm/starter_fs.py \
    --disk-image=$M5_PATH/ubuntu-18.04-arm64-docker-sxl.img\ 
    --script=./m5out_fs/script.rsC 

./build/ARM/gem5.fast -d m5out_fs/ \
    configs/example/arm/starter_fs.py \
    --cpu=o3 \
    --restore=./m5out_fs/cpt.220266758750/ \
    --disk-image=$M5_PATH/ubuntu-18.04-arm64-docker-sxl.img \
    --maxinsts=250'000'000
