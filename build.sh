#!/bin/bash

POCL_BASE="/home/$USER/Documents/pocl"
LIBREALSENSE_BASE="/home/$USER/Documents/librealsense"
#CXX_FLAGS="-pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -fPIE -std=gnu++11"
#C_FLAGS="-pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -fPIE"
#L_FLAGS="-pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -rdynamic"
CXX_FLAGS="-g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -fPIE -std=gnu++11"
C_FLAGS="-g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -fPIE"
L_FLAGS="-g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -rdynamic"
CUDA_FLAGS="-m64 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_32,code=sm_32 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_72,code=sm_72 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75"
DEFINES="-DBUILD_EASYLOGGINGPP -DBUILD_SHARED_LIBS -DCHECK_FOR_UPDATES -DCOM_MULTITHREADED -DCURL_STATICLIB -DEASYLOGGINGPP_ASYNC -DELPP_NO_DEFAULT_LOG_FILE -DELPP_THREAD_SAFE -DHWM_OVER_XU -DRS2_USE_V4L2_BACKEND -DSQLITE_HAVE_ISNAN -DUNICODE"
INCLUDES="-I$LIBREALSENSE_BASE/build -I$LIBREALSENSE_BASE/include -I$LIBREALSENSE_BASE/third-party/glfw/include  -I/usr/local/cuda/include"
LIBS="$POCL_BASE/build/lib/poclu/libpoclu.a $LIBREALSENSE_BASE/build/librealsense2.so $LIBREALSENSE_BASE/build/third-party/glfw/src/libglfw3.a -lOpenCL -lGL -lGLU -lrt -ldl -lm -lX11"
OBJS="RealFusion.cpp.o IMUFusion.cpp.o LinearAlgebra.cpp.o Compute.cpp.o Exception.cpp.o FusionAhrs.c.o FusionBias.c.o FusionCompass.c.o vecadd.cu.o"

C_COMPILER=cc
CXX_COMPILER=c++
CUDA_COMPILER="nvcc -ccbin c++"

$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o main.cpp.o -c main.cpp
$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o Tests.cpp.o -c Tests.cpp
$C_COMPILER $DEFINES $INCLUDES $C_FLAGS -o FusionAhrs.c.o -c Fusion/FusionAhrs.c
$C_COMPILER $DEFINES $INCLUDES $C_FLAGS -o FusionBias.c.o -c Fusion/FusionBias.c
$C_COMPILER $DEFINES $INCLUDES $C_FLAGS -o FusionCompass.c.o -c Fusion/FusionCompass.c
$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o IMUFusion.cpp.o -c IMUFusion.cpp
$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o RealFusion.cpp.o -c RealFusion.cpp
$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o Exception.cpp.o -c Exception.cpp
$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o LinearAlgebra.cpp.o -c LinearAlgebra.cpp
$CXX_COMPILER $DEFINES $INCLUDES $CXX_FLAGS -o Compute.cpp.o -c Compute.cpp
$CUDA_COMPILER $CUDA_FLAGS -o vecadd.cu.o -c vecadd.cu

# $CXX_COMPILER $L_FLAGS main.cpp.o $OBJS -o FuseToEuler -Wl,-rpath,"/home/$USER/Documents/librealsense/build: $LIBREALSENSE_BASE" $LIBS
# $CXX_COMPILER $L_FLAGS Tests.cpp.o $OBJS -o Tests -Wl,-rpath,"/home/$USER/Documents/librealsense/build: $LIBREALSENSE_BASE" $LIBS

$CUDA_COMPILER $CUDA_FLAGS Tests.cpp.o $OBJS -o Tests $LIBS