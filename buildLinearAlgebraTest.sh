#!/bin/bash

LIBREALSENSE_BASE="/home/$USER/Documents/librealsense"
CXX_FLAGS="-pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -fPIE -std=gnu++11"
C_FLAGS="-pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -fPIE"
CXX_DEFINES="-DBUILD_EASYLOGGINGPP -DBUILD_SHARED_LIBS -DCHECK_FOR_UPDATES -DCOM_MULTITHREADED -DCURL_STATICLIB -DEASYLOGGINGPP_ASYNC -DELPP_NO_DEFAULT_LOG_FILE -DELPP_THREAD_SAFE -DHWM_OVER_XU -DRS2_USE_V4L2_BACKEND -DSQLITE_HAVE_ISNAN -DUNICODE"
CXX_INCLUDES="-I$LIBREALSENSE_BASE/build -I$LIBREALSENSE_BASE/include -I$LIBREALSENSE_BASE/third-party/glfw/include"
LXX_LIBS="$LIBREALSENSE_BASE"

/usr/bin/c++ $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS -o test_determinant.cpp.o -c test_determinant.cpp
/usr/bin/c++ $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS -o LinearAlgebra.cpp.o -c LinearAlgebra.cpp
/usr/bin/c++ -pedantic -g -Wno-missing-field-initializers -Wno-switch -Wno-multichar -Wsequence-point -Wformat -Wformat-security -mstrict-align -ftree-vectorize -pthread -rdynamic test_determinant.cpp.o LinearAlgebra.cpp.o -o test_determinant -Wl,-rpath,"/home/$USER/Documents/librealsense/build:" "$LIBREALSENSE_BASE/build/librealsense2.so.2.49.0" "$LIBREALSENSE_BASE/build/third-party/glfw/src/libglfw3.a" -lGL -lGLU -lrt -lm -ldl -lX11 
