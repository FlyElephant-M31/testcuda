#
# Project makefile to be used with the NVIDIA CUDA SDK
#######################################################

# location of the CUDA SDK's C projects
ROOTDIR=../sdk/C/src

CUFILES		:= gpu_gausseli.cu
CCFILES		:= main.cpp cpu_gausseli.cpp
EXECUTABLE = gausseli

ROOTBINDIR = bin

CUDACCFLAGS= --ptxas-options=-v

# now include the CUDA SDK makefile
include $(ROOTDIR)/../common/common.mk

# clean up and set compile flags
CXXFLAGS += -Wall -O5

# with or without OMP
ifneq ($(NOOMP),y)	
CXXFLAGS += -DHAVE_OMP -fopenmp
LIB += -fopenmp
endif
