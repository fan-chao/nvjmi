#makefile for libnvjmi.so

CC      = gcc
CPP     = g++
AR = ar
RM      = rm -f
NVCC := /usr/local/cuda/bin/nvcc

#Define the resource compiler.

## debug flag  
DBG_ENABLE := 1

OS = $(shell uname)

## source file path  
SRC_PATH := ./ ./common/ ./converter/ ./log/ ./utils/
SRC_PATH_EXT := 

## target file name  
TARGET     := nvjmi

#############################################
# Gencode arguments
SMS ?= 60 61 62 70 72 75

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

## get all source files  
#$(warning  $(SRC_FILES_SHARED))
SRCS := $(foreach spath, $(SRC_PATH), $(wildcard $(spath)*.c)) $(foreach spath, $(SRC_PATH), $(wildcard $(spath)*.cpp)) $(foreach spath, $(SRC_PATH), $(wildcard $(spath)*.cu))
#$(warning  $(SRCS))
#$(wildcard $(SRC_PATH)*.c*) $(wildcard $(SRC_PATH)*.c) $(wildcard $(SRC_PATH_EXT)*.cpp)

## all .o based on all .c/.cpp
OBJS = $(SRCS:.c=.o)
OBJS := $(OBJS:.cpp=.o) 
OBJS := $(OBJS:.cu=.o) 

## macro define
DEFS := _FILE_OFFSET_BITS=64 $(if $(findstring D,$(ENCYPT)), _USE_ENCYPTION_DONGLE, ) $(if $(findstring T,$(ENCYPT)), _USE_ENCYPTION_TEMP, ) $(if $(findstring S,$(ENCYPT)), _USE_ENCYPTION_SOFT, ) 

#if freeimage is static-linked use this !
#DEFS += FREEIMAGE_LIB

## need libs, add at here  
LIBS := v4l2 nvbuf_utils tbb cudart
  
## used headers  file path  
INCLUDE_PATH := ./ ./include ./converter ./cuda_utils ./image ./log ./utils /usr/local/cuda/include

#$(warning $(INCLUDE_PATH))

## used include librarys file path  
LIBRARY_PATH := /usr/lib/aarch64-linux-gnu/tegra /usr/local/cuda/targets/aarch64-linux/lib
 
## debug for debug info, when use gdb to debug  
ifeq (1, ${DBG_ENABLE})   
CFLAGS += -D_DEBUG -g -DDEBUG=1 
else
CFLAGS += -O3 -DNDEBUG
endif

# for ENCYPT flags

ifeq ($(OS), Linux)
LIBS += dl $(if $(findstring D,$(ENCYPT)), RockeyARM, )
CFLAGS += -fPIC
BUILDTARGET = lib$(TARGET).so
LIBRARY_PATH += 
else
#do nothing
DEFS += _WINDOWS
LIBS +=  ws2_32 $(if $(findstring D,$(ENCYPT)), Dongle_d, )
LIBRARY_PATH += ../win_lib
BUILDTARGET = $(TARGET).dll
LDFLAGS += $(WIN32_LDFLAGS)
endif

CFLAGS += -march=armv8.2-a -pipe $(foreach m, $(DEFS), -D$(m)) 
  
## get all include path  
CFLAGS  += $(foreach dir, $(INCLUDE_PATH), -I$(dir))  

CXXFLAGS += $(CFLAGS) -std=c++11 

## get all library path  
LDFLAGS += -Wl,--rpath-link=./ -pthread $(foreach lib, $(LIBRARY_PATH), -L$(lib))  
  
## get all librarys  
LDFLAGS += $(foreach lib, $(LIBS), -l$(lib)) 


RCFLAGS ?= -DNDEBUG


default: all

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CPP) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) -m64 -std=c++11 $(foreach dir, $(INCLUDE_PATH), -I$(dir)) --default-stream per-thread $(GENCODE_FLAGS) -Xcompiler -fPIC -Xcompiler -O3 -c $< -o $@
	#$(NVCC) -m64 -g -G -std=c++11 $(foreach dir, $(INCLUDE_PATH), -I$(dir)) --default-stream per-thread $(GENCODE_FLAGS) -Xcompiler -fPIC -c $< -o $@


all: $(OBJS)
	$(CPP) $(CXXFLAGS) -shared -o $(BUILDTARGET) $(OBJS) $(LDFLAGS)

clean:  
	$(RM) $(OBJS) $(BUILDTARGET)
