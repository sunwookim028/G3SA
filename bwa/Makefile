# G3SA — GPU-accelerated BWA-MEM. Build: make all [CU_ARCH=sm_XX]
# Match CU_ARCH to your GPU: RTX A6000 / RTX 3090 = sm_86, RTX 4090 = sm_89,
# GTX 1080 Ti / Titan Xp = sm_61. Wrong arch fails at runtime with
# "named symbol not found".

CUDA_HOME ?= /usr/local/cuda
CU_ARCH   ?= sm_86
CU_COMPUTE_ARCH = $(subst sm,compute,$(CU_ARCH))

INCLUDES = -Isrc -Isrc/seeding -Isrc/chaining -Isrc/extension \
           -Isrc/finalize -Isrc/mem -Isrc/host -Isrc/types \
           -Isrc/legacy -Isrc/third_party \
           -Iext/zlib-1.3.1 -Iext/safestringlib/include -Iext/bwa-mem2/include

CXX = g++
CXXFLAGS       = -Wall -std=c++17 -pthread $(INCLUDES) -MMD -MP -O3
CXXFLAGS_DEBUG = -Wall -std=c++17 -pthread $(INCLUDES) -MMD -MP -O0 -g
CC = gcc
CFLAGS       = -Wall -Wno-unused-function -O3 $(INCLUDES) -lm -MMD -MP
CFLAGS_DEBUG = -Wall -Wno-unused-function -O0 -g $(INCLUDES) -lm -MMD -MP

NVCC = nvcc
_NVFLAGS  = -ccbin $(CXX) --gpu-architecture=$(CU_COMPUTE_ARCH) --gpu-code=$(CU_ARCH) \
            --default-stream per-thread --std=c++17 $(INCLUDES) -MMD -MP -dc
__NVFLAGS = -ccbin $(CXX) --gpu-architecture=$(CU_COMPUTE_ARCH) --gpu-code=$(CU_ARCH) \
            --default-stream per-thread --std=c++17 $(INCLUDES) -Xptxas -O4 -Xcompiler -O4
NVFLAGS       = $(_NVFLAGS) -Xptxas -O4 -Xcompiler -O4 \
                -DG3_SM_OCC_CACHE
NVFLAGS_DEBUG = $(_NVFLAGS) -G -g
NVFLAGS_EXTRA ?=

LINKFLAGS = -Lext/zlib-1.3.1 -Lext/safestringlib -L$(CUDA_HOME)/lib64 -Lext/bwa-mem2 -pthread
LIBS = -lz -lcudart -lcudadevrt -lsafestring -lreadindexele -lbntseq -ldl

HASHER = hasher.exe
EXE    = g3.exe

CPP_OBJECTS_EXE = src/host/load_kmer_index.o \
                  src/host/hash_kmer.o \
                  src/host/timer.o \
                  src/host/fastmap.o \
                  src/host/main.o \
                  src/host/bwtindex.o \
                  src/host/fmi_wrapper.o \
                  src/host/fmi_search.o \
                  src/legacy/utils.o \
                  src/host/memcpy_bwamem.o \
                  src/host/pipeline.o
CPP_OBJECTS_HASHER = src/host/build_index.o \
                     src/host/hash_kmer.o \
                     src/host/load_kmer_index.o
C_OBJECTS_EXE = src/legacy/malloc_wrap.o \
                src/legacy/bwt.o

CU_SOURCES = $(wildcard src/seeding/*.cu) $(wildcard src/chaining/*.cu) \
             $(wildcard src/extension/*.cu) $(wildcard src/finalize/*.cu) \
             $(wildcard src/mem/*.cu) $(wildcard src/host/*.cu) \
             $(wildcard src/types/*.cu)
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CU_OBJECTS_LINKER = src/gpu_link.o

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<
%.o: %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_EXTRA) -c -o $@ $<

all: $(EXE)

$(EXE): $(CPP_OBJECTS_EXE) $(C_OBJECTS_EXE) $(CU_OBJECTS_LINKER) $(CU_OBJECTS)
	$(CXX) $(LINKFLAGS) -L. -o $@ $^ $(LIBS)

$(CU_OBJECTS_LINKER): $(CU_OBJECTS)
	$(NVCC) $(__NVFLAGS) -dlink $^ --output-file $@

$(HASHER): $(CPP_OBJECTS_HASHER)
	$(CXX) $(LINKFLAGS) -o $@ $^ -lz

debug: CXXFLAGS = $(CXXFLAGS_DEBUG)
debug: CFLAGS = $(CFLAGS_DEBUG)
debug: NVFLAGS = $(NVFLAGS_DEBUG)
debug: all

clean:
	rm -f $(CU_OBJECTS_LINKER) $(CU_OBJECTS) $(HASHER) $(EXE)
	rm -f src/*/*.o src/*/*.d src/*.o src/*.d

.PHONY: all clean debug
