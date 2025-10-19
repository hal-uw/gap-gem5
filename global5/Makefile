BIN?=./bin
ROCM_PATH?=/opt/rocm
HIP_PATH=$(ROCM_PATH)
HIPCC?=$(HIP_PATH)/bin/hipcc
HIP_FLAGS?=--amdgpu-target=gfx906,gfx900 --save-temps

#icache icache1 icache2 icache3 icache4
#all: icache1 icache3 icache4 global
#	mkdir -p $(BIN)


#icache: icache2.cpp icache_kernels1.h icache_kernels2.h icache_kernels3.h icache_kernels4.h
#	mkdir -p $(BIN)
#	make -f Makefile-icache
#	cp icache $(BIN)

#icache1: icache.cpp icache_kernels1.h icache_kernels2.h icache_kernels3.h icache_kernels4.h
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ icache.cpp
#	cp icache1 $(BIN)

#icache2: icache2.cpp icache2_1.h icache2_2.h
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ icache2.cpp
#	cp icache2 $(BIN)

#icache3: icache3.cpp icache3_1.h icache3_2.h
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ icache3.cpp icache3_kernel.h
#	cp icache3 $(BIN)

#icache4: icache4.cpp icache4_L1.h
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ icache4.cpp
#	cp icache4 $(BIN)

#cmem:
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ cmem.cu

#texture2:
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ texture2.cu

#texture4:
#	mkdir -p $(BIN)
#	$(HIPCC) $(HIP_FLAGS) -o $@ texture4.cu

global: global.cpp
	mkdir -p $(BIN)
#	$(HIPCC) --gcc-install-dir=/usr/bin/../lib/gcc/x86_64-linux-gnu/13 $(HIP_FLAGS) -o $@ global.cpp
	$(HIPCC) $(HIP_FLAGS) -o $@ global.cpp
	cp global $(BIN)

clean:
	rm -f *~ $(BIN)/*
#	make clean -f Makefile-icache
#	make clean -f Makefile-icache1
#	make clean -f Makefile-icache2
#	make clean -f Makefile-icache3
#	make clean -f Makefile-icache4
	make clean -f Makefile-global
