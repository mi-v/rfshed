NVCC=nvcc
#NVCCFLAGS=-O3 -arch=sm_52 -m64 --ptxas-options=-v --compiler-options '-fPIC' --shared
#NVCCFLAGS=-m64 --ptxas-options=-v --compiler-options '-fPIC' --shared
NVCCFLAGS=-m64 --resource-usage --compiler-options '-fPIC' --shared
VPATH=cuhgt cuvshed cuinit

build: cuda rfshed

cuda: libcuhgt.so libcuvshed.so libcuinit.so

lib%.so: %.cu %.h
	$(NVCC) $(NVCCFLAGS) -o $@ $<
	chmod -x $@

rfshed: $(shell find -type f -name '*.go')
	GOARCH=amd64 go build -ldflags="-s -w"

run: build
	LD_LIBRARY_PATH=. ./rfshed
