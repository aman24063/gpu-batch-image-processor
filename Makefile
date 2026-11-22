# Build file for the GPU batch image processor.
#   make build   - compile src/ into bin/image_batch_processor
#   make run     - run on the images in data/ (see run.sh for the full flow)
#   make clean   - remove the binary and generated output

NVCC ?= nvcc
NVCC_FLAGS ?= -O2 --std c++17 -Isrc -Wno-deprecated-gpu-targets
TARGET := bin/image_batch_processor
SOURCES := src/image_batch_processor.cu
HEADERS := src/image_io.h
ARGS ?= --input data --output output --streams 4 --samples 6

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET)

run: build
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET)
	rm -rf output
