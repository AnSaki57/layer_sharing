.PHONY: all clean run

all:
	@echo "Creating build directory and running CMake..."
	mkdir -p build
	cd build && cmake .. && make -j4
	@echo "Build successful! Executable is located at build/fedFaultRandLayers"

clean:
	@echo "Cleaning build directory..."
	rm -rf build
	@echo "Clean successful!"

run: all
	@echo "Running fedFaultRandLayers with model 1 (SimpleCNN)..."
	./build/fedFaultRandLayers --model 1
