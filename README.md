# Layer Sharing - Federated Learning with Fault Injection

## Usage

### Python Version
Run the python script specifying the model number:
```bash
python3 fedFault_randomLayers_models.py --model [model_number]
```
*Note: `model_number` can be any integer from `1` to `6`:*
1. SimpleCNN (small/original)
2. SimpleCNN10 (deep 10-layer CNN)
3. VGG11-BN
4. VGG13-BN
5. VGG16-BN
6. ResNet-20-CIFAR

### C++ Version
The C++ equivalent (`fedFaultRandLayers.cpp`) runs using LibTorch and requires CMake to build. A `Makefile` is provided for convenience.

**Compilation:**
To build the executable, simply run:
```bash
make
```
*This will automatically configure CMake inside a `build` directory and compile the binary.*

**Cleaning the Build:**
```bash
make clean
```

**Running the Executable:**
You can run the compiled binary directly via:
```bash
./build/fedFaultRandLayers --model [model_number]
```

Or you can use the make run target (which defaults to model 1):
```bash
make run
```