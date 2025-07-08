# IGHAStar

## TODO: add a nice description of what the planner is about

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sidtalia/IGHAStar.git
cd ighastar
```

### 2. Set Up Conda Environment
```bash
conda env create -f ighastar.yml
conda activate ighastar
```

The project uses dynamic compilation, so C++ extensions will be compiled automatically when you run the examples.

## Usage

### Running the Standalone Example

The main example script supports different planning modes through configuration files:

```bash
# Use default kinematic configuration
python examples/standalone_example.py

# Specify a different configuration file
python examples/standalone_example.py --config examples/Configs/kinodynamic_example.yml

# Use simple planning configuration
python examples/standalone_example.py --config examples/Configs/simple_example.yml

# Use a specific test case
python examples/standalone_example.py --config examples/Configs/kinematic_example.yml --test-case case2
```

### Configuration

Configuration files are located in `examples/Configs/` and include vehicle parameters, planning parameters, and environment settings.

For detailed configuration and usage instructions, see [examples/EXAMPLE_USAGE.md](examples/EXAMPLE_USAGE.md).

### Compute Environment Selection

The system automatically detects CUDA availability:
- If CUDA is available: Uses GPU-accelerated planning
- If CUDA is not available: Falls back to CPU implementation

## Project Structure

```
ighastar/
├── src/                    # Core C++/CUDA source files
│   ├── *.cu              # CUDA implementations
│   ├── *_cpu.cpp         # CPU implementations  
│   ├── *.h               # Header files
│   └── *.cpp             # C++ implementations
├── examples/              # Example scripts and configurations
│   ├── Configs/          # Configuration files
│   ├── Maps/             # Map files
│   └── *.py              # Example scripts
├── scripts/              # Utility scripts
├── ighastar.yml         # Conda environment file
└── README.md            # This file
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both CPU and GPU environments
5. Submit a pull request