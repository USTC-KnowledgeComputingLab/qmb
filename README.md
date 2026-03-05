# Quantum Many-Body Problem Kit (qmp-kit)

The quantum many-body problem kit (`qmp-kit`) is a powerful tool designed to solve quantum-many-body problems especially for strongly correlated systems. This project includes our work on [Hamiltonian-Guided Autoregressive Selected-Configuration Interaction Achieves Chemical Accuracy in Strongly Correlated Systems][haar-url].

## About The Project

This repository hosts a [Python][python-url] package named `qmp-kit`, dedicated to solving the quantum-many-body problem.
It implements a suite of algorithms and interfaces with various model descriptors, such as the [OpenFermion][openfermion-url] format and FCIDUMP.
Additionally, `qmp` can efficiently utilize accelerators such as GPU(s) to enhance its performance.

## Getting Started

### Installation

The `qmp-kit` requires Python >= 3.12 and CUDA-compatible environment for GPU acceleration.

You can install the package via pip:
```bash
pip install qmp-kit
```

Or run it directly using `uvx` (recommended):
```bash
uvx qmp-kit
```

### Usage

The application is now configured via a `config.yaml` file located in the working directory. It uses [Hydra][hydra-url] for configuration management.

To run the application:
```bash
# Using the installed script
qmp

# Or using uvx
uvx qmp-kit
```

#### Configuration Structure

A typical `config.yaml` includes several sections: `common`, `model`, `network`, `optimizer`, and `action`.

```yaml
# Example config.yaml
model:
  name: fcidump
  params:
    model_path: /path/to/molecule.fcidump
    ref_energy: -7.44606892

network:
  name: mlp/u1u1
  params:
    hidden: [512, 512]

optimizer:
  name: Adam
  params:
    lr: 0.001

common:
  random_seed: 2333
  device: cuda
  dtype: bfloat16

action:
  name: haar
  params:
    sampling_count_from_neural_network: 1024
    krylov_iteration: 32
```

#### Configuration Options

##### Common Settings (`common`)
| Parameter | Description | Default |
|-----------|-------------|---------|
| `device` | Computing device (`cuda`, `cpu`, `cuda:0`, etc.) | `cuda:0` |
| `dtype` | Data type (`bfloat16`, `float16`, `float32`, `float64`) | `None` |
| `random_seed` | Manual random seed for reproducibility | `None` |
| `checkpoint_interval` | Interval (in steps) to save checkpoints | `5` |
| `parent_path` | Path to load a checkpoint from | `None` |
| `max_absolute_step` | Maximum absolute step for the process | `None` |
| `max_relative_step` | Maximum relative step for the process | `None` |

##### Model Settings (`model`)
- **fcidump**: Interface for FCIDUMP files.
  - `model_path`: Path to the FCIDUMP file.
  - `ref_energy`: Reference energy (optional).
- **hubbard**: 2D Hubbard model.
  - `m`, `n`: Lattice dimensions.
  - `t`, `u`: Model coefficients.
  - `electron_number`: Number of electrons.
- **ising**: 2D Ising-like model on a lattice.
  - `m`, `n`: Lattice dimensions.
  - `x`, `y`, `z`: Coefficients for the transverse and longitudinal fields.
  - `xh`, `yh`, `zh`: Coefficients for horizontal bond interactions (XX, YY, ZZ).
  - `xv`, `yv`, `zv`: Coefficients for vertical bond interactions.
  - `xd`, `yd`, `zd`: Coefficients for diagonal bond interactions.
  - `xa`, `ya`, `za`: Coefficients for anti-diagonal bond interactions.

##### Network Settings (`network`)
The available network architectures and their naming conventions depend on the chosen model, as they implement specific symmetry and conservation laws. **Explicit naming is recommended** to clarify the constraints being used (e.g., total electron number `u1` or spin-resolved `u1u1`).

For models like **fcidump** and **hubbard**, recommended options include:
- **mlp/u1u1**: Multi-Layer Perceptron with $U(1) \times U(1)$ symmetry (conserves spin-up and spin-down electrons separately).
- **mlp/u1**: MLP with $U(1)$ symmetry (conserves total electron number).
- **transformers/u1u1**: Transformer architecture with $U(1) \times U(1)$ symmetry.
- **transformers/u1**: Transformer architecture with $U(1)$ symmetry.

Parameters for these networks:
- **mlp** variants:
  - `hidden`: Tuple of hidden layer widths (e.g., `[512, 512]`).
- **transformers** variants:
  - `embedding_dim`, `heads_num`, `depth`, etc.

##### Action Settings (`action`)
The `action` section determines the algorithm to run. The primary algorithms are **haar** and **vmc**. Each action has its own set of parameters defined in its corresponding configuration class.

Example for `haar` action:
```yaml
action:
  name: haar
  params:
    sampling_count_from_neural_network: 1024
    krylov_iteration: 32
    # ... other haar specific parameters
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is distributed under the GPLv3 License. See [LICENSE.md](LICENSE.md) for more information.

[python-url]: https://www.python.org/
[openfermion-url]: https://quantumai.google/openfermion
[cuda-url]: https://docs.nvidia.com/cuda/
[anaconda-url]: https://www.anaconda.com/
[miniconda-url]: https://docs.anaconda.com/miniconda/
[venv-url]: https://docs.python.org/3/library/venv.html
[pyenv-url]: https://github.com/pyenv/pyenv
[our-pypi-url]: https://pypi.org/project/qmp-kit/
[pytorch-install-url]: https://pytorch.org/get-started/locally/
[hydra-url]: https://hydra.cc/
[haar-url]: https://pubs.acs.org/doi/10.1021/acs.jctc.5c01415
