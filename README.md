# HeDopant: Helium Droplet Doping Simulation

## Overview

HeDopant is a simulation tool for modeling the evolution of dopant cluster-size distributions as helium nanodroplets traverse a doping cell. The simulation integrates the dopant pick-up process using a fourth-order Runge–Kutta (RK4) solver and supports both CPU and GPU execution.

## Features

- **Flexible Backend**: Supports both CPU and GPU (via CUDA) for RK4 solver.
- **Configuration**: Allow configuration through a simple YAML file, but allows to override the configuration options using command line. Sample YAML file is provided in data folder.
- **Dopant Support**: Dopant species are configured using yaml files provided in data folder. Just add your own dopant yaml file and specify it in config.yaml.
- **Output**: Final distribution of dopant size would be writted on computer. By configuring flag `trajectory` to `true`, distribution would be saved at each RK4 step.

## Building

This project uses Meson Build System.

### Prerequisited

- C++23 compiler (e.g., `g++`)  
- CUDA (for GPU support)
- [Meson](https://mesonbuild.com/) build system  
- Dependencies: `eigen`, `yaml-cpp`, `arrow`

```bash
meson setup build
meson compile -C build
```

## Usage

1. Edit `config.yaml` to specify your simulation parameters:

```yaml
name: "Helium Droplet"
number_of_atoms: [10000, 19000]
type: "LOGNORMAL" # options are: "NONE", "LOGNORMAL" or "EXPONENTIAL"
dopant: "krypton"
doping_cell: 0.018 # m
dopant_pressure: 1E-4
max_dopant: 100
rk_steps: 100000
datadir: "../data"
output: "he_drop_krypton"
trajectory: false
```

2. Run the program

```bash
./build/HeDopant --config config.yaml
```
- --config <file> : Specify an alternate configuration file
- --help : Show usage information

You can override any configuration option at command line.

```bash
./build/HeDopant --config config.yaml --dopant water --dopant_pressure 1.5
```

## Output

Prefix of output files can be controlled either setting option `output` in config file or on the command line option `--output <prefix>`.

The evolved distribution would be saved as `<prefix>_<pressure>_mbar_output.txt`.

Few other output files would also be genereted to make sure things are working smoothly.

- `<prefix>_vcluster_ebe.txt`: Velocity and binding energy of He droplet for certain size. It is interpolated using `droplet.txt` file in `data` directory.
- `<prefix>_<N>_size_distribution.txt`: Size distribution of the droplet with mean size `<N>`.
- `<prefix>_evap.txt`: For each initial size in droplet distribution, we will have `alpha` parameter and number of remaining He atoms after absorption of `k` dopants.
- `<prefix>_final_y.txt`: Final dopant distribution without taking droplet distribution into account. It would be same as `output.txt` file if you select distribution type `NONE`.
- `<prefix>_trajectory.arrow`: Binary file having dopant distribution at each RK4 step. To read this file in a pandas dataframe a simple python script is provided in `utils` directory.
