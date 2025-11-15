# Number of Dopant Calculator

This program simulates the **doping process of helium droplets** using a **Runge-Kutta numerical solver**.  
It calculates the distribution of dopant atoms based on user-defined parameters.

---

## Features

- Saves the droplet distribution at each step to an **Arrow IPC stream** for efficient post-processing  
- Configurable via a YAML file  

---

## Requirements

- C++23 compiler (e.g., `g++`)  
- [Meson](https://mesonbuild.com/) build system  
- Dependencies: `gsl`, `yaml-cpp`, `arrow`, `parquet`  

---

## Build

```bash
meson setup build
meson compile -C build
```

## Usage

1. Edit `config.yaml` to specify your simulation parameters:

```yaml
number_of_atoms: 100
max_dopant: 50
doping_cell: 1.0 # m
rk_steps: 100000
dopant: "water"
doping_pressure: 1.0E-5 # mbar
```

2. Run the program

```bash
./build/HeDopant --config config.yaml
```
- --config <file> : Specify an alternate configuration file
- --help : Show usage information

The program writes the droplet evolution as function of distance to trajectory.arrow (Arrow IPC stream), which can be analyzed in Python, Julia, or other languages supporting Arrow.
Final dopant distribution is written to standard output.

## Python example

```python
import pyarrow.ipc as ipc
import pyarrow as pa

with pa.memory_map("trajectory.arrow", "r") as mmap:
    reader = ipc.RecordBatchStreamReader(mmap)
    table = reader.read_all()

print(table)
```
