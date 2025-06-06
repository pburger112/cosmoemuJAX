<div align="center"><img src="docs/images/CosmoEmuJAX.png" width="500" height="400"> </div>

![](https://img.shields.io/badge/Python-181717?style=plastic&logo=python)
![](https://img.shields.io/badge/JAX-181717?style=plastic&logo=jax)
![](https://img.shields.io/badge/Author-Pierre%20Burger%20-181717?style=plastic)


This package is oriented toward the CosmoPower and CosmoPower JAX emulators. However, it does train emulators similar to CosmoPower, but entirely in JAX. Therefore, it is independent of TensorFlow and can be easily used for gradient-dependent inferences. 

---

## 🛠️ Installation

We recommend using a dedicated `conda` environment for clean dependency management.

### 1. Clone the Repository

```bash
git clone https://github.com/pburger112/cosmoemuJAX.git
cd cosmoemuJAX
```

### 2. Create and Activate a Conda Environment

```bash
conda create -n cosmoemu_JAX_env python=3.10 
conda activate cosmoemu_JAX_env
```

### 3. Install the Package

Install the package and its dependencies using `pip`:

```bash
pip install . ".[cpu]" 
```

If you need editable/development mode:

```bash
pip install -e . ".[cpu]"
```

If you want use the Mac M1,M2,M3 GPU install like:

```bash
pip install . ".[gpu]" 
```

If you need editable/development mode:

```bash
pip install -e . ".[gpu]"
```

---

## 📦 Dependencies

The core dependencies for CPU (automatically installed) include:

* numpy
* jax
* scipy
* optax
* gdown
* matplotlib
* tqdm
* getdist


The adjusted core dependencies for GPU (automatically installed) include:

* numpy<2.0
* jax==0.4.20
* jax-metal==0.0.5
* scipy==1.11.4
* optax==0.1.7

---

## 📂 Structure

* `cosmoemu_jax/`: Python file with the JAX emulator
* `outputs/`: Folder where data is downloaded and emulators will be saved
* `cosmoemu_jax_3times2pt.ipynb`: Notebook to demonstrate emulator use
*  `HMC_example.ipynb`: Notebook to demonstrate emulator use for an HMC inference


---

## 📬 Contact

For questions, please contact:
Pierre Burger – [pierre.burger@uwaterloo.ca](mailto:pierre.burger@uwaterloo.ca)
