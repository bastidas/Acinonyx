# Automata

A Python application for simulating and visualizing mechanical linkage systems and automata.

This project provides tools for creating, simulating, and visualizing mechanical linkage systems. It includes:



- **Mechanical Link Simulation**: Define and simulate multi-link mechanical systems
- **Visualization Tools**: Generate static plots and animations of linkage motion
- **NetworkX Integration**: Graph-based representation of linkage systems

## Requirements

- Python >= 3.11
- See `pyproject.toml` for complete dependency list

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd automata
```

### 2. Install in Development Mode

```bash
pip install -e .
```

This installs the package in editable mode, making all modules (`configs`, `structs`, `viz_tools`, etc.) available for import.

### 3. Verify Installation

```bash
python structs/basic.py
```

This should run without errors and generate visualization files in the `user/` directory.

## Usage

### Basic Example

The main example demonstrates a 3-link mechanical system:

```bash
python structs/basic.py
```

This will:
- Create a 3-link mechanical system
- Run the simulation
- Generate visualization plots
- Save results to the `user/` directory





