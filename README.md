# Acinonyx 🐆

**Design tools for planar mechanisms, automata, and path synthesis optimization including a complete user interface for simulating and visualizing mechanisms.** 

An *automaton* is a mechanical device—often shaped like a human or animal—that moves as if by its own power, yet contains only gears, linkages, and levers. Acinonyx helps you design the multi-link mechanisms that bring these creations to life by targeting a trajectory and creatin a mechanism that can follow that trajectory.

> **The problem:** Given a planar path (or set of paths), construct a mechanism that traces those paths while satisfying design constraints.

This is an inverse problem in mechanism synthesis—and Acinonyx provides the simulation, visualization, and optimization tools to solve it.

---

*The name comes from Greek: **akinetos** (ἀκίνητος) meaning "unmoved" + **onyx** (ὄνυξ) meaning "claw"—the genus of the cheetah, swift and precise.*

---

![Example of UI](ui_example.png "Example of the UI") 

### Features

- **Mechanism Builder**: Interactive graph-based editor for designing multi-link planar mechanisms
- **Trajectory Simulation**: Compute and visualize joint paths using [pylinkage](https://github.com/HugoFara/pylinkage)
- **Path Optimization**: Fit mechanisms to target curves with global optimization
- **Modern Stack**: React frontend + FastAPI backend with real-time updates

## Requirements

- Python >= 3.11
- Node.js >= 18 (for frontend)
- Conda (recommended) or pip
- See `pyproject.toml` for complete dependency list

## Configuration

The application uses `configs/appconfig.py` for centralized configuration:

- **USER_DIR**: Location for saved graphs and user data (`user/` directory)
- **BACKEND_PORT**: `8021` - FastAPI backend server port
- **FRONTEND_PORT**: `5173` - Vite development server port
- **API URLs**: Automatically constructed from port configuration

All port settings are centralized in `appconfig.py` - modify this file if you need to change ports due to conflicts.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd acinonyx
```

### 2. Set Up Python Environment

We recommend using Python 3.11.x

#### Option A: Using Conda

```bash
conda env create -f environment.yml
conda activate acinonyx
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv acinonyx

# Activate environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

#### Option C: Using pyenv

Requires [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) plugin.

```bash
pyenv install 3.11.9
pyenv virtualenv 3.11.9 acinonyx
pyenv activate acinonyx

# Install dependencies
pip install -e .
```

### 4. Verify Installation

Run the test suite to ensure everything is working:

```bash
# Using pytest
pytest tests/ -v
```

All tests should pass without errors.

## Usage

### Quick Start

There are two scripts to start the development environment:

#### Option A: `run.sh` (Recommended for Unix/macOS)

The full-featured bash script with environment checks, colored output, and automatic browser opening:

```bash
# On Unix/macOS:
chmod +x run.sh
./run.sh
```

**Features:**
- Checks Python, Node.js, and npm versions
- Auto-detects and resolves port conflicts
- Installs missing dependencies if needed
- Opens browser automatically when ready
- Streams colored logs from both servers
- Use `./run.sh -q` or `./run.sh --quick` to skip dependency checks

#### Option B: `start_dev.py` (Cross-platform)

A simpler Python script that works on any platform:

```bash
python start_dev.py
```

**Features:**
- Cross-platform (works on Windows, macOS, Linux)

- Graceful shutdown with Ctrl+C

**Note:** This script assumes dependencies are already installed and ports are available.

---

Both scripts will:
1. Start the FastAPI backend server on `http://localhost:8021` (configured in `configs/appconfig.py`)
2. Start the Vite development server on `http://localhost:5173`
3. Handle graceful shutdown on Ctrl+C

### Manual Start

If you prefer to run the servers manually:

#### Terminal 1 - Backend Server

```bash
# Activate your environment first
conda activate acinonyx  # or: source venv/bin/activate

# Start backend
cd backend
python run_server.py
```

The API will be available at `http://localhost:8021` (port configured in `configs/appconfig.py`)

#### Terminal 2 - Frontend Server

```bash
cd frontend
npm run dev
```

The web interface will be available at `http://localhost:5173`

### API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8021/docs`
- ReDoc: `http://localhost:8021/redoc`
- Status Check: `http://localhost:8021/status`

**Note**: The backend port (8021) is configured in `configs/appconfig.py` and can be changed if needed.

## Project Structure

```
TODO
```


### Frontend structure
TODO


## Testing

### Run Tests

```bash
conda activate acinonyx
pytest tests/ -v
```

#### Run Tests with Coverage

```bash
pytest tests/ --cov=backend --cov=link --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`

## Available Endpoints

The backend provides the following REST API endpoints:

### Health Check
- `GET /` - Root endpoint, returns API running status
- `GET /status` - Health check endpoint

### Pylink Graph Management
- `POST /save-pylink-graph` - Save a pylink graph to user directory
- `GET /load-pylink-graph?filename=<name>` - Load a specific saved graph (omit filename for most recent)
- `GET /list-pylink-graphs` - List all saved graphs with metadata
- `POST /compute-pylink-trajectory` - Compute joint trajectories from pylink graph data

### Force Graph
- `GET /load-last-force-graph` - Load most recent force graph visualization

### Notes
- All endpoints return JSON responses with `status` field ('success' or 'error')
- Graphs are saved to `user/pygraphs/` directory (auto-created)
- Force graphs are saved to `user/force_graphs/` directory
- See API docs at `http://localhost:8021/docs` for detailed schemas

## Development

### Code Style

The project uses standard Python and TypeScript formatting:

```bash
# Format Python code
black backend/ configs/ link/ tests/

# Format TypeScript/React code
cd frontend
npm run format
```


## Troubleshooting

### Common Issues

**1. Module not found errors**
- Ensure your conda environment is activated:
  * for example if using pyenv `pyenv activate acinonyx`
  * * for example if using anaconda `conda activate acinonyx`
- Verify installation: `pip list | grep acinonyx`

**2. Frontend not loading**
- Check Node.js version: `node --version` (should be >= 18)
- Reinstall dependencies: `cd frontend && npm install`

**3. Backend connection refused**
- Verify backend is running: `curl http://localhost:8021/status`
- Check for port conflicts (default port is 8021, see `configs/appconfig.py`)
- Ensure CORS is properly configured

**4. Tests failing**
- Ensure pytest is installed: `pip install pytest pytest-cov`
- Check Python version: `python --version` (should be >= 3.11)

**5. Port conflicts**
- If ports 8021 or 5173 are already in use, edit `configs/appconfig.py` to change ports
- Restart both backend and frontend servers after changing ports

### Why Doesn't My Mechanism Work?

There are countless ways to create a poorly-formed mechanism that doesn't meet the requirements for trajectory calculation in this implementation. It's possible to create fully constrained systems—think of a rigid cube—that obviously can't be animated.

**Understanding Node Types**

First, note that there are three kinds of nodes:

- **Static** — A fixed anchor point that doesn't move. It serves as a ground reference for the mechanism. Press `W` to convert a node to Static.
- **Crank** — A node that rotates around a fixed point (its parent Static node) at a constant radius. The crank provides the input motion that drives the mechanism. Press `A` to convert a node to Crank.
- **Revolute** — A pivot joint whose position is determined by constraints from two parent nodes. Most nodes in a mechanism are Revolute joints. Press `Q` to convert a node to Revolute.

**The Crank Requirement**

In this implementation, there must be at least one Crank node, and that crank must be able to make a full revolution. It's *very easy* to accidentally make even the simplest system over-constrained by making a single link too long or too short! You can see a minimal working example in the Demo section by clicking "Four Bar." If you're having problems with your mechanism, try shortening or lengthening a link.

**Grashof's Law**

If you're wondering why your four-bar linkage won't complete a full rotation, consider **Grashof's Law**: For four-bar linkages, this law predicts whether continuous rotation is possible based on link lengths. Let *s* = shortest link, *l* = longest link, and *p*, *q* = the other two links. If *s* + *l* ≤ *p* + *q*, the linkage permits continuous rotation (crank-rocker or double-crank). Otherwise, it's a non-Grashof linkage limited to oscillation (double-rocker). Note: The current implementation doesn't support double-rocker oscillation.

**Under-Constrained Mechanisms**

Just as we can over-constrain and lock a mechanism, we can also under-constrain them. If you make a triangle of links, it's locked in shape and will compute correctly. However, if you make a square of links, it would collapse into a parallelogram since it has an extra degree of freedom. You need to fully constrain shapes by adding additional links. For example, if you add a square of links to a working mechanism, trajectory simulation will fail—to fix this, add a diagonal cross-link to triangulate and rigidify the square.

**Summary: Making Your Mechanisms Work**

- Mark appropriate nodes as Static, Crank, or Revolute
- Shorten or lengthen links as necessary to satisfy Grashof's Law
- Fully constrain open shapes by triangulating them (add diagonal links)
- Avoid dangling or hanging links that aren't part of a closed chain

## Similar Projects

If you're interested in linkage mechanism design and simulation, here are some other notable tools in this space:

### Pylinkage
- **Source**: [https://github.com/HugoFara/pylinkage](- **Source**: https://github.com/HugoFara/pylinkage)
- Pylinkage is a Python library for building and optimizing planar linkages using Particle Swarm Optimization.
- **This work uses pylinkage.**

### PMKS+ (Planar Mechanism Kinematic Simulator Plus)
- **Website**: [app.pmksplus.com](https://app.pmksplus.com/)
- **Source**: [github.com/PMKS-Web/PMKSWeb](https://github.com/PMKS-Web/PMKSWeb)
- Web-based tool with advanced analysis and automatic synthesis capabilities for mechanical design optimization. Features comprehensive kinematic and force analysis, automatic linkage synthesis for desired motion, and tools to modify designs for optimal performance.

### MotionGen
- **Website**:  [https://motiongen.io](https://motiongen.io)
- App for synthesizing and simulating planar four-bar linkages. Design and animate mechanisms for walking robots, drawing bots, and grabbers. Includes 2D/3D visualization, custom shape design, SnappyXO hardware kit prototyping support, and export for 3D printing/laser-cutting.

### GeoGebra Four-Bar Coupler Curve Creator
- **Website**: [geogebra.org/m/k3VXAnXK](https://www.geogebra.org/m/k3VXAnXK)
- Interactive four-bar coupler curve creator built on the GeoGebra platform. Great for quick exploration of coupler curves.

### Linkage Mechanism Designer and Simulator
- **Website**: [blog.rectorsquid.com/linkage-mechanism-designer-and-simulator](https://blog.rectorsquid.com/linkage-mechanism-designer-and-simulator/)
- Computer-aided design program for Microsoft Windows used for prototyping mechanical linkages.

### Pyslvs-UI
- **Source**: [github.com/KmolYuan/Pyslvs-UI](https://github.com/KmolYuan/Pyslvs-UI)
- A GUI-based (PyQt5) tool for designing 2D linkage mechanisms. Python-based with optimization capabilities.

### LInK
- **Demo**: [ahn1376-linkalphabetdemo.hf.space](https://ahn1376-linkalphabetdemo.hf.space/)
- **Source**: [github.com/ahnobari/LInK](https://github.com/ahnobari/LInK)
- LInK is a novel framework that integrates contrastive learning of performance and design space with optimization techniques for solving complex inverse problems in engineering design with discrete and continuous variables. Focuses on the path synthesis problem for planar linkage mechanisms.

### SAM
- **Website**: [artas.nl](https://www.artas.nl/en/)
- Commercial software for design, motion/force analysis, and constrained optimization of linkage mechanisms and drive systems.

### Four-bar-rs
- **Demo**: [kmolyuan.github.io/four-bar-rs](https://kmolyuan.github.io/four-bar-rs/)
- **Source**: [github.com/KmolYuan/four-bar-rs](https://github.com/KmolYuan/four-bar-rs)
- Atlas-based path synthesis of planar four-bar linkages using Elliptical Fourier Descriptors.

### Wolfram Demonstrations
- **Demos**: [Four-Bar Linkage](https://demonstrations.wolfram.com/FourBarLinkage/) | [Configuration Space](https://demonstrations.wolfram.com/ConfigurationSpaceForFourBarLinkage/)
- Interactive visualizations of four-bar linkage motion and configuration space exploration.

### SpatialGraphEmbeddings
- **Website**: [jan.legersky.cz/project/real_embeddings_of_rigid_graphs](https://jan.legersky.cz/project/real_embeddings_of_rigid_graphs/)
- **Source**: [github.com/Legersky/SpatialGraphEmbeddings](https://github.com/Legersky/SpatialGraphEmbeddings)
- Implements methods for obtaining edge lengths of minimally rigid graphs with many real spatial embeddings. Based on sampling over a two-parameter family that preserves the coupler curve. Useful for studying embeddings of graphs in Euclidean space where distances between adjacent vertices must satisfy given edge lengths.


## Why This Project?

### The Problem

**Given a planar path (or set of paths), how do we construct a multi-link mechanism that closely traces those paths while satisfying design constraints?**

This is an **inverse problem** in mechanism synthesis. The goal is to choose mechanism **dimensions** (link lengths, joint positions) so that a point on the coupler link follows a target path as closely as possible. Some of the tools mentioned here address path synthesis such as SAM, LInK, and Four-bar-rs, however for my use case they were all limited in someway (though each has strengths). Acinonyx aims to be a flexible, open-source option with support for **multi-link** mechanisms well beyond four-bars.

**Two synthesis approaches:**
- **Dimensional synthesis** — The mechanism type (topology) is fixed; the task is to find link lengths and geometry that achieve a desired function, path, or motion. Currently we do this.

-  **type synthesis** - a designer initially specifies a predefined motion transmission and is supposed not initially to know the mechanism type. This method is analogous to topology design in structural optimization. Having finished synthesizing, a certain mechanism type is received. This is future work.

### Background

A **mechanism** transforms input forces and movement into a desired set of output forces and movement. A **mechanical linkage** is an assembly of rigid bodies (links) connected by joints to manage forces and motion. When modeled as a network of rigid links and ideal joints, this is called a **kinematic chain**.

**Degrees of Freedom (DOF)**: Also called mobility, DOF is the number of independent inputs needed to fully define a mechanism's configuration. The Grübler–Kutzbach criterion calculates this from the number of links, joints, and each joint's freedom.

**Planar Linkages**: A planar mechanism constrains all links to move within parallel planes. The simplest closed-chain planar linkage is the **four-bar linkage**—four rigid bodies connected in a loop by four one-degree-of-freedom joints (revolute/pin joints or prismatic/sliding joints).

**Grashof's Law**: For four-bar linkages, this law predicts whether continuous rotation is possible based on link lengths. Let *s* = shortest link, *l* = longest link, and *p*, *q* = the other two links. If *s* + *l* ≤ *p* + *q*, the linkage permits continuous rotation (crank-rocker or double-crank). Otherwise, it's a non-Grashof linkage limited to oscillation (double-rocker).

Key terminology for four-bar linkages:
- **Ground link**: Fixed in place relative to the viewer
- **Crank**: Connected to ground by a revolute joint; can complete full revolutions
- **Rocker**: Connected to ground by a revolute joint; limited rotation range
- **Slider**: Connected to ground by a prismatic joint
- **Coupler** (floating link): Connects two other links; traces the output path

More complex mechanisms are built by combining multiple linkages.


### Technical Challenges

Path synthesis is hard because:

- **Non-convex search space** — Many local minima; gradient-based methods often get stuck.
- **Mixed variables** — Both discrete choices (topology) and continuous parameters (lengths, positions).
- **Timing** — Path generation can be *without* prescribed timing (phase-invariant) or *with* prescribed timing; both cases matter in practice.

Acinonyx tackles these with multiple **distance metrics** and **global optimization** strategies (e.g. MLSL, PSO), while type/topology synthesis remains a future goal.

## License

TBD
