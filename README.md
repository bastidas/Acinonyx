# Acinonyx 🐆

A Python application for simulating and visualizing mechanical linkage systems - fast and agile like a cheetah.

This project provides tools for creating, simulating, and visualizing mechanical linkage systems with exceptional performance. It includes:

- **Mechanical Link Simulation**: Define and simulate multi-link mechanical systems using pylinkage
- **Visualization Tools**: Generate static plots and animations of linkage motion using a React frontend
- **REST API**: FastAPI backend for graph operations and trajectory computation

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
cd automata
```

### 2. Set Up Python Environment

#### Option A: Using Conda

```bash
conda env create -f environment.yml
conda activate automata
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

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

### 4. Verify Installation

Run the test suite to ensure everything is working:

```bash
# Using pytest
pytest tests/ -v

# Or run a specific test
pytest tests/test_pylink_trajectory.py -v
```

All tests should pass without errors.

## Usage

### Quick Start

Use the provided run script to start both backend and frontend:

```bash
# On Unix/MacOS:
chmod +x run.sh
./run.sh
```

The script will:
1. Start the FastAPI backend server on `http://localhost:8021` (configured in `configs/appconfig.py`)
2. Start the Vite development server on `http://localhost:5173`
3. Open the web GUI in your browser

### Manual Start

If you prefer to run the servers manually:

#### Terminal 1 - Backend Server

```bash
# Activate your environment first
conda activate automata  # or: source venv/bin/activate

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
automata/
├── backend/
│   ├── query_api.py          # FastAPI endpoints
│   └── run_server.py         # Backend server entry point
├── configs/
│   ├── appconfig.py          # Centralized config (ports, paths)
│   ├── link_models.py        # Pydantic models for links/nodes
│   └── matplotlib_config.py  # Matplotlib backend config
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── ForceGraphViewTab.tsx
│   │   │   ├── PathVisualization.tsx
│   │   │   ├── PylinkAnimateSimulate.tsx
│   │   │   ├── PylinkBuilderTab.tsx
│   │   │   ├── PylinkBuilderTools.tsx
│   │   │   └── StatusAboutTab.tsx
│   │   ├── App.tsx           # Main application
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   └── vite.config.ts
├── link/                     # Pylinkage bridge utilities
│   └── tools.py              # Link manipulation utilities
├── pylink_tools/             # Demo linkages and examples
├── viz_tools/                # Animation and visualization
├── tests/
│   ├── test_api_endpoints.py      # API endpoint tests
│   ├── test_demo_linkages.py      # Pylinkage validation tests
│   ├── test_file_operations.py    # Cross-platform file I/O tests
│   ├── test_pylink_trajectory.py  # Trajectory computation tests
│   └── 4bar_test.json            # Test data
├── user/                     # User-saved graphs (generated)
│   ├── pygraphs/            # Saved pylink graphs
│   └── force_graphs/        # Saved force graphs
├── pyproject.toml            # Python project configuration
└── README.md
```

## Testing

### Run Tests

```bash
conda activate automata
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
  * for example if using pyenv `pyenv activate automata`
  * * for example if using anaconda `conda activate automata`
- Verify installation: `pip list | grep automata`

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

## License

TBD







