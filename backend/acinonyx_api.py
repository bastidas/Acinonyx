from __future__ import annotations

import json
import math
import time
import traceback
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configs.appconfig import USER_DIR
from pylink_tools.hypergraph_adapter import sync_hypergraph_distances
from pylink_tools.kinematic import compute_trajectory
from pylink_tools.kinematic import is_hypergraph_format
from pylink_tools.optimize import extract_dimensions
from pylink_tools.optimize import optimize_trajectory
from pylink_tools.optimize import TargetTrajectory
from pylink_tools.trajectory_utils import analyze_trajectory
from pylink_tools.trajectory_utils import resample_trajectory
from pylink_tools.trajectory_utils import smooth_trajectory
# from pylink_tools.trajectory_utils import prepare_trajectory_for_optimization


def sanitize_for_json(obj):
    """
    Recursively sanitize an object for JSON serialization.

    Converts inf/-inf to string "Infinity"/"-Infinity" and nan to null.
    This prevents JSON serialization errors.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isinf(obj):
            return 'Infinity' if obj > 0 else '-Infinity'
        elif math.isnan(obj):
            return None
        return obj
    elif hasattr(obj, '__float__'):  # numpy types
        val = float(obj)
        if math.isinf(val):
            return 'Infinity' if val > 0 else '-Infinity'
        elif math.isnan(val):
            return None
        return val
    return obj


app = FastAPI(title='Acinonyx API')

# Simple CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
def root():
    return {'message': 'Acinonyx API is running'}


@app.get('/status')
def get_status():
    return {
        'status': 'operational',
        'message': 'Acinonyx backend is running successfully',
    }


# @app.get('/load-last-force-graph')
# def load_force_graph():
#     """Load the most recent force graph from the force_graphs directory"""
#     try:
#         # Ensure USER_DIR exists first
#         USER_DIR.mkdir(parents=True, exist_ok=True)
#         force_graphs_dir = USER_DIR / 'force_graphs'

#         if not force_graphs_dir.exists():
#             return {
#                 'error': 'No force_graphs directory found',
#                 'path': str(force_graphs_dir),
#             }

#         # Find all force graph JSON files
#         force_graph_files = list(force_graphs_dir.glob('force_graph_*.json'))

#         if not force_graph_files:
#             return {
#                 'error': 'No force graphs found in force_graphs directory',
#             }

#         # Get the most recent file by modification time
#         latest_file = max(force_graph_files, key=lambda f: f.stat().st_mtime)

#         with open(latest_file) as f:
#             graph_data = json.load(f)

#         print(f'Loaded force graph from: {latest_file.name}')
#         return graph_data

#     except Exception as e:
#         return {
#             'error': f'Failed to load force graph: {str(e)}',
#         }


@app.post('/save-pylink-graph')
def save_pylink_graph(pylink_data: dict):
    """Save a pylink graph to the graphs directory"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        graphs_dir = USER_DIR / 'graphs'
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # Use provided name or generate timestamp
        name = pylink_data.get('name', 'pylink')
        time_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{name}_{time_mark}.json'
        save_path = graphs_dir / filename

        # Add metadata
        save_data = {
            **pylink_data,
            'saved_at': time_mark,
        }

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f'Pylink graph saved to: {save_path}')

        return {
            'status': 'success',
            'message': 'Pylink graph saved successfully',
            'filename': filename,
            'path': str(save_path),
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to save pylink graph: {str(e)}',
        }


@app.post('/save-pylink-graph-as')
def save_pylink_graph_as(request: dict):
    """Save a pylink graph to the graphs directory with a custom filename"""
    try:
        pylink_data = request.get('data', {})
        custom_filename = request.get('filename', '')

        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        graphs_dir = USER_DIR / 'graphs'
        graphs_dir.mkdir(parents=True, exist_ok=True)

        # Use custom filename or generate from name
        if custom_filename:
            # Ensure .json extension
            if not custom_filename.endswith('.json'):
                custom_filename += '.json'
            filename = custom_filename
        else:
            name = pylink_data.get('name', 'pylink')
            time_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{name}_{time_mark}.json'

        save_path = graphs_dir / filename

        # Add metadata
        time_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_data = {
            **pylink_data,
            'saved_at': time_mark,
        }

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f'Pylink graph saved as: {save_path}')

        return {
            'status': 'success',
            'message': 'Pylink graph saved successfully',
            'filename': filename,
            'path': str(save_path),
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to save pylink graph: {str(e)}',
        }


@app.get('/list-pylink-graphs')
def list_pylink_graphs():
    """List all saved pylink graphs"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        graphs_dir = USER_DIR / 'graphs'

        if not graphs_dir.exists():
            return {
                'status': 'success',
                'files': [],
            }

        files = []
        for f in sorted(graphs_dir.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    linkage = data.get('linkage', {})
                    files.append({
                        'filename': f.name,
                        'name': data.get('name', f.stem),
                        'nodes_count': len(linkage.get('nodes', {})),
                        'edges_count': len(linkage.get('edges', {})),
                        'saved_at': data.get('saved_at', ''),
                        'version': data.get('version', '2.0'),
                    })
            except:
                files.append({
                    'filename': f.name,
                    'name': f.stem,
                    'error': True,
                })

        return {
            'status': 'success',
            'files': files,
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to list pylink graphs: {str(e)}',
        }


@app.get('/load-pylink-graph')
def load_pylink_graph(filename: str = None):
    """Load a pylink graph from the graphs directory"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        graphs_dir = USER_DIR / 'graphs'

        if not graphs_dir.exists():
            return {
                'status': 'error',
                'message': 'No graphs directory found',
            }

        if filename:
            # Load specific file
            file_path = graphs_dir / filename
            if not file_path.exists():
                return {
                    'status': 'error',
                    'message': f'File not found: {filename}',
                }
        else:
            # Load most recent file
            files = list(graphs_dir.glob('*.json'))
            if not files:
                return {
                    'status': 'error',
                    'message': 'No pylink graphs found',
                }
            file_path = max(files, key=lambda f: f.stat().st_mtime)

        with open(file_path) as f:
            graph_data = json.load(f)

        print(f'Loaded pylink graph from: {file_path.name}')

        return {
            'status': 'success',
            'filename': file_path.name,
            'data': graph_data,
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to load pylink graph: {str(e)}',
        }


@app.get('/load-demo')
def load_demo(name: str):
    """Load a demo linkage from the demo directory"""
    from configs.paths import BASE_DIR

    try:
        demo_dir = BASE_DIR / 'demo'

        if not demo_dir.exists():
            return {
                'status': 'error',
                'message': 'Demo directory not found',
            }

        # Map demo names to files
        demo_files = {
            '4bar': 'test_graphs/4bar.json',
            'leg': 'test_graphs/leg.json',
            'walker': 'walker0.json',
            'complex': 'test_graphs/complex.json',
            'intermediate': 'test_graphs/intermediate.json',
        }

        if name not in demo_files:
            return {
                'status': 'error',
                'message': f'Unknown demo: {name}. Available: {list(demo_files.keys())}',
            }

        file_path = demo_dir / demo_files[name]
        if not file_path.exists():
            return {
                'status': 'error',
                'message': f'Demo file not found: {file_path}',
            }

        with open(file_path) as f:
            demo_data = json.load(f)

        print(f'Loaded demo: {name} from {file_path.name}')

        return {
            'status': 'success',
            'name': name,
            'filename': file_path.name,
            'data': demo_data,
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to load demo: {str(e)}',
        }


@app.post('/compute-pylink-trajectory')
def compute_pylink_trajectory(request: dict):
    """
    Compute joint trajectories from PylinkDocument format.

    This endpoint takes the pylink graph data (same format as save/load),
    converts it to a pylinkage Linkage object, runs the simulation,
    and returns the positions of each joint at each timestep.

    Request body:
        {
            "name": "...",
            "pylinkage": {
                "name": "...",
                "joints": [...],
                "solve_order": [...]
            },
            "meta": {
                "joints": {...},
                "links": {...}
            },
            "n_steps": 12,           # Optional, defaults to 12
            "skip_sync": false       # Optional, if true, uses stored distances directly
        }

    Returns:
        {
            "status": "success",
            "trajectories": {
                "joint_name": [[x0, y0], [x1, y1], ...],
                ...
            },
            "n_steps": 12,
            "execution_time_ms": 15.2
        }
    """

    try:
        start_time = time.perf_counter()

        # Extract pylink_data (request may wrap it or be it directly)
        pylink_data = request.get('pylink_data', request)
        skip_sync = request.get('skip_sync', False)
        n_steps = request.get('n_steps') or pylink_data.get('n_steps', 12)
        pylink_data['n_steps'] = n_steps
        # nodes_count = len(pylink_data.get('linkage', {}).get('nodes', {}))
        # print(f'\n=== COMPUTE PYLINK TRAJECTORY ===')
        # print(f'Nodes: {nodes_count}, Steps: {n_steps}, skip_sync: {skip_sync}')
        # Delegate to kinematic module
        result = compute_trajectory(pylink_data, verbose=True, skip_sync=skip_sync)

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        print(f'  Completed in {execution_time_ms:.2f}ms')

        if not result.success:
            return {
                'status': 'error',
                'message': result.error,
            }

        return {
            'status': 'success',
            'message': f'Computed {result.n_steps} trajectory steps for {len(result.trajectories)} joints',
            'trajectories': result.trajectories,
            'n_steps': result.n_steps,
            'execution_time_ms': execution_time_ms,
            'joint_types': result.joint_types,
        }

    except Exception as e:
        print(f'Error computing pylink trajectory: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Failed to compute trajectory: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


@app.post('/validate-mechanism')
def validate_mechanism_endpoint(pylink_data: dict):
    """
    Identify valid mechanisms in the pylink graph.

    A valid mechanism is a connected group of links that:
      - Has at least 3 links (+ implicit ground)
      - Contains a Crank joint (driver)
      - Contains at least one Static joint (ground)
      - Can successfully build a pylinkage Linkage object
      - All links maintain their length (no over-constrained links)

    Returns:
        {
            "status": "success",
            "valid": bool,           # True if at least one valid mechanism exists
            "groups": [...],         # All connected link groups
            "valid_groups": [...],   # Only the valid mechanism groups
            "errors": [...],         # Validation errors
        }
    """
    from pylink_tools.kinematic import validate_mechanism

    try:
        result = validate_mechanism(pylink_data)

        return {
            'status': 'success',
            **result,
        }

    except Exception as e:
        print(f'Error validating mechanism: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Validation failed: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


# @app.post("/check-link-rigidity")
# def check_link_rigidity_endpoint(pylink_data: dict):
#     """
#     Check if visual links would stretch during simulation.

#     This detects over-constrained mechanisms where a link connects a kinematic
#     joint (which moves during simulation) to a static joint that's not part of
#     the kinematic chain. Such links would need to "stretch" during simulation,
#     which is physically impossible for rigid links.

#     In reality, this would either:
#     - Lock the mechanism completely (if distances happen to match)
#     - Make the mechanism impossible to assemble (if distances don't match)

#     Returns:
#         {
#             "status": "success",
#             "valid": bool,           # True if all links can maintain their length
#             "locked_links": [...],   # Links that would stretch/lock the mechanism
#             "message": str           # Human-readable summary
#         }
#     """
#     from pylink_tools.kinematic import check_link_rigidity

#     try:
#         result = check_link_rigidity(pylink_data)

#         return {
#             "status": "success",
#             **result
#         }

#     except Exception as e:
#         print(f"Error checking link rigidity: {e}")
#         traceback.print_exc()
#         return {
#             "status": "error",
#             "message": f"Check failed: {str(e)}",
#             "traceback": traceback.format_exc().split('\n')
#         }


@app.post('/optimize-trajectory')
def optimize_trajectory_endpoint(request: dict):
    """
    Optimize linkage dimensions to fit a target trajectory.

    This endpoint takes a pylink document, target path, and optimization options,
    then optimizes linkage dimensions (link lengths) to make a specified joint
    follow the target path as closely as possible.

    Request body:
        {
            "pylink_data": { ... },           # Full pylink document
            "target_path": {
                "joint_name": "joint_name",   # Which joint should follow the path
                "positions": [[x, y], ...]    # Target positions for the joint
            },
            "optimization_options": {
                "method": "pylinkage",        # "pso", "pylinkage", "scipy", "powell", "nelder-mead"
                "n_particles": 32,            # PSO: swarm size (5-1024)
                "iterations": 512,            # PSO: iterations (10-10000)
                "max_iterations": 100,        # SciPy: max function evaluations
                "tolerance": 1e-6,            # SciPy: convergence tolerance
                "bounds_factor": 2.0,         # How much dimensions can vary (1.25-5.0)
                "min_length": 0.1,            # Minimum link length
                "verbose": true               # Print progress
            }
        }

    Returns:
        {
            "status": "success",
            "message": "...",
            "result": {
                "success": true,
                "initial_error": 123.45,
                "final_error": 12.34,
                "iterations": 512,
                "optimized_dimensions": { "A_distance": 2.5, ... },
                "optimized_pylink_data": { ... }
            },
            "execution_time_ms": 1234.5
        }
    """
    try:
        start_time = time.perf_counter()

        pylink_data = request.get('pylink_data')
        target_path = request.get('target_path', {})
        optimization_options = request.get('optimization_options', {})

        # Extract target path info
        target_joint = target_path.get('joint_name')
        target_positions = target_path.get('positions', [])

        # Extract optimization options with defaults
        method = optimization_options.get('method', 'pylinkage')
        n_particles = optimization_options.get('n_particles', 32)
        iterations = optimization_options.get('iterations', 512)
        max_iterations = optimization_options.get('max_iterations', 100)
        tolerance = optimization_options.get('tolerance', 1e-6)
        bounds_factor = optimization_options.get('bounds_factor', 2.0)
        min_length = optimization_options.get('min_length', 0.1)
        verbose = optimization_options.get('verbose', True)

        if not pylink_data:
            return {
                'status': 'error',
                'message': 'Missing pylink_data',
            }

        if not target_joint:
            return {
                'status': 'error',
                'message': 'Missing target_path.joint_name',
            }

        if len(target_positions) < 2:
            return {
                'status': 'error',
                'message': 'Target path must have at least 2 points',
            }

        print(f'\n=== OPTIMIZE TRAJECTORY ===')
        print(f'Target joint: {target_joint}')
        print(f'Target points: {len(target_positions)}')
        print(f'Method: {method}')
        print(f'Bounds factor: {bounds_factor}')

        # CRITICAL: First sync distances from visual positions
        # The frontend may save stale/incorrect distances that don't match visual layout.
        # This ensures we start with valid, solvable mechanism geometry.
        print('  Syncing distances from visual positions...')
        if is_hypergraph_format(pylink_data):
            pylink_data = sync_hypergraph_distances(pylink_data, verbose=verbose)

        # Validate the mechanism by running a reference simulation
        pylink_data['n_steps'] = len(target_positions)
        ref_result = compute_trajectory(pylink_data, verbose=False, skip_sync=True)

        if not ref_result.success:
            return sanitize_for_json({
                'status': 'error',
                'message': f'Mechanism validation failed: {ref_result.error}',
                'hint': 'The mechanism may be over-constrained or have invalid geometry',
            })

        # Check that target joint exists
        if target_joint not in ref_result.trajectories:
            return {
                'status': 'error',
                'message': f"Target joint '{target_joint}' not found in mechanism",
                'available_joints': list(ref_result.trajectories.keys()),
            }

        print(f'  ✓ Reference simulation successful')

        # Extract dimensions from the SYNCED data (now has correct values)
        dim_spec = extract_dimensions(pylink_data, bounds_factor=bounds_factor, min_length=min_length)
        print(f'Optimizable dimensions: {dim_spec.names}')
        print(f"Initial values: {[f'{v:.3f}' for v in dim_spec.initial_values]}")
        print(f"Bounds: {[(f'{b[0]:.2f}', f'{b[1]:.2f}') for b in dim_spec.bounds]}")

        # Create target trajectory
        target = TargetTrajectory(
            joint_name=target_joint,
            positions=target_positions,
        )

        # n_steps already set above during reference simulation

        # Build kwargs based on method
        opt_kwargs = {
            'pylink_data': pylink_data,
            'target': target,
            'dimension_spec': dim_spec,  # Pass pre-computed dimensions with custom bounds
            'method': method,
            'verbose': verbose,
        }

        # Add method-specific parameters
        if method in ('pso', 'pylinkage'):
            opt_kwargs['n_particles'] = n_particles
            opt_kwargs['iterations'] = iterations
        elif method in ('scipy', 'powell', 'nelder-mead'):
            opt_kwargs['max_iterations'] = max_iterations
            opt_kwargs['tolerance'] = tolerance

        # Run optimization
        result = optimize_trajectory(**opt_kwargs)

        # If optimization succeeded, update node positions based on new distances
        optimized_pylink_data = result.optimized_pylink_data
        if result.success and optimized_pylink_data:
            try:
                # Run a simulation on the optimized data to get correct positions
                # IMPORTANT: skip_sync=True prevents distances from being overwritten by old visual positions
                sim_result = compute_trajectory(optimized_pylink_data, verbose=False, skip_sync=True)
                if sim_result.success and sim_result.trajectories:
                    # Update node positions with first frame from simulation
                    nodes = optimized_pylink_data.get('linkage', {}).get('nodes', {})
                    for joint_name, positions in sim_result.trajectories.items():
                        if positions and len(positions) > 0:
                            x, y = positions[0]  # First frame position
                            if joint_name in nodes:
                                nodes[joint_name]['position'] = [x, y]
                    print(f'  Updated node positions from simulation')
            except Exception as e:
                print(f'  Warning: Could not update node positions: {e}')

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Calculate improvement percentage
        improvement = 0.0
        if result.initial_error > 0:
            improvement = ((result.initial_error - result.final_error) / result.initial_error) * 100

        # Handle inf/nan for logging
        initial_err_str = f'{result.initial_error:.4f}' if math.isfinite(result.initial_error) else str(result.initial_error)
        final_err_str = f'{result.final_error:.4f}' if math.isfinite(result.final_error) else str(result.final_error)

        print(f'  Initial error: {initial_err_str}')
        print(f'  Final error: {final_err_str}')
        print(f'  Improvement: {improvement:.1f}%')
        print(f'  Completed in {execution_time_ms:.2f}ms')

        # Build response and sanitize for JSON (inf/nan not allowed)
        response = {
            'status': 'success',
            'message': f'Optimization complete: {improvement:.1f}% improvement',
            'result': {
                'success': result.success,
                'initial_error': result.initial_error,
                'final_error': result.final_error,
                'iterations': result.iterations,
                'optimized_dimensions': result.optimized_dimensions,
                'optimized_pylink_data': optimized_pylink_data,
                'error': result.error,
            },
            'execution_time_ms': execution_time_ms,
        }

        # Sanitize to handle inf/nan values (JSON doesn't support them)
        return sanitize_for_json(response)

    except Exception as e:
        print(f'Error optimizing trajectory: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Optimization failed: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


@app.get('/get-optimizable-dimensions')
def get_optimizable_dimensions(pylink_data: dict = None):
    """
    Get the list of optimizable dimensions for a linkage.

    This is useful for displaying to the user what parameters
    will be adjusted during optimization.

    Request body:
        {
            "pylink_data": { ... }  # Full pylink document
        }

    Returns:
        {
            "status": "success",
            "dimensions": ["A_distance", "B_distance0", "B_distance1"],
            "initial_values": [1.5, 3.5, 2.5],
            "bounds": [[0.75, 3.0], [1.75, 7.0], [1.25, 5.0]]
        }
    """
    try:
        if not pylink_data:
            return {
                'status': 'error',
                'message': 'Missing pylink_data',
            }

        dim_spec = extract_dimensions(pylink_data)

        return {
            'status': 'success',
            'dimensions': dim_spec.names,
            'initial_values': dim_spec.initial_values,
            'bounds': dim_spec.bounds,
            'n_dimensions': len(dim_spec),
        }

    except Exception as e:
        print(f'Error extracting dimensions: {e}')
        return {
            'status': 'error',
            'message': f'Failed to extract dimensions: {str(e)}',
        }


@app.post('/prepare-trajectory')
def prepare_trajectory(request: dict):
    """
    Prepare a trajectory for optimization by resampling and/or smoothing.

    This is essential for working with external/captured trajectories that may have:
    - Different point counts than the simulation N_STEPS
    - Noise from measurement or digitization
    - Irregular sampling

    Request body:
        {
            "trajectory": [[x1, y1], [x2, y2], ...],  # Input trajectory points
            "target_n_steps": 24,        # Target number of points (match simulation)
            "smooth": true,              # Whether to apply smoothing
            "smooth_window": 4,          # Smoothing window size (2-64)
            "smooth_polyorder": 3,       # Smoothing polynomial order
            "smooth_method": "savgol",   # "savgol", "moving_avg", or "gaussian"
            "resample": true,            # Whether to resample
            "resample_method": "parametric",  # "linear", "cubic", or "parametric"
            "closed": true               # Treat trajectory as closed loop (default: true)
        }

    Returns:
        {
            "status": "success",
            "original_points": 157,
            "output_points": 24,
            "trajectory": [[x1, y1], [x2, y2], ...],  # Processed trajectory
            "analysis": {
                "n_points": 24,
                "centroid": [x, y],
                "bounding_box": {...},
                "total_path_length": 123.4,
                "is_closed": true
            }
        }

    Hyperparameter Guide:
        target_n_steps:
            - Should match your simulation N_STEPS for error computation
            - Higher = more precision, slower optimization
            - Recommended: 24-48 for optimization, 48-96 for final results

        smooth_window:
            - Must be odd number
            - 3 = light smoothing (preserves detail)
            - 5-7 = medium smoothing (good for noisy data)
            - 9-11 = heavy smoothing (aggressive noise removal)

        smooth_polyorder:
            - Must be < smooth_window
            - 2-3 = preserves peaks and valleys
            - Higher = more aggressive smoothing

        smooth_method:
            - "savgol": Savitzky-Golay filter (preserves peaks, recommended)
            - "moving_avg": Simple moving average (aggressive)
            - "gaussian": Gaussian-weighted average (natural)

        resample_method:
            - "parametric": Arc-length based (best for closed curves, recommended)
            - "cubic": Cubic spline (smooth, may overshoot)
            - "linear": Linear interpolation (fast, may create corners)

        closed:
            - true (default): Treats trajectory as a closed loop. The resampling
              includes the segment from the last point back to the first.
              This is correct for linkage mechanism trajectories.
            - false: Treats trajectory as an open curve (start != end)
    """
    try:
        start_time = time.perf_counter()

        # Extract parameters
        trajectory = request.get('trajectory', [])
        target_n_steps = request.get('target_n_steps', 24)
        do_smooth = request.get('smooth', True)
        smooth_window = request.get('smooth_window', 5)
        smooth_polyorder = request.get('smooth_polyorder', 3)
        smooth_method = request.get('smooth_method', 'savgol')
        do_resample = request.get('resample', True)
        resample_method = request.get('resample_method', 'parametric')
        is_closed = request.get('closed', True)  # Default to closed trajectories

        if not trajectory:
            return {
                'status': 'error',
                'message': 'No trajectory provided',
            }

        if len(trajectory) < 3:
            return {
                'status': 'error',
                'message': f'Trajectory too short: {len(trajectory)} points (need at least 3)',
            }

        # Validate smooth_window - clamp to 2-64 range
        smooth_window = max(2, min(smooth_window, 64))

        # For Savitzky-Golay filter, window must be odd and >= 3
        # For other methods (moving_avg, gaussian), even is fine
        if smooth_method == 'savgol':
            if smooth_window < 3:
                smooth_window = 3
            if smooth_window % 2 == 0:
                smooth_window += 1  # Make odd

        # Validate smooth_polyorder
        smooth_polyorder = max(1, min(smooth_polyorder, smooth_window - 1))

        # Convert to list of tuples
        original_points = len(trajectory)
        result = [(float(p[0]), float(p[1])) for p in trajectory]

        # Apply smoothing first (if enabled)
        if do_smooth and len(result) >= smooth_window:
            result = smooth_trajectory(
                result,
                window_size=smooth_window,
                polyorder=smooth_polyorder,
                method=smooth_method,
            )

        # Then resample (if enabled and needed)
        # Pass closed=True to ensure the closing segment is included in arc length
        if do_resample and len(result) != target_n_steps:
            result = resample_trajectory(
                result,
                target_n_steps,
                method=resample_method,
                closed=is_closed,
            )

        # Analyze the result
        analysis = analyze_trajectory(result)

        # Convert to list format for JSON
        result_list = [[p[0], p[1]] for p in result]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            'status': 'success',
            'original_points': original_points,
            'output_points': len(result),
            'trajectory': result_list,
            'analysis': analysis,
            'processing': {
                'smoothed': do_smooth,
                'smooth_window': smooth_window if do_smooth else None,
                'smooth_polyorder': smooth_polyorder if do_smooth else None,
                'smooth_method': smooth_method if do_smooth else None,
                'resampled': do_resample and len(result) != original_points,
                'resample_method': resample_method if do_resample else None,
                'target_n_steps': target_n_steps if do_resample else None,
            },
            'execution_time_ms': round(elapsed_ms, 2),
        }

    except Exception as e:
        print(f'Error preparing trajectory: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Failed to prepare trajectory: {str(e)}',
        }


@app.post('/analyze-trajectory')
def analyze_trajectory_endpoint(request: dict):
    """
    Analyze a trajectory and return statistics.

    Useful for understanding trajectory properties before optimization.

    Request body:
        {
            "trajectory": [[x1, y1], [x2, y2], ...]
        }

    Returns:
        {
            "status": "success",
            "analysis": {
                "n_points": 24,
                "centroid": [x, y],
                "bounding_box": {
                    "x_min": ..., "x_max": ...,
                    "y_min": ..., "y_max": ...,
                    "width": ..., "height": ...
                },
                "total_path_length": 123.4,
                "closure_gap": 0.1,
                "is_closed": true,
                "roughness": 0.05,
                "avg_segment_length": 5.14
            }
        }
    """
    try:
        trajectory = request.get('trajectory', [])

        if not trajectory:
            return {
                'status': 'error',
                'message': 'No trajectory provided',
            }

        # Convert to list of tuples
        traj_tuples = [(float(p[0]), float(p[1])) for p in trajectory]

        # Analyze
        analysis = analyze_trajectory(traj_tuples)

        return {
            'status': 'success',
            'analysis': analysis,
        }

    except Exception as e:
        print(f'Error analyzing trajectory: {e}')
        return {
            'status': 'error',
            'message': f'Failed to analyze trajectory: {str(e)}',
        }


@app.get('/logs/backend')
def get_backend_log(lines: int = 500, offset: int = 0):
    """
    Get the most recent lines from the backend log file.

    Args:
        lines: Number of lines to return (default: 500)
        offset: Number of lines to skip from the end (default: 0)

    Returns:
        {
            "status": "success",
            "content": "...",
            "total_lines": 1234,
            "lines_returned": 500
        }
    """
    from configs.paths import BASE_DIR

    log_file = BASE_DIR / 'backend.log'

    try:
        if not log_file.exists():
            return {
                'status': 'success',
                'content': '(No log file found yet)',
                'total_lines': 0,
                'lines_returned': 0,
            }

        # Read all lines
        with open(log_file, encoding='utf-8') as f:
            all_lines = f.readlines()

        total_lines = len(all_lines)

        # Get the requested range from the end
        if offset > 0:
            end_idx = max(0, total_lines - offset)
            start_idx = max(0, end_idx - lines)
            selected_lines = all_lines[start_idx:end_idx]
        else:
            start_idx = max(0, total_lines - lines)
            selected_lines = all_lines[start_idx:]

        return {
            'status': 'success',
            'content': ''.join(selected_lines),
            'total_lines': total_lines,
            'lines_returned': len(selected_lines),
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to read log file: {str(e)}',
        }


@app.delete('/logs/backend')
def clear_backend_log():
    """Clear the backend log file."""
    from configs.paths import BASE_DIR

    log_file = BASE_DIR / 'backend.log'

    try:
        if log_file.exists():
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('')  # Clear the file

        return {
            'status': 'success',
            'message': 'Log file cleared',
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to clear log file: {str(e)}',
        }
