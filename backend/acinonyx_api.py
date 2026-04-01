from __future__ import annotations

import json
import logging
import math
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any
from typing import cast

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pylinkage.joints import Crank
from pylinkage.joints import Static

from configs.appconfig import GRAPHS_DIR
from configs.logging_config import setup_logging
from demo.helpers import create_mechanism_from_dict
from multi.dim_tools import reduce_dimensions
from pylink_tools.hypergraph_adapter import sync_hypergraph_distances
from pylink_tools.optimization_types import DimensionBoundsSpec
from pylink_tools.optimization_types import TargetTrajectory
from pylink_tools.optimization_types import TopologyVariationSpec
from pylink_tools.optimize import optimize_trajectory
from pylink_tools.trajectory_utils import analyze_trajectory
from pylink_tools.trajectory_utils import resample_trajectory
from pylink_tools.trajectory_utils import smooth_trajectory
from target_gen.achievable_target import create_achievable_target
from target_gen.sampling import apply_dimension_variation_config
from target_gen.sampling import generate_good_samples
from target_gen.sampling import generate_samples
from target_gen.sampling import generate_valid_samples
from target_gen.variation_config import DimensionVariationConfig
from target_gen.variation_config import MechVariationConfig
from target_gen.variation_config import StaticJointMovementConfig
from target_gen.variation_config import TopologyChangeConfig

# Configure logging when this module is loaded (e.g. by uvicorn worker with --reload).
# Ensures backend.log and console get logs from this process; run_server.py only runs in the launcher process.
_log_level = os.getenv('LOG_LEVEL', 'DEBUG').upper()
_level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
}
setup_logging(level=_level_map.get(_log_level, logging.DEBUG))

logger = logging.getLogger(__name__)

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


def _build_trajectory_with_initial(
    trajectory_array: np.ndarray,
    initial_row: np.ndarray,
    n_steps: int,
    *,
    check_closed: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> tuple[np.ndarray, bool]:
    """
    Build trajectory of exactly n_steps rows with frame 0 = doc state (initial).

    Uses one allocation and no concat: when closed, shifts last row to first;
    when not closed, sets row 0 = initial and rows 1..n_steps-1 = trajectory[0..n_steps-2].
    step_fast(iterations=n_steps) returns rows [after_1, ..., after_n_steps]; last row
    equals initial when the loop closes.

    Args:
        trajectory_array: Shape (n_steps, n_joints, 2) from mechanism.simulate().
        initial_row: Shape (1, n_joints, 2), e.g. mechanism._initial_positions.reshape(1, -1, 2).
        n_steps: Requested number of steps (must match trajectory_array.shape[0]).
        check_closed: If True, run allclose(initial, trajectory[-1]) and set
            trajectory_did_not_close when not closed. If False, skip the check (faster, for batch).
        rtol, atol: Tolerances for allclose when check_closed=True.

    Returns:
        (result_array, trajectory_did_not_close): result_array shape (n_steps, n_joints, 2).
    """
    n_joints = trajectory_array.shape[1]
    result = np.empty((n_steps, n_joints, 2), dtype=np.float64)
    trajectory_did_not_close = False

    if n_steps < 2:
        result[0] = initial_row[0]
        return result, False

    if check_closed and np.allclose(
        initial_row[0], trajectory_array[n_steps - 1], rtol=rtol, atol=atol,
    ):
        # Closed: use last row as first (avoid reading initial), then trajectory[0..n_steps-2]
        result[0] = trajectory_array[n_steps - 1]
        result[1:] = trajectory_array[: n_steps - 1]
    else:
        if check_closed:
            trajectory_did_not_close = True
        result[0] = initial_row[0]
        result[1:] = trajectory_array[: n_steps - 1]

    return result, trajectory_did_not_close


def create_mechanism_from_request(request: dict, n_steps: int | None = None):
    """
    Convert frontend request data to Mechanism object.

    This is the SINGLE conversion point for all endpoints.
    Handles:
    - Extracting pylink_data from request (may be nested or direct)
    - Validating required fields
    - Syncing distances from visual positions if needed
    - Creating Mechanism with proper n_steps

    Args:
        request: Request dict from frontend (may contain 'pylink_data' key or be the data itself)
        n_steps: Optional n_steps override (defaults to request value or 32)

    Returns:
        Mechanism object ready for simulation/optimization

    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Extract pylink_data (request may wrap it or be it directly)
    pylink_data = request.get('pylink_data', request)

    # Validate required fields
    if not isinstance(pylink_data, dict):
        raise ValueError(f'Invalid request: expected dict, got {type(pylink_data).__name__}')

    # Detect format: hypergraph (linkage.nodes) vs legacy (pylinkage.joints)
    has_hypergraph = 'linkage' in pylink_data and isinstance(pylink_data.get('linkage'), dict)
    has_legacy = 'pylinkage' in pylink_data and isinstance(pylink_data.get('pylinkage'), dict)

    if not has_hypergraph and not has_legacy:
        raise ValueError('Invalid request: must have either "linkage" (hypergraph) or "pylinkage" (legacy) field')

    # If legacy format, we need to convert it - but to_simulatable_linkage only accepts hypergraph
    # For now, reject legacy format and require frontend to convert
    if has_legacy and not has_hypergraph:
        raise ValueError(
            'Invalid request: legacy format (pylinkage.joints) not supported by this endpoint. '
            'Please convert to hypergraph format (linkage.nodes, linkage.edges) before sending.',
        )

    # Validate hypergraph format
    linkage = pylink_data.get('linkage')
    if not isinstance(linkage, dict):
        raise ValueError('Invalid request: missing or invalid "linkage" field. Expected dict with "nodes" and "edges"')

    nodes = linkage.get('nodes')
    edges = linkage.get('edges')

    if not isinstance(nodes, dict) or len(nodes) == 0:
        raise ValueError('Invalid request: missing or empty "linkage.nodes" field')

    if not isinstance(edges, dict) or len(edges) == 0:
        raise ValueError('Invalid request: missing or empty "linkage.edges" field')

    # Determine n_steps
    if n_steps is None:
        n_steps = request.get('n_steps') or pylink_data.get('n_steps', 32)

    # Sync distances from visual positions if needed (hypergraph format)
    # Check if data is in hypergraph format: has 'linkage' key with 'nodes'
    if 'linkage' in pylink_data and 'nodes' in pylink_data.get('linkage', {}):
        pylink_data = sync_hypergraph_distances(pylink_data, verbose=False)

    # Create Mechanism using helper from demo.helpers
    # This may raise exceptions if the linkage structure is invalid
    mechanism = create_mechanism_from_dict(pylink_data)

    # Ensure n_steps is set correctly
    if mechanism._n_steps != n_steps:
        mechanism._n_steps = n_steps

    return mechanism


@app.post('/save-pylink-graph')
def save_pylink_graph(pylink_data: dict):
    """Save a pylink graph to the graphs directory.
    Request body may include optional 'filename'; if omitted, uses acinonyx-YYYYMMDD.json.
    """
    try:
        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

        custom_filename = pylink_data.get('filename')
        if custom_filename and isinstance(custom_filename, str) and custom_filename.strip():
            filename = custom_filename.strip()
            if not filename.endswith('.json'):
                filename += '.json'
        else:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'acinonyx-{date_str}.json'

        save_path = GRAPHS_DIR / filename
        time_mark = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Add metadata (exclude 'filename' from saved document)
        save_data = {k: v for k, v in pylink_data.items() if k != 'filename'}
        save_data['saved_at'] = time_mark

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f'Saved pylink graph to {save_path}')

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

        GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

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

        save_path = GRAPHS_DIR / filename

        # Add metadata
        time_mark = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_data = {
            **pylink_data,
            'saved_at': time_mark,
        }

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f'Saved pylink graph as {save_path}')

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
        if not GRAPHS_DIR.exists():
            return {
                'status': 'success',
                'files': [],
            }

        files = []
        for f in sorted(GRAPHS_DIR.glob('*.json'), key=lambda x: x.stat().st_mtime, reverse=True):
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
            except Exception:
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
def load_pylink_graph(filename: str | None = None):
    """Load a pylink graph from the graphs directory"""
    try:
        if not GRAPHS_DIR.exists():
            return {
                'status': 'error',
                'message': 'No graphs directory found',
            }

        if filename:
            # Load specific file
            file_path = GRAPHS_DIR / filename
            if not file_path.exists():
                return {
                    'status': 'error',
                    'message': f'File not found: {filename}',
                }
        else:
            # Load most recent file
            files = list(GRAPHS_DIR.glob('*.json'))
            if not files:
                return {
                    'status': 'error',
                    'message': 'No pylink graphs found',
                }
            file_path = max(files, key=lambda f: f.stat().st_mtime)

        with open(file_path) as f:
            graph_data = json.load(f)

        logger.info(f'Loaded pylink graph from {file_path.name}')

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


# Demo target joints mapping (from demo/helpers.py MECHANISMS)
DEMO_TARGET_JOINTS = {
    '4bar': 'coupler_rocker_joint',
    'simple': 'coupler_rocker_joint',
    'intermediate': 'final',
    'complex': 'final_joint',
    'leg': 'toe',
    'walker': None,  # No default for walker
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
            'walker': 'test_graphs/strider.json',
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

        logger.info(f'Loaded demo {name} from {file_path.name}')

        return {
            'status': 'success',
            'name': name,
            'filename': file_path.name,
            'data': demo_data,
            'target_joint': DEMO_TARGET_JOINTS.get(name),  # Include target joint in response
        }

    except Exception as e:
        return {
            'status': 'error',
            'message': f'Failed to load demo: {str(e)}',
        }


@app.post('/get-demo-target-joint')
def get_demo_target_joint(request: dict):
    """
    Get the target joint for a mechanism.

    For demo mechanisms: Uses load_mechanism() to get the target_joint.
    For user-generated mechanisms: Picks a random joint from non-default named joints
    (not "joint1", "joint2", etc.).

    Request body:
        {
            "pylink_data": {...}  # Full pylink document
        }

    Returns:
        {
            "status": "success",
            "target_joint": str | null  # Target joint name
        }
    """
    import uuid
    request_id = str(uuid.uuid4())[:8]

    try:
        pylink_data = request.get('pylink_data', {})

        # First, try to identify if this is a demo mechanism
        from demo.helpers import load_mechanism, MECHANISMS

        # Get mechanism name and joint names
        mechanism_name = pylink_data.get('name', '')
        joint_names = []
        if 'linkage' in pylink_data and 'joints' in pylink_data['linkage']:
            joint_names = list(pylink_data['linkage']['joints'].keys())
        elif 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
            joint_names = [j.get('name', '') for j in pylink_data['pylinkage']['joints'] if j.get('name')]

        # Check if mechanism name matches a known demo type
        mechanism_name_lower = mechanism_name.lower()
        for mechanism_type in MECHANISMS.keys():
            if mechanism_type in mechanism_name_lower or mechanism_name_lower in mechanism_type:
                try:
                    # Load the demo mechanism to get its target joint
                    _, demo_target_joint, _ = load_mechanism(mechanism_type)
                    return {
                        'status': 'success',
                        'target_joint': demo_target_joint,
                    }
                except Exception:
                    # If loading fails, continue
                    pass

        # Also check by characteristic joint names (heuristic for demos)
        joint_names_set = set(joint_names)
        heuristic_match = None
        if 'toe' in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('leg')
                heuristic_match = ('leg', demo_target_joint)
            except Exception:
                pass
        elif 'final_joint' in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('complex')
                heuristic_match = ('complex', demo_target_joint)
            except Exception:
                pass
        elif 'final' in joint_names_set and 'coupler_rocker_joint' not in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('intermediate')
                heuristic_match = ('intermediate', demo_target_joint)
            except Exception:
                pass
        elif 'coupler_rocker_joint' in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('simple')
                heuristic_match = ('simple', demo_target_joint)
            except Exception:
                pass

        if heuristic_match:
            return {
                'status': 'success',
                'target_joint': heuristic_match[1],
            }

        # Not a demo mechanism - pick a deterministic joint from non-default named joints
        # Default names are like "joint1", "joint2", "joint3", "joint_1", "joint_2", "crank", "static"
        import re
        default_name_pattern = re.compile(r'^joint[_]?\d+|crank|static$', re.IGNORECASE)

        # Filter to non-default named joints that are Crank or Revolute
        candidate_joints = []
        for joint_name in joint_names:
            # Skip default-named joints
            if default_name_pattern.match(joint_name):
                continue

            # Check if it's a Crank or Revolute joint
            joint_data = None
            if 'linkage' in pylink_data and 'joints' in pylink_data['linkage']:
                joint_data = pylink_data['linkage']['joints'].get(joint_name)
            elif 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
                for j in pylink_data['pylinkage']['joints']:
                    if j.get('name') == joint_name:
                        joint_data = j
                        break

            if joint_data:
                joint_type = None
                if isinstance(joint_data, dict):
                    joint_type = joint_data.get('type', '')
                elif hasattr(joint_data, 'type'):
                    joint_type = joint_data.type

                if joint_type in ('Crank', 'Revolute'):
                    candidate_joints.append(joint_name)

        # If we found non-default joints, pick one deterministically (sorted, then hash-based)
        if candidate_joints:
            # Sort for consistency, then use hash of mechanism name to pick deterministically
            sorted_candidates = sorted(candidate_joints)
            mechanism_name_hash = hash(mechanism_name) if mechanism_name else 0
            selected_index = abs(mechanism_name_hash) % len(sorted_candidates)
            target_joint = sorted_candidates[selected_index]
            return {
                'status': 'success',
                'target_joint': target_joint,
            }

        # Fallback: pick any Crank or Revolute joint
        for joint_name in joint_names:
            joint_data = None
            if 'linkage' in pylink_data and 'joints' in pylink_data['linkage']:
                joint_data = pylink_data['linkage']['joints'].get(joint_name)
            elif 'pylinkage' in pylink_data and 'joints' in pylink_data['pylinkage']:
                for j in pylink_data['pylinkage']['joints']:
                    if j.get('name') == joint_name:
                        joint_data = j
                        break

            if joint_data:
                joint_type = None
                if isinstance(joint_data, dict):
                    joint_type = joint_data.get('type', '')
                elif hasattr(joint_data, 'type'):
                    joint_type = joint_data.type

                if joint_type in ('Crank', 'Revolute'):
                    return {
                        'status': 'success',
                        'target_joint': joint_name,
                    }

        # No suitable joint found
        return {
            'status': 'success',
            'target_joint': None,
        }

    except Exception as e:
        logger.error(f'Error getting demo target joint [{request_id}]: {e}')
        return {
            'status': 'error',
            'message': f'Failed to get target joint: {str(e)}',
        }

    except Exception as e:
        logger.error(f'Error getting demo target joint: {e}')
        return {
            'status': 'error',
            'message': f'Failed to get demo target joint: {str(e)}',
        }


@app.post('/compute-pylink-trajectory')
def compute_pylink_trajectory(request: dict):
    """
    Compute joint trajectories from PylinkDocument format.

    This endpoint takes the pylink graph data (same format as save/load),
    converts it to a Mechanism at the API boundary, runs the simulation,
    and returns the positions of each joint at each timestep.

    Request body:
        {
            "name": "...",
            "linkage": {
                "nodes": {...},
                "edges": {...}
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

        # Convert ONCE at API boundary
        n_steps = request.get('n_steps', 12)
        mechanism = create_mechanism_from_request(request, n_steps=n_steps)

        # Run simulation and check for validity
        # Use simulate() directly to check for inf/nan before converting to dict
        trajectory_array = mechanism.simulate()

        # Check if mechanism is solvable (no inf or nan values)
        has_nan = np.isnan(trajectory_array).any()
        has_inf = np.isinf(trajectory_array).any()

        if has_nan or has_inf:
            # Mechanism is unsolvable - return error instead of invalid trajectories
            error_reasons = []
            if has_nan:
                error_reasons.append('NaN values in trajectory')
            if has_inf:
                error_reasons.append('infinite values in trajectory')

            error_msg = f'Unsolvable mechanism: {", ".join(error_reasons)}. '
            error_msg += 'This may be due to over-constrained geometry, invalid link lengths, or impossible joint positions.'

            return {
                'status': 'error',
                'error_type': 'unsolvable_mechanism',
                'message': error_msg,
                'trajectories': None,  # Don't return invalid trajectories
            }

        # Build exactly n_steps with frame 0 = doc state; check closure and set trajectory_did_not_close when not closed.
        initial_row = np.asarray(mechanism._initial_positions, dtype=float).reshape(1, -1, 2)
        trajectory_array, trajectory_did_not_close = _build_trajectory_with_initial(
            trajectory_array, initial_row, n_steps, check_closed=True,
        )

        # Mechanism is solvable - convert array to dict format
        # Build trajectories dict from the already-computed trajectory_array
        trajectories = {}
        for i, joint_name in enumerate(mechanism._joint_names):
            converted_positions = []
            for step in range(n_steps):
                x, y = trajectory_array[step, i, 0], trajectory_array[step, i, 1]
                # Ensure we have Python floats, not numpy types
                converted_positions.append([float(x), float(y)])
            trajectories[joint_name] = converted_positions

        # Build joint_types from mechanism linkage
        joint_types = {}
        for joint in mechanism.linkage.joints:
            if isinstance(joint, Static):
                joint_types[joint.name] = 'Static'
            elif isinstance(joint, Crank):
                joint_types[joint.name] = 'Crank'
            else:
                joint_types[joint.name] = 'Revolute'

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        response = {
            'status': 'success',
            'message': f'Computed {n_steps} trajectory steps for {len(trajectories)} joints',
            'trajectories': trajectories,
            'n_steps': n_steps,
            'execution_time_ms': execution_time_ms,
            'joint_types': joint_types,
            'trajectory_did_not_close': bool(trajectory_did_not_close),
        }

        # Sanitize to handle any edge cases (though we've already validated)
        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error computing pylink trajectory: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Failed to compute trajectory: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


@app.post('/compute-pylink-trajectories-batch')
def compute_pylink_trajectories_batch(request: dict):
    """
    Compute joint trajectories for multiple mechanism variants in one request.

    Request body:
        {
            "requests": [
                { ...doc (same as single endpoint), "n_steps": 12 },
                ...
            ]
        }

    Returns:
        {
            "results": [
                { "status": "success"|"error", "trajectories": {...}?, "n_steps"?: int, "message"?: str },
                ...
            ]
        }
    """
    try:
        requests_list = request.get('requests', [])
        if not isinstance(requests_list, list):
            return {'status': 'error', 'message': 'Missing or invalid "requests" array'}
        results: list[dict[str, Any]] = []
        for i, req in enumerate(requests_list):
            try:
                n_steps = req.get('n_steps', 12)
                mechanism = create_mechanism_from_request(req, n_steps=n_steps)
                trajectory_array = mechanism.simulate()
                has_nan = np.isnan(trajectory_array).any()
                has_inf = np.isinf(trajectory_array).any()
                if has_nan or has_inf:
                    results.append({
                        'status': 'error',
                        'message': 'Unsolvable mechanism (NaN or infinite values)',
                        'trajectories': None,
                    })
                    continue
                initial_row = np.asarray(mechanism._initial_positions, dtype=float).reshape(1, -1, 2)
                trajectory_array, _ = _build_trajectory_with_initial(
                    trajectory_array, initial_row, n_steps, check_closed=False,
                )
                trajectories = {}
                for j, joint_name in enumerate(mechanism._joint_names):
                    positions = []
                    for step in range(n_steps):
                        x, y = trajectory_array[step, j, 0], trajectory_array[step, j, 1]
                        positions.append([float(x), float(y)])
                    trajectories[joint_name] = positions
                joint_types = {}
                for joint in mechanism.linkage.joints:
                    if isinstance(joint, Static):
                        joint_types[joint.name] = 'Static'
                    elif isinstance(joint, Crank):
                        joint_types[joint.name] = 'Crank'
                    else:
                        joint_types[joint.name] = 'Revolute'
                results.append({
                    'status': 'success',
                    'trajectories': trajectories,
                    'n_steps': n_steps,
                    'joint_types': joint_types,
                })
            except Exception as e:
                results.append({
                    'status': 'error',
                    'message': str(e),
                    'trajectories': None,
                })
        return sanitize_for_json({'results': results})
    except Exception as e:
        logger.error(f'Error in compute-pylink-trajectories-batch: {e}')
        traceback.print_exc()
        return sanitize_for_json({
            'status': 'error',
            'message': str(e),
            'results': None,
        })


@app.post('/compute-link-z-levels')
def compute_link_z_levels_endpoint(request: dict):
    """
    Compute integer z-levels for each link (layer assignment).

    Request body: current document with linkage (and optional meta, n_steps,
    drawn_objects / drawnObjects for polygon-aware z-levels).
    Returns: { "status": "success", "assignments": [ { linkId: zLevel, ... } ], "n_steps": int,
    "polygon_z_levels": { polygonId: zLevel } (when drawn_objects provided) }
    or { "status": "error", "message": "..." }.
    """
    try:
        from form_tools import compute_link_z_levels
        from form_tools.z_level import config_from_request
        from form_tools import polygon_utils as _polygon_utils

        n_steps_raw = request.get('n_steps', 32)
        n_steps: int = 32 if n_steps_raw is None else int(n_steps_raw)
        drawn_objects = request.get('drawn_objects') or request.get('drawnObjects') or []
        polygons_for_z = [
            o for o in drawn_objects
            if o.get('type') == 'polygon' and o.get('id') and (o.get('contained_links') or [])
        ]
        trajectories = request.get('trajectories')
        if isinstance(trajectories, dict) and trajectories:
            trajectories = {
                str(k): [[float(p[0]), float(p[1])] for p in v] if isinstance(v, (list, tuple)) else v
                for k, v in trajectories.items()
            }
        else:
            trajectories = None

        fixed_entity_z_levels = request.get('fixed_entity_z_levels')
        if isinstance(fixed_entity_z_levels, dict):
            fixed_entity_z_levels = {str(k): int(v) for k, v in fixed_entity_z_levels.items()}
        else:
            fixed_entity_z_levels = None

        raw_z_cfg = request.get('z_level_config')
        z_level_config = config_from_request(raw_z_cfg)
        try:
            logger.info('compute-link-z-levels: raw z_level_config from client=%s', raw_z_cfg)
            if z_level_config is not None:
                logger.info(
                    'compute-link-z-levels: parsed z_level_config=%s',
                    asdict(z_level_config),
                )
            else:
                logger.info('compute-link-z-levels: no usable z_level_config in request; solver uses defaults')
        except Exception as log_exc:
            logger.debug('compute-link-z-levels: could not log z_level_config: %s', log_exc)

        extra_entity_conflict_pairs = []
        linkage_for_z = request.get('linkage')
        if not linkage_for_z and isinstance(request.get('pylink_data'), dict):
            linkage_for_z = request.get('pylink_data', {}).get('linkage')

        # Polygon-aware conflicts (overlap + connector joint vs body) need trajectories.
        if polygons_for_z and len(polygons_for_z) >= 2 and isinstance(linkage_for_z, dict) and not trajectories:
            try:
                from pylink_tools.hypergraph_adapter import sync_hypergraph_distances
                from pylink_tools.mechanism import create_mechanism_from_dict

                doc = dict(request)
                doc['n_steps'] = n_steps
                if 'linkage' in doc and 'nodes' in doc.get('linkage', {}):
                    doc = sync_hypergraph_distances(doc, verbose=False)
                mech = create_mechanism_from_dict(doc, n_steps=n_steps)
                traj_d = mech.simulate_dict()
                trajectories = {
                    str(k): [[float(p[0]), float(p[1])] for p in seq]
                    for k, seq in traj_d.items()
                }
            except Exception as e:
                logger.warning('compute-link-z-levels: simulate for polygon conflicts failed: %s', e)

        if polygons_for_z and trajectories and len(polygons_for_z) >= 2 and isinstance(linkage_for_z, dict):
            try:
                margin_units = request.get('margin_units')
                if margin_units is not None:
                    margin_units = float(margin_units)
                extra_entity_conflict_pairs = _polygon_utils.build_polygon_entity_conflict_pairs(
                    polygons_for_z, linkage_for_z, trajectories,
                    margin_fraction=0.05,
                    margin_units=margin_units,
                )
                if extra_entity_conflict_pairs:
                    logger.info(
                        'compute-link-z-levels: added %d polygon-aware conflict pair(s)',
                        len(extra_entity_conflict_pairs),
                    )
            except Exception:
                pass

        result = compute_link_z_levels(
            mechanism=None,
            path=None,
            pylink_data=request,
            n_steps=n_steps,
            use_trajectory_conflicts=True,
            max_assignments=1,
            drawn_objects=polygons_for_z if polygons_for_z else None,
            trajectories=trajectories,
            extra_entity_conflict_pairs=extra_entity_conflict_pairs if extra_entity_conflict_pairs else None,
            fixed_entity_z_levels=fixed_entity_z_levels,
            z_level_config=z_level_config,
        )
        if isinstance(result, tuple):
            assignments, polygon_z_levels = result
            return sanitize_for_json({
                'status': 'success',
                'assignments': assignments,
                'n_steps': n_steps,
                'polygon_z_levels': polygon_z_levels,
            })
        return sanitize_for_json({
            'status': 'success',
            'assignments': result,
            'n_steps': n_steps,
        })
    except Exception as e:
        logger.error(f'Error in compute-link-z-levels: {e}')
        traceback.print_exc()
        return sanitize_for_json({
            'status': 'error',
            'message': str(e),
            'assignments': None,
        })


@app.post('/create-polygons-from-rigid-groups')
def create_polygons_from_rigid_groups_endpoint(request: dict):
    """
    Compute bounding polygons and z-levels for the frontend to create drawn objects.

    Request body: linkage, meta, n_steps; optional trajectories (joint_name -> [[x,y],...])
    and n_steps (effective step count). If trajectories is provided, it is used instead of
    re-simulating so polygons match the frontend's displayed trajectory exactly.
    Optional: mode = "rigid_groups" (default) or "per_link"; margin_fraction, margin_units.

    mode "rigid_groups": Detect rigid groups, one polygon per group (groups with only one
    link are skipped; those links get link forms in per_link mode for correct z sandwiching).
    mode "per_link": One polygon per link not already part of a multi-link rigid group.

    Returns: {
        "status": "success",
        "assignments": { linkId: zLevel },
        "polygon_z_levels": { polygonId: zLevel },
        "suggested_polygons": [ { "polygon_id", "points", "contained_links", "z_level" }, ... ]
    }
    """
    try:
        from form_tools import polygon_utils
        from form_tools import compute_link_z_levels
        from form_tools.z_level import config_from_request

        pylink_data = request.get('pylink_data', request)
        linkage = pylink_data.get('linkage') if isinstance(pylink_data, dict) else None
        if not linkage:
            return sanitize_for_json({
                'status': 'error',
                'message': 'Missing linkage in request.',
                'assignments': None,
                'suggested_polygons': None,
            })

        mode = request.get('mode', 'rigid_groups')
        if mode not in ('rigid_groups', 'per_link'):
            mode = 'rigid_groups'

        # Use frontend trajectory when provided so polygons match displayed trajectory
        trajectories_in = request.get('trajectories')
        n_steps_effective = request.get('n_steps')
        trajectories: dict[str, list[list[float]]] | None = None
        if isinstance(trajectories_in, dict) and isinstance(n_steps_effective, int) and n_steps_effective > 0:
            trajectories = {}
            for jname, positions in trajectories_in.items():
                if not isinstance(positions, (list, tuple)) or len(positions) < n_steps_effective:
                    trajectories = None
                    break
                trajectories[str(jname)] = [
                    [float(p[0]), float(p[1])] for p in positions[:n_steps_effective]
                ]
            if trajectories:
                n_steps = n_steps_effective
        else:
            trajectories = None

        if trajectories is None:
            n_steps = request.get('n_steps', 32)
            mechanism = create_mechanism_from_request(request, n_steps=n_steps)
            trajectory_array = mechanism.simulate()

            has_nan = np.isnan(trajectory_array).any()
            has_inf = np.isinf(trajectory_array).any()
            if has_nan or has_inf:
                return sanitize_for_json({
                    'status': 'error',
                    'message': 'Unsolvable mechanism (NaN or infinite values in trajectory).',
                    'assignments': None,
                    'suggested_polygons': None,
                })

            initial_row = np.asarray(mechanism._initial_positions, dtype=float).reshape(1, -1, 2)
            trajectory_array, _ = _build_trajectory_with_initial(
                trajectory_array, initial_row, n_steps, check_closed=False,
            )
            n_steps_effective = n_steps

            # Build trajectory_dict: joint_name -> list of [x, y]
            trajectories = {}
            for i, joint_name in enumerate(mechanism._joint_names):
                positions = []
                for step in range(n_steps_effective):
                    x, y = trajectory_array[step, i, 0], trajectory_array[step, i, 1]
                    positions.append([float(x), float(y)])
                trajectories[joint_name] = positions

        margin_fraction = float(request.get('margin_fraction', 0.05))
        margin_units = request.get('margin_units')
        if margin_units is not None:
            margin_units = float(margin_units)
        initial_step = 0
        n_steps_int: int = n_steps_effective if isinstance(n_steps_effective, int) else 32
        z_level_config = config_from_request(request.get('z_level_config'))
        skip_existing_forms = request.get('skip_existing_forms', True)
        existing_drawn = request.get('existing_drawn_objects') or []
        traj_typed = cast(dict[str, list[list[float] | tuple[float, float]]], trajectories)

        def _drawn_objects_for_z(existing: list[Any], new_polygons: list[dict]) -> list[dict]:
            """Union of canvas/synthetic polygons and new suggestions for one z-level solve."""
            seen: set[str] = set()
            out: list[dict] = []
            for o in existing:
                if not isinstance(o, dict) or o.get('type') != 'polygon':
                    continue
                pid = o.get('id')
                cl = o.get('contained_links') or []
                if not pid or not cl:
                    continue
                if pid in seen:
                    continue
                seen.add(str(pid))
                row: dict = {'id': str(pid), 'type': 'polygon', 'contained_links': list(cl)}
                pts = o.get('points')
                if isinstance(pts, list) and len(pts) >= 3:
                    row['points'] = pts
                out.append(row)
            for o in new_polygons:
                pid = o.get('id')
                if not pid or str(pid) in seen:
                    continue
                seen.add(str(pid))
                out.append(dict(o))
            return out

        covered_link_ids: set[str] = set()
        if skip_existing_forms and existing_drawn:
            for o in existing_drawn:
                if isinstance(o, dict) and o.get('type') == 'polygon':
                    for lid in o.get('contained_links') or []:
                        covered_link_ids.add(str(lid))

        rigid_groups_all = polygon_utils.detect_rigid_groups(
            linkage, cast(dict[str, list[list[float] | tuple[float, float]]], trajectories),
        )
        links_in_multi_rigid: set[str] = set()
        for g in rigid_groups_all:
            if len(g) >= 2:
                links_in_multi_rigid.update(str(lid) for lid in g)

        if mode == 'per_link':
            edges = linkage.get('edges') or {}
            link_ids = [
                lid for lid in edges.keys()
                if lid not in links_in_multi_rigid
            ]
            if skip_existing_forms and covered_link_ids:
                link_ids = [lid for lid in link_ids if lid not in covered_link_ids]
            if not link_ids:
                return sanitize_for_json({
                    'status': 'success',
                    'assignments': {},
                    'polygon_z_levels': {},
                    'suggested_polygons': [],
                })

            def _link_form_id(lid: str) -> str:
                return f'link_form_{lid[6:]}' if lid.startswith('link_') else f'link_form_{lid}'

            suggested_drawn = [
                {'id': _link_form_id(lid), 'type': 'polygon', 'contained_links': [lid]}
                for lid in link_ids
            ]
            drawn_for_z = _drawn_objects_for_z(existing_drawn, suggested_drawn)
            extra_pairs: list[tuple[str, str]] = []
            if len(drawn_for_z) >= 2:
                try:
                    extra_pairs = polygon_utils.build_polygon_entity_conflict_pairs(
                        drawn_for_z,
                        linkage,
                        traj_typed,
                        margin_fraction=margin_fraction,
                        margin_units=margin_units,
                    )
                except Exception as e:
                    logger.warning('create-polygons-from-rigid-groups: polygon conflict pairs: %s', e)
            result = compute_link_z_levels(
                pylink_data=request,
                n_steps=n_steps_int,
                use_trajectory_conflicts=True,
                max_assignments=1,
                drawn_objects=drawn_for_z,
                trajectories=trajectories,
                extra_entity_conflict_pairs=extra_pairs if extra_pairs else None,
                z_level_config=z_level_config,
            )
            if not isinstance(result, tuple) or len(result) != 2:
                return sanitize_for_json({
                    'status': 'error',
                    'message': 'Z-level computation failed for per-link polygons.',
                    'assignments': None,
                    'suggested_polygons': None,
                })
            assignments, polygon_z_levels = result
            suggested_polygons = []
            for lid in link_ids:
                pid = _link_form_id(lid)
                points = polygon_utils.bounding_polygon_for_links(
                    [lid], linkage,
                    cast(dict[str, list[list[float] | tuple[float, float]]], trajectories),
                    margin_fraction=margin_fraction,
                    margin_units=margin_units,
                    step=initial_step,
                )
                if len(points) >= 3:
                    z_level = polygon_z_levels.get(pid, 0)
                    suggested_polygons.append({
                        'polygon_id': pid,
                        'points': points,
                        'contained_links': [lid],
                        'z_level': z_level,
                    })
            return sanitize_for_json({
                'status': 'success',
                'assignments': assignments,
                'polygon_z_levels': polygon_z_levels,
                'suggested_polygons': suggested_polygons,
            })

        # mode == 'rigid_groups'
        # Only true rigid bodies (2+ links); singleton "groups" are connector links → per_link forms.
        rigid_groups = [g for g in rigid_groups_all if len(g) >= 2]
        if not rigid_groups:
            return sanitize_for_json({
                'status': 'success',
                'assignments': {},
                'polygon_z_levels': {},
                'suggested_polygons': [],
            })
        if skip_existing_forms and covered_link_ids:
            rigid_groups = [g for g in rigid_groups if not (set(g) <= covered_link_ids)]
        if not rigid_groups:
            return sanitize_for_json({
                'status': 'success',
                'assignments': {},
                'polygon_z_levels': {},
                'suggested_polygons': [],
            })

        suggested_drawn = [
            {'id': f'rigid_group_{i}', 'type': 'polygon', 'contained_links': list(g)}
            for i, g in enumerate(rigid_groups)
        ]
        drawn_for_z_rigid = _drawn_objects_for_z(existing_drawn, suggested_drawn)
        extra_pairs_rigid: list[tuple[str, str]] = []
        if len(drawn_for_z_rigid) >= 2:
            try:
                extra_pairs_rigid = polygon_utils.build_polygon_entity_conflict_pairs(
                    drawn_for_z_rigid,
                    linkage,
                    traj_typed,
                    margin_fraction=margin_fraction,
                    margin_units=margin_units,
                )
            except Exception as e:
                logger.warning('create-polygons-from-rigid-groups: polygon conflict pairs: %s', e)
        result = compute_link_z_levels(
            pylink_data=request,
            n_steps=n_steps_int,
            use_trajectory_conflicts=True,
            max_assignments=1,
            drawn_objects=drawn_for_z_rigid,
            trajectories=trajectories,
            extra_entity_conflict_pairs=extra_pairs_rigid if extra_pairs_rigid else None,
            z_level_config=z_level_config,
        )
        if not isinstance(result, tuple) or len(result) != 2:
            return sanitize_for_json({
                'status': 'error',
                'message': 'Z-level computation failed for rigid groups.',
                'assignments': None,
                'suggested_polygons': None,
            })
        assignments, polygon_z_levels = result

        suggested_polygons = []
        for i, group in enumerate(rigid_groups):
            pid = f'rigid_group_{i}'
            points = polygon_utils.bounding_polygon_for_links(
                list(group), linkage,
                cast(dict[str, list[list[float] | tuple[float, float]]], trajectories),
                margin_fraction=margin_fraction,
                margin_units=margin_units,
                step=initial_step,
            )
            z_level = polygon_z_levels.get(pid, 0)
            suggested_polygons.append({
                'polygon_id': pid,
                'points': points,
                'contained_links': list(group),
                'z_level': z_level,
            })

        return sanitize_for_json({
            'status': 'success',
            'assignments': assignments,
            'polygon_z_levels': polygon_z_levels,
            'suggested_polygons': suggested_polygons,
        })
    except Exception as e:
        logger.error(f'Error in create-polygons-from-rigid-groups: {e}')
        traceback.print_exc()
        return sanitize_for_json({
            'status': 'error',
            'message': str(e),
            'assignments': None,
            'suggested_polygons': None,
        })


def _pylink_from_request(request: dict) -> tuple[dict, dict]:
    """Extract linkage and meta from request (pylink_data or flat)."""
    pylink_data = request.get('pylink_data', request)
    if not isinstance(pylink_data, dict):
        raise ValueError('Invalid request: expected dict with linkage or pylink_data')
    linkage = pylink_data.get('linkage')
    if not isinstance(linkage, dict):
        raise ValueError('Invalid request: missing or invalid linkage')
    meta = pylink_data.get('meta') or {}
    return linkage, meta


@app.post('/merge-polygon')
def merge_polygon_endpoint(request: dict):
    """
    Create/update polygon association from geometry: find all links whose endpoints
    lie inside the polygon, set primary link (first contained), return polygon payload.

    Containment uses linkage.nodes positions only (document / initial step 0). Caller
    must pass polygon_points and linkage at the same configuration (e.g. step 0) so
    create-polygons-from-rigid-groups and merge stay consistent.

    Request: pylink_data (or flat) with linkage, meta; polygon_id; polygon_points (list of [x,y]);
    optional selected_link_name: if set and that link is inside the polygon, contained_links and
    primary attachment are restricted to that link only (multi-link enclosure otherwise returns all);
    optional restrict_to_links (list of link IDs): if set, contained_links is restricted to this set only (e.g. rigid group).
    Returns: { status, polygon: { contained_links, mergedLinkName, ..., selected_link_fully_inside? } }
    """
    logger.info('merge-polygon: request received polygon_id=%s', request.get('polygon_id'))
    try:
        from form_tools import polygon_utils
        linkage, meta = _pylink_from_request(request)
        polygon_id = request.get('polygon_id')
        selected_link_name = request.get('selected_link_name')
        polygon_points = request.get('polygon_points')
        drawn_objects = request.get('drawn_objects') or request.get('drawnObjects')
        if polygon_points is None and drawn_objects and polygon_id:
            for obj in drawn_objects:
                if obj.get('id') == polygon_id and obj.get('type') == 'polygon':
                    polygon_points = obj.get('points')
                    break
        if not polygon_id:
            logger.warning('merge-polygon: missing polygon_id')
            return sanitize_for_json({'status': 'error', 'message': 'missing polygon_id', 'polygon': None})
        if not polygon_points or len(polygon_points) < 3:
            logger.warning('merge-polygon: invalid polygon_points for polygon_id=%s', polygon_id)
            return sanitize_for_json({'status': 'error', 'message': 'invalid or missing polygon_points', 'polygon': None})

        logger.info('merge-polygon: polygon_id=%s, points=%d', polygon_id, len(polygon_points))
        restrict_to_links = request.get('restrict_to_links')
        if restrict_to_links is not None and not isinstance(restrict_to_links, list):
            restrict_to_links = None
        contained = polygon_utils.contained_links(linkage, polygon_points)
        if restrict_to_links is not None:
            allowed = set(restrict_to_links)
            contained = [lid for lid in contained if lid in allowed]
        if not contained:
            logger.info('merge-polygon: no fully bounded links in polygon_id=%s', polygon_id)
            return sanitize_for_json({
                'status': 'success',
                'polygon': {
                    'polygon_id': polygon_id,
                    'contained_links': [],
                    'mergedLinkName': None,
                    'mergedLinkOriginalStart': None,
                    'mergedLinkOriginalEnd': None,
                    'fill_color': None,
                    'stroke_color': None,
                    'selected_link_fully_inside': None if not selected_link_name else False,
                },
            })

        if selected_link_name and selected_link_name not in contained:
            logger.warning(
                'merge-polygon: selected_link_name=%s is not fully inside polygon_id=%s; contained_links=%s',
                selected_link_name, polygon_id, contained,
            )
        elif selected_link_name and selected_link_name in contained:
            # User picked a specific link in merge mode — attach only that link, not every enclosed link.
            contained = [selected_link_name]

        primary = contained[0]
        endpoints = polygon_utils.get_link_endpoints(linkage, primary)
        if not endpoints:
            logger.error('merge-polygon: could not get endpoints for primary link %s', primary)
            return sanitize_for_json({'status': 'error', 'message': f'primary link {primary} has no positions', 'polygon': None})
        start_pt, end_pt = endpoints
        meta_edges = meta.get('edges') or {}
        link_meta = meta_edges.get(primary) or {}
        fill_color = link_meta.get('color') or '#2ca02c'
        stroke_color = fill_color

        selected_link_fully_inside = (selected_link_name in contained) if selected_link_name else None
        out = {
            'status': 'success',
            'polygon': {
                'polygon_id': polygon_id,
                'contained_links': contained,
                'mergedLinkName': primary,
                'mergedLinkOriginalStart': list(start_pt),
                'mergedLinkOriginalEnd': list(end_pt),
                'fill_color': fill_color,
                'stroke_color': stroke_color,
                'selected_link_fully_inside': selected_link_fully_inside,
            },
        }
        logger.info(
            'merge-polygon: polygon_id=%s contained_links=%s primary=%s selected_link_fully_inside=%s',
            polygon_id, len(contained), primary, selected_link_fully_inside,
        )
        return sanitize_for_json(out)
    except ValueError as e:
        logger.warning('merge-polygon: invalid request: %s', e)
        return sanitize_for_json({'status': 'error', 'message': str(e), 'polygon': None})
    except Exception as e:
        logger.error('merge-polygon: %s', e)
        traceback.print_exc()
        return sanitize_for_json({'status': 'error', 'message': str(e), 'polygon': None})


@app.post('/merge-two-polygons')
def merge_two_polygons_endpoint(request: dict):
    """
    Merge two polygons into one outer bounding polygon (union geometry).
    Interior boundaries between the two are removed; result is a single polygon
    that perfectly bounds both. Uses first polygon's id for the result.

    Request: pylink_data (or flat) with linkage, meta;
             polygon_id_a, polygon_points_a (first polygon - keep its id);
             polygon_id_b, polygon_points_b (second polygon - removed by frontend).
    Returns: { status, merged_polygon: { polygon_id, points, contained_links } }
    Frontend applies first polygon's z_level and color to the result.
    """
    logger.info(
        'merge-two-polygons: request polygon_id_a=%s polygon_id_b=%s',
        request.get('polygon_id_a'),
        request.get('polygon_id_b'),
    )
    try:
        from form_tools import polygon_utils
        linkage, _meta = _pylink_from_request(request)
        polygon_id_a = request.get('polygon_id_a')
        polygon_id_b = request.get('polygon_id_b')
        points_a = request.get('polygon_points_a')
        points_b = request.get('polygon_points_b')
        if not polygon_id_a or not points_a or len(points_a) < 3:
            return sanitize_for_json({
                'status': 'error',
                'message': 'missing or invalid polygon_id_a / polygon_points_a',
                'merged_polygon': None,
            })
        if not polygon_id_b or not points_b or len(points_b) < 3:
            return sanitize_for_json({
                'status': 'error',
                'message': 'missing or invalid polygon_id_b / polygon_points_b',
                'merged_polygon': None,
            })
        pts_a = [(float(p[0]), float(p[1])) for p in points_a]
        pts_b = [(float(p[0]), float(p[1])) for p in points_b]
        merged_points = polygon_utils.merge_two_polygons_geometry(pts_a, pts_b)
        if not merged_points:
            return sanitize_for_json({
                'status': 'error',
                'message': 'could not compute merged polygon geometry',
                'merged_polygon': None,
            })
        contained = polygon_utils.contained_links(linkage, merged_points)
        out = {
            'status': 'success',
            'merged_polygon': {
                'polygon_id': polygon_id_a,
                'points': [list(p) for p in merged_points],
                'contained_links': contained,
            },
        }
        logger.info(
            'merge-two-polygons: merged polygon_id=%s points=%d contained_links=%d',
            polygon_id_a,
            len(merged_points),
            len(contained),
        )
        return sanitize_for_json(out)
    except ValueError as e:
        logger.warning('merge-two-polygons: invalid request: %s', e)
        return sanitize_for_json({
            'status': 'error',
            'message': str(e),
            'merged_polygon': None,
        })
    except Exception as e:
        logger.error('merge-two-polygons: %s', e)
        traceback.print_exc()
        return sanitize_for_json({
            'status': 'error',
            'message': str(e),
            'merged_polygon': None,
        })


@app.post('/find-associated-polygons')
def find_associated_polygons_endpoint(request: dict):
    """
    Recompute containment for current positions (e.g. after drag).
    Request: pylink_data (or flat) with linkage, meta; drawn_objects (list of polygons with points).
    Returns: { status, polygons: { polygon_id: { contained_links: [...], all_inside: bool } } }
    """
    drawn_objects = request.get('drawn_objects') or request.get('drawnObjects') or []
    polygons_in = [o for o in drawn_objects if o.get('type') == 'polygon' and o.get('points') and len(o.get('points', [])) >= 3]
    logger.info('find-associated-polygons: request received n_polygons=%d', len(polygons_in))
    try:
        from form_tools import polygon_utils
        linkage, _meta = _pylink_from_request(request)
        result = {}
        for obj in polygons_in:
            pid = obj.get('id')
            if not pid:
                continue
            points = obj.get('points') or []
            poly_tuples = [(float(p[0]), float(p[1])) for p in points]
            contained = polygon_utils.contained_links(linkage, points)
            all_inside = True
            existing = obj.get('contained_links')
            if not existing and obj.get('mergedLinkName'):
                existing = [obj.get('mergedLinkName')]
            for link_id in (existing or contained):
                ep = polygon_utils.get_link_endpoints(linkage, link_id)
                if not ep:
                    all_inside = False
                    break
                if not polygon_utils.is_point_in_polygon(ep[0], poly_tuples) or not polygon_utils.is_point_in_polygon(ep[1], poly_tuples):
                    all_inside = False
                    break
            result[pid] = {'contained_links': contained, 'all_inside': all_inside}
            if not all_inside:
                logger.info('find-associated-polygons: polygon_id=%s all_inside=False (link endpoints outside)', pid)
        logger.info('find-associated-polygons: processed %d polygons', len(result))
        return sanitize_for_json({'status': 'success', 'polygons': result})
    except ValueError as e:
        logger.warning('find-associated-polygons: invalid request: %s', e)
        return sanitize_for_json({'status': 'error', 'message': str(e), 'polygons': {}})
    except Exception as e:
        logger.error('find-associated-polygons: %s', e)
        traceback.print_exc()
        return sanitize_for_json({'status': 'error', 'message': str(e), 'polygons': {}})


@app.post('/validate-polygon-rigidity')
def validate_polygon_rigidity_endpoint(request: dict):
    """
    Validate that each polygon's contained links form a rigid body (relative angles
    at shared joints constant over time). Uses precomputed trajectory; no simulation.

    Request: trajectories (joint_name -> [[x,y], ...]), linkage (nodes, edges),
             drawn_objects (list of polygons with id, contained_links).
    Returns: { status, polygons: { polygon_id: { rigid_valid: bool, message?: str } } }
    """
    trajectories = request.get('trajectories')
    linkage = request.get('linkage')
    drawn_objects = request.get('drawn_objects') or request.get('drawnObjects') or []
    if not trajectories or not isinstance(trajectories, dict):
        return sanitize_for_json({
            'status': 'error',
            'message': 'missing or invalid trajectories (expected dict joint_name -> list of [x,y])',
            'polygons': {},
        })
    if not linkage or not isinstance(linkage, dict):
        return sanitize_for_json({
            'status': 'error',
            'message': 'missing or invalid linkage',
            'polygons': {},
        })
    from form_tools import polygon_utils
    result: dict[str, dict[str, Any]] = {}
    polygons_in = [o for o in drawn_objects if o.get('type') == 'polygon' and o.get('id')]
    for obj in polygons_in:
        pid = obj.get('id')
        contained = obj.get('contained_links') or []
        if len(contained) < 2:
            result[pid] = {'rigid_valid': True}
            continue
        rigid_valid, message = polygon_utils.validate_polygon_rigidity(
            linkage, contained, trajectories,
        )
        result[pid] = {'rigid_valid': rigid_valid}
        if message:
            result[pid]['message'] = message
    return sanitize_for_json({'status': 'success', 'polygons': result})


@app.post('/validate-mechanism')
def validate_mechanism_endpoint(request: dict):
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
    from target_gen.achievable_target import verify_mechanism_viable

    try:
        # Convert ONCE at API boundary
        mechanism = create_mechanism_from_request(request)

        # Validate using Mechanism
        is_valid = verify_mechanism_viable(mechanism)

        return {
            'status': 'success',
            'valid': is_valid,
            'message': 'Mechanism is valid' if is_valid else 'Mechanism validation failed',
        }

    except Exception as e:
        logger.error(f'Error validating mechanism: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'valid': False,
            'message': f'Validation failed: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


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
                "mech_variation_config": { ... },  # NEW: MechVariationConfig object
                "bounds_factor": 2.0,         # DEPRECATED: How much dimensions can vary (converted to MechVariationConfig)
                "min_length": 0.1,            # DEPRECATED: Minimum link length (ignored, not supported)
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
        verbose = optimization_options.get('verbose', True)

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

        # Extract original metadata from request BEFORE creating mechanism
        # This preserves link colors, node colors, showPath settings, link names, etc.
        # Note: In the future, optimization may add or remove links/nodes, so we'll
        # need to handle metadata merging more carefully (only preserve what still exists)
        pylink_data = request.get('pylink_data', request)
        original_meta = pylink_data.get('meta', {})
        original_name = pylink_data.get('name', 'optimized')
        original_drawn_objects = pylink_data.get('drawnObjects')
        original_components = pylink_data.get('components')
        original_hierarchy = pylink_data.get('hierarchy')
        original_saved_at = pylink_data.get('savedAt')

        # Convert ONCE at API boundary
        n_steps = len(target_positions)
        mechanism = create_mechanism_from_request(request, n_steps=n_steps)

        # Validate the mechanism by running a reference simulation
        try:
            mechanism.simulate()  # Quick validation
        except Exception as e:
            return sanitize_for_json({
                'status': 'error',
                'message': f'Mechanism validation failed: {str(e)}',
                'hint': 'The mechanism may be over-constrained or have invalid geometry',
            })

        # Check that target joint exists
        if target_joint not in mechanism.joint_names:
            return {
                'status': 'error',
                'message': f"Target joint '{target_joint}' not found in mechanism",
                'available_joints': mechanism.joint_names,
            }

        # Get MechVariationConfig from request (new way) or convert old params (backward compatibility)
        mech_variation_config = None
        if 'mech_variation_config' in optimization_options:
            # New way: accept MechVariationConfig directly
            try:
                config_input = optimization_options['mech_variation_config']

                # If it's already a MechVariationConfig instance, use it directly
                if isinstance(config_input, MechVariationConfig):
                    mech_variation_config = config_input
                else:
                    # Convert nested dicts to their respective dataclass instances
                    # Ensure we have a dict to work with
                    if isinstance(config_input, dict):
                        config_dict = config_input.copy()
                    else:
                        # If it's not a dict and not a MechVariationConfig, try to convert to dict
                        # This handles cases where it might be a different type
                        try:
                            config_dict = asdict(config_input) if hasattr(config_input, '__dataclass_fields__') else dict(config_input)
                        except Exception:
                            config_dict = dict(config_input)

                    # Convert dimension_variation dict to DimensionVariationConfig
                    if 'dimension_variation' in config_dict:
                        dim_var = config_dict['dimension_variation']
                        if isinstance(dim_var, dict):
                            # Only convert if it's a dict (not already a DimensionVariationConfig instance)
                            config_dict['dimension_variation'] = DimensionVariationConfig(**dim_var)
                        # If it's already a DimensionVariationConfig instance, leave it as is

                    # Convert static_joint_movement dict to StaticJointMovementConfig
                    if 'static_joint_movement' in config_dict:
                        static_joint = config_dict['static_joint_movement']
                        if isinstance(static_joint, dict):
                            # Only convert if it's a dict (not already a StaticJointMovementConfig instance)
                            config_dict['static_joint_movement'] = StaticJointMovementConfig(**static_joint)
                        # If it's already a StaticJointMovementConfig instance, leave it as is

                    # Convert topology_changes dict to TopologyChangeConfig
                    if 'topology_changes' in config_dict:
                        topology = config_dict['topology_changes']
                        if isinstance(topology, dict):
                            # Only convert if it's a dict (not already a TopologyChangeConfig instance)
                            config_dict['topology_changes'] = TopologyChangeConfig(**topology)
                        # If it's already a TopologyChangeConfig instance, leave it as is

                    # Now instantiate MechVariationConfig with properly converted nested dataclasses
                    mech_variation_config = MechVariationConfig(**config_dict)
            except Exception as e:
                logger.error(f'Failed to create MechVariationConfig: {e}')
                traceback.print_exc()
                return sanitize_for_json({
                    'status': 'error',
                    'message': f'Invalid MechVariationConfig: {str(e)}',
                })
        elif 'bounds_factor' in optimization_options or 'min_length' in optimization_options:
            # Old way: convert bounds_factor/min_length to MechVariationConfig (backward compatibility)
            bounds_factor = optimization_options.get('bounds_factor', 2.0)
            # Note: min_length is not directly supported in MechVariationConfig
            # We'll use bounds_factor as default_variation_range
            # DimensionVariationConfig is already imported at module level
            mech_variation_config = MechVariationConfig(
                dimension_variation=DimensionVariationConfig(
                    default_variation_range=bounds_factor,
                ),
            )
            if verbose:
                logger.warning(f'[DEPRECATED] Using bounds_factor={bounds_factor} (converted to MechVariationConfig)')
                logger.warning('[DEPRECATED] min_length parameter is ignored (not supported in MechVariationConfig)')

        # Get dimension spec from Mechanism (will use mech_variation_config if provided)
        dim_spec = mechanism.get_dimension_bounds_spec()
        if mech_variation_config:
            dim_spec = DimensionBoundsSpec.from_mechanism(mechanism, mech_variation_config)

        # Create target trajectory
        target = TargetTrajectory(
            joint_name=target_joint,
            positions=target_positions,
        )

        # Optional topology variation spec (for SCIP topology optimization)
        topology_variation_spec = None
        if 'topology_variation_spec' in optimization_options:
            try:
                topo_input = optimization_options['topology_variation_spec']
                if isinstance(topo_input, dict):
                    topology_variation_spec = TopologyVariationSpec.from_dict(topo_input)
                else:
                    topology_variation_spec = topo_input
            except Exception as e:
                logger.warning(f'Invalid topology_variation_spec: {e}')

        # Build kwargs based on method
        opt_kwargs = {
            'mechanism': mechanism,  # Pass Mechanism, not pylink_data
            'target': target,
            'dimension_bounds_spec': dim_spec,
            'mech_variation_config': mech_variation_config,  # Pass config for optimizer
            'topology_variation_spec': topology_variation_spec,
            'method': method,
            'verbose': verbose,
        }

        # Add method-specific parameters
        if method == 'pylinkage':
            opt_kwargs['n_particles'] = n_particles
            opt_kwargs['iterations'] = iterations
            w = optimization_options.get('w')
            if w is not None:
                opt_kwargs['w'] = w
            c1 = optimization_options.get('c1')
            if c1 is not None:
                opt_kwargs['c1'] = c1
            c2 = optimization_options.get('c2')
            if c2 is not None:
                opt_kwargs['c2'] = c2
        elif method in ('scipy', 'powell', 'nelder-mead'):
            opt_kwargs['max_iterations'] = max_iterations
            opt_kwargs['tolerance'] = tolerance
        elif method == 'scip':
            opt_kwargs['discretization_steps'] = optimization_options.get('discretization_steps', 20)
            opt_kwargs['time_limit'] = optimization_options.get('time_limit', 300.0)
            opt_kwargs['gap_limit'] = optimization_options.get('gap_limit', 0.01)

        # Run optimization (modifies mechanism in place)
        # Wrap in try-catch to gracefully handle any exceptions
        try:
            result = optimize_trajectory(**opt_kwargs)
        except ValueError as e:
            # Handle validation errors (e.g., dimension mismatch)
            error_msg = str(e)
            logger.error(f'Optimization validation error: {error_msg}')
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': error_msg,
                'execution_time_ms': (time.perf_counter() - start_time) * 1000,
            }
        except Exception as e:
            # Handle any other exceptions during optimization
            error_msg = f'Optimization failed: {str(e)}'
            logger.error(f'Optimization error: {error_msg}')
            traceback.print_exc()
            return {
                'status': 'error',
                'error_type': 'optimization_error',
                'message': error_msg,
                'execution_time_ms': (time.perf_counter() - start_time) * 1000,
            }

        # When optimization reports failure (e.g. unknown method), return error immediately
        # with a user-friendly message so the frontend does not expect optimized_pylink_data.
        if not result.success:
            end_time = time.perf_counter()
            execution_time_ms = (end_time - start_time) * 1000
            message = result.error if result.error else 'Optimization failed'
            if 'Unknown optimization method' in (result.error or ''):
                message = f"Optimization failed: Unknown method '{method}'"
            logger.warning(f'Optimization failed: {result.error}')
            return sanitize_for_json({
                'status': 'error',
                'message': message,
                'execution_time_ms': execution_time_ms,
            })

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Calculate improvement percentage (handle cases where errors might be NaN/Inf)
        improvement = 0.0
        if (
            result.initial_error is not None and
            result.final_error is not None and
            math.isfinite(result.initial_error) and
            math.isfinite(result.final_error) and
            result.initial_error > 0
        ):
            improvement = ((result.initial_error - result.final_error) / result.initial_error) * 100

        # Handle inf/nan for logging - ensure we have valid numbers
        initial_err_str = (
            f'{result.initial_error:.4f}'
            if (result.initial_error is not None and math.isfinite(result.initial_error))
            else str(result.initial_error) if result.initial_error is not None
            else 'N/A'
        )
        final_err_str = (
            f'{result.final_error:.4f}'
            if (result.final_error is not None and math.isfinite(result.final_error))
            else str(result.final_error) if result.final_error is not None
            else 'N/A'
        )

        logger.info(f'Initial error: {initial_err_str}')
        logger.info(f'Final error: {final_err_str}')
        logger.info(f'Improvement: {improvement:.1f}%')
        logger.info(f'Completed in {execution_time_ms:.2f}ms')

        # Build response using OptimizationResult.to_dict() for proper type serialization
        # This ensures all fields are correctly formatted (convergence_history, best_error, etc.)
        result_dict = result.to_dict()

        # Ensure optimized_dimensions match edge distances
        # correctness check. The optimizer reports dimension values
        # in optimized_dimensions (e.g., {"crank_link_distance": 4.953}). These MUST match
        # the actual edge distances in the returned mechanism's linkage.edges.
        # If they don't match, it means:
        # - Mechanism.to_dict() is not correctly applying optimized distances to edges, OR
        # - The dimension_mapping.edge_mapping is incorrect, OR
        # - There's a bug in how we're reading/writing _current_dimensions
        # We validate by:
        # 1. For each dimension in optimized_dimensions, look up its edge_id from edge_mapping
        # 2. Check that edges[edge_id].distance matches the dimension value
        # 3. Raise ValueError with detailed error message if any mismatch is found
        # This validation runs BEFORE returning the response, ensuring we never return
        # inconsistent data to the frontend.
        if result.success:
            # CRITICAL: Fail hard if optimized_pylink_data is missing
            if not result_dict.get('optimized_pylink_data'):
                error_msg = 'CRITICAL ERROR: Optimization succeeded but optimized_pylink_data is missing'
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'traceback': [error_msg],
                }

            # CRITICAL: Validate structure
            opt_data = result_dict.get('optimized_pylink_data')
            if not isinstance(opt_data, dict):
                error_msg = 'CRITICAL ERROR: optimized_pylink_data is not a dict'
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'traceback': [error_msg],
                }

            if 'linkage' not in opt_data or 'nodes' not in opt_data.get('linkage', {}) or 'edges' not in opt_data.get('linkage', {}):
                error_msg = 'CRITICAL ERROR: optimized_pylink_data missing required structure (linkage.nodes or linkage.edges)'
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'traceback': [error_msg],
                }

            # Validate optimized_dimensions match edge distances if both exist
            if result_dict.get('optimized_dimensions'):
                optimized_pylink_data = result_dict.get('optimized_pylink_data')
                optimized_dimensions = result_dict.get('optimized_dimensions', {})
                if optimized_pylink_data is not None:
                    edges = optimized_pylink_data.get('linkage', {}).get('edges', {})

                    # Get dimension_bounds_spec to access edge_mapping
                    mismatches = []
                    validation_details = []

                    if dim_spec and dim_spec.edge_mapping:
                        for dim_name, dim_value in optimized_dimensions.items():
                            if dim_name not in dim_spec.edge_mapping:
                                continue

                            edge_id, prop_name = dim_spec.edge_mapping[dim_name]
                            if edge_id not in edges:
                                mismatches.append(f"Dimension '{dim_name}' maps to edge '{edge_id}' which is missing from returned edges")
                                continue

                            edge_distance = edges[edge_id].get('distance')
                            if edge_distance is None:
                                mismatches.append(f"Dimension '{dim_name}' maps to edge '{edge_id}' which has no distance")
                                continue

                            # Compare with tolerance for floating point
                            tolerance = 1e-6
                            if abs(float(dim_value) - float(edge_distance)) > tolerance:
                                mismatches.append(
                                    f"Dimension '{dim_name}': reported={dim_value}, "
                                    f"edge '{edge_id}' distance={edge_distance}, diff={abs(float(dim_value) - float(edge_distance))}",
                                )

                            validation_details.append({
                                'dim_name': dim_name,
                                'dim_value': dim_value,
                                'edge_id': edge_id,
                                'edge_distance': edge_distance,
                                'matches': abs(float(dim_value) - float(edge_distance)) <= tolerance,
                            })

                    # RAISE EXCEPTION if there are mismatches - this is a critical bug
                    if mismatches:
                        error_msg = (
                            'CRITICAL BUG: optimized_dimensions do not match edge distances in returned mechanism!\n'
                            'Mismatches:\n' + '\n'.join(f'  - {m}' for m in mismatches) + '\n'
                            'This indicates a fundamental bug in Mechanism.to_dict() or dimension mapping.\n'
                            f'optimized_dimensions: {optimized_dimensions}\n'
                            f"edge_distances: {dict((eid, e.get('distance')) for eid, e in edges.items())}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    logger.info(f'✓ Validation passed: All {len(validation_details)} optimized dimensions match edge distances')
        # #endregion

        # Reconcile edge distances to node positions so the payload is self-consistent.
        # Some edges (e.g. rigid/dependent links) are determined by geometry; setting each
        # edge's distance from the two node positions avoids position/distance mismatch
        # on the frontend and lets dependent link lengths be "found" from positions.
        opt_data = result_dict.get('optimized_pylink_data')
        if result.success and opt_data is not None:
            optimized_pylink_data = opt_data
            linkage_data = optimized_pylink_data.get('linkage', {})
            nodes = linkage_data.get('nodes', {})
            edges = linkage_data.get('edges', {})
            reconciled = 0
            for edge_id, edge in edges.items():
                src_id = edge.get('source')
                tgt_id = edge.get('target')
                src_node = nodes.get(src_id) if src_id else None
                tgt_node = nodes.get(tgt_id) if tgt_id else None
                if not src_node or not tgt_node:
                    continue
                pos_src = src_node.get('position')
                pos_tgt = tgt_node.get('position')
                if not pos_src or not pos_tgt or len(pos_src) < 2 or len(pos_tgt) < 2:
                    continue
                dx = float(pos_tgt[0]) - float(pos_src[0])
                dy = float(pos_tgt[1]) - float(pos_src[1])
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    edge['distance'] = dist
                    reconciled += 1
            if reconciled:
                logger.info(f'Reconciled {reconciled} edge distances to node positions (consistent linkage)')

            # Ensure meta structure exists
            if 'meta' not in optimized_pylink_data:
                optimized_pylink_data['meta'] = {'nodes': {}, 'edges': {}}

            optimized_meta = optimized_pylink_data.get('meta', {})
            optimized_nodes = optimized_pylink_data.get('linkage', {}).get('nodes', {})
            optimized_edges = optimized_pylink_data.get('linkage', {}).get('edges', {})

            # Preserve node metadata (colors, showPath, zlevel) for nodes that still exist
            original_node_meta = original_meta.get('nodes', {})
            if 'nodes' not in optimized_meta:
                optimized_meta['nodes'] = {}

            for node_id in optimized_nodes.keys():
                if node_id in original_node_meta:
                    # Preserve all original node metadata (color, zlevel, showPath)
                    optimized_meta['nodes'][node_id] = original_node_meta[node_id].copy()
                    # Update position from optimized node (but keep other metadata)
                    if 'position' in optimized_nodes[node_id]:
                        # Position is already in the node, but we also need it in meta for getJointPosition
                        pos = optimized_nodes[node_id]['position']
                        if isinstance(pos, list) and len(pos) >= 2:
                            # Ensure meta has x, y for backward compatibility with legacy format conversion
                            if 'x' not in optimized_meta['nodes'][node_id]:
                                optimized_meta['nodes'][node_id]['x'] = pos[0]
                            if 'y' not in optimized_meta['nodes'][node_id]:
                                optimized_meta['nodes'][node_id]['y'] = pos[1]

            # Preserve edge metadata (colors, isGround) for edges that still exist
            original_edge_meta = original_meta.get('edges', {})
            if 'edges' not in optimized_meta:
                optimized_meta['edges'] = {}

            for edge_id in optimized_edges.keys():
                if edge_id in original_edge_meta:
                    # Preserve all original edge metadata (color, isGround)
                    optimized_meta['edges'][edge_id] = original_edge_meta[edge_id].copy()

            # Preserve other optional fields (drawnObjects, components, hierarchy, savedAt)
            if original_drawn_objects is not None:
                optimized_pylink_data['drawnObjects'] = original_drawn_objects
            if original_components is not None:
                optimized_pylink_data['components'] = original_components
            if original_hierarchy is not None:
                optimized_pylink_data['hierarchy'] = original_hierarchy
            if original_saved_at is not None:
                optimized_pylink_data['savedAt'] = original_saved_at

            # Preserve original name (or use 'optimized' if not provided)
            if original_name:
                optimized_pylink_data['name'] = original_name
                linkage_dict = optimized_pylink_data.get('linkage')
                if linkage_dict is not None:
                    linkage_dict['name'] = original_name

            logger.info('Preserved original metadata (colors, showPath, link names, etc.)')

        # Ensure optimized_pylink_data exists and is valid
        if result.success:
            if not result_dict.get('optimized_pylink_data'):
                error_msg = 'CRITICAL ERROR: Optimization succeeded but optimized_pylink_data is missing'
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'traceback': [error_msg],
                }

            # CRITICAL: Validate structure
            opt_data = result_dict.get('optimized_pylink_data')
            if not isinstance(opt_data, dict):
                error_msg = 'CRITICAL ERROR: optimized_pylink_data is not a dict'
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'traceback': [error_msg],
                }

            if 'linkage' not in opt_data or 'nodes' not in opt_data.get('linkage', {}) or 'edges' not in opt_data.get('linkage', {}):
                error_msg = 'CRITICAL ERROR: optimized_pylink_data missing required structure (linkage.nodes or linkage.edges)'
                logger.error(error_msg)
                return {
                    'status': 'error',
                    'message': error_msg,
                    'traceback': [error_msg],
                }

            logger.info('✓ Validated optimized_pylink_data structure')

        response = {
            'status': 'success',
            'message': f'Optimization complete: {improvement:.1f}% improvement',
            'result': result_dict,  # Use OptimizationResult.to_dict() output
            'execution_time_ms': execution_time_ms,
        }

        # Sanitize to handle inf/nan values (JSON doesn't support them)
        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error optimizing trajectory: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Optimization failed: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


@app.post('/get-optimizable-dimensions')
def get_optimizable_dimensions(request: dict):
    """
    Get the list of optimizable dimensions for a linkage.

    This is useful for displaying to the user what parameters
    will be adjusted during optimization. Optionally accepts
    DimensionVariationConfig or MechVariationConfig to filter
    or adjust which dimensions are included.

    Request body:
        {
            "pylink_data": { ... },  # Full pylink document
            "dimension_variation_config": {  # Optional: DimensionVariationConfig
                "default_variation_range": 0.5,
                "default_enabled": true,
                "dimension_overrides": {
                    "dimension_name": [enabled: bool, min_pct: float, max_pct: float]
                },
                "exclude_dimensions": ["dim1", "dim2"]
            },
            # OR
            "mech_variation_config": {  # Optional: MechVariationConfig (uses its dimension_variation)
                "dimension_variation": {...},
                ...
            }
        }

    Returns:
        {
            "status": "success",
            "dimension_bounds_spec": {
                # DimensionBoundsSpec.to_dict() format:
                "names": ["A_distance", "B_distance0", "B_distance1"],
                "initial_values": [1.5, 3.5, 2.5],
                "bounds": [[0.75, 3.0], [1.75, 7.0], [1.25, 5.0]],
                "weights": [1.0, 1.0, 1.0] | null,
                "edge_mapping": {...} | null,
                "n_dimensions": 3
            },
            "config_applied": {
                "type": "dimension_variation" | "achievable_target" | "none",
                "config": {...} | null
            }
        }
    """
    import time
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    try:
        if not request:
            elapsed = (time.time() - start_time) * 1000
            return {
                'status': 'error',
                'message': 'Missing request data',
            }

        # Convert ONCE at API boundary
        mechanism = create_mechanism_from_request(request)
        # mechanism_name = getattr(mechanism, 'name', 'unknown')

        # Get dimension spec from Mechanism
        dim_spec = mechanism.get_dimension_bounds_spec()
        # initial_dim_count = len(dim_spec.names) if dim_spec else 0

        # Check for configs and apply filtering if provided
        config_applied: dict[str, str | dict[str, object] | None] = {'type': 'none', 'config': None}

        # Initialize dimension_variation_config to None to avoid "cannot access local variable" error
        # This will be set from mech_variation_config if provided, or from the separate dimension_variation_config field
        dimension_variation_config = request.get('dimension_variation_config')
        mech_variation_config = request.get('mech_variation_config')  # NEW name
        achievable_target_config = request.get('achievable_target_config')  # OLD name (backward compat)

        # Use new name if provided, fall back to old name for backward compatibility
        if mech_variation_config:
            config_data = mech_variation_config
        elif achievable_target_config:
            config_data = achievable_target_config
            logger.warning('[DEPRECATED] Using "achievable_target_config" (use "mech_variation_config" instead)')
        else:
            config_data = None

        if config_data:
            # Extract dimension_variation from MechVariationConfig
            try:
                # If it's already a MechVariationConfig instance, use it directly
                if isinstance(config_data, MechVariationConfig):
                    config = config_data
                else:
                    # Convert nested dicts to their respective dataclass instances
                    # Ensure we have a dict to work with
                    if isinstance(config_data, dict):
                        config_dict = config_data.copy()
                    else:
                        # If it's not a dict and not a MechVariationConfig, try to convert to dict
                        try:
                            config_dict = asdict(config_data) if hasattr(config_data, '__dataclass_fields__') else dict(config_data)
                        except Exception:
                            config_dict = dict(config_data)

                    # Convert dimension_variation dict to DimensionVariationConfig
                    if 'dimension_variation' in config_dict:
                        dim_var = config_dict['dimension_variation']
                        if isinstance(dim_var, dict):
                            # Only convert if it's a dict (not already a DimensionVariationConfig instance)
                            config_dict['dimension_variation'] = DimensionVariationConfig(**dim_var)
                        # If it's already a DimensionVariationConfig instance, leave it as is

                    # Convert static_joint_movement dict to StaticJointMovementConfig
                    if 'static_joint_movement' in config_dict:
                        static_joint = config_dict['static_joint_movement']
                        if isinstance(static_joint, dict):
                            # Only convert if it's a dict (not already a StaticJointMovementConfig instance)
                            config_dict['static_joint_movement'] = StaticJointMovementConfig(**static_joint)
                        # If it's already a StaticJointMovementConfig instance, leave it as is

                    # Convert topology_changes dict to TopologyChangeConfig
                    if 'topology_changes' in config_dict:
                        topology = config_dict['topology_changes']
                        if isinstance(topology, dict):
                            # Only convert if it's a dict (not already a TopologyChangeConfig instance)
                            config_dict['topology_changes'] = TopologyChangeConfig(**topology)
                        # If it's already a TopologyChangeConfig instance, leave it as is

                    # Now instantiate MechVariationConfig with properly converted nested dataclasses
                    config = MechVariationConfig(**config_dict)
                # Extract dimension_variation from the config (always set this, even if exception occurs later)
                # This overwrites the value from request.get('dimension_variation_config') if it was set
                dimension_variation_config = config.dimension_variation
                config_applied = {
                    'type': 'achievable_target',
                    'config': {
                        'max_attempts': config.max_attempts,
                        'fallback_ranges': config.fallback_ranges,
                        'random_seed': config.random_seed,
                    },
                }
            except Exception as e:
                logger.error(f'Failed to create MechVariationConfig in get_optimizable_dimensions: {e}')
                traceback.print_exc()
                return {
                    'status': 'error',
                    'message': f'Invalid MechVariationConfig: {str(e)}',
                }

        if dimension_variation_config:
            try:
                # Handle both dict and DimensionVariationConfig instance
                if isinstance(dimension_variation_config, dict):
                    dim_var_config = DimensionVariationConfig(**dimension_variation_config)
                elif hasattr(dimension_variation_config, 'default_variation_range'):
                    # Already a DimensionVariationConfig instance
                    dim_var_config = dimension_variation_config
                else:
                    # Try to convert to dict first
                    try:
                        from dataclasses import asdict
                        dim_var_config = DimensionVariationConfig(**asdict(dimension_variation_config))
                    except Exception:
                        dim_var_config = DimensionVariationConfig(**dict(dimension_variation_config))

                config_applied = {
                    'type': 'dimension_variation',
                    'config': {
                        'default_variation_range': dim_var_config.default_variation_range,
                        'default_enabled': dim_var_config.default_enabled,
                        'exclude_dimensions': dim_var_config.exclude_dimensions,
                    },
                }

                # Use apply_dimension_variation_config to properly modify bounds
                # This function applies the variation config to modify bounds according to percentage ranges
                dim_spec = apply_dimension_variation_config(dim_spec, dim_var_config)
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Invalid DimensionVariationConfig: {str(e)}',
                }

        # Use DimensionBoundsSpec.to_dict() for proper serialization
        elapsed = (time.time() - start_time) * 1000

        return {
            'status': 'success',
            'dimension_bounds_spec': dim_spec.to_dict(),
            'config_applied': config_applied,
        }

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        logger.error(f'Error extracting dimensions [{request_id}] in {elapsed:.2f}ms: {e}')
        traceback.print_exc()
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
            # Option 1: TargetTrajectory format (preferred)
            "target_trajectory": {
                "joint_name": str,
                "positions": [[x, y], ...]
            },
            # Option 2: Raw trajectory (backward compatible)
            "trajectory": [[x1, y1], [x2, y2], ...],

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
            "target_trajectory": {
                # TargetTrajectory format:
                "joint_name": str,  # From input or "unknown"
                "positions": [[x, y], ...],  # Processed trajectory
                "n_steps": int
            },
            "original_points": 157,
            "output_points": 24,
            "analysis": {
                "n_points": 24,
                "centroid": [x, y],
                "bounding_box": {...},
                "total_path_length": 123.4,
                "is_closed": true
            },
            "processing": {...}
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

        # Extract trajectory - support both TargetTrajectory format and raw trajectory (backward compatible)
        target_trajectory_input = request.get('target_trajectory')
        trajectory_raw = request.get('trajectory', [])

        joint_name = 'unknown'
        trajectory = []

        if target_trajectory_input:
            # Use TargetTrajectory format
            try:
                target_traj = TargetTrajectory.from_dict(target_trajectory_input)
                joint_name = target_traj.joint_name
                trajectory = [[float(p[0]), float(p[1])] for p in target_traj.positions]
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Invalid target_trajectory format: {str(e)}',
                }
        elif trajectory_raw:
            # Use raw trajectory format (backward compatible)
            trajectory = trajectory_raw
        else:
            return {
                'status': 'error',
                'message': 'No trajectory provided. Provide either "target_trajectory" or "trajectory"',
            }

        target_n_steps = request.get('target_n_steps', 24)
        do_smooth = request.get('smooth', True)
        smooth_window = request.get('smooth_window', 5)
        smooth_polyorder = request.get('smooth_polyorder', 3)
        smooth_method = request.get('smooth_method', 'savgol')
        do_resample = request.get('resample', True)
        resample_method = request.get('resample_method', 'parametric')
        is_closed = request.get('closed', True)  # Default to closed trajectories

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

        # Create TargetTrajectory from processed result
        processed_target = TargetTrajectory(
            joint_name=joint_name,
            positions=[(float(p[0]), float(p[1])) for p in result],
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        response = {
            'status': 'success',
            'target_trajectory': processed_target.to_dict(),  # Return TargetTrajectory format
            'original_points': original_points,
            'output_points': len(result),
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

        # Sanitize to handle inf/nan values (JSON doesn't support them)
        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error preparing trajectory: {e}')
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
            # Option 1: TargetTrajectory format (preferred)
            "target_trajectory": {
                "joint_name": str,
                "positions": [[x, y], ...]
            },
            # Option 2: Raw trajectory (backward compatible)
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
        # Extract trajectory - support both TargetTrajectory format and raw trajectory (backward compatible)
        target_trajectory_input = request.get('target_trajectory')
        trajectory_raw = request.get('trajectory', [])

        if target_trajectory_input:
            # Use TargetTrajectory format
            try:
                target_traj = TargetTrajectory.from_dict(target_trajectory_input)
                traj_tuples = [(float(p[0]), float(p[1])) for p in target_traj.positions]
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Invalid target_trajectory format: {str(e)}',
                }
        elif trajectory_raw:
            # Use raw trajectory format (backward compatible)
            traj_tuples = [(float(p[0]), float(p[1])) for p in trajectory_raw]
        else:
            return {
                'status': 'error',
                'message': 'No trajectory provided. Provide either "target_trajectory" or "trajectory"',
            }

        # Analyze
        analysis = analyze_trajectory(traj_tuples)

        response = {
            'status': 'success',
            'analysis': analysis,
        }

        # Sanitize to handle inf/nan values (JSON doesn't support them)
        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error analyzing trajectory: {e}')
        return {
            'status': 'error',
            'message': f'Failed to analyze trajectory: {str(e)}',
        }


@app.post('/create-dimension-variation-config')
def create_dimension_variation_config(request: dict):
    """
    Create and validate a DimensionVariationConfig.

    This endpoint validates the configuration and returns the serialized config.
    Useful for validating configs before using them in other endpoints.

    Request body:
        {
            "default_variation_range": float,  # Default: 0.5
            "default_enabled": bool,  # Default: true
            "dimension_overrides": {
                "dimension_name": [enabled: bool, min_pct: float, max_pct: float]
            },
            "exclude_dimensions": [str, ...]
        }

    Returns:
        {
            "status": "success",
            "config": {
                "default_variation_range": float,
                "default_enabled": bool,
                "dimension_overrides": {...},
                "exclude_dimensions": [...]
            }
        }
    """
    try:
        # Instantiate DimensionVariationConfig from input
        config = DimensionVariationConfig(**request)

        # Serialize config to dict for JSON response
        config_dict = asdict(config)

        return {
            'status': 'success',
            'config': config_dict,
        }

    except TypeError as e:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Invalid DimensionVariationConfig parameters: {str(e)}',
        }
    except Exception as e:
        logger.error(f'Error creating DimensionVariationConfig: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Failed to create config: {str(e)}',
        }


@app.post('/create-achievable-target-config')
def create_achievable_target_config(request: dict):
    """
    Create and validate an MechVariationConfig.

    This endpoint validates the configuration and returns the serialized config.
    Useful for validating configs before using them in other endpoints.

    Request body:
        {
            "dimension_variation": {  # Optional: DimensionVariationConfig
                "default_variation_range": 0.5,
                "default_enabled": true,
                "dimension_overrides": {...},
                "exclude_dimensions": [...]
            },
            "static_joint_movement": {  # Optional: StaticJointMovementConfig
                "enabled": false,
                "max_x_movement": 10.0,
                "max_y_movement": 10.0,
                "joint_overrides": {
                    "joint_name": [enabled: bool, max_x: float, max_y: float]
                },
                "linked_joints": [["joint1", "joint2"], ...]
            },
            "max_attempts": int,  # Default: 128
            "fallback_ranges": [float, ...],  # Default: [0.15, 0.15, 0.15]
            "random_seed": int | null
        }

    Returns:
        {
            "status": "success",
            "config": {
                "dimension_variation": {...},
                "static_joint_movement": {...},
                "topology_changes": {
                    "enabled": false,  # Always false (not implemented)
                    ...
                },
                "max_attempts": int,
                "fallback_ranges": [float, ...],
                "random_seed": int | null
            }
        }
    """
    try:
        # Properly instantiate nested dataclasses from dict input
        # Convert nested dicts to their respective dataclass instances
        config_dict = request.copy()

        # Convert dimension_variation dict to DimensionVariationConfig
        if 'dimension_variation' in config_dict and isinstance(config_dict['dimension_variation'], dict):
            config_dict['dimension_variation'] = DimensionVariationConfig(**config_dict['dimension_variation'])

        # Convert static_joint_movement dict to StaticJointMovementConfig
        if 'static_joint_movement' in config_dict and isinstance(config_dict['static_joint_movement'], dict):
            config_dict['static_joint_movement'] = StaticJointMovementConfig(**config_dict['static_joint_movement'])

        # Convert topology_changes dict to TopologyChangeConfig
        if 'topology_changes' in config_dict and isinstance(config_dict['topology_changes'], dict):
            config_dict['topology_changes'] = TopologyChangeConfig(**config_dict['topology_changes'])

        # Instantiate MechVariationConfig from properly converted input
        # This will raise NotImplementedError if topology_changes.enabled=True
        config = MechVariationConfig(**config_dict)

        # Serialize config to dict for JSON response
        config_dict = asdict(config)

        return {
            'status': 'success',
            'config': config_dict,
        }

    except NotImplementedError as e:
        return {
            'status': 'error',
            'error_type': 'not_implemented',
            'message': str(e),
        }
    except TypeError as e:
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Invalid MechVariationConfig parameters: {str(e)}',
        }
    except Exception as e:
        logger.error(f'Error creating MechVariationConfig: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Failed to create config: {str(e)}',
        }


@app.post('/get-achievable-target')
def get_achievable_target(request: dict):
    """
    Generate an achievable optimization target by modifying mechanism dimensions.

    This endpoint creates a target trajectory that is guaranteed to be achievable
    by randomly varying mechanism dimensions within configured bounds and using
    the resulting trajectory as the target.

    Request body:
        {
            "pylink_data": {...},  # Full pylink document (or nested)
            "target_joint": str,   # Required: joint name whose trajectory becomes the target
            "n_steps": int | None,  # Optional: simulation steps (defaults to mechanism's n_steps)
            "config": {  # Optional: MechVariationConfig (defaults to ±50% variation)
                "dimension_variation": {...},
                "static_joint_movement": {...},
                "max_attempts": 128,
                "fallback_ranges": [0.15, 0.15, 0.15],
                "random_seed": int | null
            }
        }

    Returns:
        {
            "status": "success",
            "result": {
                "target": {
                    "joint_name": str,
                    "positions": [[x, y], ...],
                    "n_steps": int
                },
                "target_dimensions": {
                    "dimension_name": float, ...
                },
                "target_mechanism": {...},  # Pylink document format
                "static_joint_movements": {
                    "joint_name": [dx: float, dy: float], ...
                },
                "attempts_needed": int,
                "fallback_range_used": float | null
            },
            "execution_time_ms": float
        }
    """
    try:
        start_time = time.perf_counter()

        # Extract parameters
        target_joint = request.get('target_joint')
        n_steps = request.get('n_steps')
        config_input = request.get('config')

        if not target_joint:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Missing required parameter: target_joint',
            }

        # Convert mechanism at API boundary
        mechanism = create_mechanism_from_request(request, n_steps=n_steps)

        # Validate target joint exists
        if target_joint not in mechanism.joint_names:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': f"Target joint '{target_joint}' not found in mechanism",
                'available_joints': mechanism.joint_names,
            }

        # Create config from input or use default
        # Properly instantiate nested dataclasses from dict input
        config = None
        if config_input:
            try:
                # Convert nested dicts to their respective dataclass instances
                config_dict = config_input.copy()

                # Convert dimension_variation dict to DimensionVariationConfig
                if 'dimension_variation' in config_dict and isinstance(config_dict['dimension_variation'], dict):
                    config_dict['dimension_variation'] = DimensionVariationConfig(**config_dict['dimension_variation'])

                # Convert static_joint_movement dict to StaticJointMovementConfig
                if 'static_joint_movement' in config_dict and isinstance(config_dict['static_joint_movement'], dict):
                    config_dict['static_joint_movement'] = StaticJointMovementConfig(**config_dict['static_joint_movement'])

                # Convert topology_changes dict to TopologyChangeConfig
                if 'topology_changes' in config_dict and isinstance(config_dict['topology_changes'], dict):
                    config_dict['topology_changes'] = TopologyChangeConfig(**config_dict['topology_changes'])

                # Now instantiate MechVariationConfig with properly converted nested dataclasses
                config = MechVariationConfig(**config_dict)
            except Exception as e:
                logger.error(f'Failed to create MechVariationConfig: {e}')
                traceback.print_exc()
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid MechVariationConfig: {str(e)}',
                }

        # Call create_achievable_target
        result = create_achievable_target(
            mechanism=mechanism,
            target_joint=target_joint,
            config=config,
            n_steps=n_steps,
        )

        # Convert AchievableTargetResult to dict format
        # Note: AchievableTargetResult doesn't have .to_dict(), so we convert manually
        result_dict = {
            'target': result.target.to_dict(),  # TargetTrajectory has .to_dict()
            'target_dimensions': result.target_dimensions,
            'target_mechanism': result.target_mechanism.to_dict(),  # Mechanism has .to_dict()
            'static_joint_movements': {
                k: [float(v[0]), float(v[1])] if isinstance(v, tuple) else v
                for k, v in result.static_joint_movements.items()
            },
            'attempts_needed': result.attempts_needed,
            'fallback_range_used': result.fallback_range_used,
        }

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        response = {
            'status': 'success',
            'result': result_dict,
            'execution_time_ms': execution_time_ms,
        }

        return sanitize_for_json(response)

    except ValueError as e:
        # Max attempts exceeded or other ValueError
        return {
            'status': 'error',
            'error_type': 'max_attempts_exceeded',
            'message': str(e),
        }
    except Exception as e:
        logger.error(f'Error creating achievable target: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Failed to create achievable target: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


@app.post('/reduce-dimensions')
def reduce_dimensions_endpoint(request: dict):
    """
    Reduce high-dimensional samples to lower dimensions for visualization/analysis.

    Request body:
        {
            "samples": [[float, ...], ...],  # (n_samples, n_dims) array
            "method": "pca" | "tsne" | "umap",  # Default: "pca"
            "n_components": int,  # Default: 2
            "fit_samples_mask": [bool, ...] | null,
            "transform_samples_mask": [bool, ...] | null,
            "normalize": bool,  # Default: true
            "dimension_bounds": [[min, max], ...] | null,
            "random_state": int | null,
            "method_params": {...}  # Method-specific parameters
        }

    Returns:
        {
            "status": "success",
            "reduced_samples": [[float, ...], ...],
            "metadata": {
                "method": str,
                "n_components": int,
                "normalized": bool,
                "normalization_params": {...} | null,
                "variance_explained": [float, ...] | null
            },
            "execution_time_ms": float
        }
    """
    try:
        start_time = time.perf_counter()

        # Extract parameters
        samples = request.get('samples')
        if samples is None:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Missing required parameter: samples',
            }

        samples_array = np.array(samples)
        method = request.get('method', 'pca')
        n_components = request.get('n_components', 2)
        fit_samples_mask = request.get('fit_samples_mask')
        transform_samples_mask = request.get('transform_samples_mask')
        normalize = request.get('normalize', True)
        dimension_bounds = request.get('dimension_bounds')
        random_state = request.get('random_state')
        method_params = request.get('method_params', {})

        # Convert masks to numpy arrays if provided
        if fit_samples_mask is not None:
            fit_samples_mask = np.array(fit_samples_mask, dtype=bool)
        if transform_samples_mask is not None:
            transform_samples_mask = np.array(transform_samples_mask, dtype=bool)

        # Convert dimension_bounds to list of tuples if provided
        if dimension_bounds is not None:
            dimension_bounds = [tuple(b) for b in dimension_bounds]

        # Call reduce_dimensions
        reduced_samples, reducer, metadata = reduce_dimensions(
            samples=samples_array,
            method=method,
            n_components=n_components,
            fit_samples_mask=fit_samples_mask,
            transform_samples_mask=transform_samples_mask,
            normalize=normalize,
            dimension_bounds=dimension_bounds,
            random_state=random_state,
            **method_params,
        )

        # Convert metadata for JSON (handle numpy arrays)
        metadata_dict = {
            'method': metadata['method'],
            'n_components': metadata['n_components'],
            'normalized': metadata['normalized'],
            'normalization_params': metadata.get('normalization_params'),
            'variance_explained': (
                metadata['variance_explained'].tolist()
                if metadata.get('variance_explained') is not None
                else None
            ),
        }

        # Convert normalization_params if present
        if metadata_dict['normalization_params']:
            norm_params = metadata_dict['normalization_params']
            metadata_dict['normalization_params'] = {
                'min': norm_params['min'].tolist() if hasattr(norm_params['min'], 'tolist') else norm_params['min'],
                'max': norm_params['max'].tolist() if hasattr(norm_params['max'], 'tolist') else norm_params['max'],
                'range': norm_params['range'].tolist() if hasattr(norm_params['range'], 'tolist') else norm_params['range'],
            }

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        response = {
            'status': 'success',
            'reduced_samples': reduced_samples.tolist(),
            'metadata': metadata_dict,
            'execution_time_ms': execution_time_ms,
        }

        return sanitize_for_json(response)

    except ImportError as e:
        return {
            'status': 'error',
            'error_type': 'import_error',
            'message': f'Required library not available: {str(e)}',
        }
    except Exception as e:
        logger.error(f'Error reducing dimensions: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Failed to reduce dimensions: {str(e)}',
        }


@app.post('/generate-samples')
def generate_samples_endpoint(request: dict):
    """
    Generate mechanism samples with viability validation.

    Generates n_requested samples and validates viability (checks if mechanism
    can complete full rotation). Returns all samples with validity flags.

    Request body:
        {
            "pylink_data": {...},
            "n_requested": int,
            "sampling_mode": "sobol" | "behnken" | "meshgrid" | "full_combinatoric",
            "dimension_variation_config": {...},  # Optional
            "target_trajectory": {...},  # Optional: for fitness scoring
            "target_joint": str | null,
            "metric": "mse" | "rmse" | "total" | "max",
            "phase_invariant": bool,
            "seed": int | null,
            "return_mechanisms": bool,
            "return_trajectories": bool
        }

    Returns:
        {
            "status": "success",
            "result": {
                # SamplingResult.to_dict() format
                "samples": [[float, ...], ...],
                "is_valid": [bool, ...],
                "scores": [float, ...] | null,
                "mechanisms": [{...}, ...] | null,
                "trajectories": [{...}, ...] | null,
                "n_generated": int,
                "n_valid": int,
                "n_invalid": int
            },
            "execution_time_ms": float
        }
    """
    try:
        start_time = time.perf_counter()

        # Extract parameters
        n_requested = request.get('n_requested')
        if n_requested is None:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Missing required parameter: n_requested',
            }

        # Convert mechanism at API boundary
        mechanism = create_mechanism_from_request(request)
        dim_spec = mechanism.get_dimension_bounds_spec()

        # Parse optional configs
        dimension_variation_config = None
        if request.get('dimension_variation_config'):
            try:
                dim_var_input = request['dimension_variation_config']
                # Handle both dict and DimensionVariationConfig instance
                if isinstance(dim_var_input, dict):
                    dimension_variation_config = DimensionVariationConfig(**dim_var_input)
                elif hasattr(dim_var_input, 'default_variation_range'):
                    # Already a DimensionVariationConfig instance
                    dimension_variation_config = dim_var_input
                else:
                    # Try to convert to dict first
                    try:
                        dimension_variation_config = DimensionVariationConfig(**asdict(dim_var_input))
                    except Exception:
                        dimension_variation_config = DimensionVariationConfig(**dict(dim_var_input))
            except Exception as e:
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid DimensionVariationConfig: {str(e)}',
                }

        target_trajectory = None
        if request.get('target_trajectory'):
            try:
                target_trajectory = TargetTrajectory.from_dict(request['target_trajectory'])
            except Exception as e:
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid TargetTrajectory: {str(e)}',
                }

        # Extract other parameters
        sampling_mode = request.get('sampling_mode', 'sobol')
        target_joint = request.get('target_joint')
        metric = request.get('metric', 'mse')
        phase_invariant = request.get('phase_invariant', True)
        seed = request.get('seed')
        return_mechanisms = request.get('return_mechanisms', False)
        return_trajectories = request.get('return_trajectories', True)

        # Call generate_samples
        result = generate_samples(
            mechanism=mechanism,
            dimension_bounds_spec=dim_spec,
            n_requested=n_requested,
            sampling_mode=sampling_mode,
            dimension_variation_config=dimension_variation_config,
            target_trajectory=target_trajectory,
            target_joint=target_joint,
            metric=metric,
            phase_invariant=phase_invariant,
            seed=seed,
            return_mechanisms=return_mechanisms,
            return_trajectories=return_trajectories,
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        response = {
            'status': 'success',
            'result': result.to_dict(),  # SamplingResult has .to_dict()
            'execution_time_ms': execution_time_ms,
        }

        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error generating samples: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'error_type': 'validation_error',
            'message': f'Failed to generate samples: {str(e)}',
        }


@app.post('/generate-valid-samples')
def generate_valid_samples_endpoint(request: dict):
    """
    Generate viable mechanism samples.

    Generates n_valid_requested samples that are viable (can complete a full
    rotation without singularities). Keeps trying until enough valid samples
    are found or max_attempts is reached.

    Request body:
        {
            "pylink_data": {...},
            "n_valid_requested": int,
            "max_attempts": int | null,  # Optional: default = n_valid_requested * 100
            "sampling_mode": "sobol" | "behnken" | "meshgrid" | "full_combinatoric",
            "dimension_variation_config": {...},  # Optional
            "target_trajectory": {...} | null,  # Optional: for fitness scoring
            "target_joint": str | null,
            "metric": "mse" | "rmse" | "total" | "max",
            "phase_invariant": bool,
            "seed": int | null,
            "return_all": bool,  # Default: false
            "return_mechanisms": bool,
            "return_trajectories": bool
        }

    Returns:
        {
            "status": "success",
            "result": {
                # SamplingResult.to_dict() format
                "samples": [[float, ...], ...],
                "is_valid": [bool, ...],
                "scores": [float, ...] | null,
                "mechanisms": [{...}, ...] | null,
                "trajectories": [{...}, ...] | null,
                "n_generated": int,
                "n_valid": int,
                "n_invalid": int
            },
            "execution_time_ms": float
        }
    """
    try:
        start_time = time.perf_counter()

        # Extract parameters
        n_valid_requested = request.get('n_valid_requested')
        if n_valid_requested is None:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Missing required parameter: n_valid_requested',
            }

        # Convert mechanism at API boundary
        mechanism = create_mechanism_from_request(request)
        dim_spec = mechanism.get_dimension_bounds_spec()

        # Parse optional configs
        dimension_variation_config = None
        if request.get('dimension_variation_config'):
            try:
                dim_var_input = request['dimension_variation_config']
                # Handle both dict and DimensionVariationConfig instance
                if isinstance(dim_var_input, dict):
                    dimension_variation_config = DimensionVariationConfig(**dim_var_input)
                elif hasattr(dim_var_input, 'default_variation_range'):
                    # Already a DimensionVariationConfig instance
                    dimension_variation_config = dim_var_input
                else:
                    # Try to convert to dict first
                    try:
                        dimension_variation_config = DimensionVariationConfig(**asdict(dim_var_input))
                    except Exception:
                        dimension_variation_config = DimensionVariationConfig(**dict(dim_var_input))
            except Exception as e:
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid DimensionVariationConfig: {str(e)}',
                }

        target_trajectory = None
        if request.get('target_trajectory'):
            try:
                target_trajectory = TargetTrajectory.from_dict(request['target_trajectory'])
            except Exception as e:
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid TargetTrajectory: {str(e)}',
                }

        # Extract other parameters
        max_attempts = request.get('max_attempts')
        sampling_mode = request.get('sampling_mode', 'sobol')
        target_joint = request.get('target_joint')
        metric = request.get('metric', 'mse')
        phase_invariant = request.get('phase_invariant', True)
        seed = request.get('seed')
        return_all = request.get('return_all', False)
        return_mechanisms = request.get('return_mechanisms', False)
        return_trajectories = request.get('return_trajectories', True)

        # Call generate_valid_samples
        result = generate_valid_samples(
            mechanism=mechanism,
            dimension_bounds_spec=dim_spec,
            n_valid_requested=n_valid_requested,
            max_attempts=max_attempts,
            sampling_mode=sampling_mode,
            dimension_variation_config=dimension_variation_config,
            target_trajectory=target_trajectory,
            target_joint=target_joint,
            metric=metric,
            phase_invariant=phase_invariant,
            seed=seed,
            return_all=return_all,
            return_mechanisms=return_mechanisms,
            return_trajectories=return_trajectories,
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        response = {
            'status': 'success',
            'result': result.to_dict(),  # SamplingResult has .to_dict()
            'execution_time_ms': execution_time_ms,
        }

        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error generating valid samples: {e}')
        traceback.print_exc()
        # Check if we have a partial result we can return
        error_response = {
            'status': 'error',
            'error_type': 'max_attempts_exceeded' if 'attempts' in str(e).lower() else 'validation_error',
            'message': f'Failed to generate valid samples: {str(e)}',
        }
        return error_response


@app.post('/generate-good-samples')
def generate_good_samples_endpoint(request: dict):
    """
    Generate samples that are epsilon-close to a target trajectory.

    Generates n_good_requested samples that have fitness score <= epsilon
    (i.e., are "good" matches to the target trajectory).

    Request body:
        {
            "pylink_data": {...},
            "target_trajectory": {  # Required
                "joint_name": str,
                "positions": [[x, y], ...]
            },
            "n_good_requested": int,
            "epsilon": float,  # Default: 500.0
            "max_attempts": int | null,  # Optional: default = n_good_requested * 200
            "sampling_mode": "sobol" | "behnken" | "meshgrid" | "full_combinatoric",
            "dimension_variation_config": {...},  # Optional
            "target_joint": str | null,  # Optional: defaults to target_trajectory.joint_name
            "metric": "mse" | "rmse" | "total" | "max",
            "phase_invariant": bool,
            "seed": int | null,
            "return_all": bool,  # Default: false
            "return_mechanisms": bool,
            "return_trajectories": bool
        }

    Returns:
        {
            "status": "success",
            "result": {
                # SamplingResult.to_dict() format
                # is_valid marks samples with score <= epsilon
                "samples": [[float, ...], ...],
                "is_valid": [bool, ...],
                "scores": [float, ...],  # Always present
                "mechanisms": [{...}, ...] | null,
                "trajectories": [{...}, ...] | null,
                "n_generated": int,
                "n_valid": int,
                "n_invalid": int
            },
            "execution_time_ms": float
        }
    """
    try:
        start_time = time.perf_counter()

        # Extract required parameters
        target_trajectory_input = request.get('target_trajectory')
        if not target_trajectory_input:
            return {
                'status': 'error',
                'error_type': 'missing_target',
                'message': 'Missing required parameter: target_trajectory',
            }

        n_good_requested = request.get('n_good_requested')
        if n_good_requested is None:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': 'Missing required parameter: n_good_requested',
            }

        # Parse target_trajectory
        try:
            target_trajectory = TargetTrajectory.from_dict(target_trajectory_input)
        except Exception as e:
            return {
                'status': 'error',
                'error_type': 'validation_error',
                'message': f'Invalid TargetTrajectory: {str(e)}',
            }

        # Convert mechanism at API boundary
        mechanism = create_mechanism_from_request(request)
        dim_spec = mechanism.get_dimension_bounds_spec()

        # Parse optional configs
        dimension_variation_config = None
        if request.get('dimension_variation_config'):
            try:
                dim_var_input = request['dimension_variation_config']
                # Handle both dict and DimensionVariationConfig instance
                if isinstance(dim_var_input, dict):
                    dimension_variation_config = DimensionVariationConfig(**dim_var_input)
                elif hasattr(dim_var_input, 'default_variation_range'):
                    # Already a DimensionVariationConfig instance
                    dimension_variation_config = dim_var_input
                else:
                    # Try to convert to dict first
                    try:
                        dimension_variation_config = DimensionVariationConfig(**asdict(dim_var_input))
                    except Exception:
                        dimension_variation_config = DimensionVariationConfig(**dict(dim_var_input))
            except Exception as e:
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid DimensionVariationConfig: {str(e)}',
                }

        # Extract other parameters
        epsilon = request.get('epsilon', 500.0)
        max_attempts = request.get('max_attempts')
        sampling_mode = request.get('sampling_mode', 'sobol')
        target_joint = request.get('target_joint')  # Defaults to target_trajectory.joint_name if None
        metric = request.get('metric', 'mse')
        phase_invariant = request.get('phase_invariant', True)
        seed = request.get('seed')
        return_all = request.get('return_all', False)
        return_mechanisms = request.get('return_mechanisms', False)
        return_trajectories = request.get('return_trajectories', True)

        # Call generate_good_samples
        result = generate_good_samples(
            mechanism=mechanism,
            dimension_bounds_spec=dim_spec,
            target_trajectory=target_trajectory,
            n_good_requested=n_good_requested,
            epsilon=epsilon,
            max_attempts=max_attempts,
            sampling_mode=sampling_mode,
            dimension_variation_config=dimension_variation_config,
            target_joint=target_joint,
            metric=metric,
            phase_invariant=phase_invariant,
            seed=seed,
            return_all=return_all,
            return_mechanisms=return_mechanisms,
            return_trajectories=return_trajectories,
        )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        response = {
            'status': 'success',
            'result': result.to_dict(),  # SamplingResult has .to_dict()
            'execution_time_ms': execution_time_ms,
        }

        return sanitize_for_json(response)

    except Exception as e:
        logger.error(f'Error generating good samples: {e}')
        traceback.print_exc()
        error_response = {
            'status': 'error',
            'error_type': 'max_attempts_exceeded' if 'attempts' in str(e).lower() else 'validation_error',
            'message': f'Failed to generate good samples: {str(e)}',
        }
        return error_response


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
