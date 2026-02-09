from __future__ import annotations

import json
import logging
import math
import time
import traceback
from dataclasses import asdict
from datetime import datetime

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pylinkage.joints import Crank
from pylinkage.joints import Static

from configs.appconfig import USER_DIR
from demo.helpers import create_mechanism_from_dict
from multi.dim_tools import reduce_dimensions
from pylink_tools.hypergraph_adapter import sync_hypergraph_distances
from pylink_tools.optimization_types import DimensionBoundsSpec
from pylink_tools.optimization_types import TargetTrajectory
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

        print(f'Loaded demo: {name} from {file_path.name}')

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
    import time
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    print(f'[API] [get-demo-target-joint] [{request_id}] Entry')

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

        print(f'[API] [get-demo-target-joint] [{request_id}] Mechanism: {mechanism_name}, Joints: {len(joint_names)}')

        # Check if mechanism name matches a known demo type
        mechanism_name_lower = mechanism_name.lower()
        for mechanism_type in MECHANISMS.keys():
            if mechanism_type in mechanism_name_lower or mechanism_name_lower in mechanism_type:
                try:
                    # Load the demo mechanism to get its target joint
                    _, demo_target_joint, _ = load_mechanism(mechanism_type)
                    elapsed = (time.time() - start_time) * 1000
                    print(f'[API] [get-demo-target-joint] [{request_id}] Found via demo match ({mechanism_type}): {demo_target_joint}, elapsed: {elapsed:.2f}ms')
                    return {
                        'status': 'success',
                        'target_joint': demo_target_joint,
                    }
                except Exception as e:
                    # If loading fails, continue
                    print(f'[API] [get-demo-target-joint] [{request_id}] Failed to load demo {mechanism_type}: {e}')
                    pass

        # Also check by characteristic joint names (heuristic for demos)
        joint_names_set = set(joint_names)
        heuristic_match = None
        if 'toe' in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('leg')
                heuristic_match = ('leg', demo_target_joint)
            except Exception as e:
                print(f'[API] [get-demo-target-joint] [{request_id}] Failed to load leg demo: {e}')
                pass
        elif 'final_joint' in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('complex')
                heuristic_match = ('complex', demo_target_joint)
            except Exception as e:
                print(f'[API] [get-demo-target-joint] [{request_id}] Failed to load complex demo: {e}')
                pass
        elif 'final' in joint_names_set and 'coupler_rocker_joint' not in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('intermediate')
                heuristic_match = ('intermediate', demo_target_joint)
            except Exception as e:
                print(f'[API] [get-demo-target-joint] [{request_id}] Failed to load intermediate demo: {e}')
                pass
        elif 'coupler_rocker_joint' in joint_names_set:
            try:
                _, demo_target_joint, _ = load_mechanism('simple')
                heuristic_match = ('simple', demo_target_joint)
            except Exception as e:
                print(f'[API] [get-demo-target-joint] [{request_id}] Failed to load simple demo: {e}')
                pass

        if heuristic_match:
            elapsed = (time.time() - start_time) * 1000
            print(f'[API] [get-demo-target-joint] [{request_id}] Found via heuristic ({heuristic_match[0]}): {heuristic_match[1]}, elapsed: {elapsed:.2f}ms')
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
            elapsed = (time.time() - start_time) * 1000
            msg = (
                f'[API] [get-demo-target-joint] [{request_id}] Using fallback (deterministic non-default): '
                f'{target_joint}, candidates: {len(candidate_joints)}, elapsed: {elapsed:.2f}ms'
            )
            print(msg)
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
                    elapsed = (time.time() - start_time) * 1000
                    print(f'[API] [get-demo-target-joint] [{request_id}] Using fallback (any Crank/Revolute): {joint_name}, elapsed: {elapsed:.2f}ms')
                    return {
                        'status': 'success',
                        'target_joint': joint_name,
                    }

        # No suitable joint found
        elapsed = (time.time() - start_time) * 1000
        print(f'[API] [get-demo-target-joint] [{request_id}] No suitable joint found, elapsed: {elapsed:.2f}ms')
        return {
            'status': 'success',
            'target_joint': None,
        }

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        print(f'[API] [get-demo-target-joint] [{request_id}] Error in {elapsed:.2f}ms: {e}')
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Failed to get target joint: {str(e)}',
        }

    except Exception as e:
        print(f'Error getting demo target joint: {e}')
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

            print(f'  Mechanism simulation failed: {error_msg}')

            return {
                'status': 'error',
                'error_type': 'unsolvable_mechanism',
                'message': error_msg,
                'trajectories': None,  # Don't return invalid trajectories
            }

        # Mechanism is solvable - convert array to dict format
        # Build trajectories dict from the already-computed trajectory_array
        trajectories = {}
        for i, joint_name in enumerate(mechanism._joint_names):
            converted_positions = []
            for step in range(mechanism._n_steps):
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

        print(f'  Computed trajectory in {execution_time_ms:.2f}ms (Mechanism)')

        response = {
            'status': 'success',
            'message': f'Computed {n_steps} trajectory steps for {len(trajectories)} joints',
            'trajectories': trajectories,
            'n_steps': n_steps,
            'execution_time_ms': execution_time_ms,
            'joint_types': joint_types,
        }

        # Sanitize to handle any edge cases (though we've already validated)
        return sanitize_for_json(response)

    except Exception as e:
        print(f'Error computing pylink trajectory: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'message': f'Failed to compute trajectory: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


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
        print(f'Error validating mechanism: {e}')
        traceback.print_exc()
        return {
            'status': 'error',
            'valid': False,
            'message': f'Validation failed: {str(e)}',
            'traceback': traceback.format_exc().split('\n'),
        }


@app.post('/optimize-trajectory')
def optimize_trajectory_endpoint(request: dict):
    # #region agent log
    # with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
    #     f.write(json.dumps({
    #         'location': 'acinonyx_api.py:788', 'message': 'optimize_trajectory_endpoint called',
    #         'data': {
    #             'hasRequest': request is not None,
    #             'hasOptimizationOptions': 'optimization_options' in request if request else False,
    #             'hasMechVariationConfig': 'mech_variation_config' in (
    #                 request.get('optimization_options', {}) if request else {}
    #             ),
    #         },
    #         'timestamp': int(time.time() * 1000), 'sessionId': 'debug-session',
    #         'runId': 'run6', 'hypothesisId': 'C',
    #     }) + '\n')
    # #endregion
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

        logger.info('=== OPTIMIZE TRAJECTORY ===')
        logger.info(f'Target joint: {target_joint}')
        logger.info(f'Target points: {len(target_positions)}')
        logger.info(f'Method: {method}')

        # Convert ONCE at API boundary
        n_steps = len(target_positions)
        mechanism = create_mechanism_from_request(request, n_steps=n_steps)

        # Validate the mechanism by running a reference simulation
        logger.info('Validating mechanism...')
        try:
            mechanism.simulate()  # Quick validation
            logger.info('Mechanism validation successful')
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

                    # Log all properties of MechVariationConfig for debugging
                    if verbose:
                        logger.info('Instantiating MechVariationConfig:')
                        logger.info(f'  Type: {type(mech_variation_config)}')
                        logger.info(f'  Has dimension_variation: {hasattr(mech_variation_config, "dimension_variation")}')
                        if hasattr(mech_variation_config, 'dimension_variation'):
                            dim_var = mech_variation_config.dimension_variation
                            logger.info(f'    dimension_variation type: {type(dim_var)}')
                            logger.info(f'    default_variation_range: {getattr(dim_var, "default_variation_range", "MISSING")}')
                            logger.info(f'    default_enabled: {getattr(dim_var, "default_enabled", "MISSING")}')
                            logger.info(f'    dimension_overrides: {len(getattr(dim_var, "dimension_overrides", {}))} overrides')
                            logger.info(f'    exclude_dimensions: {len(getattr(dim_var, "exclude_dimensions", []))} excluded')
                        logger.info(f'  Has static_joint_movement: {hasattr(mech_variation_config, "static_joint_movement")}')
                        if hasattr(mech_variation_config, 'static_joint_movement'):
                            static_joint = mech_variation_config.static_joint_movement
                            logger.info(f'    static_joint_movement type: {type(static_joint)}')
                            logger.info(f'    enabled: {getattr(static_joint, "enabled", "MISSING")}')
                            logger.info(f'    max_x_movement: {getattr(static_joint, "max_x_movement", "MISSING")}')
                            logger.info(f'    max_y_movement: {getattr(static_joint, "max_y_movement", "MISSING")}')
                        logger.info(f'  Has topology_changes: {hasattr(mech_variation_config, "topology_changes")}')
                        logger.info(f'  Has max_attempts: {hasattr(mech_variation_config, "max_attempts")}')
                        logger.info(f'  Has fallback_ranges: {hasattr(mech_variation_config, "fallback_ranges")}')
                        logger.info(f'  Has random_seed: {hasattr(mech_variation_config, "random_seed")}')
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

        # Log dimension spec table
        if verbose:
            logger.info('DimensionBoundsSpec details:')
            logger.info(f'  {"Dimension Name":<35} {"Initial":>10} {"Min Bound":>12} {"Max Bound":>12} {"Range":>12}')
            logger.info(f'  {"-"*35} {"-"*10} {"-"*12} {"-"*12} {"-"*12}')
            for i, name in enumerate(dim_spec.names):
                initial = dim_spec.initial_values[i]
                min_bound, max_bound = dim_spec.bounds[i]
                range_val = max_bound - min_bound
                logger.info(f'  {name:<35} {initial:>10.4f} {min_bound:>12.4f} {max_bound:>12.4f} {range_val:>12.4f}')
            logger.info(f'  {"-"*35} {"-"*10} {"-"*12} {"-"*12} {"-"*12}')

        # Log config stats if provided
        if mech_variation_config and verbose:
            logger.info('MechVariationConfig:')
            dim_var = mech_variation_config.dimension_variation
            logger.info('  DimensionVariationConfig:')
            logger.info(f'    default_variation_range: {dim_var.default_variation_range}')
            logger.info(f'    default_enabled: {dim_var.default_enabled}')
            logger.info(f'    dimension_overrides: {len(dim_var.dimension_overrides)} overrides')
            logger.info(f'    exclude_dimensions: {len(dim_var.exclude_dimensions)} excluded')

            static_joint = mech_variation_config.static_joint_movement
            logger.info('  StaticJointMovementConfig:')
            logger.info(f'    enabled: {static_joint.enabled}')

        # Create target trajectory
        target = TargetTrajectory(
            joint_name=target_joint,
            positions=target_positions,
        )

        # Build kwargs based on method
        opt_kwargs = {
            'mechanism': mechanism,  # Pass Mechanism, not pylink_data
            'target': target,
            'dimension_bounds_spec': dim_spec,
            'mech_variation_config': mech_variation_config,  # Pass config for optimizer
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

        # Get optimized mechanism from result (optimizer returns it)
        optimized_mechanism = result.optimized_mechanism

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
        # #region agent log
        with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
            f.write(
                json.dumps({
                    'location': 'acinonyx_api.py:1082',
                    'message': 'Before result.to_dict(), checking optimized_dimensions',
                    'data': {
                        'optimized_dimensions': result.optimized_dimensions,
                        'optimized_mechanism_current_dims': list(optimized_mechanism._current_dimensions) if optimized_mechanism else None,
                        'optimized_mechanism_dim_names': list(optimized_mechanism._dimension_mapping.names) if optimized_mechanism else None,
                    },
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'H',
                }) + '\n',
            )
        # #endregion
        result_dict = result.to_dict()

        # ═══════════════════════════════════════════════════════════════════════════════
        # CRITICAL VALIDATION: Ensure optimized_dimensions match edge distances
        # ═══════════════════════════════════════════════════════════════════════════════
        # This is a fundamental correctness check. The optimizer reports dimension values
        # in optimized_dimensions (e.g., {"crank_link_distance": 4.953}). These MUST match
        # the actual edge distances in the returned mechanism's linkage.edges.
        #
        # If they don't match, it means:
        # - Mechanism.to_dict() is not correctly applying optimized distances to edges, OR
        # - The dimension_mapping.edge_mapping is incorrect, OR
        # - There's a bug in how we're reading/writing _current_dimensions
        #
        # We validate by:
        # 1. For each dimension in optimized_dimensions, look up its edge_id from edge_mapping
        # 2. Check that edges[edge_id].distance matches the dimension value
        # 3. Raise ValueError with detailed error message if any mismatch is found
        #
        # This validation runs BEFORE returning the response, ensuring we never return
        # inconsistent data to the frontend.
        # ═══════════════════════════════════════════════════════════════════════════════
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

                    # #region agent log
                    with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
                        f.write(
                            json.dumps({
                                'location': 'acinonyx_api.py:1099',
                                'message': 'CRITICAL: Validating optimized_dimensions vs edge distances',
                                'data': {
                                    'optimized_dimensions': optimized_dimensions,
                                    'edge_distances': {eid: e.get('distance') for eid, e in edges.items()},
                                    'validation_details': validation_details,
                                    'mismatches': mismatches,
                                    'has_mismatches': len(mismatches) > 0,
                                },
                                'timestamp': int(time.time() * 1000),
                                'sessionId': 'debug-session',
                                'runId': 'run1',
                                'hypothesisId': 'VALIDATION',
                            }) + '\n',
                        )
                    # #endregion

                    # RAISE EXCEPTION if there are mismatches - this is a critical bug
                    if mismatches:
                        error_msg = (
                            f'CRITICAL BUG: optimized_dimensions do not match edge distances in returned mechanism!\n'
                            'Mismatches:\n' + '\n'.join(f'  - {m}' for m in mismatches) + '\n'
                            f'This indicates a fundamental bug in Mechanism.to_dict() or dimension mapping.\n'
                            f'optimized_dimensions: {optimized_dimensions}\n'
                            f"edge_distances: {dict((eid, e.get('distance')) for eid, e in edges.items())}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    logger.info(f'✓ Validation passed: All {len(validation_details)} optimized dimensions match edge distances')
        # #endregion

        # Update node positions from simulation (after to_dict() so the update is preserved)
        # IMPORTANT: If optimization succeeded, we return the optimized mechanism even if simulation fails.
        # The optimization result is valid - we just can't update positions from simulation.
        # Only fail if the optimization itself failed.
        if result.success and optimized_mechanism and result_dict.get('optimized_pylink_data'):
            try:
                # CRITICAL: Get FULL simulation trajectory (all joints, all steps)
                # Shape: (n_steps, n_joints, 2) - we need ALL joint positions, not just target
                full_trajectory = optimized_mechanism.simulate()

                if full_trajectory is not None and len(full_trajectory) > 0:
                    # Check for NaN or Inf values in trajectory (indicates unsolvable mechanism)
                    has_nan = np.isnan(full_trajectory).any()
                    has_inf = np.isinf(full_trajectory).any()

                    if has_nan or has_inf:
                        error_reasons = []
                        if has_nan:
                            error_reasons.append('NaN values in trajectory')
                        if has_inf:
                            error_reasons.append('infinite values in trajectory')

                        error_msg = f'Cannot simulate optimized mechanism: {", ".join(error_reasons)}. '
                        error_msg += 'Optimization succeeded, but mechanism cannot be simulated. Returning optimized dimensions anyway.'

                        logger.warning(f'Optimized mechanism cannot be simulated: {error_msg}')
                        # DO NOT mark as failed - optimization succeeded, we just can't simulate it
                        # The optimized dimensions are still valid and should be returned

                        # Don't update node positions if mechanism is unsolvable
                    else:
                        # Mechanism is solvable - update ALL node positions from simulation
                        # CRITICAL: All node positions must match edge distances - no inconsistent data!
                        optimized_pylink_data = result_dict.get('optimized_pylink_data')
                        if optimized_pylink_data:
                            nodes = optimized_pylink_data.get('linkage', {}).get('nodes', {})

                            # Update ALL node positions from first frame of simulation
                            # full_trajectory shape: (n_steps, n_joints, 2)
                            # We use step 0 (first frame) to get initial positions
                            first_frame = full_trajectory[0]  # Shape: (n_joints, 2)

                            updated_count = 0
                            for i, joint_name in enumerate(optimized_mechanism._joint_names):
                                if joint_name in nodes and i < len(first_frame):
                                    x, y = float(first_frame[i, 0]), float(first_frame[i, 1])
                                    nodes[joint_name]['position'] = [x, y]
                                    updated_count += 1

                            logger.info(f'Updated {updated_count} node positions from simulation (ensuring consistency with edge distances)')
            except Exception as e:
                logger.warning(f'Could not update node positions from simulation: {e}')
                # DO NOT mark as failed - optimization succeeded, we just can't simulate it
                # The optimized dimensions are still valid and should be returned
                # Log the warning but continue to return the successful optimization result

        # Preserve metadata from original linkage (link colors, node colors, showPath, link names, etc.)
        # Note: In the future, optimization may add or remove links/nodes, so we only preserve
        # metadata for nodes/edges that still exist in the optimized linkage
        opt_data = result_dict.get('optimized_pylink_data')
        if result.success and opt_data is not None:
            optimized_pylink_data = opt_data

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

        # ═══════════════════════════════════════════════════════════════════════════════
        # CRITICAL VALIDATION: Ensure optimized_pylink_data exists and is valid
        # ═══════════════════════════════════════════════════════════════════════════════
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

    print(f'[API] [get-optimizable-dimensions] [{request_id}] Entry')

    try:
        if not request:
            elapsed = (time.time() - start_time) * 1000
            print(f'[API] [get-optimizable-dimensions] [{request_id}] Error: Missing request data, elapsed: {elapsed:.2f}ms')
            return {
                'status': 'error',
                'message': 'Missing request data',
            }

        # Convert ONCE at API boundary
        mechanism = create_mechanism_from_request(request)
        mechanism_name = getattr(mechanism, 'name', 'unknown')

        print(f'[API] [get-optimizable-dimensions] [{request_id}] Mechanism: {mechanism_name}')

        # Get dimension spec from Mechanism
        dim_spec = mechanism.get_dimension_bounds_spec()
        initial_dim_count = len(dim_spec.names) if dim_spec else 0

        print(f'[API] [get-optimizable-dimensions] [{request_id}] Initial dimensions: {initial_dim_count}')

        # Check for configs and apply filtering if provided
        config_applied: dict[str, str | dict[str, object] | None] = {'type': 'none', 'config': None}

        # Initialize dimension_variation_config to None to avoid "cannot access local variable" error
        # This will be set from mech_variation_config if provided, or from the separate dimension_variation_config field
        dimension_variation_config = request.get('dimension_variation_config')
        mech_variation_config = request.get('mech_variation_config')  # NEW name
        achievable_target_config = request.get('achievable_target_config')  # OLD name (backward compat)

        # #region agent log
        import json
        with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
            f.write(
                json.dumps({
                    'location': 'acinonyx_api.py:1223', 'message': 'Initializing dimension_variation_config', 'data': {
                        'hasDimensionVariationConfig': dimension_variation_config is not None, 'hasMechVariationConfig':
                        mech_variation_config is not None, 'hasAchievableTargetConfig': achievable_target_config is not None,
                    }, 'timestamp': int(time.time() * 1000), 'sessionId': 'debug-session', 'runId': 'run6', 'hypothesisId': 'B',
                }) + '\n',
            )
        # #endregion

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
                # #region agent log
                import json
                with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
                    f.write(
                        json.dumps({
                            'location': 'acinonyx_api.py:1283', 'message': 'Successfully extracted dimension_variation_config from MechVariationConfig', 'data': {
                                'hasDimensionVariationConfig': dimension_variation_config is not None, 'type': type(
                                    dimension_variation_config,
                                ).__name__ if dimension_variation_config is not None else 'None',
                            }, 'timestamp': int(time.time() * 1000), 'sessionId': 'debug-session', 'runId': 'run6', 'hypothesisId': 'B',
                        }) + '\n',
                    )
                # #endregion
            except Exception as e:
                print(f'[ERROR] Failed to create MechVariationConfig in get_optimizable_dimensions: {e}')
                traceback.print_exc()
                # #region agent log
                import json
                with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
                    f.write(
                        json.dumps({
                            'location': 'acinonyx_api.py:1292', 'message': 'ERROR creating MechVariationConfig', 'data': {
                                'error': str(e), 'hasDimensionVariationConfig': 'dimension_variation_config' in locals(
                                ), 'dimensionVariationConfigValue': dimension_variation_config if 'dimension_variation_config' in locals() else 'NOT_DEFINED',
                            }, 'timestamp': int(time.time() * 1000), 'sessionId': 'debug-session', 'runId': 'run6', 'hypothesisId': 'B',
                        }) + '\n',
                    )
                # #endregion
                return {
                    'status': 'error',
                    'message': f'Invalid MechVariationConfig: {str(e)}',
                }

        # #region agent log
        with open('/Users/abf/projects/Acinonyx/.cursor/debug.log', 'a') as f:
            _dv_in_locals = 'dimension_variation_config' in locals()
            _data = {
                'hasDimensionVariationConfig': _dv_in_locals,
                'isNone': dimension_variation_config is None if _dv_in_locals else 'NOT_DEFINED',
                'type': (
                    type(dimension_variation_config).__name__
                    if _dv_in_locals and dimension_variation_config is not None
                    else 'N/A'
                ),
                'hasConfigData': config_data is not None if 'config_data' in locals() else False,
            }
            f.write(
                json.dumps({
                    'location': 'acinonyx_api.py:1305',
                    'message': 'Checking dimension_variation_config before use',
                    'data': _data,
                    'timestamp': int(time.time() * 1000),
                    'sessionId': 'debug-session',
                    'runId': 'run6',
                    'hypothesisId': 'B',
                }) + '\n',
            )
        # #endregion

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
        final_dim_count = len(dim_spec.names) if dim_spec else 0
        elapsed = (time.time() - start_time) * 1000

        if config_applied['type'] != 'none':
            print(
                f'[API] [get-optimizable-dimensions] [{request_id}] Applied {config_applied["type"]}, '
                f'final dimensions: {final_dim_count} (was {initial_dim_count}), elapsed: {elapsed:.2f}ms',
            )
        else:
            print(
                f'[API] [get-optimizable-dimensions] [{request_id}] No config applied, '
                f'final dimensions: {final_dim_count}, elapsed: {elapsed:.2f}ms',
            )

        return {
            'status': 'success',
            'dimension_bounds_spec': dim_spec.to_dict(),
            'config_applied': config_applied,
        }

    except Exception as e:
        elapsed = (time.time() - start_time) * 1000
        print(f'[API] [get-optimizable-dimensions] [{request_id}] Error in {elapsed:.2f}ms: {e}')
        print(f'Error extracting dimensions: {e}')
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
        print(f'Error analyzing trajectory: {e}')
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
        print(f'Error creating DimensionVariationConfig: {e}')
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
        print(f'Error creating MechVariationConfig: {e}')
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
        print(f'[DEBUG] get_achievable_target called with target_joint={request.get("target_joint")}, n_steps={request.get("n_steps")}')

        # Extract parameters
        target_joint = request.get('target_joint')
        n_steps = request.get('n_steps')
        config_input = request.get('config')

        print(f'[DEBUG] Extracted: target_joint={target_joint}, n_steps={n_steps}, has_config={config_input is not None}')

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
                print('[DEBUG] Created MechVariationConfig successfully')

                # Log config stats
                if config:
                    print('\n[get-achievable-target] MechVariationConfig:')
                    dim_var = config.dimension_variation
                    print('  DimensionVariationConfig:')
                    print(f'    default_variation_range: {dim_var.default_variation_range}')
                    print(f'    default_enabled: {dim_var.default_enabled}')
                    print(f'    dimension_overrides: {len(dim_var.dimension_overrides)} overrides')
                    print(f'    exclude_dimensions: {len(dim_var.exclude_dimensions)} excluded')

                    static_joint = config.static_joint_movement
                    print('  StaticJointMovementConfig:')
                    print(f'    enabled: {static_joint.enabled}')
                    print(f'    max_x_movement: {static_joint.max_x_movement}')
                    print(f'    max_y_movement: {static_joint.max_y_movement}')
                    print(f'    joint_overrides: {len(static_joint.joint_overrides)} overrides')
                    print(f'    linked_joints: {len(static_joint.linked_joints)} pairs')
            except Exception as e:
                print(f'[ERROR] Failed to create MechVariationConfig: {e}')
                traceback.print_exc()
                return {
                    'status': 'error',
                    'error_type': 'validation_error',
                    'message': f'Invalid MechVariationConfig: {str(e)}',
                }

        # Call create_achievable_target
        print(f'[DEBUG] Calling create_achievable_target with target_joint={target_joint}, n_steps={n_steps}')
        result = create_achievable_target(
            mechanism=mechanism,
            target_joint=target_joint,
            config=config,
            n_steps=n_steps,
        )
        print(f'[DEBUG] create_achievable_target returned: target.joint_name={result.target.joint_name}, positions count={len(result.target.positions)}')

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

        print(f'[DEBUG] Returning response with {len(result_dict["target"]["positions"])} positions')
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
        print(f'Error creating achievable target: {e}')
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
        print(f'Error reducing dimensions: {e}')
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
        print(f'Error generating samples: {e}')
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
        print(f'Error generating valid samples: {e}')
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
        print(f'Error generating good samples: {e}')
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
