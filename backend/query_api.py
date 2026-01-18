from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import traceback
# import uuid
# from pathlib import Path
from datetime import datetime
import numpy as np
from pylinkage.joints import Crank, Revolute
from pylinkage.linkage import Linkage
from configs.appconfig import USER_DIR

app = FastAPI(title="Acinonyx API")

# Simple CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Acinonyx API is running"}

@app.get("/status")
def get_status():
    return {
        "status": "operational",
        "message": "Acinonyx backend is running successfully"
    }

@app.get("/load-last-force-graph")
def load_force_graph():
    """Load the most recent force graph from the force_graphs directory"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        force_graphs_dir = USER_DIR / "force_graphs"
        
        if not force_graphs_dir.exists():
            return {
                "error": "No force_graphs directory found",
                "path": str(force_graphs_dir)
            }
        
        # Find all force graph JSON files
        force_graph_files = list(force_graphs_dir.glob("force_graph_*.json"))
        
        if not force_graph_files:
            return {
                "error": "No force graphs found in force_graphs directory"
            }
        
        # Get the most recent file by modification time
        latest_file = max(force_graph_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            graph_data = json.load(f)
        
        print(f"Loaded force graph from: {latest_file.name}")
        return graph_data
        
    except Exception as e:
        return {
            "error": f"Failed to load force graph: {str(e)}"
        }


@app.post("/save-pylink-graph")
def save_pylink_graph(pylink_data: dict):
    """Save a pylink graph to the pygraphs directory"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        pygraphs_dir = USER_DIR / "pygraphs"
        pygraphs_dir.mkdir(parents=True, exist_ok=True)
        
        # Use provided name or generate timestamp
        name = pylink_data.get('name', 'pylink')
        time_mark = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{time_mark}.json"
        save_path = pygraphs_dir / filename
        
        # Add metadata
        save_data = {
            **pylink_data,
            "saved_at": time_mark
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Pylink graph saved to: {save_path}")
        
        return {
            "status": "success",
            "message": "Pylink graph saved successfully",
            "filename": filename,
            "path": str(save_path)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save pylink graph: {str(e)}"
        }

@app.get("/list-pylink-graphs")
def list_pylink_graphs():
    """List all saved pylink graphs"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        pygraphs_dir = USER_DIR / "pygraphs"
        
        if not pygraphs_dir.exists():
            return {
                "status": "success",
                "files": []
            }
        
        files = []
        for f in sorted(pygraphs_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    files.append({
                        "filename": f.name,
                        "name": data.get("name", f.stem),
                        "joints_count": len(data.get("pylinkage", {}).get("joints", [])),
                        "links_count": len(data.get("meta", {}).get("links", {})),
                        "saved_at": data.get("saved_at", "")
                    })
            except:
                files.append({
                    "filename": f.name,
                    "name": f.stem,
                    "error": True
                })
        
        return {
            "status": "success",
            "files": files
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list pylink graphs: {str(e)}"
        }

@app.get("/load-pylink-graph")
def load_pylink_graph(filename: str = None):
    """Load a pylink graph from the pygraphs directory"""
    try:
        # Ensure USER_DIR exists first
        USER_DIR.mkdir(parents=True, exist_ok=True)
        pygraphs_dir = USER_DIR / "pygraphs"
        
        if not pygraphs_dir.exists():
            return {
                "status": "error",
                "message": "No pygraphs directory found"
            }
        
        if filename:
            # Load specific file
            file_path = pygraphs_dir / filename
            if not file_path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {filename}"
                }
        else:
            # Load most recent file
            files = list(pygraphs_dir.glob("*.json"))
            if not files:
                return {
                    "status": "error",
                    "message": "No pylink graphs found"
                }
            file_path = max(files, key=lambda f: f.stat().st_mtime)
        
        with open(file_path, 'r') as f:
            graph_data = json.load(f)
        
        print(f"Loaded pylink graph from: {file_path.name}")
        
        return {
            "status": "success",
            "filename": file_path.name,
            "data": graph_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load pylink graph: {str(e)}"
        }


@app.post("/compute-pylink-trajectory")
def compute_pylink_trajectory(pylink_data: dict):
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
            "n_steps": 12  # Optional, defaults to 12
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
        
        n_steps = pylink_data.get('n_steps', 12)
        pylinkage_data = pylink_data.get('pylinkage', {})
        meta = pylink_data.get('meta', {})
        meta_joints = meta.get('joints', {})
        
        joints_data = pylinkage_data.get('joints', [])
        solve_order = pylinkage_data.get('solve_order', [])
        
        if not joints_data:
            return {
                "status": "error",
                "message": "No joints found in pylinkage data"
            }
        
        print(f"\n=== COMPUTE PYLINK TRAJECTORY ===")
        print(f"Joints: {len(joints_data)}, Steps: {n_steps}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Build joint objects in dependency order
        # ─────────────────────────────────────────────────────────────────────
        
        # First pass: identify all joints and their types
        joint_info = {}
        for jdata in joints_data:
            joint_info[jdata['name']] = jdata
        
        # Build joints in solve_order (respects dependencies)
        joint_objects = {}
        
        def get_position_for_joint(jdata):
            """Get the position for a joint from meta or calculate it"""
            name = jdata['name']
            jtype = jdata['type']
            
            # Check meta for stored UI position
            if name in meta_joints:
                meta_j = meta_joints[name]
                if meta_j.get('x') is not None and meta_j.get('y') is not None:
                    return (meta_j['x'], meta_j['y'])
            
            # For Static joints, use stored x, y
            if jtype == 'Static':
                return (jdata['x'], jdata['y'])
            
            # For Crank, calculate from parent
            if jtype == 'Crank':
                parent_name = jdata['joint0']['ref']
                parent_pos = get_position_for_joint(joint_info[parent_name])
                distance = jdata['distance']
                angle = jdata.get('angle', 0)
                x = parent_pos[0] + distance * np.cos(angle)
                y = parent_pos[1] + distance * np.sin(angle)
                return (x, y)
            
            # For Revolute, calculate from parents (circle-circle intersection)
            if jtype == 'Revolute':
                parent0_name = jdata['joint0']['ref']
                parent1_name = jdata['joint1']['ref']
                pos0 = get_position_for_joint(joint_info[parent0_name])
                pos1 = get_position_for_joint(joint_info[parent1_name])
                d0 = jdata['distance0']
                d1 = jdata['distance1']
                
                dx = pos1[0] - pos0[0]
                dy = pos1[1] - pos0[1]
                d = np.sqrt(dx * dx + dy * dy)
                
                if d > 0 and d <= d0 + d1:
                    a = (d0 * d0 - d1 * d1 + d * d) / (2 * d)
                    h = np.sqrt(max(0, d0 * d0 - a * a))
                    px = pos0[0] + (a * dx) / d
                    py = pos0[1] + (a * dy) / d
                    x = px - (h * dy) / d
                    y = py + (h * dx) / d
                    return (x, y)
                
                # Fallback
                return ((pos0[0] + pos1[0]) / 2, (pos0[1] + pos1[1]) / 2)
            
            return (0, 0)
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 1.5: Sync distances from actual visual positions
        # This fixes the bug where stored distances don't match visual positions
        # ─────────────────────────────────────────────────────────────────────
        
        # First, get all visual positions
        visual_positions = {}
        for jdata in joints_data:
            visual_positions[jdata['name']] = get_position_for_joint(jdata)
        
        # Now recalculate distances to match visual positions
        for jdata in joints_data:
            jtype = jdata['type']
            name = jdata['name']
            my_pos = visual_positions[name]
            
            if jtype == 'Crank':
                parent_name = jdata['joint0']['ref']
                parent_pos = visual_positions[parent_name]
                new_distance = np.sqrt(
                    (my_pos[0] - parent_pos[0])**2 + 
                    (my_pos[1] - parent_pos[1])**2
                )
                old_distance = jdata['distance']
                if abs(new_distance - old_distance) > 0.01:
                    print(f"  [SYNC] Crank '{name}': distance {old_distance:.2f} → {new_distance:.2f}")
                    jdata['distance'] = new_distance
                    
            elif jtype == 'Revolute':
                parent0_name = jdata['joint0']['ref']
                parent1_name = jdata['joint1']['ref']
                parent0_pos = visual_positions[parent0_name]
                parent1_pos = visual_positions[parent1_name]
                
                new_distance0 = np.sqrt(
                    (my_pos[0] - parent0_pos[0])**2 + 
                    (my_pos[1] - parent0_pos[1])**2
                )
                new_distance1 = np.sqrt(
                    (my_pos[0] - parent1_pos[0])**2 + 
                    (my_pos[1] - parent1_pos[1])**2
                )
                
                old_distance0 = jdata['distance0']
                old_distance1 = jdata['distance1']
                
                if abs(new_distance0 - old_distance0) > 0.01 or abs(new_distance1 - old_distance1) > 0.01:
                    print(f"  [SYNC] Revolute '{name}': d0 {old_distance0:.2f} → {new_distance0:.2f}, d1 {old_distance1:.2f} → {new_distance1:.2f}")
                    jdata['distance0'] = new_distance0
                    jdata['distance1'] = new_distance1
        
        # Calculate angle per step for Crank joints (full rotation over n_steps)
        angle_per_step = 2 * np.pi / n_steps
        
        # Build joints in solve order
        for joint_name in solve_order:
            if joint_name not in joint_info:
                continue
                
            jdata = joint_info[joint_name]
            jtype = jdata['type']
            pos = get_position_for_joint(jdata)
            
            if jtype == 'Static':
                # Static joints become tuple references (implicit Fixed in pylinkage)
                joint_objects[joint_name] = (jdata['x'], jdata['y'])
                print(f"  Static '{joint_name}' at ({jdata['x']:.1f}, {jdata['y']:.1f})")
                
            elif jtype == 'Crank':
                parent_name = jdata['joint0']['ref']
                parent = joint_objects.get(parent_name)
                
                if parent is None:
                    print(f"  Warning: Crank '{joint_name}' parent '{parent_name}' not found")
                    continue
                
                joint_objects[joint_name] = Crank(
                    x=pos[0],
                    y=pos[1],
                    joint0=parent,
                    distance=jdata['distance'],
                    angle=angle_per_step,  # Use computed angle per step for animation
                    name=joint_name
                )
                print(f"  Crank '{joint_name}' at ({pos[0]:.1f}, {pos[1]:.1f}), dist={jdata['distance']:.1f}")
                
            elif jtype == 'Revolute':
                parent0_name = jdata['joint0']['ref']
                parent1_name = jdata['joint1']['ref']
                parent0 = joint_objects.get(parent0_name)
                parent1 = joint_objects.get(parent1_name)
                
                if parent0 is None or parent1 is None:
                    print(f"  Warning: Revolute '{joint_name}' parents not found")
                    continue
                
                joint_objects[joint_name] = Revolute(
                    x=pos[0],
                    y=pos[1],
                    joint0=parent0,
                    joint1=parent1,
                    distance0=jdata['distance0'],
                    distance1=jdata['distance1'],
                    name=joint_name
                )
                print(f"  Revolute '{joint_name}' at ({pos[0]:.1f}, {pos[1]:.1f}), d0={jdata['distance0']:.1f}, d1={jdata['distance1']:.1f}")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: Build the Linkage object
        # ─────────────────────────────────────────────────────────────────────
        
        # Get non-static joints for the linkage (only Crank and Revolute)
        linkage_joints = []
        for joint_name in solve_order:
            joint = joint_objects.get(joint_name)
            if joint is not None and not isinstance(joint, tuple):
                linkage_joints.append(joint)
        
        if not linkage_joints:
            return {
                "status": "error",
                "message": "No movable joints (Crank/Revolute) found. Need at least one Crank to drive the mechanism."
            }
        
        # Check for at least one Crank
        has_crank = any(isinstance(j, Crank) for j in linkage_joints)
        if not has_crank:
            return {
                "status": "error", 
                "message": "No Crank joint found. A Crank is required to drive the mechanism."
            }
        
        linkage = Linkage(
            joints=tuple(linkage_joints),
            order=tuple(linkage_joints),
            name=pylinkage_data.get('name', 'computed')
        )
        
        print(f"  Created Linkage with {len(linkage_joints)} joints")
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Run the simulation
        # ─────────────────────────────────────────────────────────────────────
        
        trajectories = {}
        
        # Initialize trajectories for all joints (including Static)
        for joint_name in solve_order:
            trajectories[joint_name] = []
        
        # Run simulation
        try:
            linkage.rebuild()  # Reset to initial state
            
            for step, coords in enumerate(linkage.step(iterations=n_steps)):
                # coords is a list of (x, y) tuples for each joint in linkage.joints
                for joint, coord in zip(linkage.joints, coords):
                    if coord[0] is not None and coord[1] is not None:
                        trajectories[joint.name].append([float(coord[0]), float(coord[1])])
                    else:
                        # Use last known position or (0,0)
                        last = trajectories[joint.name][-1] if trajectories[joint.name] else [0, 0]
                        trajectories[joint.name].append(last)
                
                # Add static joint positions (they don't change)
                for joint_name in solve_order:
                    if joint_name not in [j.name for j in linkage.joints]:
                        joint = joint_objects.get(joint_name)
                        if isinstance(joint, tuple):
                            trajectories[joint_name].append([float(joint[0]), float(joint[1])])
            
            print(f"  Simulation completed: {n_steps} steps")
            
        except Exception as sim_error:
            print(f"  Simulation error: {sim_error}")
            return {
                "status": "error",
                "message": f"Simulation failed: {str(sim_error)}",
                "traceback": traceback.format_exc().split('\n')
            }
        
        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Return results
        # ─────────────────────────────────────────────────────────────────────
        
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        
        print(f"  Completed in {execution_time_ms:.2f}ms")
        
        return {
            "status": "success",
            "message": f"Computed {n_steps} trajectory steps for {len(trajectories)} joints",
            "trajectories": trajectories,
            "n_steps": n_steps,
            "execution_time_ms": execution_time_ms,
            "joint_types": {name: joint_info[name]['type'] for name in solve_order if name in joint_info}
        }
        
    except Exception as e:
        print(f"Error computing pylink trajectory: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to compute trajectory: {str(e)}",
            "traceback": traceback.format_exc().split('\n')
        }


# @app.post("/convert-to-pylinkage")
# def convert_to_pylinkage(graph_data: dict):
#     """
#     Convert automata graph to pylinkage format and validate.
    
#     This endpoint takes the graph data (nodes, links, connections) and converts
#     it to a pylinkage Linkage object, returning a validation report.
    
#     Request body:
#         {
#             "nodes": [...],
#             "links": [...],
#             "connections": [...]
#         }
    
#     Returns:
#         {
#             "status": "success" | "error",
#             "message": str,
#             "conversion_result": {
#                 "success": bool,
#                 "warnings": [...],
#                 "errors": [...],
#                 "joint_mapping": {...},
#                 "stats": {...},
#                 "serialized_linkage": {...}  # pylinkage's to_dict() output
#             }
#         }
#     """
#     try:
#         nodes_data = graph_data.get('nodes', [])
#         links_data = graph_data.get('links', [])
#         connections = graph_data.get('connections', [])
        
#         print("\n=== PYLINKAGE CONVERSION REQUEST ===")
#         print(f"Nodes: {len(nodes_data)}, Links: {len(links_data)}, Connections: {len(connections)}")
        
#         # Convert raw dicts to model objects
#         node_objects = []
#         for i, node_dict in enumerate(nodes_data):
#             try:
#                 n_iterations = node_dict.get('n_iterations', 24)
#                 node_data = {
#                     'name': node_dict.get('name', node_dict.get('id', f'node_{i}')),
#                     'n_iterations': n_iterations,
#                     'fixed': node_dict.get('fixed', False)
#                 }
#                 if 'fixed_loc' in node_dict and node_dict['fixed_loc']:
#                     node_data['fixed_loc'] = tuple(node_dict['fixed_loc'])
#                 elif node_data['fixed'] and 'pos' in node_dict:
#                     node_data['fixed_loc'] = tuple(node_dict['pos'])
#                 if 'pos' in node_dict and node_dict['pos']:
#                     node_data['init_pos'] = tuple(node_dict['pos'])
#                 elif node_data.get('fixed_loc'):
#                     node_data['init_pos'] = node_data['fixed_loc']
#                 else:
#                     node_data['init_pos'] = (0.0, 0.0)
                
#                 node_objects.append(Node(**node_data))
#             except Exception as e:
#                 print(f"Warning: Failed to create Node from {node_dict}: {e}")
        
#         link_objects = []
#         for i, link_dict in enumerate(links_data):
#             try:
#                 excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2', 'meta']
#                 link_data = {k: v for k, v in link_dict.items() if k not in excluded_fields}
                
#                 if 'pos1' in link_data and isinstance(link_data['pos1'], dict):
#                     del link_data['pos1']
#                 if 'pos2' in link_data and isinstance(link_data['pos2'], dict):
#                     del link_data['pos2']
                
#                 link_objects.append(Link(**link_data))
#             except Exception as e:
#                 print(f"Warning: Failed to create Link from {link_dict}: {e}")
        
#         # Run conversion
#         result = convert_to_pylinkage(node_objects, link_objects, connections)
        
#         if result.success:
#             print(f"✓ Conversion successful: {result.stats}")
#             return {
#                 "status": "success",
#                 "message": "Graph converted to pylinkage successfully",
#                 "conversion_result": result.to_dict()
#             }
#         else:
#             print(f"✗ Conversion failed: {result.errors}")
#             return {
#                 "status": "error",
#                 "message": "Conversion failed",
#                 "conversion_result": result.to_dict()
#             }
    
#     except Exception as e:
#         print(f"Error in convert_to_pylinkage: {e}")
#         traceback.print_exc()
#         return {
#             "status": "error",
#             "message": f"Conversion failed: {str(e)}",
#             "traceback": traceback.format_exc().split('\n')
#         }

