from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

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

@app.post("/save-graph")
def save_graph(graph_data: dict):
    """Save the current graph state to the graphs directory"""
    from datetime import datetime
    import json
    
    try:
        graphs_dir = USER_DIR / "graphs"
        graphs_dir.mkdir(exist_ok=True)  # Create graphs directory if it doesn't exist
        
        time_mark = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = graphs_dir / f"graph_{time_mark}.json"
        
        # Extract graph components
        nodes = graph_data.get('nodes', [])
        connections = graph_data.get('connections', [])
        links = graph_data.get('links', [])
        
        # Prepare data for saving (preserve all frontend data)
        save_data = {
            "nodes": nodes,
            "connections": connections,
            "links": links,
            "saved_at": time_mark
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Graph saved to: {save_path}")
        
        return {
            "status": "success",
            "message": f"Graph saved successfully",
            "filename": save_path.name,
            "path": str(save_path),
            "nodes_count": len(nodes),
            "connections_count": len(connections),
            "links_count": len(links)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save graph: {str(e)}"
        }

@app.get("/load-last-saved-graph")
def load_last_saved_graph():
    """Load the most recently saved graph from the graphs directory"""
    import json
    from pathlib import Path
    
    try:
        graphs_dir = USER_DIR / "graphs"
        
        if not graphs_dir.exists():
            return {
                "status": "error",
                "message": "No saved graphs directory found"
            }
        
        # Find all graph JSON files
        graph_files = list(graphs_dir.glob("graph_*.json"))
        
        if not graph_files:
            return {
                "status": "error", 
                "message": "No saved graphs found"
            }
        
        # Get the most recent file by modification time
        latest_file = max(graph_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            graph_data = json.load(f)
        
        return {
            "status": "success",
            "message": f"Loaded graph from {latest_file.name}",
            "filename": latest_file.name,
            "graph_data": graph_data
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load last saved graph: {str(e)}"
        }

@app.get("/load-last-force-graph")
def load_force_graph():
    """Load the most recent force graph from the force_graphs directory"""
    import json
    from pathlib import Path
    
    try:
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


# ═══════════════════════════════════════════════════════════════════════════════
# PYLINK GRAPH SAVE/LOAD ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/save-pylink-graph")
def save_pylink_graph(pylink_data: dict):
    """Save a pylink graph to the pygraphs directory"""
    from datetime import datetime
    import json
    
    try:
        pygraphs_dir = USER_DIR / "pygraphs"
        pygraphs_dir.mkdir(exist_ok=True)
        
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
    import json
    
    try:
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
    import json
    
    try:
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
    import time
    import traceback
    import numpy as np
    from pylinkage.joints import Crank, Revolute
    from pylinkage.linkage import Linkage
    
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


@app.post("/links")
def create_link(link_data: dict):
    """Create a new mechanical link"""
    try:
        # Import here to avoid circular imports
        from configs.link_models import Link
        import uuid
        
        # Ensure length is a proper float
        if 'length' in link_data:
            link_data['length'] = float(link_data['length'])
        
        # Create the link using the configs.link_models.Link
        link = Link(**link_data)
        
        # Add an ID for frontend tracking and ensure all numbers are properly serialized
        link_dict = link.model_dump()
        link_dict['id'] = str(uuid.uuid4())
        
        # Ensure length is a float for JSON serialization
        if 'length' in link_dict:
            link_dict['length'] = float(link_dict['length'])
        
        return {
            "status": "success",
            "message": "Link created successfully",
            "link": link_dict
        }
    except ValueError as e:
        return {
            "status": "error",
            "message": f"Invalid data type: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to create link: {str(e)}"
        }

@app.post("/links/modify")
def modify_link(request: dict):
    """Modify a link property"""
    try:
        link_id = request.get('id')
        property_name = request.get('property')
        new_value = request.get('value')
        
        if not all([link_id, property_name, new_value is not None]):
            return {
                "status": "error",
                "message": "Missing required fields: id, property, value"
            }
        
        # Type conversion for numeric properties
        if property_name in ['length', 'n_iterations'] and isinstance(new_value, (str, int, float)):
            try:
                new_value = float(new_value) if property_name == 'length' else int(new_value)
            except ValueError:
                return {
                    "status": "error",
                    "message": f"Invalid {property_name} value: must be a number"
                }
        
        # For now, just return success - in a full implementation,
        # you would store and retrieve the actual link objects
        return {
            "status": "success",
            "message": f"Link {link_id} property '{property_name}' updated to {new_value}",
            "id": link_id,
            "property": property_name,
            "value": new_value
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to modify link: {str(e)}"
        }

def extract_path_visualization_data(link_objects, n_iterations):
    """
    Extract path visualization data from computed links.
    Returns serializable data for frontend visualization.
    """
    import numpy as np
    
    # Color palette similar to matplotlib's Spectral colormap
    def get_spectral_color(t):
        """Generate color similar to matplotlib Spectral colormap"""
        # Simple approximation of Spectral colormap
        if t < 0.25:
            r = 158 + int((255-158) * t * 4)
            g = 1 + int((116-1) * t * 4) 
            b = 5 + int((9-5) * t * 4)
        elif t < 0.5:
            r = 255 - int((255-255) * (t-0.25) * 4)
            g = 116 + int((217-116) * (t-0.25) * 4)
            b = 9 + int((54-9) * (t-0.25) * 4)
        elif t < 0.75:
            r = 255 - int((255-171) * (t-0.5) * 4)
            g = 217 + int((221-217) * (t-0.5) * 4)
            b = 54 + int((164-54) * (t-0.5) * 4)
        else:
            r = 171 - int((171-94) * (t-0.75) * 4)
            g = 221 - int((221-79) * (t-0.75) * 4)
            b = 164 - int((164-162) * (t-0.75) * 4)
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    # Calculate bounds for visualization
    all_positions = []
    for link in link_objects:
        if hasattr(link, 'pos1') and link.pos1 is not None:
            all_positions.extend(link.pos1.tolist())
        if hasattr(link, 'pos2') and link.pos2 is not None:
            all_positions.extend(link.pos2.tolist())
    
    if not all_positions:
        return {"bounds": None, "links": [], "history_data": []}
    
    all_positions = np.array(all_positions)
    xmin, ymin = np.min(all_positions, axis=0)
    xmax, ymax = np.max(all_positions, axis=0)
    
    # Make square bounds with margin
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    delta = max(xdelta, ydelta)
    margin = 0.2
    
    bounds = {
        "xmin": float(xmin - delta * margin),
        "xmax": float(xmax + delta * margin), 
        "ymin": float(ymin - delta * margin),
        "ymax": float(ymax + delta * margin)
    }
    
    # Extract link animation data
    links_data = []
    for link in link_objects:
        if hasattr(link, 'pos1') and hasattr(link, 'pos2') and link.pos1 is not None and link.pos2 is not None:
            link_data = {
                "name": link.name,
                "is_driven": getattr(link, 'is_driven', False),
                "has_fixed": getattr(link, 'has_fixed', False),
                "has_constraint": getattr(link, 'has_constraint', False),
                "pos1": link.pos1.tolist(),  # Convert numpy to list for JSON serialization
                "pos2": link.pos2.tolist()
            }
            links_data.append(link_data)
    
    # Generate historical trail data for free links (similar to animate_script)
    history_data = []
    n_history = int(n_iterations * 0.66)
    
    for frame in range(n_iterations):
        frame_history = []
        
        # Calculate history window
        start_idx = max(0, frame - n_history)
        end_idx = frame
        
        for link in link_objects:
            if (hasattr(link, 'pos2') and link.pos2 is not None and 
                not getattr(link, 'has_fixed', False) and 
                not getattr(link, 'has_constraint', False)):
                
                # Get historical positions for this frame
                history_positions = link.pos2[start_idx:end_idx].tolist()
                
                # Generate colors for history trail
                history_colors = []
                for i, pos in enumerate(history_positions):
                    alpha = 1.0 / (1 + (len(history_positions) - i))
                    color_t = (start_idx + i) / n_iterations
                    color = get_spectral_color(color_t)
                    history_colors.append({"color": color, "alpha": alpha})
                
                frame_history.append({
                    "link_name": link.name,
                    "positions": history_positions,
                    "colors": history_colors
                })
        
        history_data.append(frame_history)
    
    return {
        "bounds": bounds,
        "links": links_data,
        "history_data": history_data,
        "n_iterations": n_iterations
    }

@app.post("/compute-graph")
def compute_graph(graph_data: dict):
    """Compute the mechanical linkage graph following the make_3link pattern"""
    
    from datetime import datetime
    import json
    
    time_mark = datetime.now().strftime("%Y%m%d_%H%M%S")
    graphs_dir = USER_DIR / "graphs"
    graphs_dir.mkdir(exist_ok=True)  # Create graphs directory if it doesn't exist
    save_path = graphs_dir / f"graph_{time_mark}.json"

    try:
        # Import here to avoid circular imports
        from link.graph_tools import make_graph, make_force_graph
        from configs.link_models import Link, Node
        import numpy as np
        n_iterations = 24
        nodes = graph_data.get('nodes', [])
        connections = graph_data.get('connections', [])
        links = graph_data.get('links', [])
        
        print("\n=== GRAPH COMPUTATION REQUEST ===")
        print(f"Nodes ({len(nodes)}), Connections ({len(connections)}), Links ({len(links)})")
        
        # Step 1: Convert dictionary links to Link objects with error checking
        link_objects = []
        link_errors = []
        
        for i, link_dict in enumerate(links):
            try:
                # Filter out frontend-specific fields, meta, and pos arrays that might be invalid
                # The new format has frontend-only fields in 'meta' sub-object
                excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2', 'meta']
                link_data = {k: v for k, v in link_dict.items() if k not in excluded_fields}
                
                # Remove any pos1/pos2 that might have slipped through with dict values
                if 'pos1' in link_data and isinstance(link_data['pos1'], dict):
                    del link_data['pos1']
                if 'pos2' in link_data and isinstance(link_data['pos2'], dict):
                    del link_data['pos2']
                
                # Ensure n_iterations is set (default to 24 like make_3link)
                #n_iterations = link_data.get('n_iterations', 24)
                assert link_data.get('n_iterations', 24) == n_iterations, f"Link n_iterations {link_data.get('n_iterations')} does not match expected {n_iterations}"
                print(f"Converting link {i+1}: {link_dict.get('name', 'unnamed')}")
                link_obj = Link(**link_data)
                
                # Initialize pos1 and pos2 arrays with correct shape (n_iterations, 2)
                # The Link model's post_init should handle this, but let's ensure it's correct
                if not hasattr(link_obj, 'pos1') or link_obj.pos1 is None:
                    link_obj.pos1 = np.zeros((n_iterations, 2))
                if not hasattr(link_obj, 'pos2') or link_obj.pos2 is None:
                    link_obj.pos2 = np.zeros((n_iterations, 2))
                    
                # Validate array shapes
                if link_obj.pos1.shape != (n_iterations, 2):
                    link_obj.pos1 = np.zeros((n_iterations, 2))
                if link_obj.pos2.shape != (n_iterations, 2):
                    link_obj.pos2 = np.zeros((n_iterations, 2))
                
                link_objects.append(link_obj)
                print(f"  ✓ Link '{link_obj.name}' created successfully with pos arrays shape {link_obj.pos1.shape}")
                
            except Exception as e:
                error_msg = f"Link {i+1} '{link_dict.get('name', 'unnamed')}': {str(e)}"
                link_errors.append(error_msg)
                print(f"  ✗ {error_msg}")
        
        # Step 2: Convert dictionary nodes to Node objects with error checking  
        node_objects = []
        node_errors = []
        
        for i, node_dict in enumerate(nodes):
            try:
                # Create proper node data for the backend Node model
                #n_iterations = node_dict.get('n_iterations', 24)  # Use 24 like make_3link
                assert node_dict.get('n_iterations', 24) == n_iterations, f"Node n_iterations {node_dict.get('n_iterations')} does not match expected {n_iterations}"
                node_data = {
                    'name': node_dict.get('name', node_dict.get('id', f'node_{i}')),
                    'n_iterations': n_iterations,
                    'fixed': node_dict.get('fixed', False)
                }
                
                # Handle fixed_loc properly
                if 'fixed_loc' in node_dict and node_dict['fixed_loc']:
                    node_data['fixed_loc'] = tuple(node_dict['fixed_loc'])
                elif node_data['fixed'] and 'pos' in node_dict:
                    # If node is fixed but no fixed_loc, use pos as fixed_loc
                    node_data['fixed_loc'] = tuple(node_dict['pos'])
                
                # Set init_pos from the frontend pos data or fixed_loc
                if 'pos' in node_dict and node_dict['pos']:
                    node_data['init_pos'] = tuple(node_dict['pos'])
                elif node_data.get('fixed_loc'):
                    # Use fixed_loc as init_pos if no pos provided
                    node_data['init_pos'] = node_data['fixed_loc']
                else:
                    # Default position if no pos data available
                    node_data['init_pos'] = (0.0, 0.0)
                
                print(f"Converting node {i+1}: {node_data['name']}")
                node_obj = Node(**node_data)
                
                node_objects.append(node_obj)
                pos_shape = node_obj.pos.shape if hasattr(node_obj, 'pos') and node_obj.pos is not None else "None"
                init_pos = node_obj.init_pos if hasattr(node_obj, 'init_pos') else "None"
                print(f"  ✓ Node '{node_obj.name}' created successfully with init_pos {init_pos} and pos array shape {pos_shape}")
                
            except Exception as e:
                error_msg = f"Node {i+1} '{node_dict.get('name', node_dict.get('id', 'unnamed'))}': {str(e)}"
                node_errors.append(error_msg)
                print(f"  ✗ {error_msg}")
        
        # Step 3: Validate connections structure
        # Support both new format (link_id) and old format (embedded link object)
        connection_errors = []
        for i, conn in enumerate(connections):
            try:
                from_node = conn.get('from_node')
                to_node = conn.get('to_node')
                # Support both new format (link_id) and old format (embedded link)
                link_id = conn.get('link_id')
                link_ref = conn.get('link')  # Old format fallback
                
                if not from_node:
                    connection_errors.append(f"Connection {i+1}: missing 'from_node'")
                if not to_node:
                    connection_errors.append(f"Connection {i+1}: missing 'to_node'")
                if not link_id and not link_ref:
                    connection_errors.append(f"Connection {i+1}: missing 'link_id' or 'link'")
                    
                # Check if referenced nodes exist
                node_names = [node.name for node in node_objects]
                if from_node and from_node not in node_names:
                    connection_errors.append(f"Connection {i+1}: from_node '{from_node}' not found in nodes")
                if to_node and to_node not in node_names:
                    connection_errors.append(f"Connection {i+1}: to_node '{to_node}' not found in nodes")
                    
            except Exception as e:
                connection_errors.append(f"Connection {i+1}: {str(e)}")
        
        # Report any errors gracefully to frontend
        all_errors = link_errors + node_errors + connection_errors
        if all_errors:
            print("Validation errors found:")
            for error in all_errors:
                print(f"  - {error}")
            return {
                "status": "error",
                "message": f"Validation errors found: {len(all_errors)} total errors. Please fix the following issues:",
                "errors": {
                    "link_errors": link_errors,
                    "node_errors": node_errors, 
                    "connection_errors": connection_errors,
                    "summary": all_errors[:5] + (["... and more errors"] if len(all_errors) > 5 else [])
                },
                "details": {
                    "nodes_processed": len(node_objects),
                    "links_processed": len(link_objects),
                    "total_errors": len(all_errors)
                }
            }
        
        # Step 4: Create connections using the proper Link objects
        # Support both new format (link_id) and old format (embedded link)
        # Build a lookup map for links by their meta.id and name
        link_by_id = {}
        link_by_name = {}
        for idx, link_obj in enumerate(link_objects):
            link_by_name[link_obj.name] = link_obj
            # Get the id from the original link_dict's meta field or id field
            link_dict = links[idx]
            link_id = None
            if 'meta' in link_dict and isinstance(link_dict['meta'], dict):
                link_id = link_dict['meta'].get('id')
            if not link_id:
                link_id = link_dict.get('id')
            if link_id:
                link_by_id[link_id] = link_obj
        
        processed_connections = []
        for conn in connections:
            matching_link = None
            
            # Try new format first (link_id)
            link_id = conn.get('link_id')
            if link_id and link_id in link_by_id:
                matching_link = link_by_id[link_id]
            else:
                # Fall back to old format (embedded link with name)
                link_ref = conn.get('link')
                if link_ref:
                    link_name = link_ref.get('name') if isinstance(link_ref, dict) else None
                    if link_name and link_name in link_by_name:
                        matching_link = link_by_name[link_name]
            
            if matching_link:
                processed_connections.append({
                    'from_node': conn.get('from_node'),
                    'to_node': conn.get('to_node'),
                    'link': matching_link
                })
            else:
                print(f"Warning: Could not find link object for connection: {conn}")
        
        # Step 5: Call make_graph_simple with n_iterations = 24 (like make_3link)
        print(f"Calling make_graph_simple with {len(node_objects)} nodes, {len(link_objects)} links, {len(processed_connections)} connections")
        
        graph = make_graph(
            processed_connections,
            link_objects,
            node_objects
        )
        
        print("✓ Graph computation completed successfully")
        
        _ = make_force_graph(graph)
        
        from link.graph_tools import run_graph
        times = np.linspace(0, 1, n_iterations )
        for i, t in enumerate(times):
            #print(i,t)
            _ = run_graph(
                i,
                    time=t,
                    omega=2*np.pi,
                    link_graph=graph,
                    verbose=1 if i == 0 else 0)  # Verbose for first iteration only
            



        from viz_tools.animate import animate_script
        time_mark = datetime.now().strftime("%Y%m%d_%H%M%S")
        agraphs_dir = USER_DIR / "animations"
        agraphs_dir.mkdir(exist_ok=True)  # Create force graphs directory if it doesn't exist
        save_path = str(agraphs_dir / f"animation_{time_mark}.gif")
        
        #links = [Link(**link.model_dump()) for link in graph.links]

        link_objs = []
        for edge in graph.edges(data=True):
            node1, node2, edge_data = edge
            link: Link = edge_data['link']
            link_objs.append(link)

        try:
            animate_script(
                n_iterations,
                link_objs,
                fname=save_path,
                square=True)
        except Exception as e:
            print(f"Warning: failed to make animation : {e}")

        # Extract path visualization data
        path_data = extract_path_visualization_data(link_objects, n_iterations)
        
        # Save the computed graph data after successful computation
        # Use new format: connections are lightweight (link_id only), links have meta field
        def get_link_meta(link_dict):
            """Extract meta from new format or construct from old format"""
            if 'meta' in link_dict and isinstance(link_dict['meta'], dict):
                return link_dict['meta']
            # Old format - construct meta from flat fields
            return {
                "id": link_dict.get("id"),
                "start_point": link_dict.get("start_point"),
                "end_point": link_dict.get("end_point"),
                "color": link_dict.get("color")
            }
        
        computed_graph_data = {
            "nodes": nodes,  # Save original node definitions
            # Connections are now lightweight - just references
            "connections": [
                {
                    "from_node": conn.get('from_node'),
                    "to_node": conn.get('to_node'),
                    "link_id": get_link_meta(links[i]).get('id') if i < len(links) else None
                }
                for i, conn in enumerate(connections)
            ],
            # Links with new structure (meta sub-object for frontend fields)
            "links": [
                {
                    "name": link.name,
                    "length": link.length,
                    "target_length": link.target_length,
                    "target_cost_func": link.target_cost_func,
                    "n_iterations": link.n_iterations,
                    "fixed_loc": list(link.fixed_loc) if link.fixed_loc else None,
                    "has_fixed": link.has_fixed,
                    "has_constraint": link.has_constraint,
                    "path": None,  # Don't serialize numpy path array
                    "is_driven": link.is_driven,
                    "flip": link.flip,
                    "zlevel": link.zlevel,
                    # Frontend-only fields in meta sub-object
                    "meta": get_link_meta(link_dict)
                }
                for link, link_dict in zip(link_objects, links)
            ],
            "saved_at": time_mark
        }
        
        try:
            with open(save_path, "w") as f:
                json.dump(computed_graph_data, f, indent=2)
            print(f"Computed graph data saved to: {save_path}")
        except Exception as e:
            print(f"Warning: Failed to save computed graph data: {e}")

        return {
            "status": "success", 
            "message": "Graph computed successfully",
            "nodes_count": len(nodes),
            "connections_count": len(connections),
            "links_count": len(links),
            "processed_connections": len(processed_connections),
            "computation_result": "Graph created and solved using make_graph_simple",
            "n_iterations": n_iterations,
            "graph_info": {
                "nodes": len(node_objects),
                "links": len(link_objects)
            },
            "path_data": path_data,
            "saved_file": str(save_path)
        }
        
    except Exception as e:
        print(f"Error computing graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to compute graph: {str(e)}",
            "errors": {
                "computation_error": [str(e)],
                "traceback": traceback.format_exc().split('\n')
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PYLINKAGE INTEGRATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/demo-4bar-pylinkage")
def demo_4bar_pylinkage(params: dict = None):
    """
    Create and simulate a demo 4-bar linkage using pylinkage directly.
    
    This demonstrates proper pylinkage usage without conversion from automata format.
    A 4-bar linkage in pylinkage uses only 2 joints:
    - Crank: rotating driver
    - Revolute: coupler-rocker connection point (with 2 distance constraints)
    
    Request body (all optional):
        {
            "ground_length": 30.0,
            "crank_length": 10.0,
            "coupler_length": 25.0,
            "rocker_length": 20.0,
            "crank_anchor": [20.0, 30.0],
            "n_iterations": 24,
            "include_ui_format": true  # Include nodes/links/connections for UI
        }
    
    Returns:
        {
            "status": "success",
            "message": str,
            "metadata": {...},  # Linkage parameters and structure explanation
            "path_data": {...},  # Visualization data
            "ui_graph": {...}    # Optional: nodes, links, connections for UI
        }
    """
    from link.pylinkage_bridge import simulate_demo_4bar, demo_4bar_to_ui_format
    import traceback
    
    try:
        # Use provided params or defaults
        if params is None:
            params = {}
        
        ground_length = params.get('ground_length', 30.0)
        crank_length = params.get('crank_length', 10.0)
        coupler_length = params.get('coupler_length', 25.0)
        rocker_length = params.get('rocker_length', 20.0)
        crank_anchor = tuple(params.get('crank_anchor', [20.0, 30.0]))
        n_iterations = params.get('n_iterations', 24)
        include_ui_format = params.get('include_ui_format', True)
        
        print(f"\n=== DEMO 4-BAR PYLINKAGE ===")
        print(f"Ground: {ground_length}, Crank: {crank_length}, Coupler: {coupler_length}, Rocker: {rocker_length}")
        
        result = simulate_demo_4bar(
            ground_length=ground_length,
            crank_length=crank_length,
            coupler_length=coupler_length,
            rocker_length=rocker_length,
            crank_anchor=crank_anchor,
            n_iterations=n_iterations
        )
        
        # Add UI format if requested
        if include_ui_format:
            ui_graph = demo_4bar_to_ui_format(
                ground_length=ground_length,
                crank_length=crank_length,
                coupler_length=coupler_length,
                rocker_length=rocker_length,
                crank_anchor=crank_anchor,
                n_iterations=n_iterations
            )
            result["ui_graph"] = ui_graph
        
        if result["status"] == "success":
            print(f"✓ Demo 4-bar completed in {result['execution_time_ms']:.2f}ms")
            if include_ui_format:
                print(f"  UI format: {len(ui_graph['nodes'])} nodes, {len(ui_graph['links'])} links")
        else:
            print(f"✗ Demo 4-bar failed: {result['message']}")
        
        return result
        
    except Exception as e:
        print(f"Error in demo_4bar_pylinkage: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Demo 4-bar failed: {str(e)}",
            "traceback": traceback.format_exc().split('\n')
        }


@app.post("/convert-to-pylinkage")
def convert_to_pylinkage(graph_data: dict):
    """
    Convert automata graph to pylinkage format and validate.
    
    This endpoint takes the graph data (nodes, links, connections) and converts
    it to a pylinkage Linkage object, returning a validation report.
    
    Request body:
        {
            "nodes": [...],
            "links": [...],
            "connections": [...]
        }
    
    Returns:
        {
            "status": "success" | "error",
            "message": str,
            "conversion_result": {
                "success": bool,
                "warnings": [...],
                "errors": [...],
                "joint_mapping": {...},
                "stats": {...},
                "serialized_linkage": {...}  # pylinkage's to_dict() output
            }
        }
    """
    from link.pylinkage_bridge import convert_to_pylinkage as do_convert
    from configs.link_models import Link, Node
    import traceback
    
    try:
        nodes_data = graph_data.get('nodes', [])
        links_data = graph_data.get('links', [])
        connections = graph_data.get('connections', [])
        
        print("\n=== PYLINKAGE CONVERSION REQUEST ===")
        print(f"Nodes: {len(nodes_data)}, Links: {len(links_data)}, Connections: {len(connections)}")
        
        # Convert raw dicts to model objects
        node_objects = []
        for i, node_dict in enumerate(nodes_data):
            try:
                n_iterations = node_dict.get('n_iterations', 24)
                node_data = {
                    'name': node_dict.get('name', node_dict.get('id', f'node_{i}')),
                    'n_iterations': n_iterations,
                    'fixed': node_dict.get('fixed', False)
                }
                if 'fixed_loc' in node_dict and node_dict['fixed_loc']:
                    node_data['fixed_loc'] = tuple(node_dict['fixed_loc'])
                elif node_data['fixed'] and 'pos' in node_dict:
                    node_data['fixed_loc'] = tuple(node_dict['pos'])
                if 'pos' in node_dict and node_dict['pos']:
                    node_data['init_pos'] = tuple(node_dict['pos'])
                elif node_data.get('fixed_loc'):
                    node_data['init_pos'] = node_data['fixed_loc']
                else:
                    node_data['init_pos'] = (0.0, 0.0)
                
                node_objects.append(Node(**node_data))
            except Exception as e:
                print(f"Warning: Failed to create Node from {node_dict}: {e}")
        
        link_objects = []
        for i, link_dict in enumerate(links_data):
            try:
                excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2', 'meta']
                link_data = {k: v for k, v in link_dict.items() if k not in excluded_fields}
                
                if 'pos1' in link_data and isinstance(link_data['pos1'], dict):
                    del link_data['pos1']
                if 'pos2' in link_data and isinstance(link_data['pos2'], dict):
                    del link_data['pos2']
                
                link_objects.append(Link(**link_data))
            except Exception as e:
                print(f"Warning: Failed to create Link from {link_dict}: {e}")
        
        # Run conversion
        result = do_convert(node_objects, link_objects, connections)
        
        if result.success:
            print(f"✓ Conversion successful: {result.stats}")
            return {
                "status": "success",
                "message": "Graph converted to pylinkage successfully",
                "conversion_result": result.to_dict()
            }
        else:
            print(f"✗ Conversion failed: {result.errors}")
            return {
                "status": "error",
                "message": "Conversion failed",
                "conversion_result": result.to_dict()
            }
    
    except Exception as e:
        print(f"Error in convert_to_pylinkage: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Conversion failed: {str(e)}",
            "traceback": traceback.format_exc().split('\n')
        }


@app.post("/simulate-pylinkage")
def simulate_pylinkage(graph_data: dict):
    """
    Run simulation using pylinkage solver and return trajectory data.
    
    This endpoint converts the graph to pylinkage, runs the simulation,
    and returns the results in the same format as the original solver
    for compatibility with existing visualization code.
    
    Request body:
        {
            "nodes": [...],
            "links": [...], 
            "connections": [...],
            "n_iterations": 24  # optional, defaults to 24
        }
    
    Returns:
        {
            "status": "success" | "error",
            "message": str,
            "solver": "pylinkage",
            "n_iterations": int,
            "execution_time_ms": float,
            "path_data": {
                "bounds": {...},
                "links": [...],
                "history_data": [...],
                "n_iterations": int
            }
        }
    """
    from link.pylinkage_bridge import (
        convert_to_pylinkage as do_convert,
        simulate_linkage,
        extract_path_visualization_data
    )
    from configs.link_models import Link, Node
    import traceback
    
    try:
        nodes_data = graph_data.get('nodes', [])
        links_data = graph_data.get('links', [])
        connections = graph_data.get('connections', [])
        n_iterations = graph_data.get('n_iterations', 24)
        
        print("\n=== PYLINKAGE SIMULATION REQUEST ===")
        print(f"Nodes: {len(nodes_data)}, Links: {len(links_data)}, Iterations: {n_iterations}")
        
        # Convert raw dicts to model objects (same as convert route)
        node_objects = []
        for i, node_dict in enumerate(nodes_data):
            try:
                node_n_iter = node_dict.get('n_iterations', n_iterations)
                node_data = {
                    'name': node_dict.get('name', node_dict.get('id', f'node_{i}')),
                    'n_iterations': node_n_iter,
                    'fixed': node_dict.get('fixed', False)
                }
                if 'fixed_loc' in node_dict and node_dict['fixed_loc']:
                    node_data['fixed_loc'] = tuple(node_dict['fixed_loc'])
                elif node_data['fixed'] and 'pos' in node_dict:
                    node_data['fixed_loc'] = tuple(node_dict['pos'])
                if 'pos' in node_dict and node_dict['pos']:
                    node_data['init_pos'] = tuple(node_dict['pos'])
                elif node_data.get('fixed_loc'):
                    node_data['init_pos'] = node_data['fixed_loc']
                else:
                    node_data['init_pos'] = (0.0, 0.0)
                
                node_objects.append(Node(**node_data))
            except Exception as e:
                print(f"Warning: Failed to create Node from {node_dict}: {e}")
        
        link_objects = []
        for i, link_dict in enumerate(links_data):
            try:
                excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2', 'meta']
                link_data = {k: v for k, v in link_dict.items() if k not in excluded_fields}
                
                if 'pos1' in link_data and isinstance(link_data['pos1'], dict):
                    del link_data['pos1']
                if 'pos2' in link_data and isinstance(link_data['pos2'], dict):
                    del link_data['pos2']
                
                link_objects.append(Link(**link_data))
            except Exception as e:
                print(f"Warning: Failed to create Link from {link_dict}: {e}")
        
        # Step 1: Convert to pylinkage
        conversion_result = do_convert(node_objects, link_objects, connections)
        
        if not conversion_result.success:
            return {
                "status": "error",
                "message": "Failed to convert graph to pylinkage",
                "errors": conversion_result.errors,
                "warnings": conversion_result.warnings
            }
        
        # Step 2: Run simulation
        sim_result = simulate_linkage(
            conversion_result.linkage,
            n_iterations=n_iterations,
            use_fast=True
        )
        
        if not sim_result.success:
            return {
                "status": "error",
                "message": "Simulation failed",
                "errors": sim_result.errors
            }
        
        # Step 3: Extract visualization data
        path_data = extract_path_visualization_data(
            sim_result.trajectories,
            link_objects,
            conversion_result.joint_mapping,
            n_iterations
        )
        
        print(f"✓ Simulation completed in {sim_result.execution_time_ms:.2f}ms")
        
        return {
            "status": "success",
            "message": "Simulation completed successfully",
            "solver": "pylinkage",
            "n_iterations": n_iterations,
            "execution_time_ms": sim_result.execution_time_ms,
            "path_data": path_data,
            "conversion_stats": conversion_result.stats,
            "conversion_warnings": conversion_result.warnings
        }
    
    except Exception as e:
        print(f"Error in simulate_pylinkage: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Simulation failed: {str(e)}",
            "traceback": traceback.format_exc().split('\n')
        }


@app.post("/compare-solvers")
def compare_solvers_route(graph_data: dict):
    """
    Run both automata and pylinkage solvers and compare results.
    
    This is useful for validating the pylinkage integration produces
    equivalent results to the original solver.
    
    Request body:
        {
            "nodes": [...],
            "links": [...],
            "connections": [...],
            "n_iterations": 24
        }
    
    Returns:
        {
            "status": "success" | "error",
            "automata_result": {...},
            "pylinkage_result": {...},
            "comparison": {
                "max_position_error": float,
                "mean_position_error": float,
                "automata_time_ms": float,
                "pylinkage_time_ms": float,
                "speedup_factor": float
            }
        }
    """
    import traceback
    import time
    import numpy as np
    from link.pylinkage_bridge import convert_to_pylinkage as do_convert, simulate_linkage
    from link.graph_tools import make_graph, run_graph
    from configs.link_models import Link, Node
    
    try:
        nodes_data = graph_data.get('nodes', [])
        links_data = graph_data.get('links', [])
        connections_data = graph_data.get('connections', [])
        n_iterations = graph_data.get('n_iterations', 24)
        
        print("\n=== SOLVER COMPARISON REQUEST ===")
        
        # Prepare objects (shared between both solvers)
        node_objects = []
        for i, node_dict in enumerate(nodes_data):
            node_n_iter = node_dict.get('n_iterations', n_iterations)
            node_data = {
                'name': node_dict.get('name', node_dict.get('id', f'node_{i}')),
                'n_iterations': node_n_iter,
                'fixed': node_dict.get('fixed', False)
            }
            if 'fixed_loc' in node_dict and node_dict['fixed_loc']:
                node_data['fixed_loc'] = tuple(node_dict['fixed_loc'])
            elif node_data['fixed'] and 'pos' in node_dict:
                node_data['fixed_loc'] = tuple(node_dict['pos'])
            if 'pos' in node_dict and node_dict['pos']:
                node_data['init_pos'] = tuple(node_dict['pos'])
            elif node_data.get('fixed_loc'):
                node_data['init_pos'] = node_data['fixed_loc']
            else:
                node_data['init_pos'] = (0.0, 0.0)
            node_objects.append(Node(**node_data))
        
        link_objects_automata = []
        link_objects_pylinkage = []
        for i, link_dict in enumerate(links_data):
            excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2', 'meta']
            link_data = {k: v for k, v in link_dict.items() if k not in excluded_fields}
            if 'pos1' in link_data and isinstance(link_data['pos1'], dict):
                del link_data['pos1']
            if 'pos2' in link_data and isinstance(link_data['pos2'], dict):
                del link_data['pos2']
            # Create separate instances for each solver
            link_objects_automata.append(Link(**link_data))
            link_objects_pylinkage.append(Link(**link_data))
        
        results = {
            "automata": {"success": False, "time_ms": 0, "error": None},
            "pylinkage": {"success": False, "time_ms": 0, "error": None},
            "comparison": {}
        }
        
        # ─────────────────────────────────────────────────────────────
        # Run Automata Solver
        # ─────────────────────────────────────────────────────────────
        automata_positions = {}
        try:
            # Build lookup maps for connections
            link_by_id = {}
            link_by_name = {}
            for idx, link_obj in enumerate(link_objects_automata):
                link_by_name[link_obj.name] = link_obj
                link_dict = links_data[idx]
                link_id = None
                if 'meta' in link_dict and isinstance(link_dict['meta'], dict):
                    link_id = link_dict['meta'].get('id')
                if not link_id:
                    link_id = link_dict.get('id')
                if link_id:
                    link_by_id[link_id] = link_obj
            
            processed_connections = []
            for conn in connections_data:
                matching_link = None
                link_id = conn.get('link_id')
                if link_id and link_id in link_by_id:
                    matching_link = link_by_id[link_id]
                else:
                    link_ref = conn.get('link')
                    if link_ref:
                        link_name = link_ref.get('name') if isinstance(link_ref, dict) else None
                        if link_name and link_name in link_by_name:
                            matching_link = link_by_name[link_name]
                if matching_link:
                    processed_connections.append({
                        'from_node': conn.get('from_node'),
                        'to_node': conn.get('to_node'),
                        'link': matching_link
                    })
            
            start_time = time.perf_counter()
            graph = make_graph(processed_connections, link_objects_automata, node_objects)
            times = np.linspace(0, 1, n_iterations)
            for i, t in enumerate(times):
                run_graph(i, time=t, omega=2*np.pi, link_graph=graph, verbose=0)
            
            results["automata"]["time_ms"] = (time.perf_counter() - start_time) * 1000
            results["automata"]["success"] = True
            
            # Extract positions
            for link in link_objects_automata:
                automata_positions[link.name] = {
                    "pos1": link.pos1.copy(),
                    "pos2": link.pos2.copy()
                }
            
            print(f"✓ Automata solver completed in {results['automata']['time_ms']:.2f}ms")
            
        except Exception as e:
            results["automata"]["error"] = str(e)
            print(f"✗ Automata solver failed: {e}")
        
        # ─────────────────────────────────────────────────────────────
        # Run Pylinkage Solver
        # ─────────────────────────────────────────────────────────────
        pylinkage_positions = {}
        try:
            start_time = time.perf_counter()
            conversion_result = do_convert(node_objects, link_objects_pylinkage, connections_data)
            
            if conversion_result.success:
                sim_result = simulate_linkage(conversion_result.linkage, n_iterations)
                results["pylinkage"]["time_ms"] = (time.perf_counter() - start_time) * 1000
                
                if sim_result.success:
                    results["pylinkage"]["success"] = True
                    pylinkage_positions = sim_result.trajectories
                    print(f"✓ Pylinkage solver completed in {results['pylinkage']['time_ms']:.2f}ms")
                else:
                    results["pylinkage"]["error"] = sim_result.errors
            else:
                results["pylinkage"]["error"] = conversion_result.errors
                
        except Exception as e:
            results["pylinkage"]["error"] = str(e)
            print(f"✗ Pylinkage solver failed: {e}")
        
        # ─────────────────────────────────────────────────────────────
        # Compare Results
        # ─────────────────────────────────────────────────────────────
        if results["automata"]["success"] and results["pylinkage"]["success"]:
            position_errors = []
            
            # Compare trajectories for matching joints
            for link_name, automata_pos in automata_positions.items():
                if link_name in pylinkage_positions:
                    pylinkage_traj = pylinkage_positions[link_name]
                    # Compare pos2 (end positions)
                    if automata_pos["pos2"] is not None and pylinkage_traj is not None:
                        diff = np.abs(automata_pos["pos2"] - pylinkage_traj)
                        position_errors.extend(diff.flatten().tolist())
            
            if position_errors:
                results["comparison"] = {
                    "max_position_error": float(np.max(position_errors)),
                    "mean_position_error": float(np.mean(position_errors)),
                    "automata_time_ms": results["automata"]["time_ms"],
                    "pylinkage_time_ms": results["pylinkage"]["time_ms"],
                    "speedup_factor": results["automata"]["time_ms"] / max(results["pylinkage"]["time_ms"], 0.001)
                }
        
        return {
            "status": "success" if (results["automata"]["success"] or results["pylinkage"]["success"]) else "error",
            "automata_result": results["automata"],
            "pylinkage_result": results["pylinkage"],
            "comparison": results["comparison"]
        }
        
    except Exception as e:
        print(f"Error in compare_solvers: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Comparison failed: {str(e)}",
            "traceback": traceback.format_exc().split('\n')
        }
