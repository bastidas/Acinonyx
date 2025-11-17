from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/load-graph")
def load_graph():
    """Load graph data from blank JSON file in user directory"""
    import json
    import os
    from pathlib import Path
    
    try:
        # Get the user directory path relative to the backend
        user_dir = Path(__file__).parent.parent / "user"
        json_file = user_dir / "force.json"
        #json.dump(graph, open(user_dir / "force.json", "w"))

        
        if not json_file.exists():
            return {
                "error": "Graph file not found",
                "path": str(json_file)
            }
        
        with open(json_file, 'r') as f:
            graph_data = json.load(f)
            print(graph_data)
        return graph_data
        
    except Exception as e:
        return {
            "error": f"Failed to load graph data: {str(e)}"
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

@app.post("/compute-graph")
def compute_graph(graph_data: dict):
    """Compute the mechanical linkage graph following the make_3link pattern"""
    
    try:
        # Import here to avoid circular imports
        from link.graph_tools import make_graph_simple
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
                # Filter out frontend-specific fields and pos arrays that might be invalid
                excluded_fields = ['start_point', 'end_point', 'color', 'id', 'pos1', 'pos2']
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
        connection_errors = []
        for i, conn in enumerate(connections):
            try:
                from_node = conn.get('from_node')
                to_node = conn.get('to_node') 
                link_ref = conn.get('link')
                
                if not from_node:
                    connection_errors.append(f"Connection {i+1}: missing 'from_node'")
                if not to_node:
                    connection_errors.append(f"Connection {i+1}: missing 'to_node'")
                if not link_ref:
                    connection_errors.append(f"Connection {i+1}: missing 'link'")
                    
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
        # We need to map the connection links to the actual Link objects we created
        processed_connections = []
        for conn in connections:
            # Find the matching link object by name
            link_name = conn.get('link', {}).get('name')
            matching_link = None
            for link_obj in link_objects:
                if link_obj.name == link_name:
                    matching_link = link_obj
                    break
            
            if matching_link:
                processed_connections.append({
                    'from_node': conn.get('from_node'),
                    'to_node': conn.get('to_node'),
                    'link': matching_link
                })
            else:
                print(f"Warning: Could not find link object for connection with link name '{link_name}'")
        
        # Step 5: Call make_graph_simple with n_iterations = 24 (like make_3link)
        print(f"Calling make_graph_simple with {len(node_objects)} nodes, {len(link_objects)} links, {len(processed_connections)} connections")
        
        graph = make_graph_simple(
            processed_connections,
            link_objects,
            node_objects, 
            n_iterations=n_iterations
        )
        
        print("✓ Graph computation completed successfully")
        
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
            }
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
