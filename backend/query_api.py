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
    """Compute the mechanical linkage graph"""
    try:
        # Import here to avoid circular imports
        from structs.basic import make_graph
        
        nodes = graph_data.get('nodes', [])
        connections = graph_data.get('connections', [])
        links = graph_data.get('links', [])
        
        print("\n=== GRAPH COMPUTATION REQUEST ===")
        print(f"Nodes ({len(nodes)}):")
        for node in nodes:
            print(f"  - {node}")
        
        print(f"\nConnections ({len(connections)}):")
        for conn in connections:
            print(f"  - {conn}")
        
        print(f"\nLinks ({len(links)}):")
        for link in links:
            print(f"  - {link.get('name', 'unnamed')}: length={link.get('length', 'N/A')}")
        
        # Convert dictionary links to Link objects
        from configs.link_models import Link
        link_objects = []
        for i, link_dict in enumerate(links):
            try:
                # Create Link object from dictionary, filtering out frontend-specific fields
                link_data = {k: v for k, v in link_dict.items() 
                           if k not in ['start_point', 'end_point', 'color', 'id']}
                print(f"  - Converting link {i+1}: {link_dict.get('name', 'unnamed')}")
                print(f"    Filtered data: {link_data}")
                link_obj = Link(**link_data)
                link_objects.append(link_obj)
                print(f"    ✓ Successfully converted to Link object")
            except Exception as e:
                print(f"    ✗ Failed to convert link {link_dict.get('name', 'unnamed')}: {e}")
                print(f"    Link data: {link_dict}")
                # Continue with other links even if one fails
        
        print(f"Successfully converted {len(link_objects)} links to Link objects")
        
        # Convert dictionary nodes to Node objects
        from configs.link_models import Node
        node_objects = []
        for i, node_dict in enumerate(nodes):
            try:
                # Handle both old (x,y) and new (pos) formats
                if 'pos' in node_dict:
                    pos = tuple(node_dict['pos'])
                else:
                    pos = (node_dict.get('x', 0), node_dict.get('y', 0))
                
                # Ensure required fields are present
                node_data = {
                    'id': node_dict.get('id', f'node_{i}'),
                    'pos': pos,
                    'fixed': node_dict.get('fixed', False),
                    'fixed_loc': tuple(node_dict['fixed_loc']) if node_dict.get('fixed_loc') else (pos if node_dict.get('fixed', False) else None)
                }
                print(f"  - Converting node {i+1}: {node_data['id']}")
                print(f"    Node data: {node_data}")
                node_obj = Node(**node_data)
                node_objects.append(node_obj)
                print(f"    ✓ Successfully converted to Node object")
            except Exception as e:
                print(f"    ✗ Failed to convert node {node_dict.get('id', f'node_{i}')}: {e}")
                print(f"    Node data: {node_dict}")
                # Continue with other nodes even if one fails
        
        print(f"Successfully converted {len(node_objects)} nodes to Node objects")
        
        # Call the make_graph function with the frontend data
        result = make_graph(connections, link_objects, node_objects)
        
        return {
            "status": "success",
            "message": "Graph computed successfully",
            "nodes_count": len(nodes),
            "connections_count": len(connections),
            "links_count": len(links),
            "computation_result": result
        }
        
    except Exception as e:
        print(f"Error computing graph: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to compute graph: {str(e)}"
        }
