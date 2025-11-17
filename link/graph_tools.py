

import networkx as nx
from link.tools import *
import logging
from link.tools import get_tri_angles, get_tri_pos
logger = logging.getLogger()
from itertools import combinations
from configs.link_models import Link, Node
from pydantic import BaseModel


def minimal_bounds(free_link, constrained_link):
    """
    Find the minimum distance between the free link and the constrained link by drawing circles around them
    and calculating the closest distance between the circles.
    """

    # Extract the fixed locations of the links
    free_link_center = free_link.fixed_loc
    constrained_link_center = constrained_link.fixed_loc

    # Ensure both links have fixed locations
    if free_link_center is None or constrained_link_center is None:
        raise ValueError("Both links must have fixed locations to calculate the distance.")

    # Calculate the Euclidean distance between the centers of the circles
    center_distance = np.sqrt((free_link_center[0] - constrained_link_center[0])**2 +
                           (free_link_center[1] - constrained_link_center[1])**2)

    # Subtract the radii of the circles (lengths of the links) to get the closest distance
    closest_distance = max(0, center_distance - (free_link.length + constrained_link.length))
    return closest_distance


def validate_graph(graph):
    """
    Validate the graph structure to ensure all links and nodes are properly defined.
    """
    required_node_fields = Node.model_fields
    required_link_fields = Link.model_fields
    n_iterations = None
    
    n_driven_links = 0
    n_fixed_nodes = 0
    n_has_fixed_links = 0

    node_ids = list(graph.nodes())
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"Duplicate node IDs found in graph: {[nid for nid in node_ids if node_ids.count(nid) > 1]}")

    for node in graph.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        node_fields = set(node_data.keys())
        for field in required_node_fields:
            if field not in node_fields:
                raise ValueError(f"Node {node_id} is missing required field: {field}")
        
        if n_iterations is None:
            n_iterations = node_data.get('n_iterations', None)
            expected_shape = (n_iterations, 2)
        if n_iterations is None:
            raise ValueError(f"Node {node_id} is missing 'n_iterations' field.")
        
        actual_shape = node_data['pos'].shape
        if expected_shape != actual_shape:
            raise ValueError(f"Node {node_id} pos has shape {actual_shape}, expected {expected_shape}")
        

        if node_data["fixed"]:
            n_fixed_nodes += 1
  
    # # Check for duplicate link names
    # all_link_names = [edge[2].get("name") for edge in graph.edges(data=True)]
    # if len(all_link_names) != len(set(all_link_names)):
    #     duplicates = [name for name in all_link_names if all_link_names.count(name) > 1]
    #     raise ValueError(f"Duplicate link names found in graph: {list(set(duplicates))}")
        
    # for edge in graph.edges(data=True):
    #     from_node = edge[0]
    #     to_node = edge[1]
    #     link_data = edge[2]
    #     link_fields = set(link_data.keys())
    #     for field in required_link_fields:
    #         if field not in link_fields:
    #             raise ValueError(f"Link from {from_node} to {to_node} is missing required field: {field}")  
    
    #     actual_shape1 = link_data['pos1'].shape
    #     if expected_shape != actual_shape1:
    #         raise ValueError(f"Link from {from_node} to {to_node} pos1 has shape {actual_shape1}, expected {expected_shape}")
    #     actual_shape2 = link_data['pos2'].shape
    #     if expected_shape != actual_shape2:
    #         raise ValueError(f"Link from {from_node} to {to_node} pos2 has shape {actual_shape2}, expected {expected_shape}")
        
    #     if link_data["is_driven"]:
    #         n_driven_links += 1
    #     if link_data["has_fixed"]:
    #         n_has_fixed_links += 1

    #     link_name = link_data.get("name")
    #     if link_name is None:
    #         raise ValueError(f"Link from {from_node} to {to_node} is missing 'name' field")


    # Check for duplicate link names
    all_link_names = [edge[2]['link'].name for edge in graph.edges(data=True)]
    if len(all_link_names) != len(set(all_link_names)):
        duplicates = [name for name in all_link_names if all_link_names.count(name) > 1]
        raise ValueError(f"Duplicate link names found in graph: {list(set(duplicates))}")
        
    for edge in graph.edges(data=True):
        from_node = edge[0]
        to_node = edge[1]
        link_obj = edge[2]['link']
        link_fields = set(link_obj.model_fields.keys())
        for field in required_link_fields:
            if field not in link_fields:
                raise ValueError(f"Link from {from_node} to {to_node} is missing required field: {field}")  
    
        actual_shape1 = link_obj.pos1.shape
        if expected_shape != actual_shape1:
            raise ValueError(f"Link from {from_node} to {to_node} pos1 has shape {actual_shape1}, expected {expected_shape}")
        actual_shape2 = link_obj.pos2.shape
        if expected_shape != actual_shape2:
            raise ValueError(f"Link from {from_node} to {to_node} pos2 has shape {actual_shape2}, expected {expected_shape}")
        
        if link_obj.is_driven:
            n_driven_links += 1
        if link_obj.has_fixed:
            n_has_fixed_links += 1

        link_name = link_obj.name
        if link_name is None:
            raise ValueError(f"Link from {from_node} to {to_node} is missing 'name' field")  
        
    assert n_driven_links == 1, "There must be exactly one driven link in the graph."
    assert n_fixed_nodes >= 1, "There must be at least one fixed node in the graph."
    assert n_has_fixed_links >= 2, f"There must be at least two links with a fixed end in the graph, but there are {n_has_fixed_links}."
    print("Graph validation passed.")
    return True


def make_graph_simple(connections,
                      links: list[Link],
                      nodes: list[Node],
                      n_iterations=12,
                      as_dict=False):
    """
    Create a NetworkX graph from frontend connections and links data
    example of connections;
    connections = [
            {"from_node": "node1", "to_node": "node2", "link": link1},
            {"from_node": "node2", "to_node": "node3", "link": freelink},
            {"from_node": "node3", "to_node": "node4", "link": link2},]
                
    """
    graph = nx.Graph()  # Use undirected graph like the original make_3link
    for node in nodes:
        graph.add_node(node.name, 
                      **node.as_dict())
    

    for i, conn in enumerate(connections):
        from_node = conn.get('from_node')
        to_node = conn.get('to_node')
        link_obj = conn.get('link')
        
        if from_node and to_node and link_obj:
            print(f"Connection {i+1}: {from_node} -> {to_node} via link '{link_obj.name}'")
            try:
                # Filter out frontend-specific fields
                #clean_link_data = {k: v for k, v in link_data.items() 
                #                 if k not in ['start_point', 'end_point', 'color', 'id']}
                #link_obj = Link(**clean_link_data)
                if as_dict: 
                    graph.add_edge(from_node, to_node, **link_obj.as_dict())
                else:
                    graph.add_edge(from_node, to_node, link=link_obj)
                print(f"  ✓ Converted connection link to Link object: {link_obj.name}")
                print(link_obj.as_dict())
            except Exception as e:
                print(f"  ✗ Failed to convert connection link: {e}")
                print(f"    Link data: {link_obj.as_dict()}")
                # Skip this connection if Link conversion fails
                #continue
        else:
            print(f"ERROR: Skipping invalid connection {i+1}: {conn}")

    validate_graph(graph)
    return graph

def make_graph(connections,
               links: list[Link],
               nodes: list[Node],
               n_iterations=12):
    """Create a NetworkX graph from frontend connections and links data"""
    print("\n=== MAKE_GRAPH CALLED ===")
    print(f"Received {len(connections)} connections, {len(links)} links, and {len(nodes)} nodes")

    n_iterations=3

    # Create a new directed graph
    graph = nx.DiGraph()
    
    # Add nodes to the graph with their properties
    for node in nodes:
        graph.add_node(node.id, 
                      pos=node.pos,
                      fixed=node.fixed,
                      fixed_loc=node.fixed_loc)
        print(f"Added node {node.id}: pos={node.pos}, fixed={node.fixed}")
        print(f"fixed_loc: {node.fixed_loc}")
    
    # Process connections
    for i, conn in enumerate(connections):
        from_node = conn.get('from_node')
        to_node = conn.get('to_node')
        link_data = conn.get('link')
        
        if from_node and to_node and link_data:
            print(f"Connection {i+1}: {from_node} -> {to_node} via link '{link_data.get('name', 'unnamed')}'")
            
            # Ensure nodes exist in the graph (add them if missing)
            if from_node not in graph:
                # Find matching node object or create default
                from_node_obj = next((n for n in nodes if n.id == from_node), None)
                if from_node_obj:
                    graph.add_node(from_node,
                                   pos=from_node_obj.pos,
                                   fixed=from_node_obj.fixed,
                                   fixed_loc=from_node_obj.fixed_loc)
                else:
                    # Create default node if not found
                    graph.add_node(from_node, pos=(0, 0), fixed=False, fixed_loc=None)
                    print(f"  Warning: Created default node for {from_node}")
            
            if to_node not in graph:
                # Find matching node object or create default
                to_node_obj = next((n for n in nodes if n.id == to_node), None)
                if to_node_obj:
                    graph.add_node(to_node, pos=to_node_obj.pos, fixed=to_node_obj.fixed, fixed_loc=to_node_obj.fixed_loc)
                else:
                    # Create default node if not found
                    graph.add_node(to_node, pos=(0, 0), fixed=False, fixed_loc=None)
                    print(f"  Warning: Created default node for {to_node}")
            
            # Convert dictionary link_data to Link object
            try:
                # Filter out frontend-specific fields
                clean_link_data = {k: v for k, v in link_data.items() 
                                 if k not in ['start_point', 'end_point', 'color', 'id']}
                link_obj = Link(**clean_link_data)

                #  Connection 1: node1 -> node2 via link 'drive_link'
                # adding link drive_link with data: {'length': 2.1843315432490047, 'name': 'drive_link', 'n_iterations': 100, 'fixed_loc': None, 'has_fixed': False, 'has_constraint': False, 'path': None, 'is_driven': True, 'flip': False, 'zlevel': 0, 'pos1': {'0.0': 0}, 'pos2': {'0.0': 0}}
                #   ✓ Converted connection link to Link object: drive_link
                # Connection 2: node2 -> node3 via link 'link2'
                # adding link link2 with data: {'length': 2.6826852219371546, 'name': 'link2', 'n_iterations': 100, 'fixed_loc': None, 'has_fixed': False, 'has_constraint': False, 'path': None, 'is_driven': False, 'flip': False, 'zlevel': 1, 'pos1': {'0.0': 0}, 'pos2': {'0.0': 0}}
                #   ✓ Converted connection link to Link object: link2

                print("adding link", link_obj.name, "with data:", clean_link_data)
                graph.add_edge(from_node, to_node, link=link_obj)
                print(f"  ✓ Converted connection link to Link object: {link_obj.name}")
            except Exception as e:
                print(f"  ✗ Failed to convert connection link: {e}")
                print(f"    Link data: {link_data}")
                # Skip this connection if Link conversion fails
                continue
        else:
            print(f"Skipping invalid connection {i+1}: {conn}")
    
    print(f"\nCreated directed graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    print(f"Nodes: {list(graph.nodes())}")
    print(f"Edges: {list(graph.edges())}")

    validate_graph(graph)
    return {
        "graph_type": "directed",
        "nodes": list(graph.nodes()),
        "edges": list(graph.edges()),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges()
    }


def solve_graph_links(
    i,
    time,
    omega,
    link_graph,
    rtol=0.05,
    verbose=0 
    ):
    
    cycle_sets = [s for s in nx.simple_cycles(link_graph, length_bound=3)]
    tri_link_sets = []
    for cycle in cycle_sets :
        edge_data1 = link_graph.get_edge_data(cycle[0], cycle[1]) 
        edge_data2 = link_graph.get_edge_data(cycle[1], cycle[2]) 
        edge_data3 = link_graph.get_edge_data(cycle[2], cycle[0]) 
        links = [edge_data["link"] for edge_data in [edge_data1, edge_data2, edge_data3]]
        tri_link_sets.append(links)
    
    links = []
    for edge in link_graph.edges(data=True):
        node1, node2, edge_data = edge
        link = edge_data['link']
        if verbose >= 1:
            print(link.name)
        links.append(link)
        node1_data = link_graph.nodes[node1]
        node2_data = link_graph.nodes[node2]
        nodes_data = [node1_data, node2_data]

        fixed_nodes_data = [node for node in nodes_data if node['fixed']]
        #is_fixed_bool = [nodedat['fixed'] for nodedat in nodes_data]
        #is_fixed_loc = np.where(is_fixed_bool)[0]

        if len(fixed_nodes_data) == 2:
            link.pos1[i] = node1_data['fixed_loc']
            link.pos2[i] = node2_data['fixed_loc']

        if len(fixed_nodes_data) == 1:
            err_msg = f"As convention if a node is fixed, its edge must be directed away, however link `{link.name}` is backwards."
            assert node1_data['fixed'], err_msg
            if verbose > 0:
                print(f"\tlink `{link.name}` is fixed")
            fixed_node_data = node1_data
            # print("\t\t", fixed_node_data)
            fixed_loc = fixed_node_data['fixed_loc']
            link.pos1[i] = fixed_loc
            if verbose > 3:
                print("\t\t\tpos1 set:", fixed_loc)

            if link.is_driven:
                if verbose > 0:
                    print(f"\t\tlink `{link.name}` is driven")
                x = fixed_loc[0] + link.length * np.cos(omega * time)
                y = fixed_loc[1] + link.length * np.sin(omega * time)

                link.pos2[i] = (x, y)
                if verbose > 3:
                    print("\t\t\tlink.pos2[i]:", link.pos2[i])

        if len(fixed_nodes_data) == 0:

            assert not link.has_fixed, "The nodes are not fixed, but the link is fixed."
            if verbose > 0:
                print(f"\tlink `{link.name}` is free")

            in_edges_node1 = list(link_graph.in_edges(node1, data=True))
            out_edges_node1 = list(link_graph.out_edges(node1, data=True))
            in_links_node1 = [edge[2]["link"] for edge in in_edges_node1]
            out_links_node1 = [edge[2]["link"] for edge in out_edges_node1]

            in_edges_node2 = list(link_graph.in_edges(node2, data=True))
            out_edges_node2 = list(link_graph.out_edges(node2, data=True))
            in_links_node2 = [edge[2]["link"] for edge in in_edges_node2]
            out_links_node2 = [edge[2]["link"] for edge in out_edges_node2]

            out1_driven_links = [l for l in out_links_node1 if l.is_driven]
            in1_driven_links = [l for l in in_links_node1 if l.is_driven]
            out2_driven_links = [l for l in out_links_node2 if l.is_driven]
            in2_driven_links = [l for l in in_links_node2 if l.is_driven]
            all_driven_links = []
            for l in out1_driven_links + in1_driven_links + out2_driven_links + in2_driven_links:
                if l not in all_driven_links:
                    all_driven_links.append(l)

            all_free_links = [l for l in out_links_node1 if not l.has_fixed]
            all_free_links += [l for l in in_links_node1 if not l.has_fixed and l not in all_free_links]
            all_free_links += [l for l in out_links_node2 if not l.has_fixed and l not in all_free_links]
            all_free_links += [l for l in in_links_node2 if not l.has_fixed and l not in all_free_links]

            if verbose >= 3:
                print("\t\t\t node1 in ", node1, ": ", [l.name for l in in_links_node1])
                print("\t\t\t node1 out", node1, ": ", [l.name for l in out_links_node1])
                print("\t\t\t node2 in ", node2, ": ", [l.name for l in in_links_node2])
                print("\t\t\t node2 out", node2, ": ", [l.name for l in out_links_node2])

            out1_constrained_links = [l for l in out_links_node1 if l.has_fixed and not l.is_driven]
            in1_constrained_links = [l for l in in_links_node1 if l.has_fixed and not l.is_driven]
            out2_constrained_links = [l for l in out_links_node2 if l.has_fixed and not l.is_driven]
            in2_constrained_links = [l for l in in_links_node2 if l.has_fixed and not l.is_driven]
            
            all_constrained_links = []
            for l in out1_constrained_links + in1_constrained_links + out2_constrained_links + in2_constrained_links:
                if l not in all_constrained_links:
                    all_constrained_links.append(l)
            
            if verbose >= 3:
                if len(all_driven_links) == 0:
                    print("\t\tno driven link")

                print("\t\tfree links", [l.name for l in all_free_links])

                if len(all_constrained_links) == 0:
                    print("\t\tno constrained link")

                if len(all_driven_links) > 1:
                    print("\t\tmore than one driven link, this is not allowed")

            if len(all_constrained_links) == 1 and len(all_driven_links) == 1:
                #print("\t\twe can solve this as a constrained link")
                guess = None
                driven_link_pos = None
                if i>1:
                    guess = link.pos2[i-1]
                    driven_link_pos = all_driven_links[0].pos2[i]
        
                if i > 2:
                    xvel = link.pos2[i-1][0] - link.pos1[i-2][0]
                    yvel = link.pos2[i-1][1] - link.pos1[i-2][1]
                    vel = (xvel, yvel)
                else:
                    vel = None

                # print("\t\t solving for link", link.name, 'with get_rod_coordinates func')
                pos1, pos2 = get_rod_coordinates(
                    t=time,
                    w=omega,
                    driven_link=all_driven_links[0],
                    fixed_link=all_constrained_links[0],
                    free_link=link,
                    driven_link_pos=driven_link_pos,
                    guess=guess,
                    vel=vel
                    )
                
                link.pos1[i] = pos1
                link.pos2[i] = pos2
                all_constrained_links[0].pos2[i] = pos2

                cl = all_constrained_links[0]
                lc = get_cart_distance(cl.pos1[i], cl.pos2[i])
                assert np.isclose(lc, cl.length, rtol=rtol), f"No viable solution, {cl.name} {lc} {cl.length}"

            else:
                free_link_names = [l.name for l in all_free_links]
                for tri_set in tri_link_sets:
                    t_edge_names = [l.name for l in  tri_set]
                   
                    if set(free_link_names) == set(t_edge_names):
                        if verbose >= 2:
                            print("\t this looks like a triangle")
                            print("\t\t tri set ", [l.name for l in  tri_set])

                        freelink1 = tri_set[0]
                        freelink2 = tri_set[2]
                        freelink3 = tri_set[1]

                        if freelink2.pos1[i][0] != 0.0 and freelink3.pos1[i][0] != 0.0:
                            # print("\t\t Triangle already solved skiping")
                            continue

                        assert freelink1.length < (freelink2.length + freelink3.length), f"freelink1 {freelink1.length} < {freelink2.length} + {freelink3.length}"

                        freelink2.pos1[i]       =  freelink1.pos2[i] 
                        freelink3.pos2[i]       = freelink1.pos1[i]

                        angle12, angle13, angle23 = get_tri_angles(
                                    free_link1=freelink1,
                                    free_link2=freelink2,
                                    free_link3=freelink3,
                                    )

                        if freelink1.flip:
                            angle12 = -angle12

                        pos = get_tri_pos(i, freelink1, freelink2, angle12)
                        freelink2.pos2[i] = pos
                        freelink3.pos1[i] = pos

                        l2 = get_cart_distance(freelink2.pos1[i], freelink2.pos2[i])
                        l3 = get_cart_distance(freelink3.pos1[i], freelink3.pos2[i])
                        if verbose >=4:
                            print('\t\t\tlink 2 calculated length', l2, 'expected length', freelink2.length)
                            print('\t\t\tlink 3 calculated length', l3, 'expected length', freelink3.length)
                        assert np.isclose(l3, freelink3.length, rtol=rtol), f"No viable solution, {freelink3.name} {l3} vs {freelink3.length}"
                        assert np.isclose(l2, freelink2.length, rtol=rtol), f"No viable solution, {link.name} {l2} vs {link.length}"

    fixed_links = [link for link in links if link.has_fixed]
    free_links = [link for link in links if not link.has_fixed]
    min_length = links[0].length
    for c in combinations(fixed_links, 2):
        for link in free_links:
            #if minimal_bounds(c[0], c[1]) < link.length:
            #    print(minimal_bounds(c[0], c[1]) , link.name)
            #    logger.warning("There may be no valid graph")
            if link.length < min_length:
                min_length = link.length

    return links


def run_graph(
        i,
        time,
        omega,
        link_graph,
        rtol=0.05,
        verbose=0):
    
    for edge in link_graph.edges(data=True):
        node1, node2, edge_data = edge
        link: Link = edge_data['link']
        node1_data = link_graph.nodes[node1]
        node2_data = link_graph.nodes[node2]

        if verbose > 0:
            print(link.name, node1, node2)
        
        # Check if the current edge is free
        if not link.has_fixed:
            if verbose > 0:
                print(f"\tlink `{link.name}` is free")

            # Get the edges connected to node1 and node2
            connected_edges_node1 = list(link_graph.edges(node1, data=True))
            connected_edges_node2 = list(link_graph.edges(node2, data=True))
            #if verbose > 0:
            #    print("\t", len(connected_edges_node1), connected_edges_node1)
            #    print("\t", len(connected_edges_node2), connected_edges_node2)

            
            #if len(connected_edges_node1) == 2 and len(connected_edges_node2) == 2:
            if len(connected_edges_node2) == 2:
                # Check if the edges before and after are fixed
                fixed_edges_node1 = [e for e in connected_edges_node1 if link_graph.nodes[e[1]]['fixed']]
                fixed_edges_node2 = [e for e in connected_edges_node2 if link_graph.nodes[e[1]]['fixed']]
                
                if verbose > 0:
                    print("\t with 2 connected nodes, the first node is", fixed_edges_node1)

                if len(fixed_edges_node1) == 1 and len(fixed_edges_node2) == 1:
                    driven_link = fixed_edges_node1[0][2]['link']
                    fixed_link = fixed_edges_node2[0][2]['link']
                    free_link = link
                    
                    assert driven_link.is_driven
                    assert not free_link.is_driven
                    assert fixed_link.has_fixed
                    assert not fixed_link.is_driven

                    guess = None
                    if i>1:
                        guess = free_link.pos2[i-1]
           
                    pos1, pos2 = get_rod_coordinates(
                        t=time,
                        w=omega,
                        driven_link=driven_link,
                        fixed_link=fixed_link,
                        free_link=free_link,
                        driven_link_pos=driven_link.pos2[i],
                        guess=guess
                    )
  
                    fixed_link_length = get_cart_distance(
                        fixed_link.fixed_loc,
                                        pos2)
                    free_link_length = get_cart_distance(pos1, pos2)
                    assert np.isclose(free_link_length, free_link.length, rtol=rtol), "No viable solution"
                    assert np.isclose(fixed_link_length, fixed_link.length, rtol=rtol), "No viable solution"

                    if verbose > 0:
                        print(f"\tlink coords, pos1:{pos1} pos2:{pos2}")
                    
                    node1_data = link_graph.nodes[node1]
                    node2_data = link_graph.nodes[node2]

                    node1_data['pos'] = pos1
                    node2_data['pos'] = pos2
                    link.pos1[i] = pos1
                    link.pos2[i] = pos2

                    fixed_link.pos2[i] = fixed_link.fixed_loc
                    fixed_link.pos1[i] = pos2

            if len(connected_edges_node2) == 3:
                edges_node1 = [e[0:2] for e in connected_edges_node1]
                edges_node2 = [e[0:2] for e in connected_edges_node2]
                
                #print("\t with 3 connected nodes, the first node is", edges_node1)
                #print("\t with 3 connected nodes, the second node is", edges_node2)
                
                # Get all the links from edges_node1 and edges_node2
                links_node1 = [link_graph.edges[edge]['link'] for edge in edges_node1]
                links_node2 = [link_graph.edges[edge]['link'] for edge in edges_node2]
                
                #print("\t links from node1:", links_node1)
                #print("\t links from node2:", links_node2)
                
                driven_links = [link for link in links_node1 if link.is_driven]
                free_links = [link for link in links_node1 if not link.has_fixed]
                fixed_links = [link for link in links_node2 if link.has_fixed and not link.is_driven]   
                
                if verbose > 0:
                    print("driven links", [link.name for link in driven_links])
                    print("free links", [link.name for link in free_links])
                    print("fixed links", [link.name for link in fixed_links])
                # Traverse down the edges to find the next link that closes the chain
                for edge in connected_edges_node2:
                    next_link = edge[2]['link']
                    if next_link != link and not next_link.has_fixed:
                        free_link3 = next_link
                        break  #print("driven links", driven_links)
                #print("free link3", free_link3.name)

                free_link1=free_links[0]
                free_link2=free_links[1]

                driven_link=driven_links[0] #2-1
                
                fixed_link = fixed_links[0]
            
                guess1= None
                guess2 = None
                guess3 = None
                if i>0:
                    #print("driven link", driven_link.name)
                    #print("free link1", free_link1.name)
                    #print("free link2", free_link2.name)
                    #print("free link3", free_link3.name)
                    #print("fixed link", fixed_links[0].name)
                    #guess1 = free_link1.pos2[i]
                    #guess2 = free_link2.pos2[i]
                    guess3 = free_link3.pos2[i]


                driven_link_end_pos, free_link1_end_pos, free_link2_end_pos, free_link3_end_pos = get_triangle_rod_coordinates(
                    time,
                    omega,
                    driven_link=driven_link,
                    free_link1=free_link1,
                    free_link2=free_link2,
                    free_link3=free_link3,
                    fixed_link=fixed_links[0],
                    driven_link_pos=None,
                    guess1=guess1,
                    guess2=guess2,
                    guess3=guess3,
                )  

                free_link1_length = get_cart_distance(
                    driven_link_end_pos, free_link1_end_pos)
                    
                #print('fixed_link1_length', free_link1_length)

                free_link1.pos1[i] = driven_link_end_pos
                free_link1.pos2[i] = free_link1_end_pos

                free_link2.pos1[i] = driven_link_end_pos
                free_link2.pos2[i] = free_link2_end_pos
                
                free_link2_length = get_cart_distance(free_link2.pos1[i],free_link2.pos2[i] )                
                #print('filed_link2_length', free_link2_length )
                assert np.isclose( free_link2_length,  free_link2_length, rtol=rtol)

                # # #free_link3.pos1[i] = free_link3_end_pos
                # # #free_link3.pos2[i] = free_link2_end_pos
                
                free_link3.pos1[i] = free_link1_end_pos
                free_link3.pos2[i] = free_link2_end_pos
                
                free_link3_length = get_cart_distance(free_link3.pos1[i], free_link3.pos2[i])
                assert np.isclose(free_link3_length, free_link3.length, rtol=rtol)

                #print('fixed_link3_length', free_link3_length )

                #fixed_link.pos2[i] = free_link1_end_pos
                #driven_link.pos2[i] = driven_link_end_pos
                #driven_link.pos1[i] = driven_link.fixed_loc
                fixed_link.pos1[i] = fixed_link.fixed_loc
                fixed_link.pos2[i] = free_link3.pos1[i]

                fixed_link_length = get_cart_distance(fixed_link.pos1[i],fixed_link.pos2[i])
                assert np.isclose(fixed_link_length,  fixed_link.length, rtol=rtol)

                #print('fixed_link_length',fixed_link_length)
                #link.pos2[i]=pos2




        elif link.is_driven and node1_data['fixed']:
            if verbose > 0:
                print(f"\tlink `{link.name}` has a fixed location and is driven")
            fixed_loc = node1_data['fixed_loc']
            x = fixed_loc[0] + link.length * np.cos(omega * time)
            y = fixed_loc[1] + link.length * np.sin(omega * time)
            #node_data['pos'].append((x, y)
            pos2 = (x,y)
            if verbose > 0:
                print(f"\tlink coords, pos1:{fixed_loc} pos2:{pos2}")
            #node1_data['pos'].append(fixed_loc)
            #node2_data['pos'].append(pos2)
            node1_data['pos'] = fixed_loc
            node2_data['pos'] = pos2

            # link.pos1.append(fixed_loc)
            # link.pos2.append(pos2)
            link.pos1[i] = fixed_loc
            link.pos2[i]=pos2

        elif link.has_fixed and node2_data["fixed"]:
            if verbose > 0:
                print(f"\tlink `{link.name}` has a fixed location")
            #print(node1_data["fixed"], node2_data["fixed"])
            fixed_loc = node2_data['fixed_loc']
            node2_data['pos'] = fixed_loc
       
            # unccoment for 4 bar     
            #link.pos2[i] = fixed_loc
            #link.pos1[i] = node1_data['pos']


