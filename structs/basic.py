
import os
import sys
# Set matplotlib to use non-GUI backend BEFORE any matplotlib imports
import matplotlib
matplotlib.use('Agg', force=True)  # force=True to override existing backend
import networkx as nx
import matplotlib.pyplot as plt
from configs.link_models import Link, Node
#from config import user_dir
import matplotlib.colors as mcolors
import numpy as np
from link.graph_tools import run_graph
from viz_tools.viz import (
    # animate_script,
    plot_static_pos,
    plot_static_arrows)

import os
from pathlib import Path
# Define the user directory relative to this file's location
user_dir = Path(__file__).parent.parent / "user"

# Ensure the user directory exists
user_dir.mkdir(exist_ok=True)


def make_graph(connections,
               links: list[Link],
               nodes: list[Node],
               n_iterations=24):
    """Create a NetworkX graph from frontend connections and links data"""
    print("\n=== MAKE_GRAPH CALLED ===")
    print(f"Received {len(connections)} connections, {len(links)} links, and {len(nodes)} nodes")
    
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
                    graph.add_node(from_node, pos=from_node_obj.pos, fixed=from_node_obj.fixed, fixed_loc=from_node_obj.fixed_loc)
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



    times = np.linspace(0, 1, n_iterations )
    for i, t in enumerate(times):
        print(i,t)
        ng = run_graph(
            i,
                time=t,
                omega=2*np.pi,
                link_graph=graph,
                verbose=3)
        

    plot_static_arrows(
        links,
        i=min(50, n_iterations-1),  # Use safe index that doesn't exceed n_iterations
        title="Linkage Arrows - 3 Link System",
        out_path=user_dir / "linkage_arrows.png")


    plot_static_pos(
        links,
        times,
        title="Linkage Positions - 3 Link System",
        out_path=user_dir / "linkage_positions.png",
        show_end_points=True,
        show_fixed_links=True,
        show_free_links=True,
        show_paths=True,
        show_pivot_points=True)
    
    # TODO: add mechanical linkage computation
    # For now, return basic graph info
    return {
        "graph_type": "directed",
        "nodes": list(graph.nodes()),
        "edges": list(graph.edges()),
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges()
    }


def make_3link(n_iterations):

    link1 = Link(length=.4,
                fixed_loc=(0, 0),
                n_iterations=n_iterations,
                name="driven link",
                is_driven=True)  

    link2 = Link(length=.4,
                fixed_loc=(0.1, 1),
                n_iterations=n_iterations,
                name="fixed link")  

    freelink = Link(length=.99,
                    n_iterations=n_iterations,
                    name="free link")


    graph = nx.Graph([
        (1, 2, {"link": link1}),
        (2,3, {"link": freelink}),
        (3,'z', {"link": link2}),
        ])

    graph.nodes[1]['fixed'] = True
    graph.nodes[1]['fixed_loc'] = (0.0, 0.0)
    graph.nodes[1]['pos'] = None

    graph.nodes[2]['fixed'] = False
    graph.nodes[2]['pos'] = None

    graph.nodes[3]['fixed'] = False
    graph.nodes[3]['pos'] = None

    graph.nodes['z']['fixed'] = True
    graph.nodes['z']['pos'] = None
    graph.nodes['z']['fixed_loc'] = (0.0, 1.0)

    return link1, link2, freelink, graph



def make_5link(n_iterations):


    drive_link = Link(length=.3, fixed_loc=(0, 0), n_iterations = n_iterations, name="driven link", is_driven=True)  
    freelink1 = Link(length=.88, n_iterations = n_iterations, name="free_link1")
    freelink2 = Link(length=.77, n_iterations = n_iterations, name="free_link2")

    set_link1 = Link(length=.44, fixed_loc=(0, 1),n_iterations = n_iterations, name="set link1")
    set_link2 = Link(length=.44, fixed_loc=(-.5, .4),n_iterations = n_iterations, name="set link2")
    
    lg = nx.Graph([
        (1, 2, {"link": drive_link}),
        (2,3, {"link": freelink1}),
        (2,4, {"link": freelink2}),
        (3,5, {"link": set_link1}),
        (4,6, {"link": set_link2}),
        ])

    lg.nodes[1]['fixed'] = True
    lg.nodes[1]['fixed_loc'] = (0.0, 0.0)
    lg.nodes[1]['pos'] = None

    lg.nodes[2]['fixed'] = False
    lg.nodes[2]['pos'] = None

    lg.nodes[3]['fixed'] = False
    lg.nodes[3]['pos'] = None

    lg.nodes[4]['fixed'] = False
    lg.nodes[4]['pos'] = None

    lg.nodes[5]['fixed'] = True
    lg.nodes[5]['pos'] = None
    lg.nodes[5]['fixed_loc'] = set_link1.fixed_loc

    lg.nodes[6]['fixed'] = True
    lg.nodes[6]['pos'] = None
    lg.nodes[6]['fixed_loc'] = set_link2.fixed_loc

    return drive_link, freelink1, freelink2, set_link1, set_link2, lg


if __name__ == "__main__":
    n_iterations = 100 
    # link1, link2, freelink, lg = make_3link(n_iterations)


    # link1, link2, freelink,link_graph  = make_3link(n_iterations)
    drive_link, freelink1, freelink2, set_link1, set_link2, link_graph  = make_5link(n_iterations)
    links = [drive_link, freelink1, freelink2, set_link1, set_link2]

    times = np.linspace(0, 1, n_iterations )
    for i, t in enumerate(times):
        #print(i,t)
        ng = run_graph(
            i,
                time=t,
                omega=2*np.pi,
                link_graph=link_graph,
                verbose=0)
        

    plot_static_arrows(
        links,
        i=50,  # Use a specific time index instead of the whole times array
        title="Linkage Arrows - 3 Link System",
        out_path=user_dir / "linkage_arrows.png")


    plot_static_pos(
        links,
        times,
        title="Linkage Positions - 3 Link System",
        out_path=user_dir / "linkage_positions.png",
        show_end_points=True,
        show_fixed_links=True,
        show_free_links=True,
        show_paths=True,
        show_pivot_points=True)