
import os
import sys
# Configure matplotlib for backend use BEFORE any matplotlib imports
from configs.matplotlib_config import configure_matplotlib_for_backend
configure_matplotlib_for_backend()

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
from link.graph_tools import make_graph_simple

import os
from pathlib import Path
# Define the user directory relative to this file's location
user_dir = Path(__file__).parent.parent / "user"

# Ensure the user directory exists
user_dir.mkdir(exist_ok=True)

# def make_3link(n_iterations):

#     driven_link = Link(length=.4,
#                 fixed_loc=(0, 0),
#                 has_fixed=True,
#                 n_iterations=n_iterations,
#                 name="driven_link",
#                 is_driven=True)  

#     fixed_link = Link(length=.4,
#                  has_fixed=True,
#                 fixed_loc=(0.1, 1),
#                 n_iterations=n_iterations,
#                 name="fixed_link")  

#     free_link = Link(length=.99,
#                     has_fixed=False,
#                     n_iterations=n_iterations,
#                     name="free_link")


#     graph = nx.Graph([
#         (1, 2, {"link": driven_link}),
#         (2,3, {"link": free_link}),
#         (3,4, {"link": fixed_link}),
#         ])

#     graph.nodes[1]['fixed'] = True
#     graph.nodes[1]['fixed_loc'] = (0.0, 0.0)
#     graph.nodes[1]['pos'] = None

#     graph.nodes[2]['fixed'] = False
#     graph.nodes[2]['pos'] = None

#     graph.nodes[3]['fixed'] = False
#     graph.nodes[3]['pos'] = None

#     graph.nodes[4]['fixed'] = True
#     graph.nodes[4]['pos'] = None
#     graph.nodes[4]['fixed_loc'] = (0.0, 1.0)

#     links = [driven_link, free_link, fixed_link]
#     nodes = [graph.nodes[n] for n in graph.nodes]
#     return links, nodes, graph
   

def make_3link(n_iterations):

    driven_link = Link(length=.4,
                fixed_loc=(0, 0),
                has_fixed=True,
                n_iterations=n_iterations,
                name="driven_link",
                is_driven=True)  

    fixed_link = Link(length=.4,
                 has_fixed=True,
                fixed_loc=(0.1, 1),
                n_iterations=n_iterations,
                name="fixed_link")  

    free_link = Link(length=.99,
                    has_fixed=False,
                    n_iterations=n_iterations,
                    name="free_link")

    node1 = Node(name="node1",
                 n_iterations=n_iterations,
                 #pos=None,
                 fixed=True,
                 fixed_loc=(0.0, 0.0))
    
    node2 = Node(name="node2",
                 n_iterations=n_iterations,
                 #pos=None,
                 fixed=False)
    
    node3 = Node(name="node3",
                 n_iterations=n_iterations,
                 #pos=None,
                 fixed=False)   
    
    node4 = Node(name="node4",
                 n_iterations=n_iterations,
                 #pos=None,
                 fixed=True,
                 fixed_loc=(0.0, 1.0))
    

    connections = [
        {"from_node": "node1", "to_node": "node2", "link": driven_link},
        {"from_node": "node2", "to_node": "node3", "link": free_link},
        {"from_node": "node3", "to_node": "node4", "link": fixed_link}]
        
    nodes = [node1, node2, node3, node4]
    links = [driven_link, free_link, fixed_link]
    graph=make_graph_simple(
        connections,
        links,
        nodes,
        n_iterations=n_iterations)

    # Don't initialize positions - let run_graph solve the triangle geometry
    # just like the original make_3link does

    return links, nodes, graph


"""

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
"""

def make_5link(n_iterations):

    # Create links with proper has_fixed properties
    # Only the drive_link is both driven and has a fixed location
    drive_link = Link(length=.3, 
                     fixed_loc=(0, 0), 
                     has_fixed=True,
                     n_iterations=n_iterations, 
                     name="driven_link", 
                     is_driven=True)  
    
    # Free links that connect between nodes
    freelink1 = Link(length=.88, 
                    has_fixed=False,
                    n_iterations=n_iterations, 
                    name="free_link1")
    
    freelink2 = Link(length=.77, 
                    has_fixed=False,
                    n_iterations=n_iterations, 
                    name="free_link2")

    #set_link1 = Link(length=.44, fixed_loc=(0, 1),n_iterations = n_iterations, name="set link1")
    #set_link2 = Link(length=.44, fixed_loc=(-.5, .4),n_iterations = n_iterations, name="set link2")
    set_link1 = Link(length=.44, 
                    fixed_loc=(0, 1),
                    has_fixed=True,
                    n_iterations=n_iterations, 
                    name="set_link1")
    
    set_link2 = Link(length=.44, 
                    fixed_loc=(-.5, .4),
                    has_fixed=True,
                    n_iterations=n_iterations, 
                    name="set_link2")

    # Create nodes using Node objects
    node1 = Node(name="node1",
                 n_iterations=n_iterations,
                 fixed=True,
                 fixed_loc=(0.0, 0.0))
    
    node2 = Node(name="node2",
                 n_iterations=n_iterations,
                 fixed=False)
    
    node3 = Node(name="node3",
                 n_iterations=n_iterations,
                 fixed=False)
    
    node4 = Node(name="node4",
                 n_iterations=n_iterations,
                 fixed=False)
    
    node5 = Node(name="node5",
                 n_iterations=n_iterations,
                 fixed=True,
                 fixed_loc=(0.0, 1.0))
    
    node6 = Node(name="node6",
                 n_iterations=n_iterations,
                 fixed=True,
                 fixed_loc=(-0.5, 0.4))

    # Define connections using string node names
    connections = [
        {"from_node": "node1", "to_node": "node2", "link": drive_link},
        {"from_node": "node2", "to_node": "node3", "link": freelink1},
        {"from_node": "node2", "to_node": "node4", "link": freelink2},
        {"from_node": "node3", "to_node": "node5", "link": set_link1},
        {"from_node": "node4", "to_node": "node6", "link": set_link2}]
        
    nodes = [node1, node2, node3, node4, node5, node6]
    links = [drive_link, freelink1, freelink2, set_link1, set_link2]
    
    # Use make_graph_simple like make_3link
    graph = make_graph_simple(
        connections,
        links,
        nodes,
        n_iterations=n_iterations)

    # Don't initialize positions - let run_graph solve the geometry
    # just like the make_3link approach

    return links, nodes, graph


if __name__ == "__main__":
    n_iterations = 24
    

    
    # Use make_5link for the simulation
    links, nodes, graph = make_3link(n_iterations)
    links, nodes, graph = make_5link(n_iterations)

    import json

    G = nx.barbell_graph(6, 3)
    # this d3 example uses the name attribute for the mouse-hover value,
    # so add a name to each node
    for n in G:
        G.nodes[n]["name"] = n
    # write json formatted data
    d = nx.node_link_data(G, edges="links")  # node-link format to serialize
    # write json
    #json.dump(d, open("force/force.json", "w"))
   # json.dump(graph, open(user_dir / "force.json", "w"))
    json.dump(d, open(user_dir / "force.json", "w"))



    times = np.linspace(0, 1, n_iterations )
    for i, t in enumerate(times):
        #print(i,t)
        ng = run_graph(
            i,
                time=t,
                omega=2*np.pi,
                link_graph=graph,
                verbose=1 if i == 0 else 0)  # Verbose for first iteration only
        

    # Verify link lengths after simulation
    print("\n=== VERIFYING LINK LENGTHS AFTER SIMULATION ===")
    for link in links:
        # Check length at first time step
        if hasattr(link, 'pos1') and hasattr(link, 'pos2') and link.pos1 is not None and link.pos2 is not None:
            pos1 = link.pos1[0]
            pos2 = link.pos2[0]
            actual_length = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            print(f"{link.name}: expected={link.length:.3f}, actual={actual_length:.3f}, pos1={pos1}, pos2={pos2}")
        else:
            print(f"{link.name}: pos1 or pos2 is None or missing")
    print("=== VERIFICATION COMPLETE ===\n")

    plot_static_arrows(
        links,
        i=0,  # Use a specific time index instead of the whole times array
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
    

    from viz_tools.animate import animate_script
    animate_script(n_iterations,
                   links,
                   fname='animationvvv.gif',
                   square=True)
    