
from link.tools import get_tri_angles
from link.tools import rotate_point
from link.tools import get_tri_pos
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
#from link.graph_tools import validate_graph
#from structs.basic import make_3link
from configs.link_models import Link, Node
from link.graph_tools import validate_graph
import numpy as np

from link.graph_tools import run_graph
from structs.basic import make_3link


# def make_3link(n_iterations):

#     link1 = Link(length=.4,
#                  has_fixed=True,
#                  fixed_loc=(0, 0),
#                  n_iterations=n_iterations,
#                  name="driven_link",
#                  is_driven=True)  

#     link2 = Link(length=.4,
#                  has_fixed=True,
#                  fixed_loc=(0.1, 1),
#                  n_iterations=n_iterations,
#                  name="fixed_link")  

#     freelink = Link(length=.99,
#                     has_fixed=False,
#                     n_iterations=n_iterations,
#                     name="free_link")


#     node1 = Node(name="node1",
#                  n_iterations=n_iterations,
#                  #pos=None,
#                  fixed=True,
#                  fixed_loc=(0.0, 0.0))
    
#     node2 = Node(name="node2",
#                  n_iterations=n_iterations,
#                  #pos=None,
#                  fixed=False)
    
#     node3 = Node(name="node3",
#                  n_iterations=n_iterations,
#                  #pos=None,
#                  fixed=False)   
    
#     node4 = Node(name="node4",
#                  n_iterations=n_iterations,
#                  #pos=None,
#                  fixed=True,
#                  fixed_loc=(0.0, 1.0))
    

    
#     graph = nx.DiGraph()
#     links = [link1, link2, freelink]
#     nodes = [node1, node2, node3, node4]
#     for node in nodes:
#         graph.add_node(node.name, 
#                       **node.as_dict())
    
#     connections = [
#         {"from_node": "node1", "to_node": "node2", "link": link1},
#         {"from_node": "node2", "to_node": "node3", "link": freelink},
#         {"from_node": "node3", "to_node": "node4", "link": link2},]
#     ###
#     for i, conn in enumerate(connections):
#         from_node = conn.get('from_node')
#         to_node = conn.get('to_node')
#         link_obj = conn.get('link')
        
#         if from_node and to_node and link_obj:
#             print(f"Connection {i+1}: {from_node} -> {to_node} via link '{link_obj.name}'")
#             try:
#                 # Filter out frontend-specific fields
#                 #clean_link_data = {k: v for k, v in link_data.items() 
#                 #                 if k not in ['start_point', 'end_point', 'color', 'id']}
#                 #link_obj = Link(**clean_link_data)
#                 #graph.add_edge(from_node, to_node, **link_obj.as_dict())
#                 graph.add_edge(from_node, to_node,  link: link_obj.as_dict())
#                 print(f"  ✓ Converted connection link to Link object: {link_obj.name}")
#             except Exception as e:
#                 print(f"  ✗ Failed to convert connection link: {e}")
#                 print(f"    Link data: {link_obj.as_dict()}")
#                 # Skip this connection if Link conversion fails
#                 #continue
#         else:
#             print(f"ERROR: Skipping invalid connection {i+1}: {conn}")

#     return links, nodes, graph


def test_graph_validation():
    n_iterations = 7
    links, nodes, graph = make_3link(n_iterations)
    assert validate_graph(graph)


def test_run_graph():

    n_iterations = 3
    links, nodes, graph = make_3link(n_iterations)
    assert validate_graph(graph)
    times = np.linspace(0, 1, n_iterations )
    for i, t in enumerate(times):
        #print(i,t)
        ng = run_graph(
            i,
                time=t,
                omega=2*np.pi,
                link_graph=graph,
                verbose=0)
        
