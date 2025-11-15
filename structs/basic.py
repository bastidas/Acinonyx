
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
from configs.link_models import Link
from config import user_dir
import matplotlib.colors as mcolors
import numpy as np
from link.graph_tools import run_graph
from viz_tools.viz import (
    # animate_script,
    plot_static_pos,
    plot_static_arrows)

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


    lg = nx.Graph([
        (1, 2, {"link": link1}),
        (2,3, {"link": freelink}),
        (3,'z', {"link": link2}),
        ])

    lg.nodes[1]['fixed'] = True
    lg.nodes[1]['fixed_loc'] = (0.0, 0.0)
    lg.nodes[1]['pos'] = None

    lg.nodes[2]['fixed'] = False
    lg.nodes[2]['pos'] = None

    lg.nodes[3]['fixed'] = False
    lg.nodes[3]['pos'] = None

    lg.nodes['z']['fixed'] = True
    lg.nodes['z']['pos'] = None
    lg.nodes['z']['fixed_loc'] = (0.0, 1.0)

    return link1, link2, freelink, lg



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


    link1, link2, freelink,link_graph  = make_3link(n_iterations)

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
        [link1, freelink, link2],
        i=50,  # Use a specific time index instead of the whole times array
        title="Linkage Arrows - 3 Link System",
        out_path=user_dir / "linkage_arrows.png")


    plot_static_pos(
        [link1, freelink, link2],
        times,
        title="Linkage Positions - 3 Link System",
        out_path=user_dir / "linkage_positions.png",
        show_end_points=True,
        show_fixed_links=True,
        show_free_links=True,
        show_paths=True,
        show_pivot_points=True)