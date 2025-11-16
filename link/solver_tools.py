

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from viz_tools.viz import *
from link.tools import *
from link.graph_tools import minimal_bounds, solve_graph_links
from viz_tools.viz import animate_script
from matplotlib.animation import FuncAnimation
from benchmarks import make_triangle_link
from dataclasses import asdict, dataclass, field
import random

import itertools
from typing import List, Optional, Tuple


class Space(object):
    """
    parameter space 
    """

    def __init__(self,
                 parameters: List,
                 bounds: Optional[list] = None):
        self.parameters = parameters

        if bounds:
            self.bounds=bounds
        else:
            self.constraints = []

    def get_bounds(self, as_dict=True):
        if as_dict:
            return {param: bound for param, bound in zip(self.parameters, self.bounds)}
        else:
            return self.bounds


    def parameter_names(self):
        """Returns the list of names of parameters in the space."""
        return self.parameters

    def dimensionality(self):
        """Returns the number of parameters in the space."""
        return len(self.parameters)

    def sample_uniform(self, n):
        """Generates uniformly distributed random parameter points within bounds."""
        samples = []
        for _ in range(n):
            sample = {}
            for param, bound in zip(self.parameters, self.bounds):
                 name = param
                 min_val, max_val = bound
                 sample[name] = random.uniform(min_val, max_val)
            samples.append(sample)
        return samples





n_iterations  = 6
drive_link, freelink1, freelink2, freelink3, set_link1, link_graph, times  = make_triangle_link(n_iterations)
minimal_bounds(drive_link, set_link1)

for i,t in enumerate(times):
    links = solve_graph_links(
        i=i,
        time=t,
        omega=2*np.pi,
        link_graph=link_graph,
        verbose=0)