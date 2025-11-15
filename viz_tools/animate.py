

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from typing import Any, Callable, Dict, List, Optional, Tuple
from itertools import cycle
from link.tools import Link
from functools import partial



def animate_script(n_iterations,
                   links,
                   fname='animation.gif',
                   square=True):
    ymax = -666
    ymin = 666
    xmax = -666
    xmin = 666

    for link in links:
        if np.min(link.pos2[:, 0]) < xmin:
            xmin = np.min(link.pos2[:, 0])
        if np.min(link.pos1[:, 0]) < xmin:
            xmin = np.min(link.pos2[:, 0])

        if np.max(link.pos2[:, 0]) > xmax:
            xmax = np.max(link.pos2[:, 0])
        if np.max(link.pos1[:, 0]) > xmax:
            xmax = np.max(link.pos2[:, 0])

        if np.min(link.pos2[:, 1]) < ymin:
            ymin = np.min(link.pos2[:, 1])
        if np.min(link.pos1[:, 1]) < ymin:
            ymin = np.min(link.pos1[:, 1])

        if np.max(link.pos2[:, 1]) > ymax:
            ymax = np.max(link.pos2[:, 1])
        if np.max(link.pos1[:, 1]) > ymax:
            ymax = np.max(link.pos1[:, 1])
    
    xdelta = xmax - xmin

    ydelta = ymax - ymin
    #square the limits
    if square:
        delta = np.max([xdelta, ydelta])
        xmax = xmin + delta
        ymax = ymin + delta

    margin = .2
    xmin = xmin - delta * margin   
    xmax = xmax + delta * margin
    ymin = ymin - delta * margin
    ymax = ymax + delta * margin
   
    limits=[xmin, xmax, ymin, ymax]

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot()
    fig.subplots_adjust(left=0.1, right=0.85)
    figax={"fig":fig, "ax":ax}

    interval = .2
    fps = 1+int(n_iterations/3)
    colors = plt.cm.get_cmap("Spectral")
    alpha = 1.0


    historical_color_arr = [colors(k / n_iterations ) for k in range(n_iterations)]
    n_history = int(n_iterations * .66)
    historical_alphas = [alpha * (1/(1+i)) for i in range(n_history)]
    historical_alphas = historical_alphas[::-1]

    ani = FuncAnimation(
        fig,
        partial(
            plot_1d_pos_animation,
            links=links,
            figax=figax,
            alpha=alpha,
            colors=colors,
            run_len=n_iterations,
            limits=limits,
            title="automata linkage",
            n_history=n_history,
            historical_color_arr=historical_color_arr,
            historical_alphas=historical_alphas
            ),
        blit=False,
        frames= np.arange(0, (n_iterations - 1)),
        interval = interval , repeat = False)

    ani.save(fname, writer='imagemagick', fps=fps )
    return None

def plot_1d_pos_animation(
    frame,
    links: List[Link],
    figax=None,
    colors=None,
    run_len=-1,
    link_width = 4,
    alpha=1.0,
    markersize=50,
    limits = None,
    title="automata linkage",
    n_history=None,
    historical_color_arr=None,
    historical_alphas=None
    ):

    if figax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    else:
        _ = figax["fig"]
        ax = figax["ax"]

    ax.clear()
    
    frame_num = frame
    ax.set_title(f"{title}  \n step = {frame_num}")

    i = frame
    rotation_fraction = i / run_len 
    
    if n_history is not None:
        start_idx = (i - n_history) % run_len
        end_idx = start_idx + n_history
        full_start_idx = start_idx + n_history
        if full_start_idx <= run_len:
            pcolors = historical_color_arr[start_idx:end_idx]
            #pos2_history = link.pos2[start_idx:end_idx]
        else:
            mod_indx = end_idx % run_len
            pcolors = np.concatenate((historical_color_arr[start_idx:], historical_color_arr[:mod_indx]))
        
    
    artists = []
    for link in links:
            if n_history is not None:
                if not link.has_fixed and not link.has_constraint:
                    if full_start_idx <= run_len:
                        #pcolors = historical_color_arr[start_idx:end_idx]
                        pos2_history = link.pos2[start_idx:end_idx]
                    else:
                        mod_indx = end_idx % run_len
                    #pcolors = np.concatenate((historical_color_arr[start_idx:], historical_color_arr[:mod_indx]))
                        pos2_history = np.concatenate((link.pos2[start_idx:], link.pos2[:mod_indx]))

                    ax.scatter(
                        pos2_history[:,0],
                        pos2_history[:,1],
                        c=pcolors,
                        s=markersize,
                        alpha=historical_alphas
                        )
                
    for link in links:
        ax.scatter(
                    link.pos1[i][0],
                    link.pos1[i][1],
                    color=colors(rotation_fraction), 
                    s=markersize,
                    alpha=alpha)

        line = plt.plot(
            [link.pos1[i][0], link.pos2[i][0]],
            [link.pos1[i][1], link.pos2[i][1]],
            color=colors(rotation_fraction),
            linewidth=link_width,
            #markersize=5,
            alpha=alpha) 
        artists.append(line)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)

    # if limits is not None:
    #     plt.xlim(limits[0], limits[1])
    #     plt.ylim(limits[2], limits[3])  

    # square
    #plt.axis('equal')
    if limits is not None:
       plt.xlim(limits[0], limits[1])
       plt.ylim(limits[2], limits[3])  

    return None