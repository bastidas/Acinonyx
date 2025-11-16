
import numpy as np
from functools import partial
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from scipy.optimize import fsolve, least_squares
from scipy.optimize import least_squares
from functools import partial
from itertools import cycle
from scipy.optimize import least_squares

from configs.link_models import Link, DriveGear


def get_cart_distance(pos1,pos2):   
    """
    Calculate the distance between two points in Cartesian coordinates.

    Parameters:
    x1, y1 (float): Coordinates of the first point
    x2, y2 (float): Coordinates of the second point

    Returns:
    float: Distance between the two points
    """
    return np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1]- pos1[1])**2)

def get_rod_coordinates(t,
                        w,
                        driven_link: Link,
                        fixed_link: Link,
                        free_link: Link,
                        driven_link_pos: Optional[Tuple[float, float]] = None,
                        guess: Optional[Tuple[float, float]] = None,
                        vel = None,
                        ):
    
    if driven_link_pos is None and driven_link.is_driven:
        x2 = driven_link.fixed_loc[0] + driven_link.length * np.cos(w * t)
        y2 = driven_link.fixed_loc[1] + driven_link.length * np.sin(w * t)
    else:
        x2, y2 = driven_link_pos

    def equations(p):
        x3, y3 = p
        eq1 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2) - free_link.length
        eq2 = np.sqrt((x3 - fixed_link.fixed_loc[0])**2 + (y3 -  fixed_link.fixed_loc[1])**2) - fixed_link.length
        return (eq1, eq2)

    if guess is None:
        guess =  (fixed_link.fixed_loc[0], fixed_link.fixed_loc[1])
        bounds = -np.inf, np.inf
        if fixed_link.flip:
            print("\nflip true\n")
            #bounds = ((-666, driven_link.fixed_loc[0]+.01), (-666,666))
            #print(bounds)
            bounds = ((-np.inf, -np.inf), (driven_link.fixed_loc[0]+1e-3, np.inf))
    else:
        #vel = None
        if vel is None:
            delta = 1.2
            xmin = guess[0] - delta
            xmax = guess[0] + delta
            ymin = guess[1] - delta
            ymax = guess[1] + delta
            bounds = ((xmin, ymin), (xmax, ymax))
        else:
            delta = 2.2
            vtol = 10
            xmin = guess[0] + np.min([vel[0]*vtol, 0.0])-delta
            xmax = guess[0] + np.max([vel[0]*vtol, 0.0])+delta
            ymin = guess[1] + np.min([vel[1]*vtol, 0.0])-delta
            ymax = guess[1] + np.max([vel[1]*vtol, 0.0])+delta
            bounds = ((xmin, ymin), (xmax, ymax))
            #print('\t\t!!!!!!!!velbounds', guess, bounds)
            #print('guess', guess)


    res = least_squares(equations,  guess, bounds=bounds)
    
    x3, y3 = res.x 

    #x3, y3 = fsolve(equations, guess)

    driven_link_end_pos = (x2, y2)
    fixed_end_pos = (x3, y3)

    return driven_link_end_pos, fixed_end_pos
    


def rotate_point(point, theta):
    """
    Rotate a point around the origin by an angle theta.

    Parameters:
    point (tuple): The (x, y) coordinates of the point to rotate.
    theta (float): The angle in degrees to rotate the point.

    Returns:
    tuple: The rotated (x, y) coordinates.
    """
    theta_rad = np.radians(theta)  # Convert angle to radians
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad), np.cos(theta_rad)]
    ])
    rotated_point = np.dot(rotation_matrix, np.array(point))
    return tuple(rotated_point)
    


def get_tri_angles(
            free_link1: Link,
            free_link2: Link,
            free_link3: Link,
        ):
    """
    Given the known sides of a triangle calculate the angles
    """
    
    length1 = free_link1.length
    length2 = free_link2.length
    length3 = free_link3.length

    # Calculate the angles using the law of cosines
    cos_angle1 = (length2**2 + length3**2 - length1**2) / (2 * length2 * length3)
    #angle23 = np.arccos(np.clip(cos_angle1, -1.0, 1.0))
    angle23 = np.arccos(cos_angle1)

    cos_angle2 = (length1**2 + length3**2 - length2**2) / (2 * length1 * length3)
    #angle13 = np.arccos(np.clip(cos_angle2, -1.0, 1.0))
    angle13 = np.arccos(cos_angle2)
    angle12 = np.pi - angle13 - angle23
    # print("angle23", angle23*180/np.pi)
    # print("angle12", angle12*180/np.pi)
    # print("angle13", angle13*180/np.pi) 
    return angle12, angle13, angle23


def get_tri_pos(i, freelink1, freelink2, angle12):
    dx = freelink1.pos2[i][0] - freelink1.pos1[i][0]
    dy = freelink1.pos2[i][1] - freelink1.pos1[i][1]
    base_angle = np.arctan2(dy, dx)
    # print('base angle', base_angle * 180 / np.pi)

    # Calculate the rotation matrix
    angle_to_meeting_point = base_angle + angle12
    rotation_matrix = np.array([
        [np.cos(angle_to_meeting_point), -np.sin(angle_to_meeting_point)],
        [np.sin(angle_to_meeting_point), np.cos(angle_to_meeting_point)]
    ])

    # Apply the rotation matrix to freelink2's length vector
    length_vector = np.array([-freelink2.length, 0])  # freelink2 length along the negative x-axis
    rotated_vector = np.dot(rotation_matrix, length_vector)

    meeting_x = freelink1.pos2[i][0] + rotated_vector[0]
    meeting_y = freelink1.pos2[i][1] + rotated_vector[1]

    meeting_point = (meeting_x, meeting_y)
    return meeting_point

def get_tri_coords2(
    i,
    known_free_link1: Link,
    free_link2: Link,
    free_link3: Link,
    guess1: Optional[Tuple[float, float]] = None,
    #guess2: Optional[Tuple[float, float]] = None,
    ):
    
    pos1 = known_free_link1.pos1[i]
    pos2 = known_free_link1.pos2[i]

    length1 = known_free_link1.length
    length2 = free_link2.length
    length3 = free_link3.length

    def equations(p):
        x3, y3 = p
        eq1 = np.sqrt((x3 - pos1[0])**2 + (y3 - pos1[1])**2) - length2
        eq2 = np.sqrt((x3 - pos2[0])**2 + (y3 - pos2[1])**2) - length3
        return (eq1, eq2)

    if guess1 is None:
        guess1 = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
        bounds = -np.inf, np.inf
    else:
        
        delta = 0.15
        xmin = guess1[0] - delta
        xmax = guess1[0] + delta
        ymin = guess1[1] - delta
        ymax = guess1[1] + delta
        bounds = ((xmin, ymin), (xmax, ymax))
       
    res = least_squares(equations,  guess1, bounds=bounds)
    pos3 = res.x
    return pos3
