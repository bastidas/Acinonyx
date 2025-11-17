
from link.tools import get_tri_angles
from link.tools import rotate_point
from link.tools import get_tri_pos
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from link.tools import *




def check_sides(freelink1, freelink2, freelink3):

        i = 0
        dx, dy = np.random.random()*10-5, np.random.random()*10-5
        x0, y0 = 0,0
        x1, y1 = x0 + freelink1.length, y0+0
        angle = np.random.random() * 360

        freelink1.pos1[i]       =   (x0,  y0)
        freelink1.pos2[i]      = (x1,  y1)
        freelink1.pos2[i]       = rotate_point(freelink1.pos2[i], angle)

        freelink1.pos1[i][0] += dx
        freelink1.pos1[i][1] += dy

        freelink1.pos2[i][0] += dx
        freelink1.pos2[i][1] += dy

        freelink2.pos1[i]       =  freelink1.pos2[i] 

        freelink3.pos2[i]       = freelink1.pos1[i]

        angle12, angle13, angle23 = get_tri_angles(
                free_link1=freelink1,
                free_link2=freelink2,
                free_link3=freelink3,
                )

        rc = 180.0 / np.pi
        #print('angles', angle12*rc, angle13*rc, angle23*rc)
        pos =  get_tri_pos(i, freelink1, freelink2, angle12)
        freelink2.pos2[i]       =  pos
        freelink3.pos1[i]       = pos


        l1 = get_cart_distance(freelink1.pos1[i], freelink1.pos2[i])
        l2 = get_cart_distance(freelink2.pos1[i], freelink2.pos2[i])
        l3 = get_cart_distance(freelink3.pos1[i], freelink3.pos2[i])
        # #print('link 2 calculated length', l2, 'expected length', freelink2.length)
        # #print('link 3 calculated length', l3, 'expected length', freelink3.length)

        assert np.all(freelink1.pos1[i] == freelink3.pos2[i])
        assert np.all(freelink1.pos2[i]  == freelink2.pos1[i])
        assert np.all(freelink2.pos2[i] == freelink3.pos1[i])

        rtol = 0.01
        assert np.isclose(l1, freelink1.length, rtol=rtol)
        assert np.isclose(l2, freelink2.length, rtol=rtol)
        assert np.isclose(l3, freelink3.length, rtol=rtol)

def test_tri_angles():
    n_iterations = 2

    from itertools import permutations
    
    sides = [5,4,3]
    # test some right triangles
    for c in permutations(sides, 3):
        freelink1 = Link(length=c[0], n_iterations = n_iterations, name="free_link1")
        freelink2 = Link(length=c[1],n_iterations = n_iterations, name="free_link2")
        freelink3 = Link(length=c[2], n_iterations = n_iterations, name="free_link3")
        check_sides(freelink1, freelink2, freelink3)

    sides = [3.1,2.0,3.5]
    # test some not not right triangles
    for c in permutations(sides, 3):
        freelink1 = Link(length=c[0], n_iterations = n_iterations, name="free_link1")
        freelink2 = Link(length=c[1],n_iterations = n_iterations, name="free_link2")
        freelink3 = Link(length=c[2], n_iterations = n_iterations, name="free_link3")
        check_sides(freelink1, freelink2, freelink3)


#def test_find_triangle():
     
