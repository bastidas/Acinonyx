

import networkx as nx
from link.tools import *
import logging
from link.tools import get_tri_angles, get_tri_pos
logger = logging.getLogger()
from itertools import combinations

# def get_link_graph_flow(
#     time,
#     omega,
#     link_graph: nx.Graph):

#     # for link in link_graph.edges.data():
#     #     print(link)
#     visited_edges = []
#     for node, node_data in link_graph.nodes.data():
#         print(node, node_data)
        
#         #fixed_loc = None
#         if node_data['fixed']:
#             fixed_loc = node_data['fixed_loc']
#             node_data['pos'].append(fixed_loc)
#             continue

#         connected_edges = list(link_graph.edges(node, data=True))
#         for edge in  connected_edges:
#         #for edge in link_graph.edges(node, data=True):
#             # link = edge[2]['link']
#             # link.is_driven:
#             #print("\t", edge)
#             #print("\t\t", edge[2]['link'])
#             # if (this_node == edge[0] or this_node == edge[1]):
#             #     print("this node in it") # this will always be true!
#             link = edge[2]['link']

#             #print("\t\t", 'edge 0 and 1', edge[0], edge[1])
#             #print("\t\t", edge[0] == node)

#             """
#             The first node in any link should be this link
#             This ensures we are looking at the end of the link
#             """
#             assert edge[0] == node, "The first node in any link should be this link"
#             connected_node = edge[1]
#             print("\t\t", link_graph.nodes[connected_node])
#             connected_node_data = link_graph.nodes[connected_node]
#             if link.is_driven and connected_node_data['fixed']:
#                 fixed_loc =  connected_node_data['fixed_loc']
#                 x = fixed_loc[0] + link.length * np.cos(omega * time)
#                 y = fixed_loc[1] + link.length * np.sin(omega * time)
#                 print("\t\tSETTING NODE DATA:", node_data['pos'].append((x, y)))

#         if not node_data['fixed']:
#             # Retrieve the edges connected to the current node
#             connected_edges = list(link_graph.edges(node, data=True))
#             if len(connected_edges) == 2:
            
#                 # Extract the links and their constraints
#                 edge1, edge2 = connected_edges
#                 link1 = edge1[2]['link']
#                 link2 = edge2[2]['link']

#                 assert edge1[0] ==  edge2[0]
#                 con_node1 = edge1[1]
#                 con_node2 = edge2[1]
                
#                 #print("n\t\t\t", con_node1, con_node2)

#                 con_node_data1 = link_graph.nodes[con_node1]
#                 con_node_data2 = link_graph.nodes[con_node2]

#                 assert (con_node_data1['fixed']) or (con_node_data2['fixed'])
#                 if con_node_data2['fixed']:
#                     fixed_node = con_node2
#                     free_node = con_node1
#                     fixed_node_data = con_node_data2
#                     free_node_data = con_node_data1
#                 else:
#                     fixed_node = con_node1
#                     free_node = con_node2
#                     fixed_node_data = con_node_data1
#                     free_node_data = con_node_data2
                    

#                 print("fee node:", free_node,  free_node_data)
#                 print("fixed node:", fixed_node, fixed_node_data)


#                 x2, y2, x3, y3, x4, y4 = get_rod_coordinates(
#                         t=time,
#                         w=omega,
#                         fixed_link1=prior_fixed_link,
#                         fixed_link2=next_fixed_link,
#                         free_link=this_link
#                         )
                    

#         # #if link.is_driven and connected_node_data['fixed']:
#         # # Determine the fixed location and length of the first link
#         # fixed_loc = link1.fixed_loc
#         # if fixed_loc is None:
#         #     raise ValueError(f"Link {link1.name} must have a fixed location.")

#             # if link1.is_driven:
#             #     x2 = fixed_loc[0] + link1.length * np.cos(omega * time)
#             #     y2 = fixed_loc[1] + link1.length * np.sin(omega * time)

#             # def equations(p):
#             #     x3, y3 = p
#             #     eq1 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2) - link2.length
#             #     eq2 = np.sqrt((x3 - link2.fixed_loc[0])**2 + (y3 - link2.fixed_loc[1])**2) - link2.length
#             #     return (eq1, eq2)

#             # # Solve for the position of the free node
#             # x3, y3 = fsolve(equations, (link2.fixed_loc[0], link2.fixed_loc[1]))
#             # print(f"Calculated position for node {node}: ({x3}, {y3})")


#     return None



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
        cycle_sets 
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
        link = edge_data['link']
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
                fixed_edges_node1 = [e for e in connected_edges_node1 if link_graph.nodes[e[1]].get('fixed', False)]
                fixed_edges_node2 = [e for e in connected_edges_node2 if link_graph.nodes[e[1]].get('fixed', False)]
                
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


