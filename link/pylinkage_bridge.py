"""
Bridge module for converting between automata graph format and pylinkage.

This module handles:
- Graph → Linkage conversion
- Trajectory extraction  
- Validation and error reporting
- Serialization between formats
- Demo linkage generation

Integration Level: Level 4 (User API) with Level 2 (Hypergraph) for validation

Key pylinkage concepts:
- In pylinkage, a 4-bar linkage uses only 2 joints: Crank + Revolute
- The Revolute joint represents the POINT where coupler meets rocker
- Revolute.distance0 = coupler length (from crank end to revolute point)
- Revolute.distance1 = rocker length (from second ground point to revolute point)
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
import networkx as nx
import logging

# pylinkage imports
from pylinkage.joints import Crank, Revolute, Fixed, Static
from pylinkage.linkage import Linkage

from configs.link_models import Link, Node

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO LINKAGE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def create_demo_4bar_pylinkage(
    ground_length: float = 30.0,
    crank_length: float = 10.0,
    coupler_length: float = 25.0,
    rocker_length: float = 20.0,
    crank_anchor: tuple[float, float] = (20.0, 30.0),
    angle_per_step: float = None,
    n_iterations: int = 24
) -> tuple[Linkage, dict]:
    """
    Create a proper 4-bar linkage using pylinkage's native API.
    
    A 4-bar linkage in pylinkage uses only 2 joints:
    1. Crank - the rotating driver, anchored to ground
    2. Revolute - the coupler/rocker connection point
    
    The Revolute joint's two distances represent:
    - distance0: coupler length (crank end → revolute point)
    - distance1: rocker length (second ground point → revolute point)
    
    Args:
        ground_length: Distance between the two fixed anchor points
        crank_length: Length of the rotating crank arm
        coupler_length: Length of the coupler (connects crank to rocker)
        rocker_length: Length of the rocker (connects ground to coupler)
        crank_anchor: Position of crank's fixed anchor (x, y)
        angle_per_step: Rotation per simulation step (default: 2π/n_iterations)
        n_iterations: Number of simulation steps
    
    Returns:
        (linkage, metadata) tuple where metadata contains linkage parameters
    
    Example (classic 4-bar):
        >>> linkage, meta = create_demo_4bar_pylinkage(
        ...     ground_length=30, crank_length=10, 
        ...     coupler_length=25, rocker_length=20
        ... )
        >>> for coords in linkage.step(iterations=24):
        ...     print(coords)
    """
    if angle_per_step is None:
        angle_per_step = 2 * np.pi / n_iterations
    
    # Calculate second anchor position (rocker ground point)
    rocker_anchor = (crank_anchor[0] + ground_length, crank_anchor[1])
    
    # Create the Crank joint (the rotating driver)
    # joint0 can be a tuple, which creates an implicit Static joint
    crank = Crank(
        x=crank_anchor[0] + crank_length,  # Initial x position of crank end
        y=crank_anchor[1],                  # Initial y position of crank end
        joint0=crank_anchor,                # Fixed anchor point (creates implicit Static)
        angle=angle_per_step,               # Rotation per step
        distance=crank_length,              # Crank arm length
        name="crank"
    )
    
    # Create the Revolute joint (coupler-rocker connection point)
    # This joint connects the crank end to the second ground point
    # through two distance constraints
    revolute = Revolute(
        x=crank_anchor[0] + crank_length + coupler_length * 0.5,  # Initial guess x
        y=crank_anchor[1] + coupler_length * 0.5,                  # Initial guess y
        joint0=crank,           # First parent: the crank
        joint1=rocker_anchor,   # Second parent: rocker ground point (implicit Static)
        distance0=coupler_length,  # Distance from crank end to this point
        distance1=rocker_length,   # Distance from rocker anchor to this point
        name="coupler_rocker_joint"
    )
    
    # Create the linkage with proper solve order
    linkage = Linkage(
        joints=(crank, revolute),
        order=(crank, revolute),
        name="Demo 4-Bar Linkage"
    )
    
    # Build metadata for frontend
    metadata = {
        "type": "4-bar",
        "parameters": {
            "ground_length": ground_length,
            "crank_length": crank_length,
            "coupler_length": coupler_length,
            "rocker_length": rocker_length,
            "crank_anchor": list(crank_anchor),
            "rocker_anchor": list(rocker_anchor),
            "n_iterations": n_iterations
        },
        "joints": {
            "crank": {
                "type": "Crank",
                "anchor": list(crank_anchor),
                "length": crank_length,
                "represents": "rotating driver arm"
            },
            "revolute": {
                "type": "Revolute", 
                "distance0": coupler_length,
                "distance1": rocker_length,
                "represents": "coupler-rocker connection point"
            }
        },
        "links_in_mechanism": {
            "ground": f"Fixed link between anchors, length={ground_length}",
            "crank_arm": f"Rotating driver, length={crank_length}",
            "coupler": f"Connects crank to rocker, length={coupler_length}",
            "rocker": f"Connects ground to coupler, length={rocker_length}"
        }
    }
    
    return linkage, metadata


def simulate_demo_4bar(
    ground_length: float = 30.0,
    crank_length: float = 10.0,
    coupler_length: float = 25.0,
    rocker_length: float = 20.0,
    crank_anchor: tuple[float, float] = (20.0, 30.0),
    n_iterations: int = 24
) -> dict:
    """
    Create and simulate a 4-bar linkage, returning visualization data.
    
    Returns:
        Dictionary with simulation results ready for frontend visualization
    """
    import time
    
    start_time = time.perf_counter()
    
    # Create the linkage
    linkage, metadata = create_demo_4bar_pylinkage(
        ground_length=ground_length,
        crank_length=crank_length,
        coupler_length=coupler_length,
        rocker_length=rocker_length,
        crank_anchor=crank_anchor,
        n_iterations=n_iterations
    )
    
    # Get initial positions
    rocker_anchor = (crank_anchor[0] + ground_length, crank_anchor[1])
    
    # Simulate
    trajectories = {
        "crank_anchor": np.array([list(crank_anchor)] * n_iterations),
        "rocker_anchor": np.array([list(rocker_anchor)] * n_iterations),
        "crank_end": np.zeros((n_iterations, 2)),
        "coupler_rocker_joint": np.zeros((n_iterations, 2))
    }
    
    try:
        linkage.rebuild()
        for i, coords in enumerate(linkage.step(iterations=n_iterations)):
            # coords[0] = crank position, coords[1] = revolute position
            if coords[0][0] is not None:
                trajectories["crank_end"][i] = [coords[0][0], coords[0][1]]
            if coords[1][0] is not None:
                trajectories["coupler_rocker_joint"][i] = [coords[1][0], coords[1][1]]
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Build link visualization data (4 links for the 4-bar)
        links_data = [
            {
                "name": "ground",
                "is_driven": False,
                "is_ground": True,  # This is the ground/frame link
                "has_fixed": True,
                "has_constraint": True,
                "pos1": trajectories["crank_anchor"].tolist(),
                "pos2": trajectories["rocker_anchor"].tolist()
            },
            {
                "name": "crank",
                "is_driven": True,
                "is_ground": False,
                "has_fixed": True,
                "has_constraint": False,
                "pos1": trajectories["crank_anchor"].tolist(),
                "pos2": trajectories["crank_end"].tolist()
            },
            {
                "name": "coupler",
                "is_driven": False,
                "is_ground": False,
                "has_fixed": False,
                "has_constraint": False,
                "pos1": trajectories["crank_end"].tolist(),
                "pos2": trajectories["coupler_rocker_joint"].tolist()
            },
            {
                "name": "rocker",
                "is_driven": False,
                "is_ground": False,
                "has_fixed": True,
                "has_constraint": False,
                "pos1": trajectories["rocker_anchor"].tolist(),
                "pos2": trajectories["coupler_rocker_joint"].tolist()
            }
        ]
        
        # Calculate bounds
        all_positions = np.vstack([
            trajectories["crank_anchor"],
            trajectories["rocker_anchor"],
            trajectories["crank_end"],
            trajectories["coupler_rocker_joint"]
        ])
        xmin, ymin = np.min(all_positions, axis=0)
        xmax, ymax = np.max(all_positions, axis=0)
        delta = max(xmax - xmin, ymax - ymin)
        margin = 0.2
        
        bounds = {
            "xmin": float(xmin - delta * margin),
            "xmax": float(xmax + delta * margin),
            "ymin": float(ymin - delta * margin),
            "ymax": float(ymax + delta * margin)
        }
        
        # Generate history trail for coupler-rocker joint (the interesting moving point)
        history_data = []
        n_history = int(n_iterations * 0.66)
        
        def get_spectral_color(t):
            if t < 0.25:
                r = 158 + int((255-158) * t * 4)
                g = 1 + int((116-1) * t * 4)
                b = 5
            elif t < 0.5:
                r = 255
                g = 116 + int((217-116) * (t-0.25) * 4)
                b = 9 + int((54-9) * (t-0.25) * 4)
            elif t < 0.75:
                r = 255 - int((255-171) * (t-0.5) * 4)
                g = 217 + int((221-217) * (t-0.5) * 4)
                b = 54 + int((164-54) * (t-0.5) * 4)
            else:
                r = 171 - int((171-94) * (t-0.75) * 4)
                g = 221 - int((221-79) * (t-0.75) * 4)
                b = 164
            return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
        
        for frame in range(n_iterations):
            frame_history = []
            start_idx = max(0, frame - n_history)
            end_idx = frame
            
            history_positions = trajectories["coupler_rocker_joint"][start_idx:end_idx].tolist()
            history_colors = []
            for i, pos in enumerate(history_positions):
                alpha = 1.0 / (1 + (len(history_positions) - i))
                color_t = (start_idx + i) / n_iterations
                color = get_spectral_color(color_t)
                history_colors.append({"color": color, "alpha": alpha})
            
            frame_history.append({
                "link_name": "coupler",
                "positions": history_positions,
                "colors": history_colors
            })
            history_data.append(frame_history)
        
        return {
            "status": "success",
            "message": "4-bar linkage simulation completed",
            "solver": "pylinkage",
            "execution_time_ms": execution_time,
            "n_iterations": n_iterations,
            "metadata": metadata,
            "path_data": {
                "bounds": bounds,
                "links": links_data,
                "history_data": history_data,
                "n_iterations": n_iterations
            }
        }
        
    except Exception as e:
        logger.exception("4-bar simulation failed")
        return {
            "status": "error",
            "message": f"Simulation failed: {str(e)}",
            "metadata": metadata
        }


def demo_4bar_to_ui_format(
    ground_length: float = 30.0,
    crank_length: float = 10.0,
    coupler_length: float = 25.0,
    rocker_length: float = 20.0,
    crank_anchor: tuple[float, float] = (20.0, 30.0),
    n_iterations: int = 24
) -> dict:
    """
    Convert a 4-bar linkage to the frontend UI format (nodes, links, connections).
    
    This creates the graph structure that can be loaded directly into the 
    Graph Builder UI, allowing users to visualize and edit the linkage.
    
    The 4-bar linkage has:
    - 4 nodes: crank_anchor (fixed), crank_end, rocker_anchor (fixed), coupler_rocker_joint
    - 4 links: crank (driven), coupler (free), rocker (fixed), ground (fixed)
    - 4 connections: one for each link
    
    Args:
        ground_length: Distance between the two fixed anchor points
        crank_length: Length of the rotating crank arm
        coupler_length: Length of the coupler
        rocker_length: Length of the rocker
        crank_anchor: Position of crank's fixed anchor (x, y)
        n_iterations: Number of simulation iterations
    
    Returns:
        Dictionary with nodes, links, connections ready for frontend
    """
    import uuid
    
    # Calculate positions
    rocker_anchor = (crank_anchor[0] + ground_length, crank_anchor[1])
    
    # Initial position of crank end (at angle 0, pointing right)
    crank_end_init = (crank_anchor[0] + crank_length, crank_anchor[1])
    
    # Initial position of coupler-rocker joint
    # Use geometry to find a valid starting position
    # Simple approach: place it above the midpoint
    mid_x = (crank_end_init[0] + rocker_anchor[0]) / 2
    mid_y = crank_anchor[1] + coupler_length * 0.6  # Rough estimate above
    coupler_joint_init = (mid_x, mid_y)
    
    # Color palette (tab10)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Generate UUIDs for links
    crank_id = str(uuid.uuid4())
    coupler_id = str(uuid.uuid4())
    rocker_id = str(uuid.uuid4())
    ground_id = str(uuid.uuid4())
    
    # ═══════════════════════════════════════════════════════════════
    # NODES
    # ═══════════════════════════════════════════════════════════════
    nodes = [
        {
            "id": "crank_anchor",
            "pos": list(crank_anchor),
            "fixed": True,
            "fixed_loc": list(crank_anchor),
            "n_iterations": n_iterations
        },
        {
            "id": "crank_end",
            "pos": list(crank_end_init),
            "fixed": False,
            "n_iterations": n_iterations
        },
        {
            "id": "rocker_anchor",
            "pos": list(rocker_anchor),
            "fixed": True,
            "fixed_loc": list(rocker_anchor),
            "n_iterations": n_iterations
        },
        {
            "id": "coupler_joint",
            "pos": list(coupler_joint_init),
            "fixed": False,
            "n_iterations": n_iterations
        }
    ]
    
    # ═══════════════════════════════════════════════════════════════
    # LINKS
    # ═══════════════════════════════════════════════════════════════
    links = [
        {
            "name": "crank",
            "length": crank_length,
            "n_iterations": n_iterations,
            "has_fixed": True,
            "fixed_loc": list(crank_anchor),
            "target_length": None,
            "target_cost_func": None,
            "has_constraint": False,
            "is_driven": True,  # This is the driver!
            "is_ground": False,
            "flip": False,
            "zlevel": 0,
            "meta": {
                "id": crank_id,
                "start_point": list(crank_anchor),
                "end_point": list(crank_end_init),
                "color": colors[0]
            }
        },
        {
            "name": "coupler",
            "length": coupler_length,
            "n_iterations": n_iterations,
            "has_fixed": False,
            "fixed_loc": None,
            "target_length": None,
            "target_cost_func": None,
            "has_constraint": False,
            "is_driven": False,
            "is_ground": False,
            "flip": False,
            "zlevel": 1,
            "meta": {
                "id": coupler_id,
                "start_point": list(crank_end_init),
                "end_point": list(coupler_joint_init),
                "color": colors[1]
            }
        },
        {
            "name": "rocker",
            "length": rocker_length,
            "n_iterations": n_iterations,
            "has_fixed": True,
            "fixed_loc": list(rocker_anchor),
            "target_length": None,
            "target_cost_func": None,
            "has_constraint": False,
            "is_driven": False,
            "is_ground": False,
            "flip": False,
            "zlevel": 0,
            "meta": {
                "id": rocker_id,
                "start_point": list(rocker_anchor),
                "end_point": list(coupler_joint_init),
                "color": colors[2]
            }
        },
        {
            "name": "ground",
            "length": ground_length,
            "n_iterations": n_iterations,
            "has_fixed": True,
            "fixed_loc": list(crank_anchor),  # Ground starts at crank anchor
            "target_length": None,
            "target_cost_func": None,
            "has_constraint": True,  # Ground is always constrained
            "is_driven": False,
            "is_ground": True,  # This is the ground/frame link!
            "flip": False,
            "zlevel": -1,  # Below other links
            "meta": {
                "id": ground_id,
                "start_point": list(crank_anchor),
                "end_point": list(rocker_anchor),
                "color": "#888888"  # Gray for ground
            }
        }
    ]
    
    # ═══════════════════════════════════════════════════════════════
    # CONNECTIONS
    # Include both link_id (UUID) and link_name for robust matching
    # ═══════════════════════════════════════════════════════════════
    connections = [
        {
            "from_node": "crank_anchor",
            "to_node": "crank_end",
            "link_id": crank_id,
            "link_name": "crank"
        },
        {
            "from_node": "crank_end",
            "to_node": "coupler_joint",
            "link_id": coupler_id,
            "link_name": "coupler"
        },
        {
            "from_node": "rocker_anchor",
            "to_node": "coupler_joint",
            "link_id": rocker_id,
            "link_name": "rocker"
        },
        {
            "from_node": "crank_anchor",
            "to_node": "rocker_anchor",
            "link_id": ground_id,
            "link_name": "ground"
        }
    ]
    
    return {
        "nodes": nodes,
        "links": links,
        "connections": connections,
        "metadata": {
            "type": "4-bar",
            "source": "pylinkage_demo",
            "parameters": {
                "ground_length": ground_length,
                "crank_length": crank_length,
                "coupler_length": coupler_length,
                "rocker_length": rocker_length,
                "crank_anchor": list(crank_anchor),
                "rocker_anchor": list(rocker_anchor)
            }
        }
    }


@dataclass
class ConversionResult:
    """Result of converting automata graph to pylinkage."""
    success: bool = False
    linkage: Optional[Linkage] = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    joint_mapping: dict[str, Any] = field(default_factory=dict)  # automata name → pylinkage joint
    stats: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "success": self.success,
            "warnings": self.warnings,
            "errors": self.errors,
            "joint_mapping": {k: str(v) for k, v in self.joint_mapping.items()},
            "stats": self.stats,
            "serialized_linkage": self.linkage.to_dict() if self.linkage else None
        }


@dataclass  
class SimulationResult:
    """Result of running pylinkage simulation."""
    success: bool = False
    trajectories: dict[str, np.ndarray] = field(default_factory=dict)  # joint name → (n_iter, 2)
    n_iterations: int = 0
    errors: list[str] = field(default_factory=list)
    execution_time_ms: float = 0.0


def _build_graph_from_connections(
    nodes: list[Node],
    links: list[Link],
    connections: list[dict]
) -> nx.Graph:
    """Build a NetworkX graph from the connections data for topology analysis."""
    graph = nx.Graph()
    
    # Add nodes
    node_map = {n.name: n for n in nodes}
    for node in nodes:
        graph.add_node(node.name, node=node, fixed=node.fixed, fixed_loc=node.fixed_loc)
    
    # Build link lookup by various IDs
    link_by_name = {l.name: l for l in links}
    link_by_meta_id = {}
    for link in links:
        # Check if link has meta with id (from frontend format)
        if hasattr(link, '__dict__'):
            meta = getattr(link, 'meta', None)
            if isinstance(meta, dict) and 'id' in meta:
                link_by_meta_id[meta['id']] = link
    
    # Add edges from connections
    for conn in connections:
        from_node = conn.get('from_node')
        to_node = conn.get('to_node')
        link_id = conn.get('link_id')
        link_name_direct = conn.get('link_name')  # Direct link name in connection
        link_ref = conn.get('link')
        
        # Find the link - try multiple methods
        link = None
        
        # Method 1: By meta.id (UUID from frontend)
        if link_id:
            link = link_by_meta_id.get(link_id)
        
        # Method 2: By direct link_name in connection (from demo_4bar_to_ui_format)
        if not link and link_name_direct:
            link = link_by_name.get(link_name_direct)
        
        # Method 3: By link_id as link name (sometimes used interchangeably)
        if not link and link_id:
            link = link_by_name.get(link_id)
        
        # Method 4: By embedded link reference
        if not link and link_ref:
            ref_name = link_ref.get('name') if isinstance(link_ref, dict) else None
            if ref_name:
                link = link_by_name.get(ref_name)
        
        if from_node and to_node and link:
            graph.add_edge(from_node, to_node, link=link)
        elif from_node and to_node:
            logger.warning(f"Could not find link for connection {from_node} -> {to_node} (link_id={link_id}, link_name={link_name_direct})")
    
    return graph


def _find_driven_link(links: list[Link]) -> Optional[Link]:
    """Find the driven (crank) link."""
    driven = [l for l in links if l.is_driven]
    if len(driven) == 1:
        return driven[0]
    elif len(driven) > 1:
        logger.warning(f"Found {len(driven)} driven links, using first one: {driven[0].name}")
        return driven[0]
    return None


def _find_fixed_links(links: list[Link]) -> list[Link]:
    """Find all links with fixed anchors (excluding driven and ground)."""
    return [l for l in links if l.has_fixed and not l.is_driven and not getattr(l, 'is_ground', False)]


def _find_free_links(links: list[Link]) -> list[Link]:
    """Find all free-floating links (no fixed anchor, not ground)."""
    return [l for l in links if not l.has_fixed and not getattr(l, 'is_ground', False)]


def _find_ground_links(links: list[Link]) -> list[Link]:
    """Find all ground links (frame/base between fixed anchors)."""
    return [l for l in links if getattr(l, 'is_ground', False)]


def _determine_solve_order(
    graph: nx.Graph,
    static_joints: dict[str, Static],
    crank_joint: Optional[Crank],
    all_joints: list
) -> list:
    """
    Determine the solving order for joints.
    Start from static/driven joints, propagate outward through the graph.
    """
    # Simple approach: static joints first, then crank, then others by dependency
    order = []
    
    # Static joints are always solvable first
    order.extend(static_joints.values())
    
    # Crank is solvable because it depends only on static
    if crank_joint:
        order.append(crank_joint)
    
    # Add remaining joints (they'll be solved based on their parent dependencies)
    for joint in all_joints:
        if joint not in order:
            order.append(joint)
    
    return order


def convert_to_pylinkage(
    nodes: list[Node],
    links: list[Link],
    connections: list[dict]
) -> ConversionResult:
    """
    Convert automata graph representation to pylinkage Linkage.
    
    Algorithm:
    1. Create Static joints for all fixed nodes
    2. Find the driven link → create Crank joint
    3. For each fixed link (non-driven): create Fixed joint
    4. For each free link: create Revolute joint  
    5. Build parent relationships from connections
    6. Determine solve order
    7. Return Linkage
    
    Args:
        nodes: List of Node objects from automata
        links: List of Link objects from automata
        connections: Connection definitions [{from_node, to_node, link_id}, ...]
    
    Returns:
        ConversionResult with linkage or errors
    """
    result = ConversionResult()
    
    try:
        # Build topology graph for analysis
        topo_graph = _build_graph_from_connections(nodes, links, connections)
        
        pylinkage_joints = []
        static_joints = {}  # node_name → Static joint
        link_to_joint = {}  # link_name → pylinkage joint
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: Create Static joints for fixed nodes (ground/anchors)
        # ═══════════════════════════════════════════════════════════════
        for node in nodes:
            if node.fixed and node.fixed_loc:
                static = Static(
                    x=float(node.fixed_loc[0]),
                    y=float(node.fixed_loc[1]),
                    name=node.name
                )
                static_joints[node.name] = static
                pylinkage_joints.append(static)
                result.joint_mapping[f"node:{node.name}"] = static
                logger.debug(f"Created Static joint for node '{node.name}' at {node.fixed_loc}")
        
        if not static_joints:
            result.errors.append("No fixed nodes found - linkage needs at least one ground point")
            return result
        
        # Note: Ground links are implicit in pylinkage (they define the frame)
        ground_links = _find_ground_links(links)
        for gl in ground_links:
            result.warnings.append(f"Ground link '{gl.name}' is implicit in pylinkage - skipped in conversion")
            logger.debug(f"Skipping ground link '{gl.name}' (implicit in pylinkage)")
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: Find and create Crank (driven link)
        # ═══════════════════════════════════════════════════════════════
        driven_link = _find_driven_link(links)
        crank_joint = None
        
        if not driven_link:
            result.errors.append("No driven link found - linkage needs exactly one driver (is_driven=True)")
            return result
        
        # Find which static joint the crank attaches to
        crank_anchor = None
        if driven_link.fixed_loc:
            # Find static joint at this location
            for name, static in static_joints.items():
                if (abs(static.x - driven_link.fixed_loc[0]) < 1e-6 and
                    abs(static.y - driven_link.fixed_loc[1]) < 1e-6):
                    crank_anchor = static
                    break
        
        if not crank_anchor:
            # Create a new static joint for the crank anchor
            if driven_link.fixed_loc:
                crank_anchor = Static(
                    x=float(driven_link.fixed_loc[0]),
                    y=float(driven_link.fixed_loc[1]),
                    name=f"{driven_link.name}_anchor"
                )
                static_joints[f"{driven_link.name}_anchor"] = crank_anchor
                pylinkage_joints.append(crank_anchor)
                result.warnings.append(f"Created implicit anchor for crank at {driven_link.fixed_loc}")
            else:
                result.errors.append(f"Driven link '{driven_link.name}' has no fixed_loc")
                return result
        
        # Create the Crank joint
        # angle is rotation per step (2π / n_iterations for full rotation)
        n_iter = driven_link.n_iterations
        angle_per_step = 2 * np.pi / n_iter
        
        crank_joint = Crank(
            joint0=crank_anchor,
            angle=angle_per_step,
            distance=float(driven_link.length),
            name=driven_link.name
        )
        pylinkage_joints.append(crank_joint)
        link_to_joint[driven_link.name] = crank_joint
        result.joint_mapping[f"link:{driven_link.name}"] = crank_joint
        logger.debug(f"Created Crank joint '{driven_link.name}' with length={driven_link.length}")
        
        # ═══════════════════════════════════════════════════════════════
        # Step 3: Create Fixed joints for grounded non-driven links
        # ═══════════════════════════════════════════════════════════════
        fixed_links = _find_fixed_links(links)
        
        for link in fixed_links:
            # Find the static joint this link anchors to
            anchor = None
            if link.fixed_loc:
                for name, static in static_joints.items():
                    if (abs(static.x - link.fixed_loc[0]) < 1e-6 and
                        abs(static.y - link.fixed_loc[1]) < 1e-6):
                        anchor = static
                        break
            
            if not anchor and link.fixed_loc:
                # Create implicit anchor
                anchor = Static(
                    x=float(link.fixed_loc[0]),
                    y=float(link.fixed_loc[1]),
                    name=f"{link.name}_anchor"
                )
                static_joints[f"{link.name}_anchor"] = anchor
                pylinkage_joints.append(anchor)
                result.warnings.append(f"Created implicit anchor for fixed link '{link.name}'")
            
            if anchor:
                # Fixed joint: constrained distance from anchor point
                fixed_joint = Fixed(
                    joint0=anchor,
                    distance=float(link.length),
                    name=link.name
                )
                pylinkage_joints.append(fixed_joint)
                link_to_joint[link.name] = fixed_joint
                result.joint_mapping[f"link:{link.name}"] = fixed_joint
                logger.debug(f"Created Fixed joint '{link.name}' with length={link.length}")
            else:
                result.warnings.append(f"Could not find anchor for fixed link '{link.name}'")
        
        # ═══════════════════════════════════════════════════════════════
        # Step 4: Create Revolute joints for free links
        # ═══════════════════════════════════════════════════════════════
        free_links = _find_free_links(links)
        
        for link in free_links:
            # Find the two parent joints this revolute connects
            # Look in the topology graph for connected edges
            parent0 = None
            parent1 = None
            
            # Find edges in topo_graph that use this link
            for edge in topo_graph.edges(data=True):
                edge_link = edge[2].get('link')
                if edge_link and edge_link.name == link.name:
                    from_node, to_node = edge[0], edge[1]
                    
                    # Find joints connected to these nodes
                    for other_edge in topo_graph.edges(from_node, data=True):
                        other_link = other_edge[2].get('link')
                        if other_link and other_link.name != link.name:
                            if other_link.name in link_to_joint:
                                parent0 = link_to_joint[other_link.name]
                                break
                    
                    for other_edge in topo_graph.edges(to_node, data=True):
                        other_link = other_edge[2].get('link')
                        if other_link and other_link.name != link.name:
                            if other_link.name in link_to_joint:
                                if parent0 is None:
                                    parent0 = link_to_joint[other_link.name]
                                else:
                                    parent1 = link_to_joint[other_link.name]
                                break
                    break
            
            if parent0 and parent1:
                # Revolute connects two joints with distance constraints
                revolute = Revolute(
                    joint0=parent0,
                    joint1=parent1,
                    distance0=float(link.length),
                    distance1=float(parent1.distance) if hasattr(parent1, 'distance') else float(link.length),
                    name=link.name
                )
                pylinkage_joints.append(revolute)
                link_to_joint[link.name] = revolute
                result.joint_mapping[f"link:{link.name}"] = revolute
                logger.debug(f"Created Revolute joint '{link.name}' connecting {parent0.name} and {parent1.name}")
            elif parent0:
                # Single parent - might be a terminal link or incomplete topology
                result.warnings.append(
                    f"Free link '{link.name}' only has one parent joint ({parent0.name}). "
                    "Creating as Fixed joint instead."
                )
                # Fallback: create as revolute with estimated position
                fixed = Fixed(
                    joint0=parent0,
                    distance=float(link.length),
                    name=link.name
                )
                pylinkage_joints.append(fixed)
                link_to_joint[link.name] = fixed
                result.joint_mapping[f"link:{link.name}"] = fixed
            else:
                result.warnings.append(
                    f"Could not determine parent joints for free link '{link.name}'. "
                    "Check that connections properly reference this link."
                )
        
        # ═══════════════════════════════════════════════════════════════
        # Step 5: Build the Linkage with solve order
        # ═══════════════════════════════════════════════════════════════
        solve_order = _determine_solve_order(topo_graph, static_joints, crank_joint, pylinkage_joints)
        
        try:
            linkage = Linkage(joints=pylinkage_joints, order=solve_order, name="automata_linkage")
            result.linkage = linkage
            result.success = True
            
            # Collect stats
            result.stats = {
                "total_joints": len(pylinkage_joints),
                "static_joints": len(static_joints),
                "crank_joints": 1 if crank_joint else 0,
                "fixed_joints": len(fixed_links),
                "revolute_joints": len([j for j in pylinkage_joints if isinstance(j, Revolute)]),
                "original_nodes": len(nodes),
                "original_links": len(links),
                "original_connections": len(connections)
            }
            
            logger.info(f"Successfully created pylinkage Linkage with {len(pylinkage_joints)} joints")
            
        except Exception as e:
            result.errors.append(f"Failed to create Linkage object: {str(e)}")
            logger.exception("Linkage creation failed")
    
    except Exception as e:
        result.errors.append(f"Conversion failed: {str(e)}")
        logger.exception("Conversion to pylinkage failed")
    
    return result


def simulate_linkage(
    linkage: Linkage,
    n_iterations: int = 24,
    use_fast: bool = True
) -> SimulationResult:
    """
    Run pylinkage simulation and extract trajectories.
    
    Args:
        linkage: The pylinkage Linkage object
        n_iterations: Number of simulation steps
        use_fast: Whether to use step_fast() for numba acceleration
    
    Returns:
        SimulationResult with trajectory data
    """
    import time
    
    result = SimulationResult()
    result.n_iterations = n_iterations
    
    try:
        start_time = time.perf_counter()
        
        # Initialize trajectories dict
        trajectories = {}
        for joint in linkage.joints:
            trajectories[joint.name] = np.zeros((n_iterations, 2))
        
        if use_fast and hasattr(linkage, 'step_fast'):
            # Use numba-compiled fast simulation
            try:
                trajectory_array = linkage.step_fast(iterations=n_iterations)
                # trajectory_array shape: (n_iterations, n_joints, 2)
                for i, joint in enumerate(linkage.joints):
                    if i < trajectory_array.shape[1]:
                        trajectories[joint.name] = trajectory_array[:, i, :]
            except Exception as e:
                logger.warning(f"step_fast failed, falling back to regular step: {e}")
                use_fast = False
        
        if not use_fast:
            # Use regular Python simulation
            linkage.rebuild()  # Reset to initial positions
            for step, coords in enumerate(linkage.step(iterations=n_iterations)):
                for joint, coord in zip(linkage.joints, coords):
                    if coord[0] is not None and coord[1] is not None:
                        trajectories[joint.name][step] = [coord[0], coord[1]]
        
        end_time = time.perf_counter()
        
        result.trajectories = trajectories
        result.success = True
        result.execution_time_ms = (end_time - start_time) * 1000
        
        logger.info(f"Simulation completed in {result.execution_time_ms:.2f}ms")
        
    except Exception as e:
        result.errors.append(f"Simulation failed: {str(e)}")
        logger.exception("Pylinkage simulation failed")
    
    return result


def trajectories_to_link_positions(
    trajectories: dict[str, np.ndarray],
    links: list[Link],
    connections: list[dict],
    joint_mapping: dict[str, Any]
) -> list[dict]:
    """
    Convert pylinkage trajectories back to automata Link pos1/pos2 format.
    
    This allows the frontend to use the same visualization code.
    """
    link_positions = []
    
    for link in links:
        link_data = {
            "name": link.name,
            "is_driven": link.is_driven,
            "is_ground": getattr(link, 'is_ground', False),
            "has_fixed": link.has_fixed,
            "has_constraint": link.has_constraint,
            "pos1": None,
            "pos2": None
        }
        
        # Find the joint corresponding to this link
        joint_key = f"link:{link.name}"
        if joint_key in joint_mapping:
            joint = joint_mapping[joint_key]
            joint_name = joint.name if hasattr(joint, 'name') else str(joint)
            
            if joint_name in trajectories:
                # The trajectory gives the end position of this link
                link_data["pos2"] = trajectories[joint_name].tolist()
                
                # pos1 comes from the parent joint
                if hasattr(joint, 'joint0') and joint.joint0:
                    parent_name = joint.joint0.name if hasattr(joint.joint0, 'name') else None
                    if parent_name and parent_name in trajectories:
                        link_data["pos1"] = trajectories[parent_name].tolist()
                    elif hasattr(joint.joint0, 'x') and hasattr(joint.joint0, 'y'):
                        # Static joint - constant position
                        n_iter = len(trajectories[joint_name])
                        pos1 = np.array([[joint.joint0.x, joint.joint0.y]] * n_iter)
                        link_data["pos1"] = pos1.tolist()
        
        link_positions.append(link_data)
    
    return link_positions


def extract_path_visualization_data(
    trajectories: dict[str, np.ndarray],
    links: list[Link],
    joint_mapping: dict[str, Any],
    n_iterations: int
) -> dict:
    """
    Extract visualization data in the format expected by the frontend.
    
    This matches the format from query_api.extract_path_visualization_data()
    """
    # Color palette (Spectral colormap approximation)
    def get_spectral_color(t):
        if t < 0.25:
            r = 158 + int((255-158) * t * 4)
            g = 1 + int((116-1) * t * 4)
            b = 5 + int((9-5) * t * 4)
        elif t < 0.5:
            r = 255 - int((255-255) * (t-0.25) * 4)
            g = 116 + int((217-116) * (t-0.25) * 4)
            b = 9 + int((54-9) * (t-0.25) * 4)
        elif t < 0.75:
            r = 255 - int((255-171) * (t-0.5) * 4)
            g = 217 + int((221-217) * (t-0.5) * 4)
            b = 54 + int((164-54) * (t-0.5) * 4)
        else:
            r = 171 - int((171-94) * (t-0.75) * 4)
            g = 221 - int((221-79) * (t-0.75) * 4)
            b = 164 - int((164-162) * (t-0.75) * 4)
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"
    
    # Calculate bounds
    all_positions = []
    for name, traj in trajectories.items():
        if traj is not None and len(traj) > 0:
            all_positions.extend(traj.tolist())
    
    if not all_positions:
        return {"bounds": None, "links": [], "history_data": [], "n_iterations": n_iterations}
    
    all_positions = np.array(all_positions)
    xmin, ymin = np.min(all_positions, axis=0)
    xmax, ymax = np.max(all_positions, axis=0)
    
    xdelta = xmax - xmin
    ydelta = ymax - ymin
    delta = max(xdelta, ydelta)
    margin = 0.2
    
    bounds = {
        "xmin": float(xmin - delta * margin),
        "xmax": float(xmax + delta * margin),
        "ymin": float(ymin - delta * margin),
        "ymax": float(ymax + delta * margin)
    }
    
    # Build links data
    links_data = []
    for link in links:
        joint_key = f"link:{link.name}"
        if joint_key not in joint_mapping:
            continue
            
        joint = joint_mapping[joint_key]
        joint_name = joint.name if hasattr(joint, 'name') else str(joint)
        
        if joint_name not in trajectories:
            continue
        
        pos2 = trajectories[joint_name]
        
        # Get pos1 from parent
        pos1 = None
        if hasattr(joint, 'joint0') and joint.joint0:
            parent = joint.joint0
            if hasattr(parent, 'name') and parent.name in trajectories:
                pos1 = trajectories[parent.name]
            elif hasattr(parent, 'x') and hasattr(parent, 'y'):
                pos1 = np.array([[parent.x, parent.y]] * n_iterations)
        
        if pos1 is not None and pos2 is not None:
            links_data.append({
                "name": link.name,
                "is_driven": link.is_driven,
                "is_ground": getattr(link, 'is_ground', False),
                "has_fixed": link.has_fixed,
                "has_constraint": link.has_constraint,
                "pos1": pos1.tolist(),
                "pos2": pos2.tolist()
            })
    
    # Generate history data for trails
    history_data = []
    n_history = int(n_iterations * 0.66)
    
    for frame in range(n_iterations):
        frame_history = []
        start_idx = max(0, frame - n_history)
        end_idx = frame
        
        for link_data in links_data:
            if not link_data["has_fixed"] and not link_data["has_constraint"]:
                pos2_array = np.array(link_data["pos2"])
                history_positions = pos2_array[start_idx:end_idx].tolist()
                
                history_colors = []
                for i, pos in enumerate(history_positions):
                    alpha = 1.0 / (1 + (len(history_positions) - i))
                    color_t = (start_idx + i) / n_iterations
                    color = get_spectral_color(color_t)
                    history_colors.append({"color": color, "alpha": alpha})
                
                frame_history.append({
                    "link_name": link_data["name"],
                    "positions": history_positions,
                    "colors": history_colors
                })
        
        history_data.append(frame_history)
    
    return {
        "bounds": bounds,
        "links": links_data,
        "history_data": history_data,
        "n_iterations": n_iterations
    }


def compare_solvers(
    nodes: list[Node],
    links: list[Link],
    connections: list[dict],
    n_iterations: int = 24
) -> dict:
    """
    Run both automata and pylinkage solvers and compare results.
    
    Returns comparison metrics including:
    - Position differences at each timestep
    - Max/mean errors
    - Performance timing
    """
    import time
    from link.graph_tools import make_graph, run_graph
    
    comparison = {
        "automata_time_ms": 0,
        "pylinkage_time_ms": 0,
        "position_errors": {},
        "max_error": 0,
        "mean_error": 0,
        "success": False
    }
    
    try:
        # Run automata solver
        start = time.perf_counter()
        # ... (would need to set up the full automata solve here)
        comparison["automata_time_ms"] = (time.perf_counter() - start) * 1000
        
        # Run pylinkage solver
        start = time.perf_counter()
        result = convert_to_pylinkage(nodes, links, connections)
        if result.success:
            sim_result = simulate_linkage(result.linkage, n_iterations)
            comparison["pylinkage_time_ms"] = (time.perf_counter() - start) * 1000
            
            if sim_result.success:
                comparison["success"] = True
                # Compare positions would go here if automata result available
        
    except Exception as e:
        comparison["error"] = str(e)
    
    return comparison

