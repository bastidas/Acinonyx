/**
 * pylink.ts - Unified type definitions for linkage mechanisms
 *
 * This module defines TypeScript types that mirror the pylinkage hypergraph format.
 * These types are the SINGLE SOURCE OF TRUTH for the frontend data model.
 *
 * Reference: https://github.com/HugoFara/pylinkage/tree/main/src/pylinkage/hypergraph
 *
 * Architecture Overview:
 * =====================
 *
 * The hypergraph model represents linkage mechanisms as:
 *   - Nodes: Joints/pivot points with positions and roles
 *   - Edges: Rigid links connecting exactly two nodes with a fixed distance
 *   - Hyperedges: Multi-node constraints (for complex mechanisms)
 *
 * Key principle: Links are EXPLICIT Edge objects with distance constraints.
 * The distance is the ground truth - positions are computed from constraints.
 *
 * During animation, ALL positions come from simulation results (trajectories).
 * Links NEVER stretch - their distance is constant across all frames.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// PRIMITIVE TYPES
// Reference: pylinkage/hypergraph/_types.py
// ═══════════════════════════════════════════════════════════════════════════════

/** Unique identifier for a node within a graph */
export type NodeId = string

/** Unique identifier for an edge within a graph */
export type EdgeId = string

/** Unique identifier for a hyperedge within a graph */
export type HyperedgeId = string

/** Unique identifier for a port on a component */
export type PortId = string

/** 2D position as [x, y] tuple */
export type Position = [number, number]

// ═══════════════════════════════════════════════════════════════════════════════
// NODE ROLES AND JOINT TYPES
// Reference: pylinkage/hypergraph/core.py
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Role of a node in the mechanism.
 * Determines how the node's position is computed during simulation.
 */
export type NodeRole =
  | 'fixed'    // Stationary anchor point (ground)
  | 'driven'   // Position controlled by input (e.g., motor)
  | 'crank'    // Rotates around a fixed point at constant distance
  | 'follower' // Position determined by constraints from other nodes

/**
 * Type of joint at a node.
 * Determines the degrees of freedom and constraint behavior.
 */
export type JointType =
  | 'revolute'   // Pin joint - allows rotation only
  | 'prismatic'  // Slider joint - allows translation along an axis
  | 'fixed'      // No relative motion - rigidly connected

// ═══════════════════════════════════════════════════════════════════════════════
// CORE GRAPH ELEMENTS
// Reference: pylinkage/hypergraph/core.py
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * A node in the linkage hypergraph.
 *
 * Represents a joint or pivot point in the mechanism.
 * The position may be fixed (for anchors) or computed from constraints.
 *
 * Reference: pylinkage/hypergraph/core.py - Node dataclass
 */
export interface Node {
  /** Unique identifier within the graph */
  id: NodeId

  /** Current position [x, y]. For fixed nodes, this is the ground truth.
   *  For other nodes, this is the initial/current computed position. */
  position: Position

  /** Role determining how position is computed */
  role: NodeRole

  /** Type of joint (affects constraint solving) */
  jointType: JointType

  /** Current angle in radians (for crank/driven nodes) */
  angle?: number

  /** Initial angle in radians (starting position for simulation) */
  initialAngle?: number

  /** Human-readable name for display */
  name?: string
}

/**
 * An edge connecting exactly two nodes.
 *
 * Represents a rigid link between two joints. The distance constraint
 * is CONSTANT - links never stretch during animation.
 *
 * Reference: pylinkage/hypergraph/core.py - Edge dataclass
 */
export interface Edge {
  /** Unique identifier within the graph */
  id: EdgeId

  /** Source node ID */
  source: NodeId

  /** Target node ID */
  target: NodeId

  /** Fixed distance between source and target (link length).
   *  This is the kinematic constraint - NEVER changes during animation. */
  distance: number
}

/**
 * A hyperedge connecting multiple nodes with distance constraints.
 *
 * Used for complex multi-node constraints that can't be expressed
 * as simple two-node edges. Common in parallel mechanisms.
 *
 * Reference: pylinkage/hypergraph/core.py - Hyperedge dataclass
 */
export interface Hyperedge {
  /** Unique identifier within the graph */
  id: HyperedgeId

  /** Ordered tuple of node IDs involved in this constraint */
  nodes: NodeId[]

  /** Distance constraints between pairs of nodes.
   *  Key is "nodeId1:nodeId2" (alphabetically ordered), value is distance. */
  constraints: Record<string, number>

  /** Human-readable name for display */
  name?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// HYPERGRAPH LINKAGE
// Reference: pylinkage/hypergraph/graph.py - HypergraphLinkage
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * A complete linkage mechanism represented as a hypergraph.
 *
 * This is the primary data structure for representing mechanisms.
 * Contains all nodes (joints), edges (links), and hyperedges (constraints).
 *
 * Reference: pylinkage/hypergraph/graph.py - HypergraphLinkage class
 */
export interface HypergraphLinkage {
  /** Human-readable name for the linkage */
  name: string

  /** All nodes (joints) in the mechanism, keyed by ID */
  nodes: Record<NodeId, Node>

  /** All edges (links) in the mechanism, keyed by ID */
  edges: Record<EdgeId, Edge>

  /** All hyperedges (multi-node constraints), keyed by ID */
  hyperedges: Record<HyperedgeId, Hyperedge>
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT SYSTEM (Hierarchical Composition)
// Reference: pylinkage/hypergraph/hierarchy.py
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Port definition on a component.
 *
 * Ports are connection points that allow components to be linked together.
 * Each port maps to an internal node within the component.
 */
export interface Port {
  /** Unique identifier for this port within the component */
  id: PortId

  /** The internal node ID this port exposes */
  internalNode: NodeId

  /** Human-readable name */
  name?: string

  /** Description of the port's purpose */
  description?: string
}

/**
 * A reusable component template.
 *
 * Components are parameterized mechanism templates that can be instantiated
 * with specific parameter values. They expose ports for connection.
 *
 * Reference: pylinkage/hypergraph/components.py - Component class
 */
export interface Component {
  /** Unique identifier for this component type */
  id: string

  /** Human-readable name */
  name: string

  /** The internal graph structure of this component */
  internalGraph: HypergraphLinkage

  /** Ports exposed for external connections */
  ports: Record<PortId, Port>

  /** Parameter definitions with default values */
  parameters: Record<string, ComponentParameter>

  /** Description of the component */
  description?: string
}

/**
 * Parameter definition for a component.
 */
export interface ComponentParameter {
  /** Human-readable name */
  name: string

  /** Default value */
  defaultValue: number

  /** Minimum allowed value */
  min?: number

  /** Maximum allowed value */
  max?: number

  /** Description of what this parameter controls */
  description?: string
}

/**
 * An instance of a component with specific parameters.
 *
 * Represents a concrete instantiation of a Component template with
 * specific parameter values and a unique instance ID.
 *
 * Reference: pylinkage/hypergraph/hierarchy.py - ComponentInstance dataclass
 *
 * Example:
 *   const instance: ComponentInstance = {
 *     id: "left_leg",
 *     componentId: "leg_component",
 *     parameters: { crank_length: 1.5 },
 *     name: "Left Leg"
 *   }
 */
export interface ComponentInstance {
  /** Unique identifier for this instance within the hierarchy */
  id: string

  /** ID of the component type being instantiated */
  componentId: string

  /** Parameter values for this instance (overrides defaults) */
  parameters: Record<string, number>

  /** Human-readable name for this instance */
  name: string
}

/**
 * Connection between two component ports.
 *
 * Defines how two component instances are connected via their ports.
 * During flattening, connected ports share the same position.
 *
 * Reference: pylinkage/hypergraph/hierarchy.py - Connection dataclass
 */
export interface ComponentConnection {
  /** Instance ID of the source component */
  fromInstance: string

  /** Port ID on the source component */
  fromPort: PortId

  /** Instance ID of the target component */
  toInstance: string

  /** Port ID on the target component */
  toPort: PortId
}

/**
 * Top-level container for hierarchical linkage definition.
 *
 * A hierarchical linkage contains component instances and defines
 * connections between them. It can be flattened to a HypergraphLinkage
 * for simulation.
 *
 * Reference: pylinkage/hypergraph/hierarchy.py - HierarchicalLinkage dataclass
 */
export interface HierarchicalLinkage {
  /** Human-readable name for the linkage */
  name: string

  /** Component instances by ID */
  instances: Record<string, ComponentInstance>

  /** Connections between component ports */
  connections: ComponentConnection[]
}

// ═══════════════════════════════════════════════════════════════════════════════
// UI METADATA (Frontend-only, not sent to backend)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * UI-specific metadata for a node.
 * This is ONLY for rendering - not part of the kinematic model.
 */
export interface NodeMeta {
  /** Display color (hex string) */
  color: string

  /** Z-level for layering in rendering */
  zlevel: number

  /** Whether to show trajectory path for this node during animation */
  showPath?: boolean
}

/**
 * UI-specific metadata for an edge.
 * This is ONLY for rendering - not part of the kinematic model.
 */
export interface EdgeMeta {
  /** Display color (hex string) */
  color: string

  /** Whether this is a ground link (connects fixed nodes) */
  isGround?: boolean

  /** Z-level for layering in rendering */
  zlevel?: number
}

/**
 * Collection of UI metadata for a linkage.
 */
export interface LinkageMeta {
  /** Metadata for nodes, keyed by node ID */
  nodes: Record<NodeId, NodeMeta>

  /** Metadata for edges, keyed by edge ID */
  edges: Record<EdgeId, EdgeMeta>
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOCUMENT FORMAT (Complete frontend state)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Complete document format for saving/loading.
 *
 * Combines the kinematic model (HypergraphLinkage) with UI metadata.
 * This is what gets saved to disk and sent to the backend.
 */
export interface LinkageDocument {
  /** Document name */
  name: string

  /** Version of the document format (for migrations) */
  version: string

  /** The kinematic linkage model */
  linkage: HypergraphLinkage

  /** UI metadata for rendering */
  meta: LinkageMeta

  /** Optional: Component library used by this document */
  components?: Record<string, Component>

  /** Optional: Hierarchical structure if using components */
  hierarchy?: HierarchicalLinkage

  /** Timestamp when saved */
  savedAt?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION & TRAJECTORY TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Trajectory data from a simulation run.
 *
 * Contains the position of each node at each time step.
 * This is the OUTPUT of simulation - frontend renders these positions.
 */
export interface TrajectoryData {
  /** Number of simulation steps */
  nSteps: number

  /** Trajectories for each node: nodeId -> array of positions */
  trajectories: Record<NodeId, Position[]>

  /** Node roles/types from simulation (for reference) */
  nodeTypes?: Record<NodeId, NodeRole>
}

/**
 * Target trajectory for optimization.
 *
 * Defines a desired path that a specific node should follow.
 */
export interface TargetTrajectory {
  /** ID of the target path */
  id: string

  /** Node that should follow this trajectory */
  nodeId: NodeId

  /** Target positions (same length as simulation steps) */
  positions: Position[]

  /** Whether this is a closed loop */
  isClosed: boolean

  /** Display color for visualization */
  color?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY COMPATIBILITY TYPES
// These support the current pylinkage format until backend is migrated
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Legacy joint reference (for current pylinkage format)
 * @deprecated Will be removed after backend migration
 */
export interface LegacyJointRef {
  ref: string
}

/**
 * Legacy Static joint (for current pylinkage format)
 * @deprecated Will be removed after backend migration
 */
export interface LegacyStaticJoint {
  type: 'Static'
  name: string
  x: number
  y: number
}

/**
 * Legacy Crank joint (for current pylinkage format)
 * @deprecated Will be removed after backend migration
 */
export interface LegacyCrankJoint {
  type: 'Crank'
  name: string
  joint0: LegacyJointRef
  distance: number
  angle: number
}

/**
 * Legacy Revolute joint (for current pylinkage format)
 * @deprecated Will be removed after backend migration
 */
export interface LegacyRevoluteJoint {
  type: 'Revolute'
  name: string
  joint0: LegacyJointRef
  joint1: LegacyJointRef
  distance0: number
  distance1: number
}

/**
 * Legacy joint union type
 * @deprecated Will be removed after backend migration
 */
export type LegacyPylinkJoint = LegacyStaticJoint | LegacyCrankJoint | LegacyRevoluteJoint

/**
 * Legacy pylinkage data structure
 * @deprecated Will be removed after backend migration
 */
export interface LegacyPylinkageData {
  name: string
  joints: LegacyPylinkJoint[]
  solve_order: string[]
}

/**
 * Legacy link metadata
 * @deprecated Will be removed after backend migration
 */
export interface LegacyLinkMeta {
  color: string
  connects: [string, string]
  isGround?: boolean
}

/**
 * Legacy joint metadata
 * @deprecated Will be removed after backend migration
 */
export interface LegacyJointMeta {
  color: string
  zlevel: number
  x?: number
  y?: number
  show_path?: boolean
}

/**
 * Legacy UI metadata
 * @deprecated Will be removed after backend migration
 */
export interface LegacyUIMeta {
  joints: Record<string, LegacyJointMeta>
  links: Record<string, LegacyLinkMeta>
}

/**
 * Legacy document format (current backend format)
 * @deprecated Will be removed after backend migration
 */
export interface LegacyPylinkDocument {
  name: string
  pylinkage: LegacyPylinkageData
  meta: LegacyUIMeta
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONVERSION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Convert a legacy document to the new hypergraph format.
 * Use this when loading old saved files.
 */
export function convertLegacyToHypergraph(legacy: LegacyPylinkDocument): LinkageDocument {
  const nodes: Record<NodeId, Node> = {}
  const edges: Record<EdgeId, Edge> = {}
  const nodeMeta: Record<NodeId, NodeMeta> = {}
  const edgeMeta: Record<EdgeId, EdgeMeta> = {}

  // Convert joints to nodes
  for (const joint of legacy.pylinkage.joints) {
    const meta = legacy.meta.joints[joint.name]

    let position: Position
    let role: NodeRole
    let angle: number | undefined

    if (joint.type === 'Static') {
      position = [joint.x, joint.y]
      role = 'fixed'
    } else if (joint.type === 'Crank') {
      // Use meta position if available, otherwise compute from parent
      position = meta?.x !== undefined && meta?.y !== undefined
        ? [meta.x, meta.y]
        : [0, 0] // Will be computed
      role = 'crank'
      angle = joint.angle
    } else {
      // Revolute
      position = meta?.x !== undefined && meta?.y !== undefined
        ? [meta.x, meta.y]
        : [0, 0] // Will be computed
      role = 'follower'
    }

    nodes[joint.name] = {
      id: joint.name,
      position,
      role,
      jointType: 'revolute',
      angle,
      name: joint.name
    }

    if (meta) {
      nodeMeta[joint.name] = {
        color: meta.color,
        zlevel: meta.zlevel,
        showPath: meta.show_path
      }
    }
  }

  // Convert links to edges
  let edgeIndex = 0
  for (const [linkName, linkMeta] of Object.entries(legacy.meta.links)) {
    const [sourceId, targetId] = linkMeta.connects
    const edgeId = linkName || `edge_${edgeIndex++}`

    // Compute distance from positions
    const sourceNode = nodes[sourceId]
    const targetNode = nodes[targetId]
    const distance = sourceNode && targetNode
      ? Math.sqrt(
          Math.pow(targetNode.position[0] - sourceNode.position[0], 2) +
          Math.pow(targetNode.position[1] - sourceNode.position[1], 2)
        )
      : 0

    edges[edgeId] = {
      id: edgeId,
      source: sourceId,
      target: targetId,
      distance
    }

    edgeMeta[edgeId] = {
      color: linkMeta.color,
      isGround: linkMeta.isGround
    }
  }

  return {
    name: legacy.name,
    version: '2.0.0',
    linkage: {
      name: legacy.pylinkage.name,
      nodes,
      edges,
      hyperedges: {}
    },
    meta: {
      nodes: nodeMeta,
      edges: edgeMeta
    }
  }
}

/**
 * Convert a hypergraph document to legacy format.
 * Use this when saving for the current backend.
 */
export function convertHypergraphToLegacy(doc: LinkageDocument): LegacyPylinkDocument {
  const joints: LegacyPylinkJoint[] = []
  const jointsMeta: Record<string, LegacyJointMeta> = {}
  const linksMeta: Record<string, LegacyLinkMeta> = {}

  // Build adjacency info for determining joint types
  const nodeEdges: Record<NodeId, EdgeId[]> = {}
  for (const edge of Object.values(doc.linkage.edges)) {
    if (!nodeEdges[edge.source]) nodeEdges[edge.source] = []
    if (!nodeEdges[edge.target]) nodeEdges[edge.target] = []
    nodeEdges[edge.source].push(edge.id)
    nodeEdges[edge.target].push(edge.id)
  }

  // Sort nodes by role to establish proper order
  const sortedNodes = Object.values(doc.linkage.nodes).sort((a, b) => {
    const roleOrder = { fixed: 0, crank: 1, driven: 1, follower: 2 }
    return (roleOrder[a.role] || 3) - (roleOrder[b.role] || 3)
  })

  // Convert nodes to joints
  for (const node of sortedNodes) {
    const meta = doc.meta.nodes[node.id]

    if (node.role === 'fixed') {
      joints.push({
        type: 'Static',
        name: node.id,
        x: node.position[0],
        y: node.position[1]
      })
    } else if (node.role === 'crank') {
      // Find the fixed parent
      const connectedEdges = nodeEdges[node.id] || []
      let parentId = ''
      let distance = 0

      for (const edgeId of connectedEdges) {
        const edge = doc.linkage.edges[edgeId]
        const otherId = edge.source === node.id ? edge.target : edge.source
        const otherNode = doc.linkage.nodes[otherId]
        if (otherNode?.role === 'fixed') {
          parentId = otherId
          distance = edge.distance
          break
        }
      }

      joints.push({
        type: 'Crank',
        name: node.id,
        joint0: { ref: parentId },
        distance,
        angle: node.angle || 0
      })
    } else {
      // Follower - Revolute joint
      const connectedEdges = nodeEdges[node.id] || []
      let joint0 = ''
      let joint1 = ''
      let distance0 = 0
      let distance1 = 0

      // Find two parent joints
      for (const edgeId of connectedEdges) {
        const edge = doc.linkage.edges[edgeId]
        const otherId = edge.source === node.id ? edge.target : edge.source

        if (!joint0) {
          joint0 = otherId
          distance0 = edge.distance
        } else if (!joint1) {
          joint1 = otherId
          distance1 = edge.distance
          break
        }
      }

      joints.push({
        type: 'Revolute',
        name: node.id,
        joint0: { ref: joint0 },
        joint1: { ref: joint1 },
        distance0,
        distance1
      })
    }

    // Convert metadata
    if (meta) {
      jointsMeta[node.id] = {
        color: meta.color,
        zlevel: meta.zlevel,
        x: node.position[0],
        y: node.position[1],
        show_path: meta.showPath
      }
    }
  }

  // Convert edges to links
  for (const [edgeId, edge] of Object.entries(doc.linkage.edges)) {
    const meta = doc.meta.edges[edgeId]
    linksMeta[edgeId] = {
      color: meta?.color || '#888888',
      connects: [edge.source, edge.target],
      isGround: meta?.isGround
    }
  }

  return {
    name: doc.name,
    pylinkage: {
      name: doc.linkage.name,
      joints,
      solve_order: joints.map(j => j.name)
    },
    meta: {
      joints: jointsMeta,
      links: linksMeta
    }
  }
}
