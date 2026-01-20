/**
 * Links.tsx
 *
 * Utilities for working with linkage edges (links).
 *
 * ARCHITECTURE OVERVIEW (Hypergraph Model):
 * =========================================
 *
 * In the hypergraph model:
 *   - Edge: The kinematic link with source, target, and FIXED distance
 *   - EdgeMeta: Visual properties (color, isGround) for rendering
 *
 * Links (edges) connect nodes (joints). The edge DISTANCE is the ground truth -
 * it NEVER changes during animation. Positions are computed from constraints.
 *
 * During animation, ALL positions come from simulation trajectories:
 *   trajectoryData.trajectories[nodeId][stepIndex] → [x, y]
 *
 * Reference: types/pylink.ts for type definitions
 */

import {
  Edge,
  EdgeMeta,
  NodeId,
  Position
} from '../types'

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE WITH GEOMETRY (for rendering)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Edge data combined with computed positions for rendering.
 * Used when drawing links on the canvas.
 */
export interface EdgeWithGeometry {
  /** Edge ID */
  id: string

  /** Source node ID */
  source: NodeId

  /** Target node ID */
  target: NodeId

  /** Fixed distance (from Edge) */
  distance: number

  /** Display color (from EdgeMeta) */
  color: string

  /** Whether this is a ground link (from EdgeMeta) */
  isGround?: boolean

  /** Computed position of source node at current frame */
  p1: Position

  /** Computed position of target node at current frame */
  p2: Position

  /** Computed length (should match distance if simulation is correct) */
  computedLength: number
}

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Calculate distance between two positions
 */
export function calculateDistance(p1: Position, p2: Position): number {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Get edges with geometry at a given animation frame.
 *
 * @param edges - The edges from the linkage
 * @param edgeMeta - UI metadata for edges
 * @param trajectories - trajectoryData.trajectories from simulation
 * @param frameIndex - Current animation frame (step index)
 * @param fallbackPositions - Optional fallback for nodes not in trajectories
 */
export function getEdgesAtFrame(
  edges: Record<string, Edge>,
  edgeMeta: Record<string, EdgeMeta>,
  trajectories: Record<NodeId, Position[]>,
  frameIndex: number,
  fallbackPositions?: Record<NodeId, Position>
): EdgeWithGeometry[] {
  const result: EdgeWithGeometry[] = []

  for (const [edgeId, edge] of Object.entries(edges)) {
    const meta = edgeMeta[edgeId] || { color: '#888888' }

    // Get node positions from trajectory first, then fallback
    let p1: Position | undefined
    let p2: Position | undefined

    // Source position
    if (trajectories[edge.source]?.[frameIndex]) {
      p1 = trajectories[edge.source][frameIndex]
    } else if (fallbackPositions?.[edge.source]) {
      p1 = fallbackPositions[edge.source]
    }

    // Target position
    if (trajectories[edge.target]?.[frameIndex]) {
      p2 = trajectories[edge.target][frameIndex]
    } else if (fallbackPositions?.[edge.target]) {
      p2 = fallbackPositions[edge.target]
    }

    if (p1 && p2) {
      result.push({
        id: edgeId,
        source: edge.source,
        target: edge.target,
        distance: edge.distance,
        color: meta.color,
        isGround: meta.isGround,
        p1,
        p2,
        computedLength: calculateDistance(p1, p2)
      })
    }
  }

  return result
}

/**
 * Check if an edge would stretch during animation.
 *
 * This detects invalid links where the computed distance varies across frames.
 * Valid links maintain constant distance throughout the animation.
 *
 * @param edge - The edge to check
 * @param trajectories - Full trajectory data
 * @returns Object with stretches flag and optional details
 */
export function wouldEdgeStretch(
  edge: Edge,
  trajectories: Record<NodeId, Position[]>
): { stretches: boolean; details?: string } {
  const sourceTrajectory = trajectories[edge.source]
  const targetTrajectory = trajectories[edge.target]

  // If either node has no trajectory, can't determine stretching
  if (!sourceTrajectory || !targetTrajectory) {
    return { stretches: false }
  }

  // Calculate distances at each frame
  const nFrames = Math.min(sourceTrajectory.length, targetTrajectory.length)
  const distances: number[] = []

  for (let i = 0; i < nFrames; i++) {
    const p1 = sourceTrajectory[i]
    const p2 = targetTrajectory[i]
    distances.push(calculateDistance(p1, p2))
  }

  const minLen = Math.min(...distances)
  const maxLen = Math.max(...distances)
  const variance = maxLen - minLen

  // More than 0.01 units variance indicates stretching
  if (variance > 0.01) {
    return {
      stretches: true,
      details: `Edge length varies from ${minLen.toFixed(2)} to ${maxLen.toFixed(2)} across frames`
    }
  }

  return { stretches: false }
}

/**
 * Validate all edges in a mechanism.
 * Returns list of edges that would stretch during animation.
 */
export function validateEdges(
  edges: Record<string, Edge>,
  trajectories: Record<NodeId, Position[]>
): { valid: boolean; problems: Array<{ edgeId: string; issue: string }> } {
  const problems: Array<{ edgeId: string; issue: string }> = []

  for (const [edgeId, edge] of Object.entries(edges)) {
    const result = wouldEdgeStretch(edge, trajectories)
    if (result.stretches) {
      problems.push({
        edgeId,
        issue: result.details || 'Edge would stretch during animation'
      })
    }
  }

  return {
    valid: problems.length === 0,
    problems
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE CREATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create an edge (link) between two nodes.
 *
 * @param id - Unique edge ID
 * @param source - Source node ID
 * @param target - Target node ID
 * @param distance - Fixed distance (link length)
 */
export function createEdge(
  id: string,
  source: NodeId,
  target: NodeId,
  distance: number
): Edge {
  return {
    id,
    source,
    target,
    distance
  }
}

/**
 * Create edge metadata for rendering.
 *
 * @param color - Display color (hex string)
 * @param isGround - Whether this is a ground link
 */
export function createEdgeMeta(
  color: string,
  isGround: boolean = false
): EdgeMeta {
  return {
    color,
    ...(isGround && { isGround })
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DEFAULT COLORS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Default colors for edges (Tab10 palette)
 */
export const DEFAULT_EDGE_COLORS = [
  '#1f77b4',  // blue
  '#ff7f0e',  // orange
  '#2ca02c',  // green
  '#d62728',  // red
  '#9467bd',  // purple
  '#8c564b',  // brown
  '#e377c2',  // pink
  '#7f7f7f',  // gray
  '#bcbd22',  // olive
  '#17becf'   // cyan
]

export const GROUND_EDGE_COLOR = '#7f7f7f'  // gray for ground links

/**
 * Get default color for an edge by index
 */
export function getEdgeColor(index: number): string {
  return DEFAULT_EDGE_COLORS[index % DEFAULT_EDGE_COLORS.length]
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY COMPATIBILITY (deprecated - will be removed)
// These exports maintain compatibility during migration
// ═══════════════════════════════════════════════════════════════════════════════

/** @deprecated Use EdgeMeta from types/pylink.ts */
export interface LinkMeta {
  color: string
  connects: [string, string]
  isGround?: boolean
}

/** @deprecated Use EdgeWithGeometry */
export interface LinkWithGeometry extends LinkMeta {
  name: string
  length: number
  p1: Position
  p2: Position
}

/** @deprecated Use calculateDistance */
export const calculateLinkLength = calculateDistance

/** @deprecated Use validateEdges */
export function validateLinks(
  metaLinks: Record<string, LinkMeta>,
  _jointTypes: Record<string, string>,
  trajectories: Record<string, Position[]>
): {
  valid: boolean
  problems: Array<{ linkName: string; issue: string }>
  stretchingLinks: string[]  // Links that would stretch during animation
} {
  const problems: Array<{ linkName: string; issue: string }> = []
  const stretchingLinks: string[] = []

  for (const [linkName, linkMeta] of Object.entries(metaLinks)) {
    const [j1, j2] = linkMeta.connects

    // Check if both joints have trajectories
    const j1Traj = trajectories[j1]
    const j2Traj = trajectories[j2]

    // CRITICAL: If one joint has trajectory but the other doesn't,
    // the link would stretch during animation (one end moves, one doesn't)
    const j1HasTraj = j1Traj && j1Traj.length > 0
    const j2HasTraj = j2Traj && j2Traj.length > 0

    if (j1HasTraj !== j2HasTraj) {
      // One joint simulated, one not - this link will stretch!
      const simulatedJoint = j1HasTraj ? j1 : j2
      const unsimualtedJoint = j1HasTraj ? j2 : j1
      problems.push({
        linkName,
        issue: `Link connects simulated joint '${simulatedJoint}' to unsimulated joint '${unsimualtedJoint}' - would stretch during animation`
      })
      stretchingLinks.push(linkName)
      continue
    }

    if (!j1Traj || !j2Traj) continue

    // Calculate distances at each frame
    const nFrames = Math.min(j1Traj.length, j2Traj.length)
    const distances: number[] = []

    for (let i = 0; i < nFrames; i++) {
      distances.push(calculateDistance(j1Traj[i], j2Traj[i]))
    }

    const minLen = Math.min(...distances)
    const maxLen = Math.max(...distances)

    // Use a small tolerance (0.5% of average length) for floating point errors
    const avgLen = (minLen + maxLen) / 2
    const tolerance = Math.max(0.01, avgLen * 0.005)

    if (maxLen - minLen > tolerance) {
      problems.push({
        linkName,
        issue: `Link length varies from ${minLen.toFixed(2)} to ${maxLen.toFixed(2)} across frames`
      })
      stretchingLinks.push(linkName)
    }
  }

  return {
    valid: problems.length === 0,
    problems,
    stretchingLinks
  }
}

/** @deprecated Use DEFAULT_EDGE_COLORS */
export const DEFAULT_LINK_COLORS = DEFAULT_EDGE_COLORS

/** @deprecated Use GROUND_EDGE_COLOR */
export const GROUND_LINK_COLOR = GROUND_EDGE_COLOR

/** @deprecated Use createEdgeMeta */
export function createLinkMeta(
  connects: [string, string],
  color: string,
  isGround: boolean = false
): LinkMeta {
  return {
    color,
    connects,
    ...(isGround && { isGround })
  }
}
