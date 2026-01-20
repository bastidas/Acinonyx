/**
 * Linkage Document Helpers
 *
 * Pure utility functions for working with LinkageDocument (hypergraph format).
 * These replace the need for the legacy pylinkDoc adapter layer.
 *
 * Design principles:
 * - All functions are pure (no side effects)
 * - Work directly with LinkageDocument structure
 * - Return immutable results
 */

import type {
  LinkageDocument,
  Node,
  Edge,
  NodeId,
  EdgeId,
  Position,
  NodeRole,
  NodeMeta,
  EdgeMeta
} from '../../../types'

// ═══════════════════════════════════════════════════════════════════════════════
// NODE (JOINT) ACCESSORS
// ═══════════════════════════════════════════════════════════════════════════════

/** Get a node by ID */
export function getNode(doc: LinkageDocument, nodeId: NodeId): Node | undefined {
  return doc.linkage.nodes[nodeId]
}

/** Get all nodes as an array */
export function getNodes(doc: LinkageDocument): Node[] {
  return Object.values(doc.linkage.nodes)
}

/** Get all node IDs */
export function getNodeIds(doc: LinkageDocument): NodeId[] {
  return Object.keys(doc.linkage.nodes)
}

/** Check if a node exists */
export function hasNode(doc: LinkageDocument, nodeId: NodeId): boolean {
  return nodeId in doc.linkage.nodes
}

/** Get node metadata */
export function getNodeMeta(doc: LinkageDocument, nodeId: NodeId): NodeMeta | undefined {
  return doc.meta.nodes[nodeId]
}

/** Get node position */
export function getNodePosition(doc: LinkageDocument, nodeId: NodeId): Position | null {
  const node = doc.linkage.nodes[nodeId]
  return node ? node.position : null
}

/** Get nodes by role */
export function getNodesByRole(doc: LinkageDocument, role: NodeRole): Node[] {
  return getNodes(doc).filter(node => node.role === role)
}

/** Get fixed nodes (ground joints) */
export function getFixedNodes(doc: LinkageDocument): Node[] {
  return getNodesByRole(doc, 'fixed')
}

/** Get crank nodes */
export function getCrankNodes(doc: LinkageDocument): Node[] {
  return getNodesByRole(doc, 'crank')
}

/** Get follower nodes */
export function getFollowerNodes(doc: LinkageDocument): Node[] {
  return getNodesByRole(doc, 'follower')
}

/** Check if document has a crank (required for simulation) */
export function hasCrank(doc: LinkageDocument): boolean {
  return getNodes(doc).some(node => node.role === 'crank' || node.role === 'driven')
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE (LINK) ACCESSORS
// ═══════════════════════════════════════════════════════════════════════════════

/** Get an edge by ID */
export function getEdge(doc: LinkageDocument, edgeId: EdgeId): Edge | undefined {
  return doc.linkage.edges[edgeId]
}

/** Get all edges as an array */
export function getEdges(doc: LinkageDocument): Edge[] {
  return Object.values(doc.linkage.edges)
}

/** Get all edge IDs */
export function getEdgeIds(doc: LinkageDocument): EdgeId[] {
  return Object.keys(doc.linkage.edges)
}

/** Check if an edge exists */
export function hasEdge(doc: LinkageDocument, edgeId: EdgeId): boolean {
  return edgeId in doc.linkage.edges
}

/** Get edge metadata */
export function getEdgeMeta(doc: LinkageDocument, edgeId: EdgeId): EdgeMeta | undefined {
  return doc.meta.edges[edgeId]
}

/** Get edges connected to a node */
export function getEdgesForNode(doc: LinkageDocument, nodeId: NodeId): Edge[] {
  return getEdges(doc).filter(edge =>
    edge.source === nodeId || edge.target === nodeId
  )
}

/** Get edge IDs connected to a node */
export function getEdgeIdsForNode(doc: LinkageDocument, nodeId: NodeId): EdgeId[] {
  return getEdgesForNode(doc, nodeId).map(edge => edge.id)
}

/** Get the other node connected by an edge */
export function getOtherNode(edge: Edge, nodeId: NodeId): NodeId {
  return edge.source === nodeId ? edge.target : edge.source
}

/** Get ground edges (links marked as ground) */
export function getGroundEdges(doc: LinkageDocument): Edge[] {
  return getEdges(doc).filter(edge => doc.meta.edges[edge.id]?.isGround)
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADJACENCY & CONNECTIVITY
// ═══════════════════════════════════════════════════════════════════════════════

/** Build adjacency map: nodeId -> list of connected nodeIds */
export function buildAdjacencyMap(doc: LinkageDocument): Map<NodeId, NodeId[]> {
  const adjacency = new Map<NodeId, NodeId[]>()

  // Initialize with all nodes
  for (const nodeId of getNodeIds(doc)) {
    adjacency.set(nodeId, [])
  }

  // Add edges
  for (const edge of getEdges(doc)) {
    adjacency.get(edge.source)?.push(edge.target)
    adjacency.get(edge.target)?.push(edge.source)
  }

  return adjacency
}

/** Get nodes directly connected to a given node */
export function getConnectedNodes(doc: LinkageDocument, nodeId: NodeId): NodeId[] {
  const connected: NodeId[] = []

  for (const edge of getEdges(doc)) {
    if (edge.source === nodeId) {
      connected.push(edge.target)
    } else if (edge.target === nodeId) {
      connected.push(edge.source)
    }
  }

  return connected
}

/** Find all nodes reachable from a starting node (BFS) */
export function findConnectedComponent(doc: LinkageDocument, startNodeId: NodeId): Set<NodeId> {
  const visited = new Set<NodeId>()
  const queue = [startNodeId]

  while (queue.length > 0) {
    const current = queue.shift()!
    if (visited.has(current)) continue
    visited.add(current)

    for (const neighbor of getConnectedNodes(doc, current)) {
      if (!visited.has(neighbor)) {
        queue.push(neighbor)
      }
    }
  }

  return visited
}

/** Find edge connecting two nodes (if any) */
export function findEdgeBetween(doc: LinkageDocument, nodeA: NodeId, nodeB: NodeId): Edge | undefined {
  return getEdges(doc).find(edge =>
    (edge.source === nodeA && edge.target === nodeB) ||
    (edge.source === nodeB && edge.target === nodeA)
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEOMETRY HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/** Calculate distance between two positions */
export function calculateDistance(p1: Position, p2: Position): number {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  return Math.sqrt(dx * dx + dy * dy)
}

/** Calculate distance between two nodes */
export function calculateNodeDistance(doc: LinkageDocument, nodeA: NodeId, nodeB: NodeId): number | null {
  const posA = getNodePosition(doc, nodeA)
  const posB = getNodePosition(doc, nodeB)
  if (!posA || !posB) return null
  return calculateDistance(posA, posB)
}

/** Get edge length from stored distance or calculate from node positions */
export function getEdgeLength(doc: LinkageDocument, edgeId: EdgeId): number | null {
  const edge = getEdge(doc, edgeId)
  if (!edge) return null

  // Prefer stored distance
  if (edge.distance !== undefined && edge.distance > 0) {
    return edge.distance
  }

  // Calculate from positions
  return calculateNodeDistance(doc, edge.source, edge.target)
}

/** Get midpoint of an edge */
export function getEdgeMidpoint(doc: LinkageDocument, edgeId: EdgeId): Position | null {
  const edge = getEdge(doc, edgeId)
  if (!edge) return null

  const posA = getNodePosition(doc, edge.source)
  const posB = getNodePosition(doc, edge.target)
  if (!posA || !posB) return null

  return [(posA[0] + posB[0]) / 2, (posA[1] + posB[1]) / 2]
}

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

/** Check if a node is properly constrained for simulation */
export function isNodeConstrained(doc: LinkageDocument, nodeId: NodeId): boolean {
  const node = getNode(doc, nodeId)
  if (!node) return false

  // Fixed nodes are always constrained
  if (node.role === 'fixed') return true

  // Crank needs one edge to a fixed node
  if (node.role === 'crank' || node.role === 'driven') {
    const edges = getEdgesForNode(doc, nodeId)
    return edges.some(edge => {
      const otherId = getOtherNode(edge, nodeId)
      const otherNode = getNode(doc, otherId)
      return otherNode?.role === 'fixed'
    })
  }

  // Follower needs two edges to constrained nodes
  const edges = getEdgesForNode(doc, nodeId)
  return edges.length >= 2
}

/** Get all unconstrained nodes */
export function getUnconstrainedNodes(doc: LinkageDocument): Node[] {
  return getNodes(doc).filter(node => !isNodeConstrained(doc, node.id))
}

/** Check if document is valid for simulation */
export function isValidForSimulation(doc: LinkageDocument): { valid: boolean; issues: string[] } {
  const issues: string[] = []

  // Must have at least one crank
  if (!hasCrank(doc)) {
    issues.push('No crank joint defined - mechanism cannot be driven')
  }

  // Check for unconstrained nodes
  const unconstrained = getUnconstrainedNodes(doc)
  for (const node of unconstrained) {
    const edgeCount = getEdgesForNode(doc, node.id).length
    issues.push(`Node '${node.id}' is underconstrained (${edgeCount} edge(s))`)
  }

  return {
    valid: issues.length === 0,
    issues
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOCUMENT QUERIES
// ═══════════════════════════════════════════════════════════════════════════════

/** Get summary statistics for a document */
export function getDocumentStats(doc: LinkageDocument): {
  nodeCount: number
  edgeCount: number
  fixedCount: number
  crankCount: number
  followerCount: number
  groundEdgeCount: number
} {
  const nodes = getNodes(doc)
  const edges = getEdges(doc)

  return {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    fixedCount: nodes.filter(n => n.role === 'fixed').length,
    crankCount: nodes.filter(n => n.role === 'crank' || n.role === 'driven').length,
    followerCount: nodes.filter(n => n.role === 'follower').length,
    groundEdgeCount: getGroundEdges(doc).length
  }
}

/** Check if document is empty */
export function isDocumentEmpty(doc: LinkageDocument): boolean {
  return getNodeIds(doc).length === 0 && getEdgeIds(doc).length === 0
}
