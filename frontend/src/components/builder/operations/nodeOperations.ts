/**
 * Node Operations (Hypergraph Format)
 *
 * Pure functions for creating, updating, deleting, and manipulating nodes.
 * Works directly with LinkageDocument (hypergraph format).
 *
 * These replace the legacy jointOperations.ts functions.
 */

import type { LinkageDocument, Node, NodeId, NodeRole, Position, NodeMeta } from '../../../types'
import {
  getNode,
  getNodes,
  getNodeIds,
  getEdgesForNode,
  getOtherNode
} from '../helpers/linkageHelpers'
import {
  addNode,
  removeNode,
  moveNode,
  updateNode,
  updateNodeMeta,
  renameNode,
  removeEdge,
  translateNodes,
  mergeNodes as mergeNodesHelper
} from '../helpers/linkageMutations'

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface NodeDeletionResult {
  doc: LinkageDocument
  deletedNodes: NodeId[]
  deletedEdges: string[]
  message: string
}

export interface NodeMoveResult {
  doc: LinkageDocument
  movedNode: NodeId
}

export interface GroupMoveResult {
  doc: LinkageDocument
  movedNodes: NodeId[]
}

export interface NodeMergeResult {
  doc: LinkageDocument
  sourceNode: NodeId
  targetNode: NodeId
  deletedEdges: string[]
  message: string
}

export interface NodeRenameResult {
  doc: LinkageDocument
  oldName: NodeId
  newName: NodeId
  success: boolean
  error?: string
}

export interface NodeUpdateResult {
  doc: LinkageDocument
  success: boolean
  message?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE NODE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Find orphaned nodes that would result from deleting nodes and edges
 *
 * A node becomes orphaned if:
 * - It's not in the remaining nodes set
 * - It has no remaining edges connecting to non-deleted nodes
 * - It's a follower node (fixed and crank nodes are not orphaned)
 */
function findOrphanedNodes(
  doc: LinkageDocument,
  nodesToDelete: Set<NodeId>,
  edgesToDelete: Set<string>
): NodeId[] {
  const orphans: NodeId[] = []

  for (const node of getNodes(doc)) {
    if (nodesToDelete.has(node.id)) continue
    if (node.role === 'fixed' || node.role === 'crank') continue

    // Check if this node has any remaining connections
    const edges = getEdgesForNode(doc, node.id)
    const hasRemainingConnection = edges.some(edge => {
      if (edgesToDelete.has(edge.id)) return false
      const otherId = getOtherNode(edge, node.id)
      return !nodesToDelete.has(otherId)
    })

    if (!hasRemainingConnection) {
      orphans.push(node.id)
    }
  }

  return orphans
}

/**
 * Delete a node and all connected edges, plus any resulting orphan nodes
 */
export function deleteNode(
  doc: LinkageDocument,
  nodeId: NodeId
): NodeDeletionResult {
  if (!getNode(doc, nodeId)) {
    return {
      doc,
      deletedNodes: [],
      deletedEdges: [],
      message: `Node "${nodeId}" not found`
    }
  }

  // Find all edges connected to this node
  const connectedEdges = getEdgesForNode(doc, nodeId)
  const edgeIds = connectedEdges.map(e => e.id)

  // Find orphans that will result from this deletion
  const nodesToDelete = new Set([nodeId])
  const edgesToDelete = new Set(edgeIds)
  const orphans = findOrphanedNodes(doc, nodesToDelete, edgesToDelete)

  // Add orphans to deletion set and find their edges
  for (const orphanId of orphans) {
    nodesToDelete.add(orphanId)
    for (const edge of getEdgesForNode(doc, orphanId)) {
      edgesToDelete.add(edge.id)
    }
  }

  // Apply deletions
  let result = doc

  // Remove edges first
  for (const edgeId of edgesToDelete) {
    result = removeEdge(result, edgeId)
  }

  // Remove nodes
  for (const id of nodesToDelete) {
    result = removeNode(result, id)
  }

  // Build message
  const parts: string[] = [`Deleted ${nodeId}`]
  if (edgesToDelete.size > 0) {
    parts.push(`${edgesToDelete.size} edge(s)`)
  }
  if (orphans.length > 0) {
    parts.push(`${orphans.length} orphan(s)`)
  }

  return {
    doc: result,
    deletedNodes: Array.from(nodesToDelete),
    deletedEdges: Array.from(edgesToDelete),
    message: parts.join(' + ')
  }
}

/**
 * Delete multiple nodes at once
 */
export function deleteNodes(
  doc: LinkageDocument,
  nodeIds: NodeId[]
): NodeDeletionResult {
  let result = doc
  const allDeletedNodes = new Set<NodeId>()
  const allDeletedEdges = new Set<string>()

  for (const nodeId of nodeIds) {
    if (allDeletedNodes.has(nodeId)) continue

    const deletion = deleteNode(result, nodeId)
    result = deletion.doc

    for (const id of deletion.deletedNodes) {
      allDeletedNodes.add(id)
    }
    for (const id of deletion.deletedEdges) {
      allDeletedEdges.add(id)
    }
  }

  return {
    doc: result,
    deletedNodes: Array.from(allDeletedNodes),
    deletedEdges: Array.from(allDeletedEdges),
    message: `Deleted ${allDeletedNodes.size} node(s) + ${allDeletedEdges.size} edge(s)`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOVE NODE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Move a node to a new position
 */
export function moveNodeTo(
  doc: LinkageDocument,
  nodeId: NodeId,
  newPosition: Position
): NodeMoveResult {
  return {
    doc: moveNode(doc, nodeId, newPosition),
    movedNode: nodeId
  }
}

/**
 * Move multiple nodes by a delta (rigid body translation)
 */
export function moveNodesBy(
  doc: LinkageDocument,
  nodeIds: NodeId[],
  delta: Position
): GroupMoveResult {
  return {
    doc: translateNodes(doc, nodeIds, delta),
    movedNodes: nodeIds
  }
}

/**
 * Move nodes from original positions by a delta (for drag operations)
 *
 * This is used during drag operations where we want to apply a delta
 * from the original drag start positions, not the current positions.
 */
export function moveNodesFromOriginal(
  doc: LinkageDocument,
  originalPositions: Record<NodeId, Position>,
  delta: Position
): GroupMoveResult {
  let result = doc
  const nodeIds = Object.keys(originalPositions)

  for (const nodeId of nodeIds) {
    const originalPos = originalPositions[nodeId]
    if (originalPos) {
      const newPos: Position = [
        originalPos[0] + delta[0],
        originalPos[1] + delta[1]
      ]
      result = moveNode(result, nodeId, newPos)
    }
  }

  return {
    doc: result,
    movedNodes: nodeIds
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MERGE NODES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Merge one node into another
 *
 * All edges from the source node are redirected to the target node,
 * then the source node is deleted. Edges that would become self-loops
 * are removed.
 */
export function mergeNodesOperation(
  doc: LinkageDocument,
  sourceId: NodeId,
  targetId: NodeId
): NodeMergeResult {
  if (sourceId === targetId) {
    return {
      doc,
      sourceNode: sourceId,
      targetNode: targetId,
      deletedEdges: [],
      message: 'Cannot merge node with itself'
    }
  }

  const sourceNode = getNode(doc, sourceId)
  const targetNode = getNode(doc, targetId)

  if (!sourceNode || !targetNode) {
    return {
      doc,
      sourceNode: sourceId,
      targetNode: targetId,
      deletedEdges: [],
      message: 'Node not found'
    }
  }

  // Find edges that will become self-loops (connect source to target)
  const deletedEdges: string[] = []
  for (const edge of getEdgesForNode(doc, sourceId)) {
    const otherId = getOtherNode(edge, sourceId)
    if (otherId === targetId) {
      deletedEdges.push(edge.id)
    }
  }

  const result = mergeNodesHelper(doc, sourceId, targetId)

  return {
    doc: result,
    sourceNode: sourceId,
    targetNode: targetId,
    deletedEdges,
    message: `Merged ${sourceId} into ${targetNode.name || targetId}`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENAME NODE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Rename a node
 */
export function renameNodeOperation(
  doc: LinkageDocument,
  oldId: NodeId,
  newId: NodeId
): NodeRenameResult {
  if (oldId === newId || !newId.trim()) {
    return {
      doc,
      oldName: oldId,
      newName: newId,
      success: false,
      error: 'Invalid name'
    }
  }

  if (getNode(doc, newId)) {
    return {
      doc,
      oldName: oldId,
      newName: newId,
      success: false,
      error: `Node "${newId}" already exists`
    }
  }

  return {
    doc: renameNode(doc, oldId, newId),
    oldName: oldId,
    newName: newId,
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE NODE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Update a node's role
 */
export function setNodeRoleOperation(
  doc: LinkageDocument,
  nodeId: NodeId,
  role: NodeRole
): NodeUpdateResult {
  const node = getNode(doc, nodeId)
  if (!node) {
    return { doc, success: false, message: 'Node not found' }
  }

  return {
    doc: updateNode(doc, nodeId, { role }),
    success: true
  }
}

/**
 * Update a node's metadata (color, zlevel, showPath)
 */
export function updateNodeMetaOperation(
  doc: LinkageDocument,
  nodeId: NodeId,
  meta: Partial<NodeMeta>
): NodeUpdateResult {
  return {
    doc: updateNodeMeta(doc, nodeId, meta),
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CREATE NODE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Generate a unique node ID
 */
export function generateNodeId(doc: LinkageDocument, prefix: string = 'joint'): NodeId {
  const existingIds = getNodeIds(doc)
  let counter = 1
  let id = `${prefix}_${counter}`

  while (existingIds.includes(id)) {
    counter++
    id = `${prefix}_${counter}`
  }

  return id
}

/**
 * Create a new fixed node at a position
 */
export function createFixedNode(
  doc: LinkageDocument,
  position: Position,
  name?: string,
  meta?: Partial<NodeMeta>
): { doc: LinkageDocument; nodeId: NodeId } {
  const nodeId = name || generateNodeId(doc)

  const node: Node = {
    id: nodeId,
    position,
    role: 'fixed',
    jointType: 'revolute',
    name: nodeId
  }

  return {
    doc: addNode(doc, node, { color: '', zlevel: 0, ...meta }),
    nodeId
  }
}

/**
 * Create a new crank node
 */
export function createCrankNode(
  doc: LinkageDocument,
  position: Position,
  angle: number = 0,
  name?: string,
  meta?: Partial<NodeMeta>
): { doc: LinkageDocument; nodeId: NodeId } {
  const nodeId = name || generateNodeId(doc, 'crank')

  const node: Node = {
    id: nodeId,
    position,
    role: 'crank',
    jointType: 'revolute',
    angle,
    name: nodeId
  }

  return {
    doc: addNode(doc, node, { color: '', zlevel: 0, showPath: true, ...meta }),
    nodeId
  }
}

/**
 * Create a new follower node
 */
export function createFollowerNode(
  doc: LinkageDocument,
  position: Position,
  name?: string,
  meta?: Partial<NodeMeta>
): { doc: LinkageDocument; nodeId: NodeId } {
  const nodeId = name || generateNodeId(doc)

  const node: Node = {
    id: nodeId,
    position,
    role: 'follower',
    jointType: 'revolute',
    name: nodeId
  }

  return {
    doc: addNode(doc, node, { color: '', zlevel: 0, showPath: true, ...meta }),
    nodeId
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CAPTURE POSITIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Capture current positions of a group of nodes
 */
export function captureNodePositions(
  doc: LinkageDocument,
  nodeIds: NodeId[]
): Record<NodeId, Position> {
  const positions: Record<NodeId, Position> = {}

  for (const nodeId of nodeIds) {
    const node = getNode(doc, nodeId)
    if (node) {
      positions[nodeId] = [...node.position] as Position
    }
  }

  return positions
}

/**
 * Calculate bounding box of a group of nodes
 */
export function calculateNodeBounds(
  doc: LinkageDocument,
  nodeIds: NodeId[]
): { minX: number; minY: number; maxX: number; maxY: number } | null {
  const positions: Position[] = []

  for (const nodeId of nodeIds) {
    const node = getNode(doc, nodeId)
    if (node) {
      positions.push(node.position)
    }
  }

  if (positions.length === 0) return null

  return {
    minX: Math.min(...positions.map(p => p[0])),
    minY: Math.min(...positions.map(p => p[1])),
    maxX: Math.max(...positions.map(p => p[0])),
    maxY: Math.max(...positions.map(p => p[1]))
  }
}

/**
 * Calculate center point of a group of nodes
 */
export function calculateNodeCenter(
  doc: LinkageDocument,
  nodeIds: NodeId[]
): Position | null {
  const bounds = calculateNodeBounds(doc, nodeIds)
  if (!bounds) return null

  return [
    (bounds.minX + bounds.maxX) / 2,
    (bounds.minY + bounds.maxY) / 2
  ]
}
