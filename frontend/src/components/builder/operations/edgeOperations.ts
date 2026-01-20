/**
 * Edge Operations (Hypergraph Format)
 *
 * Pure functions for creating, updating, deleting, and manipulating edges.
 * Works directly with LinkageDocument (hypergraph format).
 *
 * These replace the legacy linkOperations.ts functions.
 */

import type { LinkageDocument, Edge, EdgeId, NodeId, EdgeMeta } from '../../../types'
import {
  getEdge,
  getEdges,
  getEdgeIds,
  getEdgesForNode,
  getNode,
  getNodes,
  calculateDistance
} from '../helpers/linkageHelpers'
import {
  removeEdge,
  updateEdgeMeta,
  renameEdge,
  createLink,
  syncEdgeDistance
} from '../helpers/linkageMutations'

// Default colors for new edges (D3 category10 palette)
const DEFAULT_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

function getDefaultEdgeColor(index: number): string {
  return DEFAULT_COLORS[index % DEFAULT_COLORS.length]
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface EdgeDeletionResult {
  doc: LinkageDocument
  deletedEdges: EdgeId[]
  orphanedNodes: NodeId[]
  message: string
}

export interface EdgeCreationResult {
  doc: LinkageDocument
  edgeId: EdgeId
  sourceNode: NodeId
  targetNode: NodeId
  message: string
}

export interface EdgeRenameResult {
  doc: LinkageDocument
  oldName: EdgeId
  newName: EdgeId
  success: boolean
  error?: string
}

export interface EdgeUpdateResult {
  doc: LinkageDocument
  success: boolean
  message?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// FIND ORPHANS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Find nodes that would become orphaned after deleting edges
 *
 * A node is orphaned if:
 * - It has no remaining edges after deletion
 * - It's a follower node (fixed and crank nodes are never orphaned)
 */
function findOrphanedNodesAfterEdgeDeletion(
  doc: LinkageDocument,
  edgesToDelete: Set<EdgeId>
): NodeId[] {
  const orphans: NodeId[] = []

  for (const node of getNodes(doc)) {
    // Fixed and crank nodes are never orphaned
    if (node.role === 'fixed' || node.role === 'crank') continue

    // Check if node has any remaining connections
    const edges = getEdgesForNode(doc, node.id)
    const remainingEdges = edges.filter(e => !edgesToDelete.has(e.id))

    if (remainingEdges.length === 0) {
      orphans.push(node.id)
    }
  }

  return orphans
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE EDGE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Delete an edge and any resulting orphan nodes
 */
export function deleteEdge(
  doc: LinkageDocument,
  edgeId: EdgeId
): EdgeDeletionResult {
  if (!getEdge(doc, edgeId)) {
    return {
      doc,
      deletedEdges: [],
      orphanedNodes: [],
      message: `Edge "${edgeId}" not found`
    }
  }

  // Find orphans before deletion
  const edgesToDelete = new Set([edgeId])
  const orphans = findOrphanedNodesAfterEdgeDeletion(doc, edgesToDelete)

  // Remove the edge
  let result = removeEdge(doc, edgeId)

  // Remove orphan nodes and their edges
  for (const orphanId of orphans) {
    // Remove any remaining edges connected to orphan
    for (const edge of getEdgesForNode(result, orphanId)) {
      result = removeEdge(result, edge.id)
    }
    // Remove the orphan node
    const { [orphanId]: _, ...remainingNodes } = result.linkage.nodes
    const { [orphanId]: __, ...remainingNodeMeta } = result.meta.nodes
    result = {
      ...result,
      linkage: { ...result.linkage, nodes: remainingNodes },
      meta: { ...result.meta, nodes: remainingNodeMeta }
    }
  }

  const orphanMsg = orphans.length > 0
    ? ` + ${orphans.length} orphan(s)`
    : ''

  return {
    doc: result,
    deletedEdges: [edgeId],
    orphanedNodes: orphans,
    message: `Deleted ${edgeId}${orphanMsg}`
  }
}

/**
 * Delete multiple edges at once
 */
export function deleteEdges(
  doc: LinkageDocument,
  edgeIds: EdgeId[]
): EdgeDeletionResult {
  let result = doc
  const allDeletedEdges = new Set<EdgeId>()
  const allOrphanedNodes = new Set<NodeId>()

  for (const edgeId of edgeIds) {
    if (allDeletedEdges.has(edgeId)) continue
    if (!getEdge(result, edgeId)) continue

    const deletion = deleteEdge(result, edgeId)
    result = deletion.doc

    allDeletedEdges.add(edgeId)
    for (const orphan of deletion.orphanedNodes) {
      allOrphanedNodes.add(orphan)
    }
  }

  const orphanMsg = allOrphanedNodes.size > 0
    ? ` + ${allOrphanedNodes.size} orphan(s)`
    : ''

  return {
    doc: result,
    deletedEdges: Array.from(allDeletedEdges),
    orphanedNodes: Array.from(allOrphanedNodes),
    message: `Deleted ${allDeletedEdges.size} edge(s)${orphanMsg}`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CREATE EDGE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Generate a unique edge ID
 */
export function generateEdgeId(doc: LinkageDocument, prefix: string = 'link'): EdgeId {
  const existingIds = getEdgeIds(doc)
  let counter = 1
  let id = `${prefix}_${counter}`

  while (existingIds.includes(id)) {
    counter++
    id = `${prefix}_${counter}`
  }

  return id
}

/**
 * Create a new edge between two existing nodes
 */
export function createEdge(
  doc: LinkageDocument,
  sourceId: NodeId,
  targetId: NodeId,
  name?: string,
  meta?: Partial<EdgeMeta>
): EdgeCreationResult {
  // Validate nodes exist
  const sourceNode = getNode(doc, sourceId)
  const targetNode = getNode(doc, targetId)

  if (!sourceNode || !targetNode) {
    return {
      doc,
      edgeId: '',
      sourceNode: sourceId,
      targetNode: targetId,
      message: 'Source or target node not found'
    }
  }

  // Check for duplicate edge
  const existingEdge = getEdges(doc).find(e =>
    (e.source === sourceId && e.target === targetId) ||
    (e.source === targetId && e.target === sourceId)
  )

  if (existingEdge) {
    return {
      doc,
      edgeId: existingEdge.id,
      sourceNode: sourceId,
      targetNode: targetId,
      message: `Edge already exists: ${existingEdge.id}`
    }
  }

  const edgeId = name || generateEdgeId(doc)
  const edgeCount = getEdgeIds(doc).length

  const result = createLink(doc, edgeId, sourceId, targetId, {
    color: getDefaultEdgeColor(edgeCount),
    ...meta
  })

  const distance = calculateDistance(sourceNode.position, targetNode.position)

  return {
    doc: result,
    edgeId,
    sourceNode: sourceId,
    targetNode: targetId,
    message: `Created ${edgeId} (${distance.toFixed(1)} units)`
  }
}

/**
 * Create an edge marked as ground
 */
export function createGroundEdge(
  doc: LinkageDocument,
  sourceId: NodeId,
  targetId: NodeId,
  name?: string
): EdgeCreationResult {
  return createEdge(doc, sourceId, targetId, name || 'ground', {
    color: '#7f7f7f',
    isGround: true
  })
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENAME EDGE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Rename an edge
 */
export function renameEdgeOperation(
  doc: LinkageDocument,
  oldId: EdgeId,
  newId: EdgeId
): EdgeRenameResult {
  if (oldId === newId || !newId.trim()) {
    return {
      doc,
      oldName: oldId,
      newName: newId,
      success: false,
      error: 'Invalid name'
    }
  }

  if (getEdge(doc, newId)) {
    return {
      doc,
      oldName: oldId,
      newName: newId,
      success: false,
      error: `Edge "${newId}" already exists`
    }
  }

  return {
    doc: renameEdge(doc, oldId, newId),
    oldName: oldId,
    newName: newId,
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE EDGE
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Update an edge's metadata (color, isGround)
 */
export function updateEdgeMetaOperation(
  doc: LinkageDocument,
  edgeId: EdgeId,
  meta: Partial<EdgeMeta>
): EdgeUpdateResult {
  if (!getEdge(doc, edgeId)) {
    return { doc, success: false, message: 'Edge not found' }
  }

  return {
    doc: updateEdgeMeta(doc, edgeId, meta),
    success: true
  }
}

/**
 * Set edge color
 */
export function setEdgeColor(
  doc: LinkageDocument,
  edgeId: EdgeId,
  color: string
): EdgeUpdateResult {
  return updateEdgeMetaOperation(doc, edgeId, { color })
}

/**
 * Set edge as ground
 */
export function setEdgeGround(
  doc: LinkageDocument,
  edgeId: EdgeId,
  isGround: boolean
): EdgeUpdateResult {
  return updateEdgeMetaOperation(doc, edgeId, { isGround })
}

/**
 * Sync edge distance from current node positions
 */
export function syncEdgeDistanceOperation(
  doc: LinkageDocument,
  edgeId: EdgeId
): EdgeUpdateResult {
  if (!getEdge(doc, edgeId)) {
    return { doc, success: false, message: 'Edge not found' }
  }

  return {
    doc: syncEdgeDistance(doc, edgeId),
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUERIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Find edges that connect nodes in a given set
 * (both endpoints must be in the set)
 */
export function findEdgesWithinGroup(
  doc: LinkageDocument,
  nodeIds: Set<NodeId>
): Edge[] {
  return getEdges(doc).filter(edge =>
    nodeIds.has(edge.source) && nodeIds.has(edge.target)
  )
}

/**
 * Find edges that connect a node in the set to a node outside the set
 */
export function findEdgesCrossingGroup(
  doc: LinkageDocument,
  nodeIds: Set<NodeId>
): Edge[] {
  return getEdges(doc).filter(edge =>
    (nodeIds.has(edge.source) && !nodeIds.has(edge.target)) ||
    (!nodeIds.has(edge.source) && nodeIds.has(edge.target))
  )
}

/**
 * Get edges connected to any node in a list
 */
export function findEdgesConnectedToNodes(
  doc: LinkageDocument,
  nodeIds: NodeId[]
): Edge[] {
  const nodeSet = new Set(nodeIds)
  return getEdges(doc).filter(edge =>
    nodeSet.has(edge.source) || nodeSet.has(edge.target)
  )
}
