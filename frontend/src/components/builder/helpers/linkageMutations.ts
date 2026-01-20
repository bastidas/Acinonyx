/**
 * Linkage Document Mutations
 *
 * Pure functions that create new LinkageDocument instances with modifications.
 * These replace the legacy setPylinkDoc operations.
 *
 * Design principles:
 * - All functions are pure (no side effects)
 * - Return new document instances (immutable)
 * - Work directly with LinkageDocument structure
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

import {
  getNode,
  getEdge,
  getEdgesForNode,
  getOtherNode,
  calculateDistance
} from './linkageHelpers'

// ═══════════════════════════════════════════════════════════════════════════════
// NODE MUTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/** Add a new node to the document */
export function addNode(
  doc: LinkageDocument,
  node: Node,
  meta?: NodeMeta
): LinkageDocument {
  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      nodes: {
        ...doc.linkage.nodes,
        [node.id]: node
      }
    },
    meta: {
      ...doc.meta,
      nodes: meta ? {
        ...doc.meta.nodes,
        [node.id]: meta
      } : doc.meta.nodes
    }
  }
}

/** Remove a node from the document */
export function removeNode(doc: LinkageDocument, nodeId: NodeId): LinkageDocument {
  const { [nodeId]: _, ...remainingNodes } = doc.linkage.nodes
  const { [nodeId]: __, ...remainingNodeMeta } = doc.meta.nodes

  // Also remove any edges connected to this node
  const edgesToRemove = getEdgesForNode(doc, nodeId).map(e => e.id)
  let newEdges = { ...doc.linkage.edges }
  let newEdgeMeta = { ...doc.meta.edges }

  for (const edgeId of edgesToRemove) {
    delete newEdges[edgeId]
    delete newEdgeMeta[edgeId]
  }

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      nodes: remainingNodes,
      edges: newEdges
    },
    meta: {
      ...doc.meta,
      nodes: remainingNodeMeta,
      edges: newEdgeMeta
    }
  }
}

/** Update a node's position */
export function moveNode(
  doc: LinkageDocument,
  nodeId: NodeId,
  newPosition: Position
): LinkageDocument {
  const node = getNode(doc, nodeId)
  if (!node) return doc

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      nodes: {
        ...doc.linkage.nodes,
        [nodeId]: {
          ...node,
          position: newPosition
        }
      }
    }
  }
}

/** Update a node's properties */
export function updateNode(
  doc: LinkageDocument,
  nodeId: NodeId,
  updates: Partial<Node>
): LinkageDocument {
  const node = getNode(doc, nodeId)
  if (!node) return doc

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      nodes: {
        ...doc.linkage.nodes,
        [nodeId]: {
          ...node,
          ...updates,
          id: nodeId // Ensure ID is not changed
        }
      }
    }
  }
}

/** Update a node's metadata */
export function updateNodeMeta(
  doc: LinkageDocument,
  nodeId: NodeId,
  updates: Partial<NodeMeta>
): LinkageDocument {
  const existingMeta = doc.meta.nodes[nodeId] || {}

  return {
    ...doc,
    meta: {
      ...doc.meta,
      nodes: {
        ...doc.meta.nodes,
        [nodeId]: {
          ...existingMeta,
          ...updates
        }
      }
    }
  }
}

/** Rename a node (updates ID and all references) */
export function renameNode(
  doc: LinkageDocument,
  oldId: NodeId,
  newId: NodeId
): LinkageDocument {
  if (oldId === newId) return doc
  if (doc.linkage.nodes[newId]) return doc // New ID already exists

  const node = getNode(doc, oldId)
  if (!node) return doc

  // Create new node with new ID
  const { [oldId]: _, ...otherNodes } = doc.linkage.nodes
  const newNodes = {
    ...otherNodes,
    [newId]: { ...node, id: newId, name: newId }
  }

  // Update node meta
  const { [oldId]: oldMeta, ...otherNodeMeta } = doc.meta.nodes
  const newNodeMeta = oldMeta ? {
    ...otherNodeMeta,
    [newId]: oldMeta
  } : otherNodeMeta

  // Update edge references
  const newEdges: Record<EdgeId, Edge> = {}
  for (const [edgeId, edge] of Object.entries(doc.linkage.edges)) {
    newEdges[edgeId] = {
      ...edge,
      source: edge.source === oldId ? newId : edge.source,
      target: edge.target === oldId ? newId : edge.target
    }
  }

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      nodes: newNodes,
      edges: newEdges
    },
    meta: {
      ...doc.meta,
      nodes: newNodeMeta
    }
  }
}

/** Change a node's role */
export function setNodeRole(
  doc: LinkageDocument,
  nodeId: NodeId,
  role: NodeRole
): LinkageDocument {
  return updateNode(doc, nodeId, { role })
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE MUTATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/** Add a new edge to the document */
export function addEdge(
  doc: LinkageDocument,
  edge: Edge,
  meta?: EdgeMeta
): LinkageDocument {
  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      edges: {
        ...doc.linkage.edges,
        [edge.id]: edge
      }
    },
    meta: {
      ...doc.meta,
      edges: meta ? {
        ...doc.meta.edges,
        [edge.id]: meta
      } : doc.meta.edges
    }
  }
}

/** Remove an edge from the document */
export function removeEdge(doc: LinkageDocument, edgeId: EdgeId): LinkageDocument {
  const { [edgeId]: _, ...remainingEdges } = doc.linkage.edges
  const { [edgeId]: __, ...remainingEdgeMeta } = doc.meta.edges

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      edges: remainingEdges
    },
    meta: {
      ...doc.meta,
      edges: remainingEdgeMeta
    }
  }
}

/** Update an edge's properties */
export function updateEdge(
  doc: LinkageDocument,
  edgeId: EdgeId,
  updates: Partial<Edge>
): LinkageDocument {
  const edge = getEdge(doc, edgeId)
  if (!edge) return doc

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      edges: {
        ...doc.linkage.edges,
        [edgeId]: {
          ...edge,
          ...updates,
          id: edgeId // Ensure ID is not changed
        }
      }
    }
  }
}

/** Update an edge's metadata */
export function updateEdgeMeta(
  doc: LinkageDocument,
  edgeId: EdgeId,
  updates: Partial<EdgeMeta>
): LinkageDocument {
  const existingMeta = doc.meta.edges[edgeId] || {}

  return {
    ...doc,
    meta: {
      ...doc.meta,
      edges: {
        ...doc.meta.edges,
        [edgeId]: {
          ...existingMeta,
          ...updates
        }
      }
    }
  }
}

/** Rename an edge */
export function renameEdge(
  doc: LinkageDocument,
  oldId: EdgeId,
  newId: EdgeId
): LinkageDocument {
  if (oldId === newId) return doc
  if (doc.linkage.edges[newId]) return doc // New ID already exists

  const edge = getEdge(doc, oldId)
  if (!edge) return doc

  // Create new edge with new ID
  const { [oldId]: _, ...otherEdges } = doc.linkage.edges
  const newEdges = {
    ...otherEdges,
    [newId]: { ...edge, id: newId }
  }

  // Update edge meta
  const { [oldId]: oldMeta, ...otherEdgeMeta } = doc.meta.edges
  const newEdgeMeta = oldMeta ? {
    ...otherEdgeMeta,
    [newId]: oldMeta
  } : otherEdgeMeta

  return {
    ...doc,
    linkage: {
      ...doc.linkage,
      edges: newEdges
    },
    meta: {
      ...doc.meta,
      edges: newEdgeMeta
    }
  }
}

/** Update edge distance from current node positions */
export function syncEdgeDistance(doc: LinkageDocument, edgeId: EdgeId): LinkageDocument {
  const edge = getEdge(doc, edgeId)
  if (!edge) return doc

  const sourcePos = doc.linkage.nodes[edge.source]?.position
  const targetPos = doc.linkage.nodes[edge.target]?.position

  if (!sourcePos || !targetPos) return doc

  const distance = calculateDistance(sourcePos, targetPos)

  return updateEdge(doc, edgeId, { distance })
}

/** Update all edge distances from current node positions */
export function syncAllEdgeDistances(doc: LinkageDocument): LinkageDocument {
  let result = doc

  for (const edgeId of Object.keys(doc.linkage.edges)) {
    result = syncEdgeDistance(result, edgeId)
  }

  return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOUND OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/** Create a link between two nodes (creates edge with computed distance) */
export function createLink(
  doc: LinkageDocument,
  linkId: EdgeId,
  sourceId: NodeId,
  targetId: NodeId,
  meta?: EdgeMeta
): LinkageDocument {
  const sourcePos = doc.linkage.nodes[sourceId]?.position
  const targetPos = doc.linkage.nodes[targetId]?.position

  const distance = sourcePos && targetPos
    ? calculateDistance(sourcePos, targetPos)
    : 0

  const edge: Edge = {
    id: linkId,
    source: sourceId,
    target: targetId,
    distance
  }

  return addEdge(doc, edge, meta)
}

/** Move multiple nodes by a delta (rigid body translation) */
export function translateNodes(
  doc: LinkageDocument,
  nodeIds: NodeId[],
  delta: Position
): LinkageDocument {
  let result = doc

  for (const nodeId of nodeIds) {
    const node = getNode(result, nodeId)
    if (node) {
      const newPos: Position = [
        node.position[0] + delta[0],
        node.position[1] + delta[1]
      ]
      result = moveNode(result, nodeId, newPos)
    }
  }

  return result
}

/** Merge two nodes (redirect all edges from source to target, then delete source) */
export function mergeNodes(
  doc: LinkageDocument,
  sourceId: NodeId,
  targetId: NodeId
): LinkageDocument {
  if (sourceId === targetId) return doc

  const sourceNode = getNode(doc, sourceId)
  const targetNode = getNode(doc, targetId)
  if (!sourceNode || !targetNode) return doc

  let result = doc

  // Update all edges that reference sourceId to reference targetId instead
  for (const edge of getEdgesForNode(doc, sourceId)) {
    const otherId = getOtherNode(edge, sourceId)

    // Skip if this would create a self-loop
    if (otherId === targetId) {
      // Remove the edge entirely
      result = removeEdge(result, edge.id)
      continue
    }

    // Update the edge to point to targetId
    result = updateEdge(result, edge.id, {
      source: edge.source === sourceId ? targetId : edge.source,
      target: edge.target === sourceId ? targetId : edge.target
    })
  }

  // Remove the source node
  result = removeNode(result, sourceId)

  return result
}

/** Batch remove multiple nodes */
export function removeNodes(doc: LinkageDocument, nodeIds: NodeId[]): LinkageDocument {
  let result = doc
  for (const nodeId of nodeIds) {
    result = removeNode(result, nodeId)
  }
  return result
}

/** Batch remove multiple edges */
export function removeEdges(doc: LinkageDocument, edgeIds: EdgeId[]): LinkageDocument {
  let result = doc
  for (const edgeId of edgeIds) {
    result = removeEdge(result, edgeId)
  }
  return result
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOCUMENT-LEVEL OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

/** Update document name */
export function setDocumentName(doc: LinkageDocument, name: string): LinkageDocument {
  return {
    ...doc,
    name,
    linkage: {
      ...doc.linkage,
      name
    }
  }
}

/** Create a deep copy of the document */
export function cloneDocument(doc: LinkageDocument): LinkageDocument {
  return JSON.parse(JSON.stringify(doc))
}

/** Create an empty document */
export function createEmptyDocument(name: string = 'untitled'): LinkageDocument {
  return {
    name,
    version: '2.0.0',
    linkage: {
      name,
      nodes: {},
      edges: {},
      hyperedges: {}
    },
    meta: {
      nodes: {},
      edges: {}
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ID GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/** Generate a unique node ID with a given prefix */
export function generateNodeId(doc: LinkageDocument, prefix: string = 'joint_'): NodeId {
  const existingIds = Object.keys(doc.linkage.nodes)
  let counter = existingIds.length
  let newId = `${prefix}${counter}`
  while (existingIds.includes(newId)) {
    counter++
    newId = `${prefix}${counter}`
  }
  return newId
}

/** Generate a unique edge ID with a given prefix */
export function generateEdgeId(doc: LinkageDocument, prefix: string = 'link_'): EdgeId {
  const existingIds = Object.keys(doc.linkage.edges)
  let counter = existingIds.length
  let newId = `${prefix}${counter}`
  while (existingIds.includes(newId)) {
    counter++
    newId = `${prefix}${counter}`
  }
  return newId
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOUND LINK CREATION
// ═══════════════════════════════════════════════════════════════════════════════

/** Default colors for new edges */
const DEFAULT_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

export function getDefaultEdgeColor(edgeCount: number): string {
  return DEFAULT_COLORS[edgeCount % DEFAULT_COLORS.length]
}

/**
 * Result of link creation operation
 */
export interface CreateLinkResult {
  doc: LinkageDocument
  edgeId: EdgeId
  startNodeId: NodeId
  endNodeId: NodeId
  createdNodes: NodeId[]
  message: string
}

/**
 * Create a link between two points, creating nodes as needed.
 *
 * Logic:
 * - If an existing node is provided, use it
 * - If creating a new node connected to a kinematic (crank/follower) node, make it a follower
 * - Otherwise create a fixed node
 *
 * @param doc - Current document
 * @param startPoint - Position of start
 * @param endPoint - Position of end
 * @param existingStartNodeId - ID of existing node at start (or null to create)
 * @param existingEndNodeId - ID of existing node at end (or null to create)
 * @param getConnectedNodeIds - Function to find nodes connected to a given node via edges
 */
export function createLinkBetweenPoints(
  doc: LinkageDocument,
  startPoint: Position,
  endPoint: Position,
  existingStartNodeId: NodeId | null,
  existingEndNodeId: NodeId | null,
  getConnectedNodeIds: (nodeId: NodeId) => NodeId[]
): CreateLinkResult {
  let currentDoc = doc
  const createdNodes: NodeId[] = []

  let startNodeId = existingStartNodeId
  let endNodeId = existingEndNodeId

  // Create start node if needed
  if (!startNodeId) {
    startNodeId = generateNodeId(currentDoc)

    // Determine role: follower if connected to kinematic, fixed otherwise
    let role: NodeRole = 'fixed'

    if (existingEndNodeId) {
      const endNode = currentDoc.linkage.nodes[existingEndNodeId]
      if (endNode && (endNode.role === 'crank' || endNode.role === 'follower')) {
        // Check if we have a second reference point for constraints
        const connectedToEnd = getConnectedNodeIds(existingEndNodeId)
        if (connectedToEnd.length > 0) {
          role = 'follower'
        }
      }
    }

    const newNode: Node = {
      id: startNodeId,
      position: startPoint,
      role,
      jointType: 'revolute',
      name: startNodeId
    }

    currentDoc = addNode(currentDoc, newNode, {
      color: '',
      zlevel: 0,
      showPath: role === 'follower'
    })
    createdNodes.push(startNodeId)
  }

  // Create end node if needed
  if (!endNodeId) {
    endNodeId = generateNodeId(currentDoc)

    // Determine role: follower if connected to kinematic, fixed otherwise
    let role: NodeRole = 'fixed'

    if (existingStartNodeId) {
      const startNode = currentDoc.linkage.nodes[existingStartNodeId]
      if (startNode && (startNode.role === 'crank' || startNode.role === 'follower')) {
        const connectedToStart = getConnectedNodeIds(existingStartNodeId)
        if (connectedToStart.length > 0) {
          role = 'follower'
        }
      }
    }

    const newNode: Node = {
      id: endNodeId,
      position: endPoint,
      role,
      jointType: 'revolute',
      name: endNodeId
    }

    currentDoc = addNode(currentDoc, newNode, {
      color: '',
      zlevel: 0,
      showPath: role === 'follower'
    })
    createdNodes.push(endNodeId)
  }

  // Handle case where we're connecting a fixed node to a kinematic node
  // The fixed node should become a follower
  if (existingStartNodeId && existingEndNodeId) {
    const startNode = currentDoc.linkage.nodes[startNodeId!]
    const endNode = currentDoc.linkage.nodes[endNodeId!]

    if (startNode && endNode) {
      const startIsFixed = startNode.role === 'fixed'
      const endIsFixed = endNode.role === 'fixed'
      const startIsKinematic = startNode.role === 'crank' || startNode.role === 'follower'
      const endIsKinematic = endNode.role === 'crank' || endNode.role === 'follower'

      // Convert fixed to follower if connecting to kinematic
      if (startIsFixed && endIsKinematic) {
        const connectedToKinematic = getConnectedNodeIds(endNodeId!)
        if (connectedToKinematic.length > 0) {
          currentDoc = setNodeRole(currentDoc, startNodeId!, 'follower')
          currentDoc = updateNodeMeta(currentDoc, startNodeId!, { showPath: true })
        }
      } else if (endIsFixed && startIsKinematic) {
        const connectedToKinematic = getConnectedNodeIds(startNodeId!)
        if (connectedToKinematic.length > 0) {
          currentDoc = setNodeRole(currentDoc, endNodeId!, 'follower')
          currentDoc = updateNodeMeta(currentDoc, endNodeId!, { showPath: true })
        }
      }
    }
  }

  // Create the edge
  const edgeId = generateEdgeId(currentDoc)
  const distance = calculateDistance(startPoint, endPoint)
  const edgeCount = Object.keys(currentDoc.linkage.edges).length

  const newEdge: Edge = {
    id: edgeId,
    source: startNodeId!,
    target: endNodeId!,
    distance
  }

  currentDoc = addEdge(currentDoc, newEdge, {
    color: getDefaultEdgeColor(edgeCount)
  })

  return {
    doc: currentDoc,
    edgeId,
    startNodeId: startNodeId!,
    endNodeId: endNodeId!,
    createdNodes,
    message: `Created ${edgeId} (${distance.toFixed(1)} units)`
  }
}

/**
 * Change a node's role with proper constraint handling
 *
 * @param doc - Current document
 * @param nodeId - Node to update
 * @param newRole - New role for the node
 * @param getNodePosition - Function to get a node's current position
 */
export function changeNodeRole(
  doc: LinkageDocument,
  nodeId: NodeId,
  newRole: NodeRole,
  getNodePosition: (id: NodeId) => Position | null
): { doc: LinkageDocument; success: boolean; error?: string } {
  const node = getNode(doc, nodeId)
  if (!node) {
    return { doc, success: false, error: `Node ${nodeId} not found` }
  }

  const currentPos = getNodePosition(nodeId)
  if (!currentPos) {
    return { doc, success: false, error: `Cannot determine position for ${nodeId}` }
  }

  let currentDoc = doc

  // Update the node's role
  currentDoc = updateNode(currentDoc, nodeId, {
    role: newRole,
    position: currentPos
  })

  // If changing to crank, we need an angle
  if (newRole === 'crank') {
    // Find the connected fixed node to calculate angle
    const connectedEdges = getEdgesForNode(currentDoc, nodeId)
    let anchorNode: Node | null = null

    for (const edge of connectedEdges) {
      const otherId = getOtherNode(edge, nodeId)
      const otherNode = currentDoc.linkage.nodes[otherId]
      if (otherNode?.role === 'fixed') {
        anchorNode = otherNode
        break
      }
    }

    if (!anchorNode) {
      return { doc, success: false, error: 'Crank requires a connected fixed (anchor) node' }
    }

    // Calculate angle from anchor to current position
    const angle = Math.atan2(
      currentPos[1] - anchorNode.position[1],
      currentPos[0] - anchorNode.position[0]
    )

    currentDoc = updateNode(currentDoc, nodeId, {
      angle,
      initialAngle: angle
    })
  }

  // Update metadata for path display
  if (newRole === 'fixed') {
    currentDoc = updateNodeMeta(currentDoc, nodeId, { showPath: false })
  } else {
    currentDoc = updateNodeMeta(currentDoc, nodeId, { showPath: true })
  }

  // Sync all edge distances based on current positions
  currentDoc = syncAllEdgeDistances(currentDoc)

  return { doc: currentDoc, success: true }
}
