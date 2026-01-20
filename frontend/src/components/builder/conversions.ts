/**
 * Conversion utilities between Legacy (PylinkDocument) and Hypergraph (LinkageDocument) formats
 *
 * Used during the migration period and for backend communication until backend is migrated.
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
} from '../../types'

import type {
  PylinkDocument,
  PylinkJoint,
  JointMeta,
  LinkMeta
} from './types'

/**
 * Convert legacy PylinkDocument to new LinkageDocument format.
 * Use this when loading old saved files or receiving data from legacy backend.
 */
export function convertLegacyToLinkageDocument(legacy: PylinkDocument): LinkageDocument {
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
      position = meta?.x !== undefined && meta?.y !== undefined
        ? [meta.x, meta.y]
        : [0, 0]
      role = 'crank'
      angle = joint.angle
    } else {
      position = meta?.x !== undefined && meta?.y !== undefined
        ? [meta.x, meta.y]
        : [0, 0]
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
  for (const [linkName, linkMeta] of Object.entries(legacy.meta.links)) {
    const [sourceId, targetId] = linkMeta.connects

    // Compute distance from positions
    const sourceNode = nodes[sourceId]
    const targetNode = nodes[targetId]
    const distance = sourceNode && targetNode
      ? Math.sqrt(
          Math.pow(targetNode.position[0] - sourceNode.position[0], 2) +
          Math.pow(targetNode.position[1] - sourceNode.position[1], 2)
        )
      : 0

    edges[linkName] = {
      id: linkName,
      source: sourceId,
      target: targetId,
      distance
    }

    edgeMeta[linkName] = {
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
 * Convert new LinkageDocument to legacy PylinkDocument format.
 * Use this for backend communication until backend is migrated.
 */
export function convertLinkageDocumentToLegacy(doc: LinkageDocument): PylinkDocument {
  const joints: PylinkJoint[] = []
  const jointsMeta: Record<string, JointMeta> = {}
  const linksMeta: Record<string, LinkMeta> = {}

  // Build adjacency info
  const nodeEdges: Record<NodeId, EdgeId[]> = {}
  for (const edge of Object.values(doc.linkage.edges)) {
    if (!nodeEdges[edge.source]) nodeEdges[edge.source] = []
    if (!nodeEdges[edge.target]) nodeEdges[edge.target] = []
    nodeEdges[edge.source].push(edge.id)
    nodeEdges[edge.target].push(edge.id)
  }

  // Sort nodes by role
  const sortedNodes = Object.values(doc.linkage.nodes).sort((a, b) => {
    const roleOrder: Record<NodeRole, number> = { fixed: 0, crank: 1, driven: 1, follower: 2 }
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
      const connectedEdges = nodeEdges[node.id] || []
      let joint0 = ''
      let joint1 = ''
      let distance0 = 0
      let distance1 = 0

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

/**
 * Create an empty LinkageDocument with default values
 */
export function createEmptyLinkageDocument(name: string = 'untitled'): LinkageDocument {
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
