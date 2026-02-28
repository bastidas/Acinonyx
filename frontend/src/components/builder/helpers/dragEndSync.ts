/**
 * dragEndSync
 *
 * Pure function to build a synced LinkageDocument after a single-node drag ends.
 * Used by the TRAJECTORY HOP FIX: sync doc to edited frame (1/N) with dropped joint
 * at user position, then run sim with that doc so the backend gets consistent state.
 */

import type { LinkageDocument } from '../types'

export type Position = [number, number]

/**
 * Builds a new LinkageDocument with:
 * - All joint positions from trajectory[frame] (edited frame, typically 0)
 * - The dragged joint at dropPosition
 * - Crank/driven node angles recomputed from parent position (backend uses node.angle;
 *   if we only sync position, crank stays at wrong angle and the mechanism jumps).
 *
 * Caller must then setLinkageDoc(syncedDoc) and runSimulation(syncedDoc).
 */
export function buildSyncedDocAfterDrop(
  linkageDoc: LinkageDocument,
  trajectories: Record<string, Position[]>,
  draggedJoint: string,
  dropPosition: Position,
  frame: number
): LinkageDocument {
  const linkage = linkageDoc.linkage
  const nodes = { ...linkage.nodes }

  for (const [jointName, points] of Object.entries(trajectories)) {
    if (points && Array.isArray(points) && points[frame] != null && nodes[jointName]) {
      const pos = points[frame] as Position
      nodes[jointName] = { ...nodes[jointName], position: [pos[0], pos[1]] }
    }
  }
  nodes[draggedJoint] = { ...nodes[draggedJoint], position: [dropPosition[0], dropPosition[1]] }

  const edges = linkage.edges ?? {}
  for (const nodeId of Object.keys(nodes)) {
    const node = nodes[nodeId]
    if (node?.role === 'crank' || node?.role === 'driven') {
      let parentId: string | null = null
      for (const edge of Object.values(edges)) {
        if (edge.source !== nodeId && edge.target !== nodeId) continue
        const otherId = edge.source === nodeId ? edge.target : edge.source
        if (nodes[otherId]?.role === 'fixed') {
          parentId = otherId
          break
        }
      }
      if (parentId != null) {
        const parentPos = nodes[parentId]?.position
        const crankPos = nodes[nodeId].position
        if (Array.isArray(parentPos) && Array.isArray(crankPos)) {
          const angle = Math.atan2(
            crankPos[1] - parentPos[1],
            crankPos[0] - parentPos[0]
          )
          nodes[nodeId] = { ...node, angle }
        }
      }
    }
  }

  return { ...linkageDoc, linkage: { ...linkage, nodes } }
}
