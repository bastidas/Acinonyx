/**
 * Pure comparison of current document vs last optimizer result.
 * Used by BuilderTab to decide synced vs unsynced vs structural mismatch (force re-apply).
 */

import type { LinkageDocument } from '../types'

export type OptimizerSyncResult =
  | { kind: 'no_check'; reason: 'no_optimizer_result' | 'not_optimized' }
  | { kind: 'synced' }
  | { kind: 'structural_mismatch'; structuralMismatches: string[]; optimizerDoc: LinkageDocument }
  | { kind: 'value_mismatch'; valueMismatches: string[] }

/** Edge distance comparison tolerance (same as original effect). */
const DISTANCE_TOLERANCE = 1e-6
/** Node position comparison tolerance (same as original effect). */
const POSITION_TOLERANCE = 1e-3

/**
 * Compares optimizer document to current document. Returns a result that the
 * effect can use to set synced state or force re-apply on structural mismatch.
 */
export function computeOptimizerSyncStatus(
  optimizerDoc: LinkageDocument | null,
  currentDoc: LinkageDocument,
  isOptimizedMechanism: boolean
): OptimizerSyncResult {
  if (!optimizerDoc) {
    return { kind: 'no_check', reason: 'no_optimizer_result' }
  }
  if (!isOptimizedMechanism) {
    return { kind: 'no_check', reason: 'not_optimized' }
  }

  const optimizerEdges = optimizerDoc.linkage?.edges || {}
  const currentEdges = currentDoc.linkage?.edges || {}
  const optimizerNodes = optimizerDoc.linkage?.nodes || {}
  const currentNodes = currentDoc.linkage?.nodes || {}

  const structuralMismatches: string[] = []
  const valueMismatches: string[] = []

  for (const [edgeId, optimizerEdge] of Object.entries(optimizerEdges)) {
    const currentEdge = currentEdges[edgeId]
    if (!currentEdge) {
      structuralMismatches.push(`Edge ${edgeId} missing in current doc`)
      continue
    }

    const optimizerDist = (optimizerEdge as { distance?: number }).distance
    const currentDist = (currentEdge as { distance?: number }).distance

    if (optimizerDist !== undefined && currentDist !== undefined) {
      const diff = Math.abs(optimizerDist - currentDist)
      if (diff > DISTANCE_TOLERANCE) {
        valueMismatches.push(`Edge ${edgeId}: optimizer=${optimizerDist}, current=${currentDist}, diff=${diff}`)
      }
    } else if (optimizerDist !== currentDist) {
      valueMismatches.push(`Edge ${edgeId}: optimizer=${optimizerDist}, current=${currentDist} (one undefined)`)
    }
  }

  for (const [nodeId, optimizerNode] of Object.entries(optimizerNodes)) {
    const currentNode = currentNodes[nodeId]
    if (!currentNode) {
      structuralMismatches.push(`Node ${nodeId} missing in current doc`)
      continue
    }

    const optimizerPos = optimizerNode.position
    const currentPos = currentNode.position

    if (optimizerPos && currentPos && optimizerPos.length >= 2 && currentPos.length >= 2) {
      const dx = Math.abs(optimizerPos[0] - currentPos[0])
      const dy = Math.abs(optimizerPos[1] - currentPos[1])
      if (dx > POSITION_TOLERANCE || dy > POSITION_TOLERANCE) {
        valueMismatches.push(`Node ${nodeId}: optimizer=[${optimizerPos[0]}, ${optimizerPos[1]}], current=[${currentPos[0]}, ${currentPos[1]}]`)
      }
    } else if (optimizerPos !== currentPos) {
      valueMismatches.push(`Node ${nodeId}: position mismatch (one undefined)`)
    }
  }

  if (structuralMismatches.length > 0) {
    return { kind: 'structural_mismatch', structuralMismatches, optimizerDoc }
  }
  if (valueMismatches.length > 0) {
    return { kind: 'value_mismatch', valueMismatches }
  }
  return { kind: 'synced' }
}
