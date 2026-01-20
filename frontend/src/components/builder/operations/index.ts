/**
 * Builder Operations Module
 *
 * Pure functions for manipulating the linkage document.
 * These functions take the current state and return the new state,
 * making them easy to test and compose.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * MIGRATION GUIDE
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * NEW (Hypergraph Format - LinkageDocument):
 * - nodeOperations.ts - Use these for new code
 * - edgeOperations.ts - Use these for new code
 *
 * DEPRECATED (Legacy Format - PylinkDocument):
 * - jointOperations.ts - Will be removed
 * - linkOperations.ts - Will be removed
 * - groupOperations.ts - Partially deprecated
 *
 * The new operations work directly with LinkageDocument and use the helpers
 * from builder/helpers/ for cleaner, more testable code.
 */

// Shared types
export * from './types'

// ═══════════════════════════════════════════════════════════════════════════════
// NEW HYPERGRAPH OPERATIONS (use these)
// ═══════════════════════════════════════════════════════════════════════════════

// Node Operations (replaces jointOperations)
export {
  // Delete
  deleteNode,
  deleteNodes,

  // Move
  moveNodeTo,
  moveNodesBy,
  moveNodesFromOriginal,

  // Merge
  mergeNodesOperation,

  // Rename
  renameNodeOperation,

  // Update
  setNodeRoleOperation,
  updateNodeMetaOperation,

  // Create
  generateNodeId,
  createFixedNode,
  createCrankNode,
  createFollowerNode,

  // Capture
  captureNodePositions,
  calculateNodeBounds,
  calculateNodeCenter
} from './nodeOperations'

export type {
  NodeDeletionResult,
  NodeMoveResult,
  GroupMoveResult,
  NodeMergeResult,
  NodeRenameResult,
  NodeUpdateResult
} from './nodeOperations'

// Edge Operations (replaces linkOperations)
export {
  // Delete
  deleteEdge,
  deleteEdges,

  // Create
  generateEdgeId,
  createEdge,
  createGroundEdge,

  // Rename
  renameEdgeOperation,

  // Update
  updateEdgeMetaOperation,
  setEdgeColor,
  setEdgeGround,
  syncEdgeDistanceOperation,

  // Queries
  findEdgesWithinGroup,
  findEdgesCrossingGroup,
  findEdgesConnectedToNodes
} from './edgeOperations'

export type {
  EdgeDeletionResult,
  EdgeCreationResult,
  EdgeRenameResult,
  EdgeUpdateResult
} from './edgeOperations'

// ═══════════════════════════════════════════════════════════════════════════════
// DEPRECATED LEGACY OPERATIONS (will be removed)
// Use the new hypergraph operations above instead
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @deprecated Use deleteNode from nodeOperations instead
 */
export { deleteJoint } from './jointOperations'

/**
 * @deprecated Use moveNodeTo or moveNodesBy from nodeOperations instead
 */
export { moveJoint, translateGroupRigid } from './jointOperations'

/**
 * @deprecated Use mergeNodesOperation from nodeOperations instead
 */
export { mergeJoints } from './jointOperations'

/**
 * @deprecated Use renameNodeOperation from nodeOperations instead
 */
export { renameJoint } from './jointOperations'

/**
 * @deprecated Use setNodeRoleOperation from nodeOperations instead
 */
export { updateJointType, updateJointMeta } from './jointOperations'

/**
 * @deprecated Use deleteEdge from edgeOperations instead
 */
export { deleteLink, deleteLinks } from './linkOperations'

/**
 * @deprecated Use createEdge from edgeOperations instead
 */
export { createLinkWithStaticJoints, createLinkWithRevoluteDefault } from './linkOperations'

/**
 * @deprecated Use renameEdgeOperation from edgeOperations instead
 */
export { renameLink } from './linkOperations'

/**
 * @deprecated Use updateEdgeMetaOperation from edgeOperations instead
 */
export { updateLinkProperty } from './linkOperations'

// Group Operations (some still useful, some deprecated)
export {
  batchDeleteItems,
  captureGroupPositions,
  calculateGroupBounds,
  calculateGroupCenter,
  moveGroup,
  findConnectedJoints,
  findLinksInGroup,
  findLinksConnectedToGroup,
  findJointsInBox,
  findLinksInBox,
  duplicateGroup
} from './groupOperations'
export type { BatchDeletionResult, DuplicateGroupResult } from './groupOperations'
