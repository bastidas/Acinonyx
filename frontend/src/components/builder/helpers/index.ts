/**
 * Builder Helpers Module
 *
 * Pure utility functions for working with LinkageDocument.
 * These provide a clean API for reading and modifying linkage documents
 * without the need for legacy format conversion.
 */

// Accessor functions (read-only)
export {
  // Node accessors
  getNode,
  getNodes,
  getNodeIds,
  hasNode,
  getNodeMeta,
  getNodePosition,
  getNodesByRole,
  getFixedNodes,
  getCrankNodes,
  getFollowerNodes,
  hasCrank,

  // Edge accessors
  getEdge,
  getEdges,
  getEdgeIds,
  hasEdge,
  getEdgeMeta,
  getEdgesForNode,
  getEdgeIdsForNode,
  getOtherNode,
  getGroundEdges,

  // Connectivity
  buildAdjacencyMap,
  getConnectedNodes,
  findConnectedComponent,
  findEdgeBetween,

  // Geometry
  calculateDistance,
  calculateNodeDistance,
  getEdgeLength,
  getEdgeMidpoint,

  // Validation
  isNodeConstrained,
  getUnconstrainedNodes,
  isValidForSimulation,

  // Document queries
  getDocumentStats,
  isDocumentEmpty
} from './linkageHelpers'

// Mutation functions (create new documents)
export {
  // Node mutations
  addNode,
  removeNode,
  moveNode,
  updateNode,
  updateNodeMeta,
  renameNode,
  setNodeRole,

  // Edge mutations
  addEdge,
  removeEdge,
  updateEdge,
  updateEdgeMeta,
  renameEdge,
  syncEdgeDistance,
  syncAllEdgeDistances,

  // Compound operations
  createLink,
  translateNodes,
  mergeNodes,
  removeNodes,
  removeEdges,

  // Document operations
  setDocumentName,
  cloneDocument,
  createEmptyDocument,

  // ID generation
  generateNodeId,
  generateEdgeId,

  // Link creation with automatic node handling
  createLinkBetweenPoints,
  getDefaultEdgeColor,

  // Role change with constraint handling
  changeNodeRole
} from './linkageMutations'

// Apply loaded document (file load / demo load)
export { applyLoadedDocument } from './applyLoadedDocument'
export type { ApplyLoadedDocumentParams } from './applyLoadedDocument'

// Optimizer sync status (pure comparison)
export { computeOptimizerSyncStatus } from './optimizerSyncStatus'
export type { OptimizerSyncResult } from './optimizerSyncStatus'

// Type exports
export type { CreateLinkResult } from './linkageMutations'
