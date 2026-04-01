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

// Explore region (circle sampling for trajectory exploration)
export { exploreRegion, getExploreRegionOptionsForMaxPoints, getCombinatorialSecondOptions } from './exploreRegion'
export type { ExploreRegionPoint, ExploreRegionOptions, ExploreRegionOptionsForMaxPoints } from './exploreRegion'

// Drawn objects: edge id references vs linkage.edges
export {
  remapEdgeReferencesInDrawnObjects,
  removeDrawnObjectsReferencingDeletedEdges
} from './drawnObjectsSync'

// Drag-end sync: build synced document after single-node drop (trajectory hop fix)
export { buildSyncedDocAfterDrop } from './dragEndSync'

// Forms vs mechanisms: group-select containment, mechanism equivalence, dissociation
export {
  boxFromCorners,
  isPointInAxisAlignedBox,
  isAxisAlignedRectContainingPolygonVertices,
  areLinkEndpointsInBox,
  isFormFullyInsideGroupSelectBox,
  selectionIsExactlyOneMechanism,
  polygonIdsTouchingMechanismLinks,
  dissociateFormFields,
  validatePolygonFormAssociations,
  getJointPositionFromLinkageDoc,
  DISSOCIATED_FORM_FILL,
  DISSOCIATED_FORM_STROKE
} from './formMechanismHelpers'
export type { AxisAlignedBox, FormForGroupSelect, PolygonFormForDissociate } from './formMechanismHelpers'

// Exploration colormap (angle/radius color for explore node trajectories)
export {
  getExplorationColormapColor,
  positionToAngleAndRadialT,
  EXPLORE_INVALID_GREY,
  DEFAULT_EXPLORATION_RADIAL_MAPPING,
  EXPLORATION_RADIAL_MAPPINGS,
  CENTER_SATURATION,
  EDGE_SATURATION_SCALE
} from './explorationColormap'
export type { ExplorationColormapType, ExplorationRadialMapping } from './explorationColormap'

// Type exports
export type { CreateLinkResult } from './linkageMutations'
