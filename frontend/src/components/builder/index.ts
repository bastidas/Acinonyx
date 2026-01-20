/**
 * Builder component module
 *
 * This barrel file exports all builder-related types, constants, and components.
 *
 * Note: Some types are defined in multiple places for different contexts.
 * This file exports the canonical versions and avoids re-exporting duplicates.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CORE TYPES (from types.ts - the canonical source)
// ═══════════════════════════════════════════════════════════════════════════════
export type {
  // Legacy joint types
  JointRef,
  StaticJoint,
  CrankJoint,
  RevoluteJoint,
  PylinkJoint,
  PylinkageData,
  JointMeta,
  LinkMeta,
  UIMeta,
  PylinkDocument,
  // UI types
  AnimatedPositions,
  HighlightType,
  ObjectType,
  HighlightStyle
} from './types'

// Re-export core linkage types for convenience
export type {
  LinkageDocument,
  Node,
  Edge,
  NodeId,
  EdgeId,
  Position,
  NodeRole,
  NodeMeta,
  EdgeMeta,
  HypergraphLinkage
} from './types'

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════
export * from './constants'

// ═══════════════════════════════════════════════════════════════════════════════
// CONVERSION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════
export {
  convertLegacyToLinkageDocument,
  convertLinkageDocumentToLegacy,
  createEmptyLinkageDocument
} from './conversions'

// ═══════════════════════════════════════════════════════════════════════════════
// TOOLBAR COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════
export {
  ToolsToolbar,
  LinksToolbar,
  NodesToolbar,
  MoreToolbar,
  SettingsToolbar,
  OptimizationToolbar
} from './toolbars'

// Toolbar prop types
export type {
  ToolsToolbarProps,
  LinksToolbarProps,
  NodesToolbarProps,
  MoreToolbarProps,
  SettingsToolbarProps,
  OptimizationToolbarProps,
  // Toolbar-specific types
  CanvasBgColor,
  TrajectoryStyle,
  SelectionHighlightColor,
  OptMethod,
  SmoothMethod,
  ResampleMethod
} from './toolbars'

// ═══════════════════════════════════════════════════════════════════════════════
// RENDERING UTILITIES AND COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════
export {
  // SVG Filters
  SVGFilters,
  // Renderers
  GridRenderer,
  renderGrid,
  JointsRenderer,
  renderJoints,
  LinksRenderer,
  renderLinks,
  TrajectoriesRenderer,
  renderTrajectories,
  DrawnObjectsRenderer,
  renderDrawnObjects,
  TargetPathsRenderer,
  renderTargetPaths,
  // Previews
  PreviewLineRenderer,
  renderPreviewLine,
  SelectionBoxRenderer,
  renderSelectionBox,
  PolygonPreviewRenderer,
  renderPolygonPreview,
  PathPreviewRenderer,
  renderPathPreview,
  // Measurements
  MeasurementMarkersRenderer,
  renderMeasurementMarkers,
  MeasurementLineRenderer,
  renderMeasurementLine,
  // Utilities
  createCoordinateConverters,
  getGlowFilterForColor,
  getHighlightStyle,
  getMidpoint,
  getDistance,
  hasMovement,
  isValidPosition,
  generatePathData
} from './rendering'

// Rendering type exports
export type {
  CanvasDimensions,
  JointColors,
  JointRenderData,
  JointsRendererProps,
  LinkRenderData,
  LinksRendererProps,
  TrajectoryRenderData,
  TrajectoriesRendererProps,
  GridRendererProps,
  PreviewLine,
  PreviewLineRendererProps,
  SelectionBoxRendererProps,
  PolygonPreviewRendererProps,
  PathPreviewRendererProps,
  DrawnObject,
  DrawnObjectsRendererProps,
  TargetPathsRendererProps,
  MeasurementMarker,
  MeasurementMarkersRendererProps,
  MeasurementLineRendererProps,
  ColorCycleType
} from './rendering'

// ═══════════════════════════════════════════════════════════════════════════════
// OPERATIONS (pure functions for state manipulation)
// ═══════════════════════════════════════════════════════════════════════════════
export {
  // Joint operations
  deleteJoint,
  moveJoint,
  translateGroupRigid,
  mergeJoints,
  renameJoint,
  updateJointType,
  updateJointMeta,
  // Link operations
  deleteLink,
  deleteLinks,
  createLinkWithStaticJoints,
  createLinkWithRevoluteDefault,
  renameLink,
  updateLinkProperty,
  // Group operations
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
} from './operations'

// Operation type exports
export type {
  JointDeletionResult,
  LinkDeletionResult,
  JointMoveResult,
  GroupTranslateResult,
  JointMergeResult,
  LinkCreationResult,
  RenameResult,
  UpdatePropertyResult,
  GetJointPositionFn,
  CalculateDistanceFn,
  FindConnectedJointsFn,
  BatchDeletionResult,
  DuplicateGroupResult
} from './operations'

// ═══════════════════════════════════════════════════════════════════════════════
// CUSTOM HOOKS
// ═══════════════════════════════════════════════════════════════════════════════
export {
  useKeyboardShortcuts,
  useCanvasInteraction,
  useStatusMessage
} from './hooks'

// Hook type exports
export type {
  KeyboardShortcutsConfig,
  UseCanvasInteractionConfig,
  UseCanvasInteractionReturn,
  CanvasCoordinates,
  JointWithPosition,
  LinkWithPosition,
  NearestResult,
  StatusType,
  StatusMessage,
  UseStatusMessageReturn
} from './hooks'

// ═══════════════════════════════════════════════════════════════════════════════
// LINKAGE DOCUMENT HELPERS (pure functions for hypergraph format)
// ═══════════════════════════════════════════════════════════════════════════════
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

  // Geometry helpers
  calculateDistance as calculateDistanceHelper,
  calculateNodeDistance,
  getEdgeLength,
  getEdgeMidpoint,

  // Validation
  isNodeConstrained,
  getUnconstrainedNodes,
  isValidForSimulation,

  // Document queries
  getDocumentStats,
  isDocumentEmpty,

  // Node mutations
  addNode,
  removeNode,
  moveNode,
  updateNode,
  updateNodeMeta,
  renameNode as renameNodeHelper,
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
  createLink as createLinkHelper,
  translateNodes,
  mergeNodes as mergeNodesHelper,
  removeNodes,
  removeEdges,

  // Document operations
  setDocumentName,
  cloneDocument,
  createEmptyDocument,

  // ID generation
  generateNodeId as generateNodeIdHelper,
  generateEdgeId as generateEdgeIdHelper,

  // Link creation with automatic node handling
  createLinkBetweenPoints,
  getDefaultEdgeColor,

  // Role change with constraint handling
  changeNodeRole
} from './helpers'

export type { CreateLinkResult } from './helpers'

// Re-export new hypergraph operations
export {
  // Node operations
  deleteNode,
  deleteNodes,
  moveNodeTo,
  moveNodesBy,
  moveNodesFromOriginal,
  mergeNodesOperation,
  renameNodeOperation,
  setNodeRoleOperation,
  updateNodeMetaOperation,
  generateNodeId,
  createFixedNode,
  createCrankNode,
  createFollowerNode,
  captureNodePositions,
  calculateNodeBounds,
  calculateNodeCenter,

  // Edge operations
  deleteEdge,
  deleteEdges,
  generateEdgeId,
  createEdge,
  createGroundEdge,
  renameEdgeOperation,
  updateEdgeMetaOperation,
  setEdgeColor,
  setEdgeGround,
  syncEdgeDistanceOperation,
  findEdgesWithinGroup,
  findEdgesCrossingGroup,
  findEdgesConnectedToNodes
} from './operations'

export type {
  NodeDeletionResult,
  NodeMoveResult,
  GroupMoveResult,
  NodeMergeResult,
  NodeRenameResult,
  NodeUpdateResult,
  EdgeDeletionResult,
  EdgeCreationResult,
  EdgeRenameResult,
  EdgeUpdateResult
} from './operations'
