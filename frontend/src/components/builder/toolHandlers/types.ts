/**
 * Tool handler types
 *
 * Infrastructure for delegating canvas input (mouse down/move/up, click,
 * double-click) to per-tool handlers. Used to extract tool mode logic from
 * BuilderTab into focused handler modules.
 */

import type { RefObject } from 'react'
import type {
  ToolMode,
  DragState,
  StatusType,
  LinkCreationState,
  MeasureState,
  MeasurementMarker,
  GroupSelectionState,
  PolygonDrawState,
  MergePolygonState,
  PathDrawState,
  ExploreTrajectoriesState
} from '../../BuilderTools'
import type { LinkageDocument } from '../types'

/** Point in canvas units [x, y] */
export type CanvasPoint = [number, number]

/** Joint with position for hit-testing (matches BuilderTools findNearestJoint) */
export interface JointWithPosition {
  name: string
  position: [number, number] | null
}

/** Link with endpoints for hit-testing (matches BuilderTools findNearestLink) */
export interface LinkWithPosition {
  name: string
  start: [number, number] | null
  end: [number, number] | null
}

/** Result of findNearestJoint */
export interface NearestJointResult {
  name: string
  position: [number, number]
  distance: number
}

/** Response from POST /merge-polygon */
export interface MergePolygonApiResult {
  status: string
  polygon?: {
    polygon_id: string
    contained_links: string[]
    mergedLinkName: string | null
    mergedLinkOriginalStart: [number, number] | null
    mergedLinkOriginalEnd: [number, number] | null
    fill_color: string | null
    stroke_color: string | null
    /** When request had selected_link_name: true if that link is fully inside, false otherwise. */
    selected_link_fully_inside?: boolean | null
  }
}

/** Response from POST /merge-two-polygons */
export interface MergeTwoPolygonsApiResult {
  status: string
  message?: string
  merged_polygon?: {
    polygon_id: string
    points: CanvasPoint[]
    contained_links: string[]
  }
}

/** Response from POST /find-associated-polygons */
export interface FindAssociatedPolygonsApiResult {
  status: string
  polygons?: Record<string, { contained_links: string[]; all_inside: boolean }>
}

/** Result of findNearestLink */
export interface NearestLinkResult {
  name: string
  distance: number
}

/**
 * Context passed to tool handlers. Provides canvas ref, coordinate conversion,
 * current tool mode, and (in later steps) document state, selection, and
 * operations. Extended as handler logic is moved out of BuilderTab.
 */
export interface ToolContext {
  /** Ref to the canvas element (SVG or wrapper) for coordinate conversion */
  canvasRef: RefObject<SVGSVGElement | HTMLDivElement | null>
  /** Convert pixel distance to canvas units */
  pixelsToUnits: (pixels: number) => number
  /** Current tool mode */
  toolMode: ToolMode
  /** Set current tool mode */
  setToolMode: (mode: ToolMode) => void
  /**
   * Get canvas-unit point from a mouse event, or null if canvas ref not ready.
   * Shared helper so handlers don't duplicate rect + conversion logic.
   */
  getPointFromEvent: (event: React.MouseEvent<SVGSVGElement>) => CanvasPoint | null

  // ─── Extended for Select (and shared) tool handlers ─────────────────────
  /** All joints with current positions */
  getJointsWithPositions: () => JointWithPosition[]
  /** All links with endpoint positions */
  getLinksWithPositions: () => LinkWithPosition[]
  /** Position of a joint by name */
  getJointPosition: (jointName: string) => [number, number] | null
  /** Link connectivity (same as pylinkDoc.meta.links) for group-select / mechanism helpers */
  linksMeta: Record<string, { connects: string[] }>
  /** Find nearest joint to point within threshold */
  findNearestJoint: (
    point: CanvasPoint,
    joints: JointWithPosition[],
    threshold?: number
  ) => NearestJointResult | null
  /** Find nearest link to point within threshold */
  findNearestLink: (
    point: CanvasPoint,
    links: LinkWithPosition[],
    threshold?: number
  ) => NearestLinkResult | null
  /** Find nearest joint that can be merged into (excludes excludeJoint) */
  findMergeTarget: (
    position: CanvasPoint,
    joints: JointWithPosition[],
    excludeJoint: string,
    threshold?: number
  ) => { name: string; position: [number, number]; distance: number } | null
  /** Distance threshold for joint snap (e.g. JOINT_SNAP_THRESHOLD) */
  snapThreshold: number
  /** Distance threshold for merge detection during drag */
  jointMergeRadius: number

  /** Current drag state (single-joint drag in select mode) */
  dragState: DragState
  /** Set drag state */
  setDragState: (value: DragState | ((prev: DragState) => DragState)) => void
  /** Selected joint ids */
  selectedJoints: string[]
  /** Set selected joints */
  setSelectedJoints: (value: string[] | ((prev: string[]) => string[])) => void
  /** Selected link ids */
  selectedLinks: string[]
  /** Set selected links */
  setSelectedLinks: (value: string[] | ((prev: string[]) => string[])) => void

  /** Move a joint to a position (updates document) */
  moveJoint: (jointName: string, position: CanvasPoint) => void
  /** Move two joints in one document update (e.g. combinatorial explore apply). */
  moveTwoJoints: (
    jointName1: string,
    position1: CanvasPoint,
    jointName2: string,
    position2: CanvasPoint
  ) => void
  /** Merge source joint into target joint */
  mergeJoints: (sourceJoint: string, targetJoint: string) => void
  /** Reset animation to first frame (1/N); call when starting group drag if desired */
  resetAnimationToFirstFrame: () => void
  /** Pause animation without changing frame; use when starting single-node drag so mechanism stays at current X/N */
  pauseAnimation: () => void
  /** Show status message */
  showStatus: (message: string, type?: StatusType, duration?: number) => void
  /** Clear status message */
  clearStatus: () => void

  // ─── Extended for Draw Link tool handler ─────────────────────────────────
  /** Link creation state (draw_link: first vs second click) */
  linkCreationState: LinkCreationState
  /** Set link creation state */
  setLinkCreationState: (value: LinkCreationState | ((prev: LinkCreationState) => LinkCreationState)) => void
  /** Set preview line (draw_link rubber-band) */
  setPreviewLine: (value: { start: CanvasPoint; end: CanvasPoint } | null) => void
  /** Create a link between two points/joints with revolute default */
  createLinkWithRevoluteDefault: (
    startPoint: CanvasPoint,
    endPoint: CanvasPoint,
    startJointName: string | null,
    endJointName: string | null
  ) => void

  // ─── Extended for Delete tool handler ────────────────────────────────────
  /** Delete a joint (and connected links / orphans) */
  deleteJoint: (jointName: string) => void
  /** Delete a link (and orphaned joints) */
  deleteLink: (linkName: string) => void
  /** Delete currently selected joints/links/drawn objects (with confirmation for multiple) */
  handleDeleteSelected: () => void

  // ─── Extended for Measure tool handler ──────────────────────────────────
  /** Measure state (first vs second click) */
  measureState: MeasureState
  /** Set measure state */
  setMeasureState: (value: MeasureState | ((prev: MeasureState) => MeasureState)) => void
  /** Set measurement markers (start/end points, fade after 3s) */
  setMeasurementMarkers: (value: MeasurementMarker[] | ((prev: MeasurementMarker[]) => MeasurementMarker[])) => void
  /** Calculate distance between two points */
  calculateDistance: (p1: CanvasPoint, p2: CanvasPoint) => number

  // ─── Extended for Group Select tool handler ─────────────────────────────
  /** Group selection state (box selection) */
  groupSelectionState: GroupSelectionState
  /** Set group selection state */
  setGroupSelectionState: (value: GroupSelectionState | ((prev: GroupSelectionState) => GroupSelectionState)) => void
  /** Find joints and links inside a box */
  findElementsInBox: (
    box: { x1: number; y1: number; x2: number; y2: number },
    joints: JointWithPosition[],
    links: LinkWithPosition[]
  ) => { joints: string[]; links: string[] }
  /** Drawn objects (for box-select and merge detection) */
  drawnObjects: { objects: Array<{ id: string; name?: string; type?: string; points: CanvasPoint[]; mergedLinkName?: string; contained_links?: string[] }> }
  /** Call backend merge-polygon; if provided, merge tool uses it instead of local logic */
  apiMergePolygon?: (params: {
    polygonId: string
    polygonPoints: CanvasPoint[]
    selectedLinkName?: string
    /** When true (e.g. fresh draw), do not set mergedLinkName/contained_links if more than one link is inside. */
    attachOnlyIfSingleContainedLink?: boolean
  }) => Promise<MergePolygonApiResult>
  /** Call backend merge-two-polygons; returns outer bounding polygon geometry for two polygons */
  apiMergeTwoPolygons?: (params: { polygonIdA: string; polygonPointsA: CanvasPoint[]; polygonIdB: string; polygonPointsB: CanvasPoint[] }) => Promise<MergeTwoPolygonsApiResult>
  /** Call backend find-associated-polygons (e.g. after drag); sets broken state from all_inside */
  apiFindAssociatedPolygons?: () => Promise<FindAssociatedPolygonsApiResult>
  /** Set linkage document (for updating meta e.g. link colors after merge). */
  setLinkageDoc?: (value: LinkageDocument | ((prev: LinkageDocument) => LinkageDocument)) => void
  /** Set drawn objects state (objects + selectedIds); accepts setState action from BuilderTab */
  setDrawnObjects: (value: unknown) => void
  /** Enter move group mode with selected joints and drawn object ids */
  enterMoveGroupMode: (jointNames: string[], drawnObjectIds?: string[]) => void

  // ─── Extended for Draw Polygon tool handler ──────────────────────────────
  /** Polygon drawing state */
  polygonDrawState: PolygonDrawState
  /** Set polygon drawing state */
  setPolygonDrawState: (value: PolygonDrawState | ((prev: PolygonDrawState) => PolygonDrawState)) => void
  /** Create a new drawn object (polygon); returns the new object with id */
  createDrawnObject: (type: 'polygon', points: CanvasPoint[], existingIds: string[]) => { id: string; type: string; points: CanvasPoint[]; name: string }
  /** Distance threshold for closing polygon (click near start) */
  mergeThreshold: number

  // ─── Extended for Merge tool handler ────────────────────────────────────
  /** Merge polygon state (polygon/link selection steps) */
  mergePolygonState: MergePolygonState
  /** Set merge polygon state */
  setMergePolygonState: (value: MergePolygonState | ((prev: MergePolygonState) => MergePolygonState)) => void
  /** Link meta by name (connects, color) for merge validation */
  getLinkMeta: (linkName: string) => { connects: [string, string]; color?: string } | null
  /** Whether point is inside polygon */
  isPointInPolygon: (point: CanvasPoint, polygonPoints: CanvasPoint[]) => boolean
  /** Whether link endpoints are inside polygon */
  areLinkEndpointsInPolygon: (start: CanvasPoint, end: CanvasPoint, polygonPoints: CanvasPoint[]) => boolean
  /** Default color by index */
  getDefaultColor: (index: number) => string
  /** Link snap threshold in merge mode (e.g. 8.0) */
  mergeLinkThreshold: number
  /** Transform polygon points when unmerging (original link → current link) */
  transformPolygonPoints: (
    points: CanvasPoint[],
    originalStart: CanvasPoint,
    originalEnd: CanvasPoint,
    currentStart: CanvasPoint,
    currentEnd: CanvasPoint
  ) => CanvasPoint[]

  // ─── Extended for Draw Path tool handler ─────────────────────────────────
  /** Path drawing state (target trajectory) */
  pathDrawState: PathDrawState
  /** Set path drawing state */
  setPathDrawState: (value: PathDrawState | ((prev: PathDrawState) => PathDrawState)) => void
  /** Existing target paths (for createTargetPath) */
  targetPaths: Array<{ id: string; name: string; points: CanvasPoint[]; targetJoint: string | null; color: string; isComplete: boolean }>
  /** Set target paths */
  setTargetPaths: (value: Array<{ id: string; name: string; points: CanvasPoint[]; targetJoint: string | null; color: string; isComplete: boolean }> | ((prev: unknown[]) => unknown[])) => void
  /** Create a new target path from points */
  createTargetPath: (points: CanvasPoint[], existingPaths: unknown[]) => { id: string; name: string; points: CanvasPoint[]; targetJoint: string | null; color: string; isComplete: boolean }
  /** Set selected target path id */
  setSelectedPathId: (id: string | null) => void

  // ─── Extended for Explore Node Trajectories tool ───────────────────────────
  /** Current linkage document (hypergraph) for building variant docs */
  linkageDoc: LinkageDocument
  /** Explore trajectories state (samples, hovered index, loading) */
  exploreTrajectoriesState: ExploreTrajectoriesState
  /** Set explore trajectories state */
  setExploreTrajectoriesState: React.Dispatch<React.SetStateAction<ExploreTrajectoriesState>>
  /** Run exploration: batch-simulate circle of positions, store results */
  runExploreTrajectories: (nodeId: string, center: [number, number]) => Promise<void>
  /** Run path exploration: first node fixed at pinnedPosition, second node explored around center2. */
  runExploreTrajectoriesSecond: (
    fixedNodeId: string,
    pinnedPosition: [number, number],
    nodeId2: string,
    center2: [number, number]
  ) => Promise<void>
  /** Run combinatorial exploration: N1×N2 combinations capped at NMAX, no colormap. */
  runExploreTrajectoriesCombinatorial: (
    nodeId1: string,
    center1: [number, number],
    samples1: Array<{ position: [number, number]; valid: boolean }>,
    nodeId2: string,
    center2: [number, number]
  ) => Promise<void>
  /** True if node has "show trajectory path" enabled. */
  getNodeShowPath: (nodeId: string) => boolean
  /** Joint type from mechanism (Static/Crank/Revolute) for explore mode; allows moving static joints. */
  getJointType: (jointName: string) => 'Static' | 'Crank' | 'Revolute' | null
}

/**
 * Optional event handlers for a single tool. A handler returns true if it
 * handled the event (and dispatch should stop); false/void otherwise.
 * Only implement the events the tool needs.
 */
export interface ToolHandler {
  onMouseDown?: (
    event: React.MouseEvent<SVGSVGElement>,
    point: CanvasPoint,
    context: ToolContext
  ) => boolean | void
  onMouseMove?: (
    event: React.MouseEvent<SVGSVGElement>,
    point: CanvasPoint,
    context: ToolContext
  ) => boolean | void
  onMouseUp?: (event: React.MouseEvent<SVGSVGElement>, context: ToolContext) => boolean | void
  onClick?: (
    event: React.MouseEvent<SVGSVGElement>,
    point: CanvasPoint,
    context: ToolContext
  ) => boolean | void
  onDoubleClick?: (
    event: React.MouseEvent<SVGSVGElement>,
    point: CanvasPoint,
    context: ToolContext
  ) => boolean | void
  onMouseLeave?: (event: React.MouseEvent<SVGSVGElement>, context: ToolContext) => boolean | void
}
