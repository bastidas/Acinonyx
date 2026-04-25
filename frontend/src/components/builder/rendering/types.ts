/**
 * Types for SVG rendering functions
 *
 * These types define the props needed by pure rendering functions.
 */

import { HighlightType, ObjectType } from '../types'

// ═══════════════════════════════════════════════════════════════════════════════
// COORDINATE & DIMENSIONS
// ═══════════════════════════════════════════════════════════════════════════════

export interface CanvasDimensions {
  width: number
  height: number
}

export type Position = [number, number]

// ═══════════════════════════════════════════════════════════════════════════════
// JOINT RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface JointColors {
  static: string
  crank: string
  pivot: string
  moveGroup: string
  mergeHighlight: string
}

export interface JointRenderData {
  name: string
  type: 'Static' | 'Crank' | 'Revolute'
  position: Position
  color: string
  isSelected: boolean
  isInMoveGroup: boolean
  isHovered: boolean
  isDragging: boolean
  isMergeTarget: boolean
}

export interface JointsRendererProps {
  joints: JointRenderData[]
  jointSize: number
  /** Joint outline width in px (0 = none, 10 = max). */
  jointOutline: number
  jointColors: JointColors
  darkMode: boolean
  showJointLabels: boolean
  moveGroupIsActive: boolean
  toolMode: string
  getHighlightStyle: (
    objectType: ObjectType,
    highlightType: HighlightType,
    baseColor: string,
    baseStrokeWidth: number
  ) => { stroke: string; strokeWidth: number; filter?: string }
  unitsToPixels: (units: number) => number
  onJointHover: (name: string | null) => void
  onJointDoubleClick: (name: string) => void
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINK RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface LinkRenderData {
  name: string
  connects: [string, string]
  position0: Position
  position1: Position
  color: string
  isGround: boolean
  isSelected: boolean
  isInMoveGroup: boolean
  isHovered: boolean
  isStretching: boolean
}

export interface LinksRendererProps {
  links: LinkRenderData[]
  linkThickness: number
  /** Link opacity as percentage 10–100. */
  linkTransparency: number
  darkMode: boolean
  showLinkLabels: boolean
  moveGroupIsActive: boolean
  moveGroupIsDragging: boolean
  toolMode: string
  getHighlightStyle: (
    objectType: ObjectType,
    highlightType: HighlightType,
    baseColor: string,
    baseStrokeWidth: number
  ) => { stroke: string; strokeWidth: number; filter?: string }
  unitsToPixels: (units: number) => number
  onLinkHover: (name: string | null) => void
  onLinkDoubleClick: (name: string) => void
  /** When true, single click calls onMergeLinkClick(linkName) if provided. */
  mergeMode?: boolean
  /** Called when user single-clicks a link in merge mode. */
  onMergeLinkClick?: (linkName: string) => void
  /** When set, render an invisible wider hit area (px) for easier clicking. */
  hitAreaStrokeWidthPx?: number
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAJECTORY RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export type TrajectoryStyle = 'dots' | 'line' | 'both'
export type ColorCycleType = 'rainbow' | 'fire' | 'glow' | 'twilight' | 'husl'

export interface TrajectoryRenderData {
  jointName: string
  positions: Position[]
  jointType: 'Static' | 'Crank' | 'Revolute'
  hasMovement: boolean
  showPath: boolean
}

export interface TrajectoriesRendererProps {
  trajectories: TrajectoryRenderData[]
  trajectoryDotSize: number
  trajectoryDotOutline: boolean
  trajectoryDotOpacity: number
  /** When true, show step numbers (1..N) next to trajectory dots. Default false. */
  showTrajectoryStepNumbers?: boolean
  trajectoryStyle: TrajectoryStyle
  trajectoryColorCycle: ColorCycleType
  jointColors: JointColors
  unitsToPixels: (units: number) => number
  getCyclicColor: (stepIndex: number, totalSteps: number, cycleType: ColorCycleType) => string
}

// ═══════════════════════════════════════════════════════════════════════════════
// GRID RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface GridRendererProps {
  canvasDimensions: CanvasDimensions
  darkMode: boolean
  unitsToPixels: (units: number) => number
  pixelsToUnits: (pixels: number) => number
}

// ═══════════════════════════════════════════════════════════════════════════════
// CANVAS IMAGES RENDERING (reference/overlay images)
// ═══════════════════════════════════════════════════════════════════════════════

export interface CanvasImageRenderItem {
  id: string
  dataUrl: string
  position: Position
  scale: number
  alpha: number
  naturalWidth: number
  naturalHeight: number
}

export interface CanvasImagesRendererProps {
  canvases: CanvasImageRenderItem[]
  unitsToPixels: (units: number) => number
  pixelsToUnits: (pixels: number) => number
  /** Double-click on handle opens edit dialog for this canvas */
  onRequestEdit: (id: string) => void
  /** Drag handle updates canvas position */
  onPositionChange: (id: string, position: [number, number]) => void
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREVIEW RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface PreviewLine {
  start: Position
  end: Position
}

export interface PreviewLineRendererProps {
  previewLine: PreviewLine | null
  unitsToPixels: (units: number) => number
}

export interface SelectionBoxRendererProps {
  startPoint: Position | null
  currentPoint: Position | null
  isSelecting: boolean
  unitsToPixels: (units: number) => number
}

export interface PolygonPreviewRendererProps {
  points: Position[]
  isDrawing: boolean
  mergeThreshold: number
  mode?: 'polygon' | 'circle'
  circleCenter?: Position | null
  circleRadius?: number | null
  circleRadiusPoint?: Position | null
  unitsToPixels: (units: number) => number
}

export interface PathPreviewRendererProps {
  points: Position[]
  isDrawing: boolean
  jointMergeRadius: number
  unitsToPixels: (units: number) => number
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAWN OBJECTS RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface DrawnObject {
  id: string
  type: 'polygon'
  name: string
  /** Display points (BuilderTab may pass transformed points when merged to a link). */
  points: Position[]
  fillColor: string
  strokeColor: string
  strokeWidth: number
  fillOpacity: number
  /** If set, polygon is merged to this link (for merge-mode styling: unmerge candidate). */
  mergedLinkName?: string
  /** All link IDs contained in the polygon. */
  contained_links?: string[]
  /** When false, at least one contained link has endpoints outside the polygon (e.g. after drag). */
  contained_links_valid?: boolean
  /** Z-level (layer) for rendering order. */
  z_level?: number
  /** When true, this form's z-level is pinned during recompute (e.g. after drag). */
  z_level_fixed?: boolean
  /** Preferred z when recomputing (soft pin). */
  target_z_level?: number
}

export interface DrawnObjectsRendererProps {
  objects: DrawnObject[]
  selectedIds: string[]
  moveGroupIsActive: boolean
  moveGroupDrObjectIds: string[]
  toolMode: string
  getHighlightStyle: (
    objectType: ObjectType,
    highlightType: HighlightType,
    baseColor: string,
    baseStrokeWidth: number
  ) => { stroke: string; strokeWidth: number; filter?: string }
  unitsToPixels: (units: number) => number
  onObjectClick: (id: string, isSelected: boolean, event: React.MouseEvent) => void
  /** In select mode, double-click opens the form edit modal (same as Forms toolbar). */
  onObjectDoubleClick?: (id: string) => void
  /** When true, apply merge-mode styling and use onMergePolygonClick for polygon clicks. */
  mergeMode?: boolean
  /** Polygon id under pointer (for merge hover styling). */
  hoveredPolygonId?: string | null
  /** Called when pointer enters/leaves a polygon in merge mode (polygonId or null). */
  onMergePolygonHover?: (polygonId: string | null) => void
  /** Called when user clicks a polygon in merge mode; isUnmerge is true on Shift+click for merged polygons. */
  onMergePolygonClick?: (objId: string, isUnmerge: boolean) => void
  /** When true (e.g. draw_polygon mode), polygon paths use pointer-events: none so clicks pass through to canvas. */
  pointerEventsNoneForDrawPolygon?: boolean
}

// ═══════════════════════════════════════════════════════════════════════════════
// TARGET PATH RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface TargetPath {
  id: string
  name: string
  points: Position[]
  color: string
}

export interface TargetPathsRendererProps {
  targetPaths: TargetPath[]
  selectedPathId: string | null
  unitsToPixels: (units: number) => number
  onPathClick: (id: string | null) => void
  /** Match visualization settings for other trajectories (dot size, opacity, outline) */
  trajectoryDotSize?: number
  trajectoryDotOpacity?: number
  trajectoryDotOutline?: boolean
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEASUREMENT RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export interface MeasurementMarker {
  id: string
  point: Position
  timestamp: number
}

export interface MeasurementMarkersRendererProps {
  markers: MeasurementMarker[]
  unitsToPixels: (units: number) => number
}

export interface MeasurementLineRendererProps {
  startPoint: Position | null
  isMeasuring: boolean
  unitsToPixels: (units: number) => number
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXPLORATION (explore_node_trajectories tool)
// ═══════════════════════════════════════════════════════════════════════════════

export type ExploreTrajectoriesMode = 'single' | 'path' | 'combinatorial'

export interface ExplorationDotsRendererProps {
  samples: Array<{ position: Position; valid: boolean }>
  hoveredIndex: number | null
  unitsToPixels: (units: number) => number
  /** When set (e.g. combinatorial mode with deduplicated dots), highlight the dot at this position instead of by index. */
  hoveredPosition?: [number, number] | null
  /** When true, color valid dots by angle/radius and invalid dots grey. Combinatorial: color by second-node position (one dot per unique position, M trajectories per color). */
  exploreColormapEnabled?: boolean
  exploreColormapType?: 'rainbow' | 'twilight' | 'husl'
  exploreCenter?: [number, number] | null
  exploreRadius?: number
  exploreMode?: ExploreTrajectoriesMode
}

export interface ExplorationTrajectoriesRendererProps {
  /** Per-sample trajectory data; only items with valid && trajectory are drawn */
  samples: Array<{
    position: Position
    valid: boolean
    trajectory: { trajectories: Record<string, Position[]>; nSteps: number; jointTypes?: Record<string, string> } | null
  }>
  hoveredIndex: number | null
  /** When set (combinatorial), highlight all trajectories at this second-node position when hover came from dot; also used to highlight associated dot. */
  hoveredPosition?: [number, number] | null
  /** When true (combinatorial), hover came from a trajectory path: highlight only that single trajectory; else from dot: highlight all at position. */
  hoveredFromTrajectoryPath?: boolean
  /** First exploration node id (combinatorial); when drawing hovered trajectory, always show this node's path even if show_path was false. */
  exploreNodeId?: string | null
  unitsToPixels: (units: number) => number
  /** Non-empty: only these joint names (same opt-out as simulation: `show_path === false`). Empty or omitted: no paths except combinatorial forceShow on exploreNodeId. */
  jointNamesToShow?: string[]
  /** Opacity for non-hovered trajectories (default 0.05) */
  baseOpacity?: number
  /** Opacity for hovered trajectory (default 0.8) */
  hoveredOpacity?: number
  /** When true, stroke each trajectory with the same color as its sample dot. Combinatorial: color by second-node position (M trajectories same color). */
  exploreColormapEnabled?: boolean
  exploreColormapType?: 'rainbow' | 'twilight' | 'husl'
  exploreCenter?: [number, number] | null
  exploreRadius?: number
  exploreMode?: ExploreTrajectoriesMode
}
