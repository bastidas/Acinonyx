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
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRAJECTORY RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

export type TrajectoryStyle = 'dots' | 'line' | 'both'
export type ColorCycleType = 'rainbow' | 'fire' | 'glow'

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
  points: Position[]
  fillColor: string
  strokeColor: string
  strokeWidth: number
  fillOpacity: number
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
  onObjectClick: (id: string, isSelected: boolean) => void
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
