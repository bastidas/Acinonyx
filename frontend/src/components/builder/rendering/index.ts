/**
 * SVG Rendering Module
 *
 * Pure, testable functions for rendering SVG elements on the canvas.
 *
 * These rendering functions are designed to be:
 * - Pure: No side effects, output depends only on inputs
 * - Testable: Can be unit tested without React component context
 * - Composable: Can be combined to build complex visualizations
 */

// Types
export * from './types'

// Utilities
export {
  createCoordinateConverters,
  getGlowFilterForColor,
  getHighlightStyle,
  getMidpoint,
  getDistance,
  hasMovement,
  isValidPosition,
  generatePathData,
  filterTrajectoriesForRendering
} from './utils'
export type { HighlightStyle, TrajectoryDataForFilter, JointMetaForTrajectory } from './utils'

// SVG Filters (should be included in <defs> section)
export { SVGFilters } from './SVGFilters'

// Canvas (SVG structure and layer order)
export { CanvasRenderer } from './CanvasRenderer'
export type { CanvasRendererProps, CanvasLayerRender, ViewportTransform } from './CanvasRenderer'

// Renderers
export { GridRenderer, renderGrid } from './GridRenderer'
export { JointsRenderer, renderJoints } from './JointsRenderer'
export { LinksRenderer, renderLinks } from './LinksRenderer'
export { TrajectoriesRenderer, renderTrajectories } from './TrajectoriesRenderer'

export {
  PreviewLineRenderer,
  renderPreviewLine,
  SelectionBoxRenderer,
  renderSelectionBox,
  PolygonPreviewRenderer,
  renderPolygonPreview,
  PathPreviewRenderer,
  renderPathPreview
} from './PreviewsRenderer'

export {
  DrawnObjectsRenderer,
  renderDrawnObjects,
  renderDrawnObjectLabels
} from './DrawnObjectsRenderer'
export { CanvasImagesRenderer, renderCanvasImages } from './CanvasImagesRenderer'
export { TargetPathsRenderer, renderTargetPaths } from './TargetPathsRenderer'

export {
  MeasurementMarkersRenderer,
  renderMeasurementMarkers,
  MeasurementLineRenderer,
  renderMeasurementLine
} from './MeasurementsRenderer'

export {
  ExplorationDotsRenderer,
  renderExplorationDots
} from './ExplorationDotsRenderer'
export {
  ExplorationTrajectoriesRenderer,
  renderExplorationTrajectories
} from './ExplorationTrajectoriesRenderer'
