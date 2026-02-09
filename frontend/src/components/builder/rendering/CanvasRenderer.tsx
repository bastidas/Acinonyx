/**
 * CanvasRenderer
 *
 * Renders the main SVG canvas element and invokes layer render functions in order.
 * Step 3.2: Inline rendering functions are invoked here via render props.
 */

import React from 'react'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/** Render function for a single layer (grid, links, joints, etc.) */
export type CanvasLayerRender = () => React.ReactNode

export interface CanvasRendererProps {
  /** Cursor style for the canvas */
  cursor: string
  /** SVG mouse handlers */
  onMouseDown?: (event: React.MouseEvent<SVGSVGElement>) => void
  onMouseMove?: (event: React.MouseEvent<SVGSVGElement>) => void
  onMouseUp?: (event: React.MouseEvent<SVGSVGElement>) => void
  onMouseLeave?: (event: React.MouseEvent<SVGSVGElement>) => void
  onClick?: (event: React.MouseEvent<SVGSVGElement>) => void
  onDoubleClick?: (event: React.MouseEvent<SVGSVGElement>) => void
  /** SVG filter definitions (e.g. <SVGFilters />) */
  filters: React.ReactNode
  /** Whether to show the grid layer */
  showGrid: boolean
  /** Layer render functions – order matches visual stacking (first = back) */
  renderGrid?: CanvasLayerRender
  renderDrawnObjects?: CanvasLayerRender
  renderLinks?: CanvasLayerRender
  renderPreviewLine?: CanvasLayerRender
  renderPolygonPreview?: CanvasLayerRender
  renderTargetPaths?: CanvasLayerRender
  renderPathPreview?: CanvasLayerRender
  renderTrajectories?: CanvasLayerRender
  renderJoints?: CanvasLayerRender
  renderSelectionBox?: CanvasLayerRender
  renderMeasurementMarkers?: CanvasLayerRender
  renderMeasurementLine?: CanvasLayerRender
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders the SVG canvas and invokes each layer render function in order.
 * Layer order (back to front): grid → drawn objects → links → previews →
 * target paths → trajectories → joints → selection box → measurements.
 */
export const CanvasRenderer: React.FC<CanvasRendererProps> = ({
  cursor,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onMouseLeave,
  onClick,
  onDoubleClick,
  filters,
  showGrid,
  renderGrid,
  renderDrawnObjects,
  renderLinks,
  renderPreviewLine,
  renderPolygonPreview,
  renderTargetPaths,
  renderPathPreview,
  renderTrajectories,
  renderJoints,
  renderSelectionBox,
  renderMeasurementMarkers,
  renderMeasurementLine
}) => {
  return (
    <svg
      width="100%"
      height="100%"
      style={{ display: 'block', cursor }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      {/* SVG Filter Definitions for Glow Effects */}
      {filters}

      {/* Grid (toggleable via Settings) */}
      {showGrid && renderGrid?.()}

      {/* Completed drawn objects (polygons, shapes) - render BEFORE links */}
      {renderDrawnObjects?.()}

      {/* Links - on top of polygons so they can be clicked */}
      {renderLinks?.()}

      {/* Preview line during link creation */}
      {renderPreviewLine?.()}

      {/* Polygon preview during drawing */}
      {renderPolygonPreview?.()}

      {/* Target paths for trajectory optimization */}
      {renderTargetPaths?.()}

      {/* Target path preview during drawing */}
      {renderPathPreview?.()}

      {/* Trajectory dots for simulation results */}
      {renderTrajectories?.()}

      {/* Joints */}
      {renderJoints?.()}

      {/* Group selection box */}
      {renderSelectionBox?.()}

      {/* Measurement markers and line */}
      {renderMeasurementMarkers?.()}
      {renderMeasurementLine?.()}
    </svg>
  )
}

export default CanvasRenderer
