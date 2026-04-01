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

/** Viewport transform for zoom/pan (canvas pixel space) */
export interface ViewportTransform {
  zoom: number
  panX: number
  panY: number
}

export interface CanvasRendererProps {
  /** Cursor style for the canvas */
  cursor: string
  /** Optional viewport for zoom/pan; when omitted, no transform is applied */
  viewport?: ViewportTransform
  /** Optional wheel handler for zoom (e.g. from useViewportState) */
  onWheel?: (event: React.WheelEvent<SVGSVGElement>) => void
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
  renderCanvases?: CanvasLayerRender
  renderDrawnObjects?: CanvasLayerRender
  /** Polygon name labels only; render late so labels stack above links/joints/trajectories */
  renderDrawnObjectLabels?: CanvasLayerRender
  renderLinks?: CanvasLayerRender
  renderPreviewLine?: CanvasLayerRender
  renderPolygonPreview?: CanvasLayerRender
  renderTargetPaths?: CanvasLayerRender
  renderPathPreview?: CanvasLayerRender
  renderExplorationTrajectories?: CanvasLayerRender
  renderTrajectories?: CanvasLayerRender
  renderExplorationDots?: CanvasLayerRender
  renderJoints?: CanvasLayerRender
  renderSelectionBox?: CanvasLayerRender
  renderMeasurementMarkers?: CanvasLayerRender
  renderMeasurementLine?: CanvasLayerRender
  /** When true (explore tool with samples), exploration layers are rendered again above joints so trajectory/dot clicks are received first */
  exploreModeActive?: boolean
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders the SVG canvas and invokes each layer render function in order.
 * Layer order (back to front): grid → drawn object paths → links → previews →
 * target paths → trajectories → joints → selection box → measurements →
 * drawn object name labels (always on top of mechanism graphics).
 */
const DEFAULT_VIEWPORT: ViewportTransform = { zoom: 1, panX: 0, panY: 0 }

export const CanvasRenderer: React.FC<CanvasRendererProps> = ({
  cursor,
  viewport: viewportProp,
  onWheel,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onMouseLeave,
  onClick,
  onDoubleClick,
  filters,
  showGrid,
  renderGrid,
  renderCanvases,
  renderDrawnObjects,
  renderDrawnObjectLabels,
  renderLinks,
  renderPreviewLine,
  renderPolygonPreview,
  renderTargetPaths,
  renderPathPreview,
  renderExplorationTrajectories,
  renderTrajectories,
  renderExplorationDots,
  renderJoints,
  renderSelectionBox,
  renderMeasurementMarkers,
  renderMeasurementLine,
  exploreModeActive = false
}) => {
  const viewport = viewportProp ?? DEFAULT_VIEWPORT
  const transform = `translate(${viewport.panX}, ${viewport.panY}) scale(${viewport.zoom})`

  return (
    <svg
      width="100%"
      height="100%"
      style={{ display: 'block', cursor }}
      onWheel={onWheel}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseLeave}
      onClick={onClick}
      onDoubleClick={onDoubleClick}
    >
      {/* SVG Filter Definitions for Glow Effects */}
      {filters}

      {/* Viewport transform: zoom and pan applied to all content */}
      <g transform={transform}>
      {/* Grid (toggleable via Settings) */}
      {showGrid && renderGrid?.()}

      {/* Canvas image overlays (reference images) */}
      {renderCanvases?.()}

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

      {/* Exploration preview trajectories (low opacity; hovered at 80%) */}
      {renderExplorationTrajectories?.()}

      {/* Trajectory dots for simulation results */}
      {renderTrajectories?.()}

      {/* Exploration dots (green/red sample positions) */}
      {renderExplorationDots?.()}

      {/* Joints */}
      {renderJoints?.()}

      {/* In explore mode with samples: exploration layers on top so trajectory/dot clicks are received */}
      {exploreModeActive && renderExplorationTrajectories?.()}
      {exploreModeActive && renderExplorationDots?.()}

      {/* Group selection box */}
      {renderSelectionBox?.()}

      {/* Measurement markers and line */}
      {renderMeasurementMarkers?.()}
      {renderMeasurementLine?.()}

      {/* Form name labels — after links/joints/trajectories so they are always readable */}
      {renderDrawnObjectLabels?.()}
      </g>
    </svg>
  )
}

export default CanvasRenderer
