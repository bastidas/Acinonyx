/**
 * Previews Renderer
 *
 * Pure functions to render preview elements during drawing/selection operations.
 */

import React from 'react'
import {
  PreviewLineRendererProps,
  SelectionBoxRendererProps,
  PolygonPreviewRendererProps,
  PathPreviewRendererProps
} from './types'

// ═══════════════════════════════════════════════════════════════════════════════
// PREVIEW LINE (during link creation)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders preview line during link creation
 */
export function renderPreviewLine({
  previewLine,
  unitsToPixels
}: PreviewLineRendererProps): JSX.Element | null {
  if (!previewLine) return null

  return (
    <line
      x1={unitsToPixels(previewLine.start[0])}
      y1={unitsToPixels(previewLine.start[1])}
      x2={unitsToPixels(previewLine.end[0])}
      y2={unitsToPixels(previewLine.end[1])}
      stroke="#ff8c00"
      strokeWidth={3}
      strokeDasharray="8,4"
      strokeLinecap="round"
      opacity={0.7}
    />
  )
}

export const PreviewLineRenderer: React.FC<PreviewLineRendererProps> = (props) => {
  return renderPreviewLine(props)
}

// ═══════════════════════════════════════════════════════════════════════════════
// SELECTION BOX (during group selection)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders group selection box (dashed rectangle)
 */
export function renderSelectionBox({
  startPoint,
  currentPoint,
  isSelecting,
  unitsToPixels
}: SelectionBoxRendererProps): JSX.Element | null {
  if (!isSelecting || !startPoint || !currentPoint) {
    return null
  }

  const x1 = unitsToPixels(startPoint[0])
  const y1 = unitsToPixels(startPoint[1])
  const x2 = unitsToPixels(currentPoint[0])
  const y2 = unitsToPixels(currentPoint[1])

  const minX = Math.min(x1, x2)
  const minY = Math.min(y1, y2)
  const width = Math.abs(x2 - x1)
  const height = Math.abs(y2 - y1)

  return (
    <rect
      x={minX}
      y={minY}
      width={width}
      height={height}
      fill="rgba(25, 118, 210, 0.1)"
      stroke="#1976d2"
      strokeWidth={2}
      strokeDasharray="8,4"
      pointerEvents="none"
    />
  )
}

export const SelectionBoxRenderer: React.FC<SelectionBoxRendererProps> = (props) => {
  return renderSelectionBox(props)
}

// ═══════════════════════════════════════════════════════════════════════════════
// POLYGON PREVIEW (during polygon drawing)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders polygon preview during drawing
 */
export function renderPolygonPreview({
  points,
  isDrawing,
  mergeThreshold,
  unitsToPixels
}: PolygonPreviewRendererProps): JSX.Element | null {
  if (!isDrawing || points.length === 0) return null

  // Draw lines between points
  const lines = []
  for (let i = 0; i < points.length - 1; i++) {
    lines.push(
      <line
        key={`poly-line-${i}`}
        x1={unitsToPixels(points[i][0])}
        y1={unitsToPixels(points[i][1])}
        x2={unitsToPixels(points[i + 1][0])}
        y2={unitsToPixels(points[i + 1][1])}
        stroke="#9c27b0"
        strokeWidth={3}
        strokeDasharray="6,3"
        opacity={0.8}
      />
    )
  }

  // Draw points
  const pointMarkers = points.map((point, i) => (
    <circle
      key={`poly-point-${i}`}
      cx={unitsToPixels(point[0])}
      cy={unitsToPixels(point[1])}
      r={i === 0 ? 10 : 6}
      fill={i === 0 ? '#9c27b0' : '#ce93d8'}
      stroke="#fff"
      strokeWidth={2}
      opacity={0.9}
    />
  ))

  // If we have 3+ points, show a preview fill
  let polygonFill = null
  if (points.length >= 3) {
    const pathData = points.map((p, i) =>
      `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
    ).join(' ') + ' Z'

    polygonFill = (
      <path
        d={pathData}
        fill="rgba(156, 39, 176, 0.15)"
        stroke="none"
        pointerEvents="none"
      />
    )
  }

  return (
    <g>
      {polygonFill}
      {lines}
      {pointMarkers}
      {/* Show hint circle around starting point */}
      {points.length >= 3 && (
        <circle
          cx={unitsToPixels(points[0][0])}
          cy={unitsToPixels(points[0][1])}
          r={unitsToPixels(mergeThreshold)}
          fill="none"
          stroke="#9c27b0"
          strokeWidth={1}
          strokeDasharray="4,4"
          opacity={0.5}
        />
      )}
    </g>
  )
}

export const PolygonPreviewRenderer: React.FC<PolygonPreviewRendererProps> = (props) => {
  return renderPolygonPreview(props)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATH PREVIEW (during target path drawing)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders target path preview during drawing
 */
export function renderPathPreview({
  points,
  isDrawing,
  jointMergeRadius,
  unitsToPixels
}: PathPreviewRendererProps): JSX.Element | null {
  if (!isDrawing || points.length === 0) return null

  const canClose = points.length >= 3

  // Draw lines between points
  const lines = []
  for (let i = 0; i < points.length - 1; i++) {
    lines.push(
      <line
        key={`path-line-${i}`}
        x1={unitsToPixels(points[i][0])}
        y1={unitsToPixels(points[i][1])}
        x2={unitsToPixels(points[i + 1][0])}
        y2={unitsToPixels(points[i + 1][1])}
        stroke="#e91e63"
        strokeWidth={3}
        strokeDasharray="8,4"
        opacity={0.9}
      />
    )
  }

  // Draw closing line preview (dashed, faded) - shows the path will be closed
  const closingLine = points.length >= 2 ? (
    <line
      key="path-closing-line"
      x1={unitsToPixels(points[points.length - 1][0])}
      y1={unitsToPixels(points[points.length - 1][1])}
      x2={unitsToPixels(points[0][0])}
      y2={unitsToPixels(points[0][1])}
      stroke="#e91e63"
      strokeWidth={2}
      strokeDasharray="4,8"
      opacity={0.4}
    />
  ) : null

  // Draw snap circle around start point when path can be closed
  const snapCircle = canClose ? (
    <circle
      cx={unitsToPixels(points[0][0])}
      cy={unitsToPixels(points[0][1])}
      r={unitsToPixels(jointMergeRadius)}
      fill="none"
      stroke="#e91e63"
      strokeWidth={2}
      strokeDasharray="4,4"
      opacity={0.6}
    />
  ) : null

  // Draw points
  const pointMarkers = points.map((point, i) => (
    <circle
      key={`path-point-${i}`}
      cx={unitsToPixels(point[0])}
      cy={unitsToPixels(point[1])}
      r={i === 0 ? 8 : 5}
      fill={i === 0 ? '#e91e63' : '#f48fb1'}
      stroke="#fff"
      strokeWidth={2}
      opacity={0.9}
    />
  ))

  return (
    <g>
      {closingLine}
      {lines}
      {snapCircle}
      {pointMarkers}
    </g>
  )
}

export const PathPreviewRenderer: React.FC<PathPreviewRendererProps> = (props) => {
  return renderPathPreview(props)
}
