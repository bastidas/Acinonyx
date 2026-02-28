/**
 * Target Paths Renderer
 *
 * Pure function to render completed target paths for trajectory optimization.
 * All target paths are rendered as closed curves (cyclic trajectories).
 * Default style matches Visualization settings (trajectory dot size, opacity, outline);
 * when selected we apply a subtle increase in opacity, size, and outline.
 */

import React from 'react'
import { TargetPathsRendererProps, TargetPath } from './types'

const DEFAULT_DOT_SIZE = 4
const DEFAULT_OPACITY = 0.85
const SELECTED_OPACITY_BOOST = 0.15
const SELECTED_SIZE_BOOST = 1
const SELECTED_STROKE_BOOST = 0.5

/**
 * Renders a single target path
 */
function renderTargetPath(
  path: TargetPath,
  selectedPathId: string | null,
  unitsToPixels: (units: number) => number,
  onPathClick: (id: string | null) => void,
  trajectoryDotSize: number,
  trajectoryDotOpacity: number,
  trajectoryDotOutline: boolean
): JSX.Element | null {
  const isSelected = selectedPathId === path.id
  const points = path.points

  if (points.length < 2) return null

  // Match trajectory visualization: base style from settings; selected = subtle boost
  const dotSize = trajectoryDotSize
  const dotSizeSelected = trajectoryDotSize + SELECTED_SIZE_BOOST
  const opacity = trajectoryDotOpacity
  const opacitySelected = Math.min(1, trajectoryDotOpacity + SELECTED_OPACITY_BOOST)
  const outline = trajectoryDotOutline
  const strokeW = outline ? 1 : 0
  const strokeWSelected = (outline ? 1 : 0) + SELECTED_STROKE_BOOST
  const pathStrokeWidth = Math.max(1.5, dotSize * 0.6)
  const pathStrokeWidthSelected = pathStrokeWidth + SELECTED_STROKE_BOOST

  // Create path data for closed curve (Z closes the path back to start)
  const pathData = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
  ).join(' ') + ' Z'  // Close the path

  return (
    <g key={path.id}>
      {/* Path line - closed shape; style matches trajectory viz, subtle boost when selected */}
      <path
        d={pathData}
        fill="none"
        stroke={path.color}
        strokeWidth={isSelected ? pathStrokeWidthSelected : pathStrokeWidth}
        strokeDasharray="10,5"
        opacity={isSelected ? opacitySelected : opacity}
        style={{ cursor: 'pointer' }}
        onClick={(e) => {
          e.stopPropagation()
          onPathClick(isSelected ? null : path.id)
        }}
      />

      {/* Path points - same size/opacity/outline as trajectory dots */}
      {points.map((point, i) => (
        <circle
          key={`${path.id}-point-${i}`}
          cx={unitsToPixels(point[0])}
          cy={unitsToPixels(point[1])}
          r={isSelected ? dotSizeSelected : dotSize}
          fill={path.color}
          stroke={isSelected || outline ? '#fff' : 'none'}
          strokeWidth={isSelected ? strokeWSelected : strokeW}
          opacity={isSelected ? opacitySelected : opacity}
          style={{ pointerEvents: 'none' }}
        />
      ))}

      {/* Start point indicator (slightly larger, same style) */}
      <circle
        cx={unitsToPixels(points[0][0])}
        cy={unitsToPixels(points[0][1])}
        r={isSelected ? dotSizeSelected + 2 : dotSize + 2}
        fill={path.color}
        stroke={isSelected || outline ? '#fff' : 'none'}
        strokeWidth={isSelected ? strokeWSelected : strokeW}
        opacity={isSelected ? opacitySelected : opacity}
        style={{ pointerEvents: 'none' }}
      />

      {/* Path label - only when selected */}
      {isSelected && points.length > 0 && (
        <text
          x={unitsToPixels(points[0][0])}
          y={unitsToPixels(points[0][1]) - 14}
          textAnchor="middle"
          fontSize="11"
          fontWeight="bold"
          fill={path.color}
        >
          {path.name} (closed)
        </text>
      )}
    </g>
  )
}

/**
 * Renders all target paths
 */
export function renderTargetPaths({
  targetPaths,
  selectedPathId,
  unitsToPixels,
  onPathClick,
  trajectoryDotSize = DEFAULT_DOT_SIZE,
  trajectoryDotOpacity = DEFAULT_OPACITY,
  trajectoryDotOutline = true
}: TargetPathsRendererProps): (JSX.Element | null)[] {
  if (targetPaths.length === 0) return []

  return targetPaths.map(path =>
    renderTargetPath(
      path,
      selectedPathId,
      unitsToPixels,
      onPathClick,
      trajectoryDotSize,
      trajectoryDotOpacity,
      trajectoryDotOutline
    )
  )
}

/**
 * React component wrapper for the target paths renderer
 */
export const TargetPathsRenderer: React.FC<TargetPathsRendererProps> = (props) => {
  return <>{renderTargetPaths(props)}</>
}

export default TargetPathsRenderer
