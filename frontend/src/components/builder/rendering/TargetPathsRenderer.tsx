/**
 * Target Paths Renderer
 *
 * Pure function to render completed target paths for trajectory optimization.
 * All target paths are rendered as closed curves (cyclic trajectories).
 */

import React from 'react'
import { TargetPathsRendererProps, TargetPath } from './types'

/**
 * Renders a single target path
 */
function renderTargetPath(
  path: TargetPath,
  selectedPathId: string | null,
  unitsToPixels: (units: number) => number,
  onPathClick: (id: string | null) => void
): JSX.Element | null {
  const isSelected = selectedPathId === path.id
  const points = path.points

  if (points.length < 2) return null

  // Create path data for closed curve (Z closes the path back to start)
  const pathData = points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
  ).join(' ') + ' Z'  // Close the path

  return (
    <g key={path.id}>
      {/* Path line - closed shape */}
      <path
        d={pathData}
        fill="none"
        stroke={path.color}
        strokeWidth={isSelected ? 4 : 3}
        strokeDasharray="10,5"
        opacity={isSelected ? 1 : 0.7}
        style={{ cursor: 'pointer' }}
        onClick={(e) => {
          e.stopPropagation()
          onPathClick(isSelected ? null : path.id)
        }}
      />

      {/* Path points */}
      {points.map((point, i) => (
        <circle
          key={`${path.id}-point-${i}`}
          cx={unitsToPixels(point[0])}
          cy={unitsToPixels(point[1])}
          r={isSelected ? 5 : 4}
          fill={path.color}
          stroke="#fff"
          strokeWidth={1.5}
          opacity={isSelected ? 1 : 0.7}
          style={{ pointerEvents: 'none' }}
        />
      ))}

      {/* Start point indicator (larger) */}
      <circle
        cx={unitsToPixels(points[0][0])}
        cy={unitsToPixels(points[0][1])}
        r={isSelected ? 7 : 6}
        fill={path.color}
        stroke="#fff"
        strokeWidth={2}
        opacity={isSelected ? 1 : 0.8}
        style={{ pointerEvents: 'none' }}
      />

      {/* Path label */}
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
  onPathClick
}: TargetPathsRendererProps): (JSX.Element | null)[] {
  if (targetPaths.length === 0) return []

  return targetPaths.map(path =>
    renderTargetPath(path, selectedPathId, unitsToPixels, onPathClick)
  )
}

/**
 * React component wrapper for the target paths renderer
 */
export const TargetPathsRenderer: React.FC<TargetPathsRendererProps> = (props) => {
  return <>{renderTargetPaths(props)}</>
}

export default TargetPathsRenderer
