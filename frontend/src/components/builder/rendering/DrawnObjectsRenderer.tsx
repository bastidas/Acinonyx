/**
 * Drawn Objects Renderer
 *
 * Pure function to render completed drawn objects (polygons, shapes).
 */

import React from 'react'
import { HighlightType } from '../types'
import { DrawnObjectsRendererProps, DrawnObject } from './types'

/**
 * Renders a single drawn object (polygon)
 */
function renderDrawnObject(
  obj: DrawnObject,
  props: Omit<DrawnObjectsRendererProps, 'objects'>
): JSX.Element | null {
  const {
    selectedIds,
    moveGroupIsActive,
    moveGroupDrObjectIds,
    toolMode,
    getHighlightStyle,
    unitsToPixels,
    onObjectClick
  } = props

  if (obj.type !== 'polygon' || obj.points.length < 3) {
    return null
  }

  const pathData = obj.points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
  ).join(' ') + ' Z'

  const isSelected = selectedIds.includes(obj.id)
  const isInMoveGroup = moveGroupIsActive && moveGroupDrObjectIds.includes(obj.id)

  // Determine highlight type for glow effect
  const polygonHighlightType: HighlightType = isInMoveGroup
    ? 'move_group'
    : isSelected
      ? 'selected'
      : 'none'

  // Get highlight styling - keeps original fill, adds glow outline in object's color
  const polygonHighlightStyle = getHighlightStyle('polygon', polygonHighlightType, obj.strokeColor, obj.strokeWidth)

  return (
    <g key={obj.id}>
      <path
        d={pathData}
        fill={obj.fillColor}  // Keep original fill color
        stroke={obj.strokeColor}  // Keep original stroke color
        strokeWidth={polygonHighlightStyle.strokeWidth}
        fillOpacity={obj.fillOpacity}
        filter={polygonHighlightStyle.filter}
        style={{ cursor: moveGroupIsActive ? 'move' : 'pointer' }}
        onClick={(e) => {
          // In merge mode, let the canvas handler process this click
          if (toolMode === 'merge') {
            return
          }

          e.stopPropagation()
          onObjectClick(obj.id, isSelected)
        }}
      />
      {/* Show object name on selection */}
      {isSelected && (
        <text
          x={unitsToPixels(obj.points[0][0])}
          y={unitsToPixels(obj.points[0][1]) - 8}
          fontSize="10"
          fill={obj.strokeColor}
          fontWeight="500"
        >
          {obj.name}
        </text>
      )}
    </g>
  )
}

/**
 * Renders all drawn objects
 */
export function renderDrawnObjects(props: DrawnObjectsRendererProps): (JSX.Element | null)[] {
  if (props.objects.length === 0) return []

  return props.objects.map(obj => renderDrawnObject(obj, props))
}

/**
 * React component wrapper for the drawn objects renderer
 */
export const DrawnObjectsRenderer: React.FC<DrawnObjectsRendererProps> = (props) => {
  return <>{renderDrawnObjects(props)}</>
}

export default DrawnObjectsRenderer
