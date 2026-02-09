/**
 * Drawn Objects Renderer
 *
 * Pure function to render completed drawn objects (polygons, shapes).
 * Supports merge mode: optional merge styling and onMergePolygonClick.
 * Display points: objects pass points to draw (BuilderTab can pass transformed
 * points when merged to a link so the polygon follows the link during move group).
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
    onObjectClick,
    mergeMode = false,
    hoveredPolygonId = null,
    onMergePolygonClick
  } = props

  if (obj.type !== 'polygon' || obj.points.length < 3) {
    return null
  }

  const pathData = obj.points.map((p, i) =>
    `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
  ).join(' ') + ' Z'

  const isSelected = selectedIds.includes(obj.id)
  const isInMoveGroup = moveGroupIsActive && moveGroupDrObjectIds.includes(obj.id)
  const isMerged = !!obj.mergedLinkName
  const isMergeHighlighted = mergeMode && hoveredPolygonId === obj.id
  const isUnmergeCandidate = mergeMode && isMerged

  // Determine highlight type for glow effect
  const polygonHighlightType: HighlightType = isInMoveGroup
    ? 'move_group'
    : (isSelected || isMergeHighlighted)
      ? 'selected'
      : 'none'

  // Get highlight styling - keeps original fill, adds glow outline in object's color
  const polygonHighlightStyle = getHighlightStyle('polygon', polygonHighlightType, obj.strokeColor, obj.strokeWidth)

  // Merge-mode styling: stroke/fill for hover (red = unmerge, green = merge)
  const strokeWidth = mergeMode && isMergeHighlighted
    ? Math.max(polygonHighlightStyle.strokeWidth, 4)
    : polygonHighlightStyle.strokeWidth
  const strokeColor = mergeMode && isMergeHighlighted
    ? (isUnmergeCandidate ? '#f44336' : '#4caf50')
    : obj.strokeColor
  const fillOpacity = mergeMode && isMergeHighlighted
    ? Math.min(obj.fillOpacity + 0.15, 0.6)
    : obj.fillOpacity

  const handleClick = (e: React.MouseEvent) => {
    if (mergeMode && onMergePolygonClick) {
      e.stopPropagation()
      onMergePolygonClick(obj.id, isMerged)
      return
    }
    e.stopPropagation()
    onObjectClick(obj.id, isSelected)
  }

  return (
    <g key={obj.id}>
      <path
        d={pathData}
        fill={obj.fillColor}
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        fillOpacity={fillOpacity}
        filter={polygonHighlightStyle.filter}
        style={{ cursor: moveGroupIsActive ? 'move' : 'pointer', pointerEvents: 'all' }}
        onMouseEnter={mergeMode ? () => props.onMergePolygonHover?.(obj.id) : undefined}
        onMouseLeave={mergeMode ? () => props.onMergePolygonHover?.(null) : undefined}
        onClick={handleClick}
      />
      {/* Show object name on selection or in merge mode when hovered */}
      {(isSelected || (mergeMode && isMergeHighlighted)) && (
        <text
          x={unitsToPixels(obj.points[0][0])}
          y={unitsToPixels(obj.points[0][1]) - 8}
          fontSize="10"
          fill={mergeMode && isMergeHighlighted ? strokeColor : obj.strokeColor}
          fontWeight="500"
          style={{ pointerEvents: 'none' }}
        >
          {obj.name}{isUnmergeCandidate ? ' (click to unmerge)' : ''}
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
