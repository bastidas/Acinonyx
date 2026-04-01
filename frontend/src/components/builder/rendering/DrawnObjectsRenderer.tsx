/**
 * Drawn Objects Renderer
 *
 * Pure function to render completed drawn objects (polygons, shapes).
 * Supports merge mode: optional merge styling and onMergePolygonClick.
 * Display points: objects pass points to draw (BuilderTab can pass transformed
 * points when merged to a link so the polygon follows the link during move group).
 *
 * Name labels are rendered via renderDrawnObjectLabels in a separate canvas layer
 * so they stack above links, joints, and trajectories regardless of form z-level.
 */

import React from 'react'
import { HighlightType } from '../types'
import { DrawnObjectsRendererProps, DrawnObject } from './types'

type DrawnObjectRenderProps = Omit<DrawnObjectsRendererProps, 'objects'>

interface PolygonRenderContext {
  pathData: string
  strokeWidth: number
  strokeColor: string
  fillOpacity: number
  strokeOpacity: number
  strokeDasharray: string | undefined
  polygonHighlightStyle: ReturnType<DrawnObjectRenderProps['getHighlightStyle']>
  showNameLabel: boolean
  labelTextColor: string
  labelFontSize: string
  labelFontWeight: string
  isUnmergeCandidate: boolean
}

function computePolygonRenderContext(
  obj: DrawnObject,
  props: DrawnObjectRenderProps
): PolygonRenderContext | null {
  if (obj.type !== 'polygon' || obj.points.length < 3) {
    return null
  }

  const {
    selectedIds,
    moveGroupIsActive,
    moveGroupDrObjectIds,
    getHighlightStyle,
    mergeMode = false,
    hoveredPolygonId = null
  } = props

  const pathData =
    obj.points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${props.unitsToPixels(p[0])} ${props.unitsToPixels(p[1])}`)
      .join(' ') + ' Z'

  const isSelected = selectedIds.includes(obj.id)
  const isInMoveGroup = moveGroupIsActive && moveGroupDrObjectIds.includes(obj.id)
  const isMerged = !!obj.mergedLinkName
  const isMergeHighlighted = mergeMode && hoveredPolygonId === obj.id
  const isHovered = hoveredPolygonId === obj.id
  const isUnmergeCandidate = mergeMode && isMerged
  const isBroken = isMerged && obj.contained_links_valid === false

  const polygonHighlightType: HighlightType = isInMoveGroup
    ? 'move_group'
    : isSelected || isMergeHighlighted || isHovered
      ? 'selected'
      : 'none'

  const polygonHighlightStyle = getHighlightStyle('polygon', polygonHighlightType, obj.strokeColor, obj.strokeWidth)

  const strokeWidth = isSelected
    ? Math.max(polygonHighlightStyle.strokeWidth + 1.5, 5)
    : isMergeHighlighted || isHovered
      ? Math.max(polygonHighlightStyle.strokeWidth + 1, 4)
      : polygonHighlightStyle.strokeWidth
  const strokeColor = isBroken
    ? '#d32f2f'
    : mergeMode && isMergeHighlighted
      ? isUnmergeCandidate
        ? '#f44336'
        : '#4caf50'
      : obj.strokeColor
  const fillOpacity = isSelected
    ? Math.min(Math.max(obj.fillOpacity + 0.2, 0.75), 0.9)
    : (mergeMode && isMergeHighlighted) || isHovered
      ? Math.min(obj.fillOpacity + 0.15, 0.7)
      : obj.fillOpacity
  const strokeOpacity = isSelected ? 0.98 : isHovered ? 0.95 : 1
  const strokeDasharray = isBroken ? '4 2' : undefined

  const showNameLabel = isSelected || isHovered || (mergeMode && isMergeHighlighted)
  const labelTextColor = mergeMode && isMergeHighlighted ? strokeColor : obj.strokeColor
  const labelFontSize = isSelected ? '13' : '11'
  const labelFontWeight = isSelected ? '700' : '600'

  return {
    pathData,
    strokeWidth,
    strokeColor,
    fillOpacity,
    strokeOpacity,
    strokeDasharray,
    polygonHighlightStyle,
    showNameLabel,
    labelTextColor,
    labelFontSize,
    labelFontWeight,
    isUnmergeCandidate
  }
}

/**
 * Renders a single drawn object (polygon path only; labels use renderDrawnObjectLabels).
 */
function renderDrawnObject(obj: DrawnObject, props: DrawnObjectRenderProps): JSX.Element | null {
  const ctx = computePolygonRenderContext(obj, props)
  if (!ctx) return null

  const {
    moveGroupIsActive,
    mergeMode = false,
    onMergePolygonClick,
    pointerEventsNoneForDrawPolygon = false,
    toolMode
  } = props

  const isSelected = props.selectedIds.includes(obj.id)
  const isMerged = !!obj.mergedLinkName

  const handleClick = (e: React.MouseEvent) => {
    if (mergeMode && onMergePolygonClick) {
      e.stopPropagation()
      onMergePolygonClick(obj.id, isMerged && e.shiftKey)
      return
    }
    e.stopPropagation()
    props.onObjectClick(obj.id, isSelected, e)
  }

  const handleDoubleClick = (e: React.MouseEvent) => {
    if (toolMode !== 'select' || mergeMode || !props.onObjectDoubleClick) return
    e.stopPropagation()
    props.onObjectDoubleClick(obj.id)
  }

  return (
    <g key={obj.id}>
      <path
        d={ctx.pathData}
        fill={obj.fillColor}
        stroke={ctx.strokeColor}
        strokeWidth={ctx.strokeWidth}
        strokeDasharray={ctx.strokeDasharray}
        fillOpacity={ctx.fillOpacity}
        strokeOpacity={ctx.strokeOpacity}
        filter={ctx.polygonHighlightStyle.filter}
        style={{
          cursor: moveGroupIsActive ? 'move' : 'pointer',
          pointerEvents: pointerEventsNoneForDrawPolygon ? 'none' : 'all'
        }}
        onMouseEnter={() => props.onMergePolygonHover?.(obj.id)}
        onMouseLeave={() => props.onMergePolygonHover?.(null)}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
      />
    </g>
  )
}

/**
 * Renders polygon name labels only — invoke in a late SVG layer so labels sit above links/joints.
 */
export function renderDrawnObjectLabels(props: DrawnObjectsRendererProps): (JSX.Element | null)[] {
  if (props.objects.length === 0) return []

  const { unitsToPixels } = props

  return props.objects.map(obj => {
    const ctx = computePolygonRenderContext(obj, props)
    if (!ctx || !ctx.showNameLabel) return null

    const px0 = unitsToPixels(obj.points[0][0])
    const py0 = unitsToPixels(obj.points[0][1])
    const nameStr = String(obj.name ?? '')

    return (
      <g key={`${obj.id}-label`} style={{ pointerEvents: 'none' }}>
        <rect
          x={px0 - (nameStr.length * 4 + 6)}
          y={py0 - 20}
          width={nameStr.length * 8 + 12}
          height={18}
          fill="rgba(255,255,255,0.88)"
          rx={4}
        />
        <text
          x={px0}
          y={py0 - 7}
          textAnchor="middle"
          fontSize={ctx.labelFontSize}
          fill={ctx.labelTextColor}
          fontWeight={ctx.labelFontWeight}
        >
          {obj.name}
          {ctx.isUnmergeCandidate ? ' (Shift+click to unmerge)' : ''}
        </text>
      </g>
    )
  })
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
  return (
    <>
      {renderDrawnObjects(props)}
      {renderDrawnObjectLabels(props)}
    </>
  )
}

export default DrawnObjectsRenderer
