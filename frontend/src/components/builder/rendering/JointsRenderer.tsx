/**
 * Joints Renderer
 *
 * Pure function to render joint circles with highlighting and labels.
 */

import React from 'react'
import { HighlightType } from '../types'
import { JointsRendererProps, JointRenderData } from './types'

/**
 * Renders a single joint (circle with optional label and type indicator)
 */
function renderJoint(
  joint: JointRenderData,
  props: Omit<JointsRendererProps, 'joints'>
): JSX.Element {
  const {
    jointSize,
    jointColors,
    darkMode,
    showJointLabels,
    moveGroupIsActive,
    toolMode,
    getHighlightStyle,
    unitsToPixels,
    onJointHover,
    onJointDoubleClick
  } = props

  const {
    name,
    type,
    position,
    color,
    isSelected,
    isInMoveGroup,
    isHovered,
    isDragging,
    isMergeTarget
  } = joint

  const isStatic = type === 'Static'
  const isCrank = type === 'Crank'

  // Highlight state flags
  const showMergeHighlight = isMergeTarget || (isDragging && isMergeTarget)
  const showMoveGroupHighlight = isInMoveGroup && !showMergeHighlight

  // Use theme colors for joint fill - keep original colors, selection indicated by glow
  const staticColor = jointColors.static  // #E74C3C - red for fixed joints
  const crankColor = jointColors.crank    // #F39C12 - amber for crank joints

  // Joint fill color based on type (NOT changed by selection - selection uses glow)
  const jointFillColor = isStatic
    ? staticColor
    : isCrank
      ? crankColor
      : color

  // Determine highlight type for glow effect
  const highlightType: HighlightType = showMergeHighlight
    ? 'merge'
    : showMoveGroupHighlight
      ? 'move_group'
      : (isSelected || isDragging)
        ? 'selected'
        : isHovered
          ? 'hovered'
          : 'none'

  // Get highlight styling - joints glow in their original color
  const defaultStroke = darkMode ? '#333333' : '#FFFFFF'
  const highlightStyle = getHighlightStyle('joint', highlightType, jointFillColor, 2)

  // Stroke color: use object color for glow, but keep default stroke for the actual outline
  const strokeColor = highlightType === 'none'
    ? defaultStroke
    : highlightType === 'move_group' || highlightType === 'merge'
      ? highlightStyle.stroke
      : defaultStroke  // Keep white/dark outline, the glow provides the color

  // Calculate radius based on state
  const radius = isDragging
    ? jointSize * 1.5
    : (isHovered || isSelected || isMergeTarget || isInMoveGroup)
      ? jointSize * 1.25
      : jointSize

  const cx = unitsToPixels(position[0])
  const cy = unitsToPixels(position[1])

  return (
    <g key={name}>
      <circle
        cx={cx}
        cy={cy}
        r={radius}
        fill={jointFillColor}
        stroke={strokeColor}
        strokeWidth={highlightStyle.strokeWidth}
        filter={highlightStyle.filter}
        style={{ cursor: moveGroupIsActive ? 'move' : (toolMode === 'select' ? 'grab' : 'pointer') }}
        onMouseEnter={() => onJointHover(name)}
        onMouseLeave={() => onJointHover(null)}
        onDoubleClick={(e) => {
          if (toolMode === 'select') {
            e.stopPropagation()
            onJointDoubleClick(name)
          }
        }}
      />

      {/* Joint label - always show on hover/drag, or when showJointLabels is enabled */}
      {(showJointLabels || isHovered || isDragging || isSelected) && (
        <text
          x={cx}
          y={cy - 14}
          textAnchor="middle"
          fontSize="11"
          fontWeight={isHovered || isDragging ? 'bold' : 'normal'}
          fill={showMergeHighlight ? jointColors.mergeHighlight : (darkMode ? '#e0e0e0' : '#333')}
          style={{ pointerEvents: 'none' }}
        >
          {name}
        </text>
      )}

      {/* Static joint indicator (rectangle below) */}
      {isStatic && !isDragging && (
        <rect
          x={cx - 4}
          y={cy + 10}
          width={8}
          height={4}
          fill="#e74c3c"
        />
      )}

      {/* Crank joint indicator (triangle below) */}
      {isCrank && !isDragging && (
        <path
          d={`M ${cx} ${cy + 10} L ${cx - 6} ${cy + 18} L ${cx + 6} ${cy + 18} Z`}
          fill="#f39c12"
        />
      )}
    </g>
  )
}

/**
 * Renders all joints
 */
export function renderJoints(props: JointsRendererProps): JSX.Element[] {
  return props.joints.map(joint => renderJoint(joint, props))
}

/**
 * React component wrapper for the joints renderer
 */
export const JointsRenderer: React.FC<JointsRendererProps> = (props) => {
  return <>{renderJoints(props)}</>
}

export default JointsRenderer
