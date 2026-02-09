/**
 * Links Renderer
 *
 * Pure function to render link lines between joints.
 * Supports merge mode: optional onMergeLinkClick for single-click in merge mode.
 * Double-click for edit is unchanged.
 */

import React from 'react'
import { HighlightType } from '../types'
import { LinksRendererProps, LinkRenderData } from './types'

/**
 * Renders a single link (line between two joints with label)
 */
function renderLink(
  link: LinkRenderData,
  props: Omit<LinksRendererProps, 'links'>
): JSX.Element {
  const {
    linkThickness,
    darkMode,
    showLinkLabels,
    moveGroupIsActive,
    moveGroupIsDragging,
    toolMode,
    getHighlightStyle,
    unitsToPixels,
    onLinkHover,
    onLinkDoubleClick,
    mergeMode = false,
    onMergeLinkClick,
    hitAreaStrokeWidthPx
  } = props

  const {
    name,
    position0,
    position1,
    color,
    isGround,
    isSelected,
    isInMoveGroup,
    isHovered,
    isStretching
  } = link

  // Base link color - stretching links are red, ground links use gray if no custom color
  const baseLinkColor = isStretching
    ? '#ff0000'  // Red for invalid/stretching links
    : isGround && !color
      ? '#7f7f7f'  // Default gray for ground links
      : color

  // Determine highlight type for glow effect
  const linkHighlightType: HighlightType = isInMoveGroup
    ? 'move_group'
    : isSelected
      ? 'selected'
      : isHovered
        ? 'hovered'
        : 'none'

  // Get highlight styling - links glow in their original color
  const linkHighlightStyle = getHighlightStyle('link', linkHighlightType, baseLinkColor, linkThickness)

  // Use base stroke width - the glow filter handles the visual effect without changing line width
  // This ensures ground links don't shrink on hover
  const effectiveStrokeWidth = linkThickness

  // Calculate midpoint for label
  const midX = (position0[0] + position1[0]) / 2
  const midY = (position0[1] + position1[1]) / 2

  const x1 = unitsToPixels(position0[0])
  const y1 = unitsToPixels(position0[1])
  const x2 = unitsToPixels(position1[0])
  const y2 = unitsToPixels(position1[1])

  const handleClick = (e: React.MouseEvent) => {
    if (mergeMode && onMergeLinkClick) {
      e.stopPropagation()
      onMergeLinkClick(name)
    }
  }

  const eventHandlers = {
    onMouseEnter: () => !moveGroupIsDragging && onLinkHover(name),
    onMouseLeave: () => onLinkHover(null),
    onClick: handleClick,
    onDoubleClick: (e: React.MouseEvent) => {
      if (toolMode === 'select') {
        e.stopPropagation()
        onLinkDoubleClick(name)
      }
    }
  }

  const lineStyle = { cursor: moveGroupIsActive ? 'move' : 'pointer' as const }

  return (
    <g key={name}>
      {hitAreaStrokeWidthPx != null && (
        <line
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="transparent"
          strokeWidth={hitAreaStrokeWidthPx}
          strokeLinecap="round"
          style={lineStyle}
          {...eventHandlers}
        />
      )}
      <line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={baseLinkColor}
        strokeWidth={effectiveStrokeWidth}
        strokeLinecap="round"
        filter={linkHighlightStyle.filter}
        style={hitAreaStrokeWidthPx != null ? { ...lineStyle, pointerEvents: 'none' } : lineStyle}
        {...(hitAreaStrokeWidthPx == null ? eventHandlers : {})}
      />

      {/* Link label - show on hover/selected, or when showLinkLabels is enabled */}
      {(showLinkLabels || isHovered || isSelected) && (
        <g>
          {/* Background for readability */}
          <rect
            x={unitsToPixels(midX) - name.length * 3.5 - 4}
            y={unitsToPixels(midY) - 8}
            width={name.length * 7 + 8}
            height={14}
            fill={darkMode ? 'rgba(30, 30, 30, 0.85)' : 'rgba(255, 255, 255, 0.85)'}
            rx={3}
            style={{ pointerEvents: 'none' }}
          />
          <text
            x={unitsToPixels(midX)}
            y={unitsToPixels(midY) + 3}
            textAnchor="middle"
            fontSize="10"
            fontWeight={isHovered ? 'bold' : 'normal'}
            fill={isHovered || isSelected ? baseLinkColor : (darkMode ? '#b0b0b0' : '#555')}
            style={{ pointerEvents: 'none' }}
          >
            {name}
          </text>
        </g>
      )}
    </g>
  )
}

/**
 * Renders all links
 */
export function renderLinks(props: LinksRendererProps): JSX.Element[] {
  return props.links.map(link => renderLink(link, props))
}

/**
 * React component wrapper for the links renderer
 */
export const LinksRenderer: React.FC<LinksRendererProps> = (props) => {
  return <>{renderLinks(props)}</>
}

export default LinksRenderer
