/**
 * Grid Renderer
 *
 * Pure function to render the background grid on the canvas.
 */

import React from 'react'
import { GridRendererProps } from './types'

/**
 * Renders a grid with major and minor lines
 *
 * Grid layout:
 * - Major grid every 20 units (solid lines)
 * - Minor grid every 10 units (dashed lines)
 * - Labels on major grid lines
 */
export function renderGrid({
  canvasDimensions,
  darkMode,
  unitsToPixels,
  pixelsToUnits
}: GridRendererProps): JSX.Element[] {
  const lines: JSX.Element[] = []
  const maxUnitsX = pixelsToUnits(canvasDimensions.width)
  const maxUnitsY = pixelsToUnits(canvasDimensions.height)

  // Get colors based on dark mode
  const gridMajorColor = darkMode ? '#444444' : '#dddddd'
  const gridMinorColor = darkMode ? '#333333' : '#eeeeee'
  const gridTextColor = darkMode ? '#666666' : '#999999'

  // Major grid every 20 units - vertical lines
  for (let i = 0; i <= Math.ceil(maxUnitsX); i += 20) {
    lines.push(
      <line
        key={`v-major-${i}`}
        x1={unitsToPixels(i)}
        y1={0}
        x2={unitsToPixels(i)}
        y2={canvasDimensions.height + 1}
        stroke={gridMajorColor}
        strokeWidth={1}
      />
    )
    if (i > 0) {
      lines.push(
        <text key={`vl-${i}`} x={unitsToPixels(i) + 2} y={12} fontSize="10" fill={gridTextColor}>
          {i}
        </text>
      )
    }
  }

  // Major grid every 20 units - horizontal lines
  for (let i = 0; i <= Math.ceil(maxUnitsY); i += 20) {
    lines.push(
      <line
        key={`h-major-${i}`}
        x1={0}
        y1={unitsToPixels(i)}
        x2={canvasDimensions.width + 1}
        y2={unitsToPixels(i)}
        stroke={gridMajorColor}
        strokeWidth={1}
      />
    )
    if (i > 0) {
      lines.push(
        <text key={`hl-${i}`} x={2} y={unitsToPixels(i) - 2} fontSize="10" fill={gridTextColor}>
          {i}
        </text>
      )
    }
  }

  // Minor grid every 10 units (offset from major) - vertical lines
  for (let i = 10; i <= Math.ceil(maxUnitsX); i += 20) {
    lines.push(
      <line
        key={`v-minor-${i}`}
        x1={unitsToPixels(i)}
        y1={0}
        x2={unitsToPixels(i)}
        y2={canvasDimensions.height + 1}
        stroke={gridMinorColor}
        strokeWidth={0.5}
        strokeDasharray="2,4"
      />
    )
  }

  // Minor grid every 10 units (offset from major) - horizontal lines
  for (let i = 10; i <= Math.ceil(maxUnitsY); i += 20) {
    lines.push(
      <line
        key={`h-minor-${i}`}
        x1={0}
        y1={unitsToPixels(i)}
        x2={canvasDimensions.width + 1}
        y2={unitsToPixels(i)}
        stroke={gridMinorColor}
        strokeWidth={0.5}
        strokeDasharray="2,4"
      />
    )
  }

  return lines
}

/**
 * React component wrapper for the grid renderer
 */
export const GridRenderer: React.FC<GridRendererProps> = (props) => {
  return <>{renderGrid(props)}</>
}

export default GridRenderer
