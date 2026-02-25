/**
 * Grid Renderer
 *
 * Pure function to render the background grid on the canvas.
 * Grid extent is fixed from -CANVAS_GRID_EXTENT to +CANVAS_GRID_EXTENT in units
 * so zooming out shows a larger coordinate space.
 */

import React from 'react'
import { CANVAS_GRID_EXTENT } from '../constants'
import { GridRendererProps } from './types'

const GRID_MIN = -CANVAS_GRID_EXTENT
const GRID_MAX = CANVAS_GRID_EXTENT
const MAJOR_STEP = 20
const MINOR_OFFSET = 10

/**
 * Renders a grid with major and minor lines
 *
 * Grid layout:
 * - Major grid every 20 units (solid lines)
 * - Minor grid every 10 units (dashed lines)
 * - Labels on major grid lines
 * - Extent: -500 to +500 units on both axes (for zoom-out)
 */
export function renderGrid({
  canvasDimensions: _canvasDimensions,
  darkMode,
  unitsToPixels,
  pixelsToUnits: _pixelsToUnits
}: GridRendererProps): JSX.Element[] {
  const lines: JSX.Element[] = []
  const yOriginPx = unitsToPixels(0)
  const gridMinPx = unitsToPixels(GRID_MIN)
  const gridMaxPx = unitsToPixels(GRID_MAX)

  const gridMajorColor = darkMode ? '#444444' : '#dddddd'
  const gridMinorColor = darkMode ? '#333333' : '#eeeeee'
  const gridTextColor = darkMode ? '#666666' : '#999999'

  // Major grid every 20 units - vertical lines
  for (let i = Math.floor(GRID_MIN / MAJOR_STEP) * MAJOR_STEP; i <= GRID_MAX; i += MAJOR_STEP) {
    const x = unitsToPixels(i)
    lines.push(
      <line
        key={`v-major-${i}`}
        x1={x}
        y1={gridMinPx}
        x2={x}
        y2={gridMaxPx}
        stroke={gridMajorColor}
        strokeWidth={1}
      />
    )
    lines.push(
      <text key={`vl-${i}`} x={x + 2} y={yOriginPx - 2} fontSize="10" fill={gridTextColor}>
        {i}
      </text>
    )
  }

  // Major grid every 20 units - horizontal lines
  for (let i = Math.floor(GRID_MIN / MAJOR_STEP) * MAJOR_STEP; i <= GRID_MAX; i += MAJOR_STEP) {
    const y = unitsToPixels(i)
    lines.push(
      <line
        key={`h-major-${i}`}
        x1={gridMinPx}
        y1={y}
        x2={gridMaxPx}
        y2={y}
        stroke={gridMajorColor}
        strokeWidth={1}
      />
    )
    if (i !== 0) {
      lines.push(
        <text key={`hl-${i}`} x={unitsToPixels(1)} y={y - 2} fontSize="10" fill={gridTextColor}>
          {i}
        </text>
      )
    }
  }

  // Minor grid every 10 units (offset from major) - vertical lines
  for (let i = Math.floor(GRID_MIN / MAJOR_STEP) * MAJOR_STEP + MINOR_OFFSET; i <= GRID_MAX; i += MAJOR_STEP) {
    const x = unitsToPixels(i)
    lines.push(
      <line
        key={`v-minor-${i}`}
        x1={x}
        y1={gridMinPx}
        x2={x}
        y2={gridMaxPx}
        stroke={gridMinorColor}
        strokeWidth={0.5}
        strokeDasharray="2,4"
      />
    )
  }

  // Minor grid every 10 units (offset from major) - horizontal lines
  for (let i = Math.floor(GRID_MIN / MAJOR_STEP) * MAJOR_STEP + MINOR_OFFSET; i <= GRID_MAX; i += MAJOR_STEP) {
    const y = unitsToPixels(i)
    lines.push(
      <line
        key={`h-minor-${i}`}
        x1={gridMinPx}
        y1={y}
        x2={gridMaxPx}
        y2={y}
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
