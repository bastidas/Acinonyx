/**
 * Exploration Dots Renderer
 *
 * Renders small circles at each explore-region sample: green = valid trajectory,
 * red = invalid (or when colormap enabled: color by angle/radius, invalid = grey).
 * Hovered sample is drawn slightly larger.
 */

import React from 'react'
import type { ExplorationDotsRendererProps } from './types'
import {
  getExplorationColormapColor,
  positionToAngleAndRadialT,
  EXPLORE_INVALID_GREY,
  EXPLORE_CENTER_COLOR
} from '../helpers/explorationColormap'

const DOT_RADIUS = 3
const HOVERED_RADIUS = 5
const DEFAULT_VALID_GREEN = '#2e7d32'
const DEFAULT_INVALID_RED = '#c62828'

export function renderExplorationDots(props: ExplorationDotsRendererProps): React.ReactNode {
  const {
    samples,
    hoveredIndex,
    hoveredPosition,
    unitsToPixels,
    exploreColormapEnabled,
    exploreColormapType = 'rainbow',
    exploreCenter,
    exploreRadius = 20,
    exploreMode
  } = props
  if (!samples.length) return null

  // Use colormap when enabled (single, path, or combinatorial). Combinatorial: exploreCenter is second center, one dot per unique second position.
  const useColormap =
    exploreColormapEnabled &&
    exploreCenter != null &&
    exploreRadius > 0

  const positionMatch = (a: [number, number], b: [number, number], tol = 0.01) =>
    Math.abs(a[0] - b[0]) < tol && Math.abs(a[1] - b[1]) < tol

  return (
    <g data-layer="exploration-dots">
      {samples.map((sample, i) => {
        const [x, y] = sample.position
        const isHovered =
          hoveredPosition != null
            ? positionMatch(sample.position, hoveredPosition)
            : hoveredIndex === i
        const r = isHovered ? HOVERED_RADIUS : DOT_RADIUS
        let fill: string
        if (useColormap) {
          if (sample.valid) {
            const { angleT, radialT } = positionToAngleAndRadialT(
              sample.position,
              exploreCenter,
              exploreRadius
            )
            fill = radialT < 0.01 ? EXPLORE_CENTER_COLOR : getExplorationColormapColor(angleT, radialT, exploreColormapType)
          } else {
            fill = EXPLORE_INVALID_GREY
          }
        } else if (exploreMode === 'combinatorial') {
          fill = sample.valid ? EXPLORE_INVALID_GREY : '#bdbdbd'
        } else {
          // Single/path with colormap off
          fill = sample.valid ? DEFAULT_VALID_GREEN : DEFAULT_INVALID_RED
        }
        const stroke = isHovered ? '#fff' : 'none'
        const strokeWidth = isHovered ? 1.5 : 0
        const isInvalid = !sample.valid
        return (
          <circle
            key={i}
            cx={unitsToPixels(x)}
            cy={unitsToPixels(y)}
            r={r}
            fill={fill}
            stroke={stroke}
            strokeWidth={strokeWidth}
            opacity={isInvalid ? 0.3 : 1}
          >
            <title>{sample.valid ? `Valid at (${x.toFixed(1)}, ${y.toFixed(1)})` : `Invalid at (${x.toFixed(1)}, ${y.toFixed(1)})`}</title>
          </circle>
        )
      })}
    </g>
  )
}

export const ExplorationDotsRenderer: React.FC<ExplorationDotsRendererProps> = (props) => (
  <>{renderExplorationDots(props)}</>
)

export default ExplorationDotsRenderer
