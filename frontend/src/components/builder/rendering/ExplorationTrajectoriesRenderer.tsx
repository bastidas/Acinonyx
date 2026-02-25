/**
 * Exploration Trajectories Renderer
 *
 * Renders all valid exploration trajectories at low opacity (baseOpacity);
 * the hovered sample's trajectory at higher opacity (hoveredOpacity).
 * When colormap is enabled, each trajectory uses the same color as its sample dot.
 */

import React from 'react'
import type { ExplorationTrajectoriesRendererProps } from './types'
import {
  getExplorationColormapColor,
  positionToAngleAndRadialT,
  EXPLORE_CENTER_COLOR
} from '../helpers/explorationColormap'

const DEFAULT_BASE_OPACITY = 0.16
const DEFAULT_HOVERED_OPACITY = 1
const BASE_STROKE_WIDTH = 2
const HOVERED_STROKE_WIDTH = 4
const DEFAULT_STROKE_GREY = 'rgba(120, 120, 120, 0.9)'
const DEFAULT_STROKE_GREY_HOVERED = 'rgba(120, 120, 120, 0.95)'
/** Invisible stroke width for trajectory hit-testing (hover/click to select corresponding dot). */
const HIT_STROKE_WIDTH = 16
/** Attribute on trajectory paths for backwards selection (hover/click trajectory highlights and selects the dot). */
export const DATA_EXPLORE_SAMPLE_INDEX = 'data-explore-sample-index'

export function renderExplorationTrajectories(props: ExplorationTrajectoriesRendererProps): React.ReactNode {
  const {
    samples,
    hoveredIndex,
    hoveredPosition,
    hoveredFromTrajectoryPath,
    exploreNodeId,
    unitsToPixels,
    jointNamesToShow,
    baseOpacity = DEFAULT_BASE_OPACITY,
    hoveredOpacity = DEFAULT_HOVERED_OPACITY,
    exploreColormapEnabled,
    exploreColormapType = 'rainbow',
    exploreCenter,
    exploreRadius = 20,
    exploreMode
  } = props

  const positionMatch = (a: [number, number], b: [number, number], tol = 0.01) =>
    Math.abs(a[0] - b[0]) < tol && Math.abs(a[1] - b[1]) < tol
  const isHovered = (position: [number, number], index: number) =>
    exploreMode === 'combinatorial' && hoveredFromTrajectoryPath
      ? hoveredIndex === index
      : exploreMode === 'combinatorial' && hoveredPosition != null
        ? positionMatch(position, hoveredPosition)
        : hoveredIndex === index

  const validWithTrajectory = samples
    .map((s, i) => ({ ...s, index: i }))
    .filter(s => s.valid && s.trajectory != null) as Array<typeof samples[0] & { index: number; trajectory: NonNullable<typeof samples[0]['trajectory']> }>

  if (validWithTrajectory.length === 0) return null

  // Single/path: color by first-node position. Combinatorial: color by second-node position (exploreCenter is second center); M trajectories per color.
  const useColormap =
    exploreColormapEnabled &&
    exploreCenter != null &&
    exploreRadius > 0

  const getStrokeColor = (position: [number, number]): string => {
    if (!useColormap) return DEFAULT_STROKE_GREY
    const { angleT, radialT } = positionToAngleAndRadialT(position, exploreCenter, exploreRadius)
    return radialT < 0.01 ? EXPLORE_CENTER_COLOR : getExplorationColormapColor(angleT, radialT, exploreColormapType)
  }

  const showJoint = (jointName: string, jt: string | undefined, forceShow = false): boolean => {
    if (jt !== 'Crank' && jt !== 'Revolute') return false
    if (forceShow) return true
    if (jointNamesToShow != null && jointNamesToShow.length > 0) {
      return jointNamesToShow.includes(jointName)
    }
    return true
  }

  const elements: React.ReactNode[] = []

  const hoveredSamples =
    exploreMode === 'combinatorial' && hoveredFromTrajectoryPath
      ? (() => { const h = validWithTrajectory.find(s => s.index === hoveredIndex); return h ? [h] : [] })()
      : exploreMode === 'combinatorial' && hoveredPosition != null
        ? validWithTrajectory.filter(s => positionMatch(s.position, hoveredPosition))
        : (() => { const h = validWithTrajectory.find(s => s.index === hoveredIndex); return h ? [h] : [] })()

  // Draw non-hovered trajectories at base opacity (hit area first, then visible path)
  validWithTrajectory.forEach(({ trajectory, index, position }) => {
    if (isHovered(position, index)) return
    const stroke = useColormap ? getStrokeColor(position) : DEFAULT_STROKE_GREY
    const jointTypes = trajectory.jointTypes || {}
    Object.entries(trajectory.trajectories).forEach(([jointName, positions]) => {
      const jt = jointTypes[jointName]
      if (!showJoint(jointName, jt)) return
      if (!positions || positions.length < 2) return
      const d = positions
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`)
        .join(' ')
      elements.push(
        <g key={`${index}-${jointName}`}>
          <path
            d={d}
            fill="none"
            stroke="transparent"
            strokeWidth={HIT_STROKE_WIDTH}
            pointerEvents="stroke"
            {...{ [DATA_EXPLORE_SAMPLE_INDEX]: index }}
          />
          <path
            d={d}
            fill="none"
            stroke={stroke}
            strokeWidth={BASE_STROKE_WIDTH}
            opacity={baseOpacity}
            pointerEvents="stroke"
            {...{ [DATA_EXPLORE_SAMPLE_INDEX]: index }}
          />
        </g>
      )
    })
  })

  // Draw hovered trajectory(ies) on top. Only show joints with show_path, except in combinatorial always show first node (exploreNodeId).
  hoveredSamples.forEach((hovered) => {
    const stroke = useColormap ? getStrokeColor(hovered.position) : DEFAULT_STROKE_GREY_HOVERED
    const jointTypes = hovered.trajectory.jointTypes || {}
    Object.entries(hovered.trajectory.trajectories).forEach(([jointName, positions]) => {
      const jt = jointTypes[jointName]
      const forceShowFirstNode = exploreMode === 'combinatorial' && exploreNodeId != null && jointName === exploreNodeId
      if (!showJoint(jointName, jt, forceShowFirstNode)) return
      if (!positions || positions.length < 2) return
      const d = positions
        .map((p, i) => `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`)
        .join(' ')
      const idx = hovered.index
      elements.push(
        <g key={`hovered-${idx}-${jointName}`}>
          <path
            d={d}
            fill="none"
            stroke="transparent"
            strokeWidth={HIT_STROKE_WIDTH}
            pointerEvents="stroke"
            {...{ [DATA_EXPLORE_SAMPLE_INDEX]: idx }}
          />
          <path
            d={d}
            fill="none"
            stroke={stroke}
            strokeWidth={HOVERED_STROKE_WIDTH}
            opacity={hoveredOpacity}
            pointerEvents="stroke"
            {...{ [DATA_EXPLORE_SAMPLE_INDEX]: idx }}
          />
        </g>
      )
    })
  })

  return <g data-layer="exploration-trajectories">{elements}</g>
}

export const ExplorationTrajectoriesRenderer: React.FC<ExplorationTrajectoriesRendererProps> = (props) => (
  <>{renderExplorationTrajectories(props)}</>
)

export default ExplorationTrajectoriesRenderer
