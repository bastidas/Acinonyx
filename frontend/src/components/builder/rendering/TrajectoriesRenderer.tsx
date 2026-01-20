/**
 * Trajectories Renderer
 *
 * Pure function to render trajectory dots for simulation results.
 */

import React from 'react'
import { TrajectoriesRendererProps, TrajectoryRenderData, Position } from './types'
import { isValidPosition, hasMovement } from './utils'

/**
 * Renders a single trajectory (dots and/or lines showing joint path during simulation)
 */
function renderTrajectory(
  trajectory: TrajectoryRenderData,
  props: Omit<TrajectoriesRendererProps, 'trajectories'>
): JSX.Element | null {
  const {
    trajectoryDotSize,
    trajectoryStyle,
    trajectoryColorCycle,
    jointColors,
    unitsToPixels,
    getCyclicColor
  } = props

  const { jointName, positions, jointType, showPath } = trajectory

  // Skip if no trajectory data or not visible
  if (!positions || !Array.isArray(positions) || positions.length === 0) {
    return null
  }

  // Show trajectories for Revolute and Crank joints (they move during simulation)
  if (jointType !== 'Revolute' && jointType !== 'Crank') return null

  // Check if this joint has show_path disabled
  if (showPath === false) return null

  // Validate first position exists and has coordinates
  const firstPos = positions[0]
  if (!isValidPosition(firstPos)) {
    return null
  }

  // Check if joint actually moves (has varying positions)
  const trajectoryHasMovement = positions.length > 1 && positions.some((pos, i) =>
    i > 0 && isValidPosition(pos) && hasMovement(pos, firstPos)
  )

  // Get color based on joint type for non-moving indicators
  const jointColor = jointType === 'Crank' ? jointColors.crank : jointColors.pivot

  // Filter out any invalid positions for rendering
  const validPositions = positions.filter(isValidPosition) as Position[]

  if (validPositions.length === 0) return null

  // Get color for a step
  const getTrajectoryColor = (stepIndex: number, totalSteps: number) => {
    return getCyclicColor(stepIndex, totalSteps, trajectoryColorCycle)
  }

  return (
    <g key={`trajectory-${jointName}`}>
      {trajectoryHasMovement && validPositions.length > 1 ? (
        <>
          {/* Draw trajectory path as a line (if trajectoryStyle includes line) */}
          {(trajectoryStyle === 'line' || trajectoryStyle === 'both') && (
            <path
              d={validPositions
                .map((pos, i) =>
                  `${i === 0 ? 'M' : 'L'} ${unitsToPixels(pos[0])} ${unitsToPixels(pos[1])}`
                ).join(' ')}
              fill="none"
              stroke="rgba(33, 150, 243, 0.3)"
              strokeWidth={2}
              strokeDasharray="4,2"
            />
          )}

          {/* Draw dots at each timestep (if trajectoryStyle includes dots) */}
          {(trajectoryStyle === 'dots' || trajectoryStyle === 'both') && validPositions.map((pos, stepIndex) => (
            <circle
              key={`${jointName}-step-${stepIndex}`}
              cx={unitsToPixels(pos[0])}
              cy={unitsToPixels(pos[1])}
              r={trajectoryDotSize}
              fill={getTrajectoryColor(stepIndex, validPositions.length)}
              stroke="#fff"
              strokeWidth={1}
              opacity={0.8}
            >
              <title>{`${jointName} step ${stepIndex + 1}/${validPositions.length}: (${pos[0]?.toFixed(1) ?? '?'}, ${pos[1]?.toFixed(1) ?? '?'})`}</title>
            </circle>
          ))}
        </>
      ) : (
        /* Non-moving joint: show a stationary indicator ring */
        <circle
          cx={unitsToPixels(validPositions[0][0])}
          cy={unitsToPixels(validPositions[0][1])}
          r={10}
          fill="none"
          stroke={jointColor}
          strokeWidth={2}
          strokeDasharray="3,3"
          opacity={0.6}
        >
          <title>{`${jointName}: stationary (not constrained to moving joints)`}</title>
        </circle>
      )}
    </g>
  )
}

/**
 * Renders all trajectories
 */
export function renderTrajectories(props: TrajectoriesRendererProps): (JSX.Element | null)[] {
  return props.trajectories.map(trajectory => renderTrajectory(trajectory, props))
}

/**
 * React component wrapper for the trajectories renderer
 */
export const TrajectoriesRenderer: React.FC<TrajectoriesRendererProps> = (props) => {
  return <>{renderTrajectories(props)}</>
}

export default TrajectoriesRenderer
