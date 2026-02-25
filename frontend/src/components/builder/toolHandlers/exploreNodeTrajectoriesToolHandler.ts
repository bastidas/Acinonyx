/**
 * Explore Node Trajectories tool handler
 *
 * Click a joint (non-fixed, or moving static) to run exploration: batch-simulate
 * positions in a circle, then show valid/invalid dots and preview trajectories.
 * Moving static joints have no self-trajectory but repositioning them changes other
 * joints' trajectories, so exploration is allowed for them too.
 * Mouse move: highlight which sample is under the cursor (exploreHoveredIndex).
 * Mouse leave: clear hover highlight.
 */

import type { ToolHandler, CanvasPoint } from './types'
import { getNode } from '../helpers'
import { DATA_EXPLORE_SAMPLE_INDEX } from '../rendering/ExplorationTrajectoriesRenderer'

/** Distance in units within which a sample is considered "hovered" or "clicked" */
const EXPLORE_HOVER_THRESHOLD_UNITS = 4

/** If the event target is a trajectory path with data-explore-sample-index, return that index. */
function getTrajectorySampleIndexFromEvent(event: React.MouseEvent<SVGSVGElement>): number | null {
  const el = (event.target as Element).closest?.(
    `[${DATA_EXPLORE_SAMPLE_INDEX}]`
  ) as Element | null
  if (!el) return null
  const v = el.getAttribute(DATA_EXPLORE_SAMPLE_INDEX)
  if (v == null || v === '') return null
  const n = parseInt(v, 10)
  return Number.isNaN(n) ? null : n
}

function findNearestSampleIndex(
  point: CanvasPoint,
  samples: Array<{ position: [number, number] }>,
  threshold: number
): number | null {
  let nearestIndex: number | null = null
  let minDist = threshold
  for (let i = 0; i < samples.length; i++) {
    const pos = samples[i].position
    const dx = point[0] - pos[0]
    const dy = point[1] - pos[1]
    const d = Math.hypot(dx, dy)
    if (d < minDist) {
      minDist = d
      nearestIndex = i
    }
  }
  return nearestIndex
}

export const exploreNodeTrajectoriesToolHandler: ToolHandler = {
  onClick(event, point, context) {
    if (context.toolMode !== 'explore_node_trajectories') return false

    const {
      exploreSamples,
      exploreLoading,
      exploreNodeId,
      exploreCenter,
      exploreMode,
      exploreSecondNodeId
    } = context.exploreTrajectoriesState

    // Click on a trajectory path (backwards selection) or a dot
    if (exploreSamples.length > 0 && !exploreLoading && exploreNodeId) {
      const trajectoryIndex = getTrajectorySampleIndexFromEvent(event)
      const clickedIndex =
        trajectoryIndex !== null
          ? (trajectoryIndex >= 0 && trajectoryIndex < exploreSamples.length ? trajectoryIndex : null)
          : findNearestSampleIndex(point, exploreSamples, EXPLORE_HOVER_THRESHOLD_UNITS)
      if (clickedIndex !== null && exploreSamples[clickedIndex].valid) {
        const sample = exploreSamples[clickedIndex]
        // Combinatorial: apply both joint positions that produced this trajectory
        if (exploreMode === 'combinatorial' && exploreSecondNodeId && sample.positionFirst != null) {
          context.moveTwoJoints(exploreNodeId, sample.positionFirst, exploreSecondNodeId, sample.position)
        } else if (exploreMode === 'combinatorial' && exploreSecondNodeId) {
          context.showStatus('Cannot apply this sample; first position missing.', 'warning', 2000)
          return true
        } else {
          const nodeToApply = exploreSecondNodeId ?? exploreNodeId
          context.moveJoint(nodeToApply, sample.position)
        }
        context.setExploreTrajectoriesState(prev => ({
          ...prev,
          exploreNodeId: null,
          exploreCenter: null,
          exploreSamples: [],
          exploreHoveredIndex: null,
          exploreHoveredFromTrajectoryPath: false,
          exploreLoading: false,
          exploreMode: 'single',
          exploreSecondNodeId: null,
          exploreSecondCenter: null,
          explorePinnedFirstPosition: null
        }))
        context.setToolMode('select')
        context.showStatus('Applied position; switched to Select mode', 'success', 2000)
        return true
      }
    }

    // Click on a joint
    const jointsWithPositions = context.getJointsWithPositions()
    const nearest = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
    if (!nearest) {
      context.showStatus('Click a joint to explore trajectories', 'info', 2000)
      return true
    }

    const node = getNode(context.linkageDoc, nearest.name)
    if (!node) return true

    const jointType = context.getJointType(nearest.name)
    const isMovingStatic = jointType === 'Static'
    const isFixed = node.role === 'fixed'
    if (isFixed && !isMovingStatic) {
      context.showStatus('Only non-fixed joints or moving static joints can be explored', 'warning', 2000)
      return true
    }

    // Already have first exploration: click another node to run combinatorial exploration
    if (exploreSamples.length > 0 && exploreNodeId && exploreCenter != null) {
      if (nearest.name === exploreNodeId) {
        context.showStatus('Pick a different node to explore combinations.', 'info', 2000)
        return true
      }
      context.runExploreTrajectoriesCombinatorial(
        exploreNodeId,
        exploreCenter,
        exploreSamples,
        nearest.name,
        nearest.position
      )
      return true
    }

    context.runExploreTrajectories(nearest.name, nearest.position)
    return true
  },

  onMouseMove(event, point, context) {
    if (context.toolMode !== 'explore_node_trajectories') return false

    const { exploreSamples } = context.exploreTrajectoriesState
    if (!exploreSamples.length) return false

    const trajectoryIndex = getTrajectorySampleIndexFromEvent(event)
    const index =
      trajectoryIndex !== null
        ? (trajectoryIndex >= 0 && trajectoryIndex < exploreSamples.length ? trajectoryIndex : null)
        : findNearestSampleIndex(point, exploreSamples, EXPLORE_HOVER_THRESHOLD_UNITS)
    const next = index !== null ? index : null
    const fromPath = trajectoryIndex !== null
    const prevIndex = context.exploreTrajectoriesState.exploreHoveredIndex
    const prevFromPath = context.exploreTrajectoriesState.exploreHoveredFromTrajectoryPath
    if (next === prevIndex && fromPath === prevFromPath) return true

    context.setExploreTrajectoriesState(prevState => ({
      ...prevState,
      exploreHoveredIndex: next,
      exploreHoveredFromTrajectoryPath: fromPath
    }))
    return true
  },

  onMouseLeave(_event, context) {
    if (context.toolMode !== 'explore_node_trajectories') return false
    if (context.exploreTrajectoriesState.exploreHoveredIndex === null) return false

    context.setExploreTrajectoriesState(prev => ({ ...prev, exploreHoveredIndex: null, exploreHoveredFromTrajectoryPath: false }))
    return true
  }
}
