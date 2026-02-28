/**
 * useMechanismPositions
 *
 * Single place for resolving a joint's visual position. Used by mechanism rendering,
 * tool context, and validation. Frame is supplied by the caller (single display-frame source).
 */

import { useCallback } from 'react'
import { DRAG_MOVE_THRESHOLD } from '../constants'
import type { LinkageDocument, PylinkDocument, PylinkJoint } from '../types'
import type { TrajectoryData } from '../../AnimateSimulate'

export type Position = [number, number]

/** Drag state needed to resolve position during single-node drag */
export interface MechanismPositionsDragState {
  isDragging: boolean
  draggedJoint: string | null
  dragStartPosition: Position | null
  currentPosition: Position | null
}

/** When set, use this position for the given joint until new trajectory arrives (avoids jump on drag release). */
export type PendingDropPosition = { jointName: string; position: Position } | null

export interface UseMechanismPositionsParams {
  linkageDoc: LinkageDocument | null
  pylinkDoc: PylinkDocument
  trajectoryData: TrajectoryData | null
  /** Resolved display frame (override ?? animation currentFrame). Caller owns single source. */
  displayFrame: number
  animatedPositions: Record<string, Position> | null | undefined
  dragState: MechanismPositionsDragState
  /** If set, use this position for the joint until trajectoryData updates (fixes jump on release). */
  pendingDropPosition?: PendingDropPosition
}

export interface UseMechanismPositionsReturn {
  getJointPosition: (jointName: string, visited?: Set<string>) => Position | null
}

interface ResolveParams {
  linkageDoc: LinkageDocument | null
  pylinkDoc: PylinkDocument
  trajectoryData: TrajectoryData | null
  displayFrame: number
  animatedPositions: Record<string, Position> | null | undefined
  dragState: MechanismPositionsDragState
  pendingDropPosition: PendingDropPosition
}

/**
 * Resolve position for one joint from trajectory, doc, or geometry.
 * Uses visited set for cycle detection in Revolute parent recursion.
 */
function resolveJointPosition(
  jointName: string,
  params: ResolveParams,
  visited: Set<string>
): Position | null {
  if (visited.has(jointName)) {
    console.warn(`Cycle detected in joint relationships for ${jointName}`)
    return null
  }
  visited.add(jointName)

  const {
    linkageDoc,
    pylinkDoc,
    trajectoryData,
    displayFrame,
    animatedPositions,
    dragState,
    pendingDropPosition
  } = params

  // Use pending drop position for just-released joint until new trajectory arrives (avoids jump).
  if (pendingDropPosition && pendingDropPosition.jointName === jointName) {
    return pendingDropPosition.position
  }

  const traj = trajectoryData?.trajectories?.[jointName]
  const totalSteps = trajectoryData?.nSteps ?? 0
  const frame =
    totalSteps > 0 ? Math.max(0, Math.min(displayFrame, totalSteps - 1)) : displayFrame

  // During single-node drag: use trajectory[frame] for dragged joint until user moves, then doc/meta
  if (dragState.isDragging && dragState.draggedJoint === jointName) {
    const hasMoved =
      dragState.dragStartPosition != null &&
      dragState.currentPosition != null &&
      Math.hypot(
        dragState.currentPosition[0] - dragState.dragStartPosition[0],
        dragState.currentPosition[1] - dragState.dragStartPosition[1]
      ) >= DRAG_MOVE_THRESHOLD
    if (traj && frame >= 0 && frame < traj.length && !hasMoved) {
      return traj[frame] as Position
    }
    const node = linkageDoc?.linkage?.nodes?.[jointName]
    const pos = node?.position
    if (Array.isArray(pos) && pos.length >= 2 && typeof pos[0] === 'number' && typeof pos[1] === 'number') {
      return [Number(pos[0]), Number(pos[1])]
    }
    const meta = pylinkDoc.meta.joints[jointName]
    if (meta?.x !== undefined && meta?.y !== undefined) {
      return [meta.x, meta.y]
    }
  }

  if (traj && frame >= 0 && frame < traj.length) {
    return traj[frame] as Position
  }

  if (animatedPositions?.[jointName]) {
    return animatedPositions[jointName]
  }

  const joint = pylinkDoc.pylinkage.joints.find((j: PylinkJoint) => j.name === jointName)
  if (!joint) return null

  if (joint.type === 'Static') {
    return [joint.x, joint.y]
  }

  const meta = pylinkDoc.meta.joints[jointName]
  if (meta?.x !== undefined && meta?.y !== undefined) {
    return [meta.x, meta.y]
  }

  if (joint.type === 'Crank') {
    const parent = pylinkDoc.pylinkage.joints.find((j: PylinkJoint) => j.name === joint.joint0.ref)
    if (parent && parent.type === 'Static') {
      const x = parent.x + joint.distance * Math.cos(joint.angle)
      const y = parent.y + joint.distance * Math.sin(joint.angle)
      return [x, y]
    }
  } else if (joint.type === 'Revolute') {
    const parent0 = pylinkDoc.pylinkage.joints.find((j: PylinkJoint) => j.name === joint.joint0.ref)
    const parent1 = pylinkDoc.pylinkage.joints.find((j: PylinkJoint) => j.name === joint.joint1.ref)
    if (parent0 && parent1) {
      const pos0 = resolveJointPosition(parent0.name, params, new Set(visited))
      const pos1 = resolveJointPosition(parent1.name, params, new Set(visited))
      if (pos0 && pos1) {
        const d0 = joint.distance0
        const d1 = joint.distance1
        const dx = pos1[0] - pos0[0]
        const dy = pos1[1] - pos0[1]
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d > 0 && d <= d0 + d1 && d >= Math.abs(d0 - d1)) {
          const a = (d0 * d0 - d1 * d1 + d * d) / (2 * d)
          const h = Math.sqrt(Math.max(0, d0 * d0 - a * a))
          const px = pos0[0] + (a * dx) / d
          const py = pos0[1] + (a * dy) / d
          const x = px - (h * dy) / d
          const y = py + (h * dx) / d
          return [x, y]
        }
        return [(pos0[0] + pos1[0]) / 2, (pos0[1] + pos1[1]) / 2 - 20]
      }
    }
  }
  return null
}

/**
 * Hook that returns getJointPosition for the current mechanism and display frame.
 * Caller must pass the single resolved display frame (e.g. displayFrameOverride ?? animation.currentFrame ?? 0).
 */
export function useMechanismPositions({
  linkageDoc,
  pylinkDoc,
  trajectoryData,
  displayFrame,
  animatedPositions,
  dragState,
  pendingDropPosition = null
}: UseMechanismPositionsParams): UseMechanismPositionsReturn {
  const getJointPosition = useCallback(
    (jointName: string, visited: Set<string> = new Set()): Position | null => {
      return resolveJointPosition(jointName, {
        linkageDoc,
        pylinkDoc,
        trajectoryData,
        displayFrame,
        animatedPositions,
        dragState,
        pendingDropPosition: pendingDropPosition ?? null
      }, visited)
    },
    [
      linkageDoc,
      pylinkDoc,
      trajectoryData,
      displayFrame,
      animatedPositions,
      dragState.isDragging,
      dragState.draggedJoint,
      dragState.dragStartPosition,
      dragState.currentPosition,
      pendingDropPosition
    ]
  )
  return { getJointPosition }
}
