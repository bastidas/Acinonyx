/**
 * Select tool handler
 *
 * Handles select mode: click to select joint/link, drag joint to move or merge.
 * Extracted from BuilderTab handleCanvasMouseDown/MouseMove/MouseUp/Click.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialDragState } from '../../BuilderTools'

export const selectToolHandler: ToolHandler = {
  onMouseDown(event, point, context) {
    if (context.toolMode !== 'select') return false
    const jointsWithPositions = context.getJointsWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions)
    if (!nearestJoint) return false

    context.setDragState({
      isDragging: true,
      draggedJoint: nearestJoint.name,
      dragStartPosition: nearestJoint.position,
      currentPosition: nearestJoint.position,
      mergeTarget: null,
      mergeProximity: Infinity
    })
    context.setSelectedJoints([nearestJoint.name])
    context.setSelectedLinks([])
    context.showStatus(`Dragging ${nearestJoint.name}`, 'action')
    return true
  },

  onMouseMove(event, point, context) {
    if (context.toolMode !== 'select') return false
    const { dragState } = context
    if (!dragState.isDragging || !dragState.draggedJoint) return false

    const jointsWithPositions = context.getJointsWithPositions()
    const mergeTarget = context.findMergeTarget(
      point,
      jointsWithPositions,
      dragState.draggedJoint,
      context.jointMergeRadius
    )

    if (mergeTarget) {
      context.setDragState((prev) => ({
        ...prev,
        currentPosition: point,
        mergeTarget: mergeTarget.name,
        mergeProximity: mergeTarget.distance
      }))
      context.showStatus(`Release to merge into ${mergeTarget.name}`, 'action')
    } else {
      context.setDragState((prev) => ({
        ...prev,
        currentPosition: point,
        mergeTarget: null,
        mergeProximity: Infinity
      }))
      context.showStatus(
        `Moving ${dragState.draggedJoint} to (${point[0].toFixed(1)}, ${point[1].toFixed(1)})`,
        'action'
      )
    }

    context.moveJoint(dragState.draggedJoint, point)
    return true
  },

  onMouseUp(_event, context) {
    if (context.toolMode !== 'select') return false
    const { dragState } = context
    if (!dragState.isDragging || !dragState.draggedJoint) return false

    if (dragState.mergeTarget) {
      context.mergeJoints(dragState.draggedJoint, dragState.mergeTarget)
    } else if (dragState.currentPosition) {
      context.showStatus(
        `Moved ${dragState.draggedJoint} to (${dragState.currentPosition[0].toFixed(1)}, ${dragState.currentPosition[1].toFixed(1)})`,
        'success',
        2000
      )
    }

    context.setDragState(initialDragState)
    return true
  },

  onClick(_event, point, context) {
    if (context.toolMode !== 'select') return false

    const jointsWithPositions = context.getJointsWithPositions()
    const linksWithPositions = context.getLinksWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
    const nearestLink = context.findNearestLink(point, linksWithPositions, context.snapThreshold)

    if (nearestJoint) {
      context.setSelectedJoints([nearestJoint.name])
      context.setSelectedLinks([])
      context.showStatus(`Selected ${nearestJoint.name}`, 'info', 1500)
    } else if (nearestLink) {
      context.setSelectedLinks([nearestLink.name])
      context.setSelectedJoints([])
      context.showStatus(`Selected ${nearestLink.name}`, 'info', 1500)
    } else {
      context.setSelectedJoints([])
      context.setSelectedLinks([])
      context.clearStatus()
    }
    return true
  }
}
