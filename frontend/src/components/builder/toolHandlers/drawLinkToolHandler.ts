/**
 * Draw Link tool handler
 *
 * Handles draw_link mode: first click starts link (optionally from a joint),
 * second click completes link. Mouse move shows rubber-band preview.
 * Extracted from BuilderTab handleCanvasMouseMove / handleCanvasClick.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialLinkCreationState } from '../../BuilderTools'

export const drawLinkToolHandler: ToolHandler = {
  onMouseMove(_event, point, context) {
    if (context.toolMode !== 'draw_link') return false
    const { linkCreationState } = context
    if (!linkCreationState.isDrawing || !linkCreationState.startPoint) return false

    const jointsWithPositions = context.getJointsWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions)
    const endPoint: CanvasPoint = nearestJoint?.position ?? point

    context.setPreviewLine({
      start: linkCreationState.startPoint,
      end: endPoint
    })
    return true
  },

  onClick(_event, point, context) {
    if (context.toolMode !== 'draw_link') return false

    const jointsWithPositions = context.getJointsWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)

    if (!context.linkCreationState.isDrawing) {
      // First click - start the link
      const startJointName = nearestJoint?.name ?? null
      const startPoint: CanvasPoint = nearestJoint?.position ?? point

      context.setLinkCreationState({
        isDrawing: true,
        startPoint,
        startJointName,
        endPoint: null
      })

      if (startJointName) {
        context.showStatus(`Drawing from ${startJointName} — click to complete`, 'action')
      } else {
        context.showStatus(
          `Drawing from (${startPoint[0].toFixed(1)}, ${startPoint[1].toFixed(1)}) — click to complete`,
          'action'
        )
      }
      return true
    }

    // Second click - complete the link
    const endJointName = nearestJoint?.name ?? null
    const endPoint: CanvasPoint = nearestJoint?.position ?? point

    if (endJointName && endJointName === context.linkCreationState.startJointName) {
      context.showStatus('Cannot connect a joint to itself', 'warning', 2000)
      return true
    }

    context.createLinkWithRevoluteDefault(
      context.linkCreationState.startPoint!,
      endPoint,
      context.linkCreationState.startJointName,
      endJointName
    )

    context.setLinkCreationState(initialLinkCreationState)
    context.setPreviewLine(null)
    return true
  }
}
