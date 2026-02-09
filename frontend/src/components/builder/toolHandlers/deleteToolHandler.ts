/**
 * Delete tool handler
 *
 * Handles delete mode: click on joint/link to delete it, or delete selected
 * items if any are selected. Extracted from BuilderTab handleCanvasClick.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'

export const deleteToolHandler: ToolHandler = {
  onClick(_event, point, context) {
    if (context.toolMode !== 'delete') return false

    const { selectedJoints, selectedLinks } = context
    if (selectedJoints.length > 0 || selectedLinks.length > 0) {
      context.handleDeleteSelected()
      return true
    }

    const jointsWithPositions = context.getJointsWithPositions()
    const linksWithPositions = context.getLinksWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
    const nearestLink = context.findNearestLink(point, linksWithPositions, context.snapThreshold)

    if (nearestJoint) {
      context.deleteJoint(nearestJoint.name)
    } else if (nearestLink) {
      context.deleteLink(nearestLink.name)
    } else {
      context.showStatus('Click on a joint or link to delete it', 'info', 2000)
    }
    return true
  }
}
