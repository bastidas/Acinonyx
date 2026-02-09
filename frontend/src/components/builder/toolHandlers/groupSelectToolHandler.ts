/**
 * Group Select tool handler
 *
 * Handles group_select mode: mouse down starts box, mouse move previews,
 * mouse up completes selection and enters move mode. Extracted from
 * BuilderTab handleCanvasMouseDown / MouseMove / MouseUp.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialGroupSelectionState } from '../../BuilderTools'

export const groupSelectToolHandler: ToolHandler = {
  onMouseDown(_event, point, context) {
    if (context.toolMode !== 'group_select') return false
    context.setGroupSelectionState({
      isSelecting: true,
      startPoint: point,
      currentPoint: point
    })
    context.showStatus('Drag to select multiple elements', 'action')
    return true
  },

  onMouseMove(_event, point, context) {
    if (context.toolMode !== 'group_select') return false
    const { groupSelectionState } = context
    if (!groupSelectionState.isSelecting || !groupSelectionState.startPoint) return false

    context.setGroupSelectionState(prev => ({
      ...prev,
      currentPoint: point
    }))

    const jointsWithPositions = context.getJointsWithPositions()
    const linksWithPositions = context.getLinksWithPositions()
    const box = {
      x1: groupSelectionState.startPoint[0],
      y1: groupSelectionState.startPoint[1],
      x2: point[0],
      y2: point[1]
    }
    const preview = context.findElementsInBox(box, jointsWithPositions, linksWithPositions)
    context.showStatus(`Selecting ${preview.joints.length} joints, ${preview.links.length} links`, 'action')
    return true
  },

  onMouseUp(_event, context) {
    if (context.toolMode !== 'group_select') return false
    const { groupSelectionState } = context
    if (!groupSelectionState.isSelecting || !groupSelectionState.startPoint || !groupSelectionState.currentPoint) {
      return false
    }

    const jointsWithPositions = context.getJointsWithPositions()
    const linksWithPositions = context.getLinksWithPositions()
    const box = {
      x1: groupSelectionState.startPoint[0],
      y1: groupSelectionState.startPoint[1],
      x2: groupSelectionState.currentPoint[0],
      y2: groupSelectionState.currentPoint[1]
    }
    const selected = context.findElementsInBox(box, jointsWithPositions, linksWithPositions)

    const minX = Math.min(box.x1, box.x2)
    const maxX = Math.max(box.x1, box.x2)
    const minY = Math.min(box.y1, box.y2)
    const maxY = Math.max(box.y1, box.y2)
    const drawnObjectsInBox = context.drawnObjects.objects
      .filter(obj => obj.points.some(p => p[0] >= minX && p[0] <= maxX && p[1] >= minY && p[1] <= maxY))
      .map(obj => obj.id)

    const mergedDrawnObjects = context.drawnObjects.objects
      .filter(obj => obj.mergedLinkName && selected.links.includes(obj.mergedLinkName))
      .map(obj => obj.id)

    const allSelectedDrawnObjects = [...new Set([...drawnObjectsInBox, ...mergedDrawnObjects])]

    context.setSelectedJoints(selected.joints)
    context.setSelectedLinks(selected.links)
    context.setDrawnObjects(prev => ({ ...prev, selectedIds: allSelectedDrawnObjects }))
    context.setGroupSelectionState(initialGroupSelectionState)

    if (selected.joints.length > 0 || selected.links.length > 0 || allSelectedDrawnObjects.length > 0) {
      context.enterMoveGroupMode(selected.joints, allSelectedDrawnObjects)
    } else {
      context.showStatus('No elements selected', 'info', 1500)
    }
    return true
  }
}
