/**
 * Group Select tool handler
 *
 * Handles group_select mode: mouse down starts box, mouse move previews,
 * mouse up completes selection and enters move mode. Extracted from
 * BuilderTab handleCanvasMouseDown / MouseMove / MouseUp.
 */

import type { ToolHandler } from './types'
import type { LinkMetaData } from '../../BuilderTools'
import { initialGroupSelectionState } from '../../BuilderTools'
import {
  boxFromCorners,
  isFormFullyInsideGroupSelectBox,
  selectionIsExactlyOneMechanism,
  polygonIdsTouchingMechanismLinks
} from '../helpers/formMechanismHelpers'

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

    const aabb = boxFromCorners(box.x1, box.y1, box.x2, box.y2)
    const strictDrawnObjectIds = context.drawnObjects.objects
      .filter(obj => obj.type === 'polygon' || obj.type == null)
      .filter(obj =>
        isFormFullyInsideGroupSelectBox(
          {
            points: obj.points as [number, number][],
            contained_links: obj.contained_links,
            mergedLinkName: obj.mergedLinkName
          },
          aabb,
          j => context.getJointPosition(j),
          context.linksMeta
        )
      )
      .map(obj => obj.id)

    const linksForMechanismCheck = context.linksMeta as Record<string, LinkMetaData>
    const isFullMechanism = selectionIsExactlyOneMechanism(
      selected.joints,
      selected.links,
      linksForMechanismCheck
    )

    const drawnObjectIds = isFullMechanism
      ? polygonIdsTouchingMechanismLinks(context.drawnObjects.objects, selected.links)
      : strictDrawnObjectIds

    context.setSelectedJoints(selected.joints)
    context.setSelectedLinks(selected.links)
    context.setDrawnObjects(prev => ({ ...prev, selectedIds: drawnObjectIds }))
    context.setGroupSelectionState(initialGroupSelectionState)

    if (selected.joints.length > 0 || selected.links.length > 0 || drawnObjectIds.length > 0) {
      if (isFullMechanism) {
        context.showStatus('Full mechanism — all linked forms included', 'info', 2500)
      } else if (drawnObjectIds.length > 0) {
        context.showStatus('Strict form inclusion (vertices / link endpoints in box)', 'info', 2500)
      }
      context.enterMoveGroupMode(selected.joints, drawnObjectIds)
    } else {
      context.showStatus('No elements selected', 'info', 1500)
    }
    return true
  }
}
