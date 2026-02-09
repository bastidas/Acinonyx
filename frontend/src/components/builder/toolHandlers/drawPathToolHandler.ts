/**
 * Draw Path tool handler
 *
 * Handles draw_path mode: click to add points, click near start or double-click
 * to finish. For trajectory optimization target paths. Extracted from
 * BuilderTab handleCanvasClick / handleCanvasDoubleClick.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialPathDrawState } from '../../BuilderTools'

export const drawPathToolHandler: ToolHandler = {
  onClick(_event, point, context) {
    if (context.toolMode !== 'draw_path') return false

    const { pathDrawState } = context

    if (!pathDrawState.isDrawing) {
      context.setPathDrawState({
        isDrawing: true,
        points: [point]
      })
      context.showStatus('Click to add points. Click near start or double-click to close path.', 'action')
      return true
    }

    const startPoint = pathDrawState.points[0]
    const distanceToStart = context.calculateDistance(point, startPoint)

    if (distanceToStart <= context.jointMergeRadius && pathDrawState.points.length >= 3) {
      const newPath = context.createTargetPath(pathDrawState.points, context.targetPaths)
      context.setTargetPaths((prev: unknown[]) => [...prev, newPath] as unknown[])
      context.setSelectedPathId(newPath.id)
      context.setPathDrawState(initialPathDrawState)
      context.showStatus(`Created closed path with ${pathDrawState.points.length} points`, 'success', 2500)
    } else {
      context.setPathDrawState(prev => ({
        ...prev,
        points: [...prev.points, point]
      }))
      const pointCount = pathDrawState.points.length + 1
      if (pointCount >= 3) {
        context.showStatus(`${pointCount} points — click near start to close, or double-click to finish`, 'action')
      } else {
        context.showStatus(`${pointCount} points — need at least 3 for a path`, 'action')
      }
    }
    return true
  },

  onDoubleClick(_event, point, context) {
    if (context.toolMode !== 'draw_path') return false
    const { pathDrawState } = context
    if (!pathDrawState.isDrawing || pathDrawState.points.length < 2) return false

    const newPath = context.createTargetPath(pathDrawState.points, context.targetPaths)
    context.setTargetPaths((prev: unknown[]) => [...prev, newPath] as unknown[])
    context.setSelectedPathId(newPath.id)
    context.setPathDrawState(initialPathDrawState)
    context.showStatus(`Created target path with ${pathDrawState.points.length} points`, 'success', 2500)
    return true
  }
}
