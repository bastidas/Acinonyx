/**
 * Draw Polygon tool handler
 *
 * Handles draw_polygon mode: first click starts polygon, subsequent clicks
 * add points, click near start to close. Creates DrawnObjects only.
 * Extracted from BuilderTab handleCanvasClick.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialPolygonDrawState } from '../../BuilderTools'

export const drawPolygonToolHandler: ToolHandler = {
  onClick(_event, point, context) {
    if (context.toolMode !== 'draw_polygon') return false

    const { polygonDrawState } = context

    if (!polygonDrawState.isDrawing) {
      context.setPolygonDrawState({
        isDrawing: true,
        points: [point]
      })
      context.showStatus('Click to add polygon sides, click near start to close', 'action')
      return true
    }

    const startPoint = polygonDrawState.points[0]
    const distanceToStart = context.calculateDistance(point, startPoint)

    if (distanceToStart <= context.mergeThreshold && polygonDrawState.points.length >= 3) {
      const polygonPoints = polygonDrawState.points
      const newDrawnObject = context.createDrawnObject(
        'polygon',
        polygonPoints,
        context.drawnObjects.objects.map(o => o.id)
      )
      context.setDrawnObjects(prev => ({
        ...prev,
        objects: [...prev.objects, newDrawnObject],
        selectedIds: [newDrawnObject.id]
      }))
      context.setPolygonDrawState(initialPolygonDrawState)
      context.showStatus(`Completed ${polygonDrawState.points.length}-sided polygon (${newDrawnObject.id})`, 'success', 2500)
      context.apiMergePolygon?.({
        polygonId: newDrawnObject.id,
        polygonPoints,
        attachOnlyIfSingleContainedLink: true
      }).then(data => {
        const n = data.polygon?.contained_links?.length ?? 0
        if (data.status === 'success' && n > 0) {
          if (n > 1) {
            context.showStatus(
              `Polygon contains ${n} links — not auto-attached; use Merge to bind a link`,
              'info',
              3500
            )
          } else {
            context.showStatus('Polygon contains 1 link', 'info', 2000)
          }
        }
      })
    } else {
      context.setPolygonDrawState(prev => ({
        ...prev,
        points: [...prev.points, point]
      }))
      const sides = polygonDrawState.points.length
      context.showStatus(`${sides + 1} points — click near start to close polygon`, 'action')
    }
    return true
  }
}
