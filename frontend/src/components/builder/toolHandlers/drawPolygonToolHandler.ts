/**
 * Draw Polygon tool handler
 *
 * Handles draw_polygon mode: first click starts polygon, subsequent clicks
 * add points, click near start to close. Creates DrawnObjects only.
 * Extracted from BuilderTab handleCanvasClick.
 */

import type { ToolHandler, CanvasPoint } from './types'
import { initialPolygonDrawState } from '../../BuilderTools'

const CIRCLE_SIDES = 32
const MIN_CIRCLE_RADIUS = 1e-6

function generateRegularPolygonPoints(
  center: CanvasPoint,
  radius: number,
  startAngleRad: number,
  sides: number
): CanvasPoint[] {
  return Array.from({ length: sides }, (_, i) => {
    const theta = startAngleRad + (2 * Math.PI * i) / sides
    return [
      center[0] + radius * Math.cos(theta),
      center[1] + radius * Math.sin(theta)
    ] as CanvasPoint
  })
}

export const drawPolygonToolHandler: ToolHandler = {
  onClick(_event, point, context) {
    if (context.toolMode !== 'draw_polygon') return false

    const { polygonDrawState } = context

    if (!polygonDrawState.isDrawing) {
      if (_event.shiftKey) {
        const jointsWithPositions = context.getJointsWithPositions()
        // Snap the circle center to an existing node when the click is within merge radius.
        const nearestCenter = context.findNearestJoint(point, jointsWithPositions, context.jointMergeRadius ?? context.snapThreshold)
        const circleCenter = nearestCenter?.position ?? point

        context.setPolygonDrawState({
          isDrawing: true,
          mode: 'circle',
          points: [],
          circleCenter,
          circleRadiusPoint: circleCenter,
          circleRadius: 0
        })
        context.showStatus('Circle center set • Move mouse to set radius • Click to create', 'action')
        return true
      }

      context.setPolygonDrawState({
        isDrawing: true,
        mode: 'polygon',
        points: [point],
        circleCenter: null,
        circleRadiusPoint: null,
        circleRadius: null
      })
      context.showStatus('Click to add polygon sides, click near start to close', 'action')
      return true
    }

    if (polygonDrawState.mode === 'circle') {
      const circleCenter = polygonDrawState.circleCenter
      if (!circleCenter) {
        context.setPolygonDrawState(initialPolygonDrawState)
        return true
      }

      const jointsWithPositions = context.getJointsWithPositions()
      // Optional snapping for the radius point: makes it easier to set radius using existing nodes.
      const nearestRadius = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
      const circleRadiusPoint = nearestRadius?.position ?? point

      const radius = context.calculateDistance(circleCenter, circleRadiusPoint)
      context.setPolygonDrawState(prev => ({ ...prev, circleRadiusPoint, circleRadius: radius }))

      if (radius < MIN_CIRCLE_RADIUS) {
        context.showStatus('Radius too small — pick a larger radius', 'warning', 2000)
        return true
      }

      const startAngle = Math.atan2(circleRadiusPoint[1] - circleCenter[1], circleRadiusPoint[0] - circleCenter[0])
      const polygonPoints = generateRegularPolygonPoints(circleCenter, radius, startAngle, CIRCLE_SIDES)

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
      context.showStatus(`Completed circle polygon (r=${radius.toFixed(2)} units)`, 'success', 2500)

      context.apiMergePolygon?.({
        polygonId: newDrawnObject.id,
        polygonPoints,
        attachOnlyIfSingleContainedLink: true
      }).then(data => {
        const n = data.polygon?.contained_links?.length ?? 0
        if (data.status === 'success' && n > 0) {
          if (n > 1) {
            context.showStatus(
              `Circle contains ${n} links — not auto-attached; use Merge to bind a link`,
              'info',
              3500
            )
          } else {
            context.showStatus('Circle contains 1 link', 'info', 2000)
          }
        }
      })

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
        mode: 'polygon',
        points: [...prev.points, point]
      }))
      const sides = polygonDrawState.points.length
      context.showStatus(`${sides + 1} points — click near start to close polygon`, 'action')
    }
    return true
  },
  onMouseMove(_event, point, context) {
    if (context.toolMode !== 'draw_polygon') return false
    const { polygonDrawState } = context
    if (!polygonDrawState.isDrawing) return false
    if (polygonDrawState.mode !== 'circle' || !polygonDrawState.circleCenter) return false

    const jointsWithPositions = context.getJointsWithPositions()
    // Snap radius to existing nodes when close enough.
    const nearestRadius = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
    const circleRadiusPoint = nearestRadius?.position ?? point
    const circleRadius = context.calculateDistance(polygonDrawState.circleCenter, circleRadiusPoint)

    context.setPolygonDrawState(prev => ({
      ...prev,
      circleRadiusPoint,
      circleRadius
    }))
    return true
  }
}
