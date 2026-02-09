/**
 * Measure tool handler
 *
 * Handles measure mode: first click sets start point, second click completes
 * measurement and shows distance. Extracted from BuilderTab handleCanvasClick.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'
import { initialMeasureState } from '../../BuilderTools'

export const measureToolHandler: ToolHandler = {
  onClick(_event, point, context) {
    if (context.toolMode !== 'measure') return false

    const jointsWithPositions = context.getJointsWithPositions()
    const nearestJoint = context.findNearestJoint(point, jointsWithPositions, context.snapThreshold)
    const snappedPoint: CanvasPoint = nearestJoint?.position ?? point
    const snappedX = snappedPoint[0]
    const snappedY = snappedPoint[1]
    const snappedToJoint = nearestJoint ? ` (${nearestJoint.name})` : ''

    if (!context.measureState.isMeasuring) {
      context.setMeasureState({
        isMeasuring: true,
        startPoint: snappedPoint,
        endPoint: null,
        measurementId: Date.now()
      })
      context.setMeasurementMarkers(prev => [
        ...prev,
        { id: Date.now(), point: snappedPoint, timestamp: Date.now() }
      ])
      context.showStatus(
        `Start: (${snappedX.toFixed(1)}, ${snappedY.toFixed(1)})${snappedToJoint} — click second point`,
        'action'
      )
      return true
    }

    const startPoint = context.measureState.startPoint!
    const distance = context.calculateDistance(startPoint, snappedPoint)
    const dx = snappedPoint[0] - startPoint[0]
    const dy = snappedPoint[1] - startPoint[1]

    context.setMeasurementMarkers(prev => [
      ...prev,
      { id: Date.now(), point: snappedPoint, timestamp: Date.now() }
    ])
    context.showStatus(
      `Distance: ${distance.toFixed(2)} units (Δx: ${dx.toFixed(1)}, Δy: ${dy.toFixed(1)})`,
      'success',
      5000
    )
    context.setMeasureState(initialMeasureState)

    setTimeout(() => {
      context.setMeasurementMarkers(prev =>
        prev.filter(m => Date.now() - m.timestamp < 3000)
      )
    }, 3000)

    return true
  }
}
