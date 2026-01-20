/**
 * Measurements Renderer
 *
 * Pure functions to render measurement markers and lines.
 */

import React from 'react'
import {
  MeasurementMarkersRendererProps,
  MeasurementLineRendererProps
} from './types'

// ═══════════════════════════════════════════════════════════════════════════════
// MEASUREMENT MARKERS (X marks that fade)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders measurement markers (X marks that fade over time)
 */
export function renderMeasurementMarkers({
  markers,
  unitsToPixels
}: MeasurementMarkersRendererProps): JSX.Element[] {
  const now = Date.now()

  return markers.map((marker) => {
    const age = now - marker.timestamp
    const opacity = Math.max(0, 1 - age / 3000) // Fade over 3 seconds
    const size = 8
    const x = unitsToPixels(marker.point[0])
    const y = unitsToPixels(marker.point[1])

    return (
      <g key={marker.id} opacity={opacity}>
        {/* X mark */}
        <line
          x1={x - size}
          y1={y - size}
          x2={x + size}
          y2={y + size}
          stroke="#f44336"
          strokeWidth={3}
          strokeLinecap="round"
        />
        <line
          x1={x + size}
          y1={y - size}
          x2={x - size}
          y2={y + size}
          stroke="#f44336"
          strokeWidth={3}
          strokeLinecap="round"
        />
        {/* Coordinate label */}
        <text
          x={x}
          y={y - 14}
          textAnchor="middle"
          fontSize="10"
          fill="#f44336"
          fontWeight="500"
        >
          ({marker.point[0].toFixed(1)}, {marker.point[1].toFixed(1)})
        </text>
      </g>
    )
  })
}

export const MeasurementMarkersRenderer: React.FC<MeasurementMarkersRendererProps> = (props) => {
  return <>{renderMeasurementMarkers(props)}</>
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEASUREMENT LINE (if measuring)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Renders measurement line (start point indicator during measurement)
 */
export function renderMeasurementLine({
  startPoint,
  isMeasuring,
  unitsToPixels
}: MeasurementLineRendererProps): JSX.Element | null {
  if (!isMeasuring || !startPoint) return null

  const startX = unitsToPixels(startPoint[0])
  const startY = unitsToPixels(startPoint[1])

  return (
    <g>
      {/* Start point indicator */}
      <circle
        cx={startX}
        cy={startY}
        r={6}
        fill="#f44336"
        stroke="#fff"
        strokeWidth={2}
      />
    </g>
  )
}

export const MeasurementLineRenderer: React.FC<MeasurementLineRendererProps> = (props) => {
  return renderMeasurementLine(props)
}
