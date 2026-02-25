/**
 * Canvas Images Renderer
 *
 * Renders overlay images (JPEG/PNG) with a lower-left handle: double-click opens
 * edit dialog, drag repositions the canvas.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react'
import type { CanvasImagesRendererProps, CanvasImageRenderItem } from './types'

const HANDLE_SIZE = 10
const HANDLE_INSET = 4

function CanvasImageWithHandle({
  c,
  xPx,
  yPx,
  widthPx,
  heightPx,
  onRequestEdit,
  onPositionChange,
  pixelsToUnits,
  unitsToPixels
}: {
  c: CanvasImageRenderItem
  xPx: number
  yPx: number
  widthPx: number
  heightPx: number
  onRequestEdit: (id: string) => void
  onPositionChange: (id: string, position: [number, number]) => void
  pixelsToUnits: (pixels: number) => number
  unitsToPixels: (units: number) => number
}) {
  const [dragging, setDragging] = useState(false)
  const dragStartRef = useRef<{ clientX: number; clientY: number; posUnits: [number, number] } | null>(null)

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      e.stopPropagation()
      onRequestEdit(c.id)
    },
    [c.id, onRequestEdit]
  )

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      setDragging(true)
      dragStartRef.current = {
        clientX: e.clientX,
        clientY: e.clientY,
        posUnits: [...c.position]
      }
    },
    [c.position]
  )

  useEffect(() => {
    if (!dragging) return
    const onMove = (e: MouseEvent) => {
      const start = dragStartRef.current
      if (!start) return
      const dxPx = e.clientX - start.clientX
      const dyPx = e.clientY - start.clientY
      const dxUnits = pixelsToUnits(dxPx) - pixelsToUnits(0)
      const dyUnits = pixelsToUnits(dyPx) - pixelsToUnits(0)
      onPositionChange(c.id, [start.posUnits[0] + dxUnits, start.posUnits[1] + dyUnits])
    }
    const onUp = () => {
      setDragging(false)
      dragStartRef.current = null
    }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
    return () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
  }, [dragging, c.id, onPositionChange, pixelsToUnits])

  const handleX = xPx + HANDLE_INSET
  const handleY = yPx + heightPx - HANDLE_INSET - HANDLE_SIZE

  return (
    <g key={c.id}>
      <image
        href={c.dataUrl}
        x={xPx}
        y={yPx}
        width={widthPx}
        height={heightPx}
        opacity={c.alpha}
        preserveAspectRatio="none"
        style={{ pointerEvents: 'none' }}
      />
      <g
        onDoubleClick={handleDoubleClick}
        onMouseDown={handleMouseDown}
        style={{ cursor: dragging ? 'grabbing' : 'grab' }}
      >
        <rect
          x={handleX}
          y={handleY}
          width={HANDLE_SIZE}
          height={HANDLE_SIZE}
          fill="rgba(255,255,255,0.5)"
          stroke="rgba(0,0,0,0.2)"
          strokeWidth={0.8}
          rx={2}
        />
        {/* Grip dots: 2 columns × 3 rows = drag handle */}
        {[1, 2, 3].flatMap((row) =>
          [1, 2].map((col) => (
            <circle
              key={`${row}-${col}`}
              cx={handleX + 2.5 * col}
              cy={handleY + 2 + 3 * (row - 1)}
              r={0.9}
              fill="rgba(0,0,0,0.45)"
            />
          ))
        )}
        <title>Drag to move. Double-click for options.</title>
      </g>
    </g>
  )
}

/**
 * Renders canvas image overlays and a lower-left handle per image.
 * Double-click handle opens edit; drag handle repositions.
 */
export function renderCanvasImages({
  canvases,
  unitsToPixels,
  pixelsToUnits,
  onRequestEdit,
  onPositionChange
}: CanvasImagesRendererProps): (JSX.Element | null)[] {
  if (!canvases.length) return []

  const unitsPerPixel = pixelsToUnits(1)

  return canvases.map((c) => {
    const [x, y] = c.position
    const widthUnits = c.naturalWidth * unitsPerPixel * c.scale
    const heightUnits = c.naturalHeight * unitsPerPixel * c.scale
    const xPx = unitsToPixels(x)
    const yPx = unitsToPixels(y)
    const widthPx = unitsToPixels(widthUnits)
    const heightPx = unitsToPixels(heightUnits)

    return (
      <CanvasImageWithHandle
        key={c.id}
        c={c}
        xPx={xPx}
        yPx={yPx}
        widthPx={widthPx}
        heightPx={heightPx}
        onRequestEdit={onRequestEdit}
        onPositionChange={onPositionChange}
        pixelsToUnits={pixelsToUnits}
        unitsToPixels={unitsToPixels}
      />
    )
  })
}

export function CanvasImagesRenderer(props: CanvasImagesRendererProps): JSX.Element {
  const elements = renderCanvasImages(props)
  return <g data-layer="canvas-images">{elements}</g>
}
