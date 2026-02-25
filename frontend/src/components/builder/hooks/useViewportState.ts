/**
 * useViewportState
 *
 * Manages zoom and pan for the builder canvas. Provides wheel-zoom (centered on cursor)
 * and drag-to-pan (middle-click drag, or Shift + left drag). Coordinate system: canvas pixel
 * space (same as getBoundingClientRect); content is drawn in world pixels and transformed
 * with translate(panX, panY) scale(zoom).
 */

import { useState, useCallback, useRef, useEffect } from 'react'

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

export const VIEWPORT_ZOOM_MIN = 0.25
export const VIEWPORT_ZOOM_MAX = 4
export const VIEWPORT_ZOOM_DEFAULT = 1
const ZOOM_FACTOR = 1.1

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface ViewportState {
  zoom: number
  panX: number
  panY: number
}

export interface UseViewportStateConfig {
  /** Ref to the canvas container (Paper) for getBoundingClientRect */
  canvasRef: React.RefObject<HTMLDivElement | SVGSVGElement | null>
}

export interface UseViewportStateReturn {
  viewport: ViewportState
  setViewport: React.Dispatch<React.SetStateAction<ViewportState>>
  /** Zoom in/out; pass center in screen pixels (e.g. cursor) to zoom at point */
  setZoom: (newZoom: number, centerScreen?: { x: number; y: number }) => void
  /** Add delta to pan (in screen pixels) */
  panBy: (dx: number, dy: number) => void
  resetView: () => void
  /** Call on wheel; zooms and keeps point under cursor fixed */
  handleWheel: (event: React.WheelEvent) => void
  /** Call on mouse down; returns true if pan started (caller should skip tool handler) */
  handlePanStart: (event: React.MouseEvent) => boolean
  /** Call on mouse move when panning */
  handlePanMove: (event: React.MouseEvent) => void
  /** Call on mouse up / leave to end pan */
  handlePanEnd: () => void
  /** Ref that is true while panning (for same-tick checks in parent) */
  isPanningRef: React.MutableRefObject<boolean>
  /** Whether we are currently panning (state, may lag by one frame) */
  isPanning: boolean
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════════

export function useViewportState(config: UseViewportStateConfig): UseViewportStateReturn {
  const { canvasRef } = config
  const [viewport, setViewport] = useState<ViewportState>({
    zoom: VIEWPORT_ZOOM_DEFAULT,
    panX: 0,
    panY: 0
  })

  const isPanningRef = useRef(false)
  const [isPanning, setIsPanning] = useState(false)
  const panStartRef = useRef<{ panX: number; panY: number; screenX: number; screenY: number } | null>(null)
  const shiftKeyRef = useRef(false)

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Shift' || e.code === 'ShiftLeft' || e.code === 'ShiftRight') shiftKeyRef.current = true
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Shift' || e.code === 'ShiftLeft' || e.code === 'ShiftRight') {
        shiftKeyRef.current = false
        panStartRef.current = null
        isPanningRef.current = false
        setIsPanning(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [])

  const setZoom = useCallback((newZoom: number, centerScreen?: { x: number; y: number }) => {
    const clamped = Math.max(VIEWPORT_ZOOM_MIN, Math.min(VIEWPORT_ZOOM_MAX, newZoom))
    if (centerScreen != null) {
      setViewport(prev => {
        const worldX = (centerScreen.x - prev.panX) / prev.zoom
        const worldY = (centerScreen.y - prev.panY) / prev.zoom
        const panX = centerScreen.x - worldX * clamped
        const panY = centerScreen.y - worldY * clamped
        return { zoom: clamped, panX, panY }
      })
    } else {
      setViewport(prev => ({ ...prev, zoom: clamped }))
    }
  }, [])

  const panBy = useCallback((dx: number, dy: number) => {
    setViewport(prev => ({
      ...prev,
      panX: prev.panX + dx,
      panY: prev.panY + dy
    }))
  }, [])

  const resetView = useCallback(() => {
    setViewport({
      zoom: VIEWPORT_ZOOM_DEFAULT,
      panX: 0,
      panY: 0
    })
  }, [])

  const handleWheel = useCallback((event: React.WheelEvent) => {
    if (!canvasRef.current) return
    event.preventDefault()
    const rect = canvasRef.current.getBoundingClientRect()
    const centerScreen = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    }
    const delta = -event.deltaY
    const factor = delta > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR
    setViewport(prev => {
      const newZoom = Math.max(VIEWPORT_ZOOM_MIN, Math.min(VIEWPORT_ZOOM_MAX, prev.zoom * factor))
      const worldX = (centerScreen.x - prev.panX) / prev.zoom
      const worldY = (centerScreen.y - prev.panY) / prev.zoom
      const panX = centerScreen.x - worldX * newZoom
      const panY = centerScreen.y - worldY * newZoom
      return { zoom: newZoom, panX, panY }
    })
  }, [canvasRef])

  // Attach non-passive wheel listener so preventDefault works (stops page scroll when zooming)
  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const handler = (e: Event) => {
      const ev = e as WheelEvent
      const target = ev.target as Node
      if (target && typeof (target as Element).closest === 'function' && (target as Element).closest('[data-no-canvas-zoom]')) {
        return
      }
      ev.preventDefault()
      if (!canvasRef.current) return
      if (shiftKeyRef.current) {
        panBy(ev.deltaX, ev.deltaY)
        return
      }
      const rect = canvasRef.current.getBoundingClientRect()
      const centerScreen = { x: ev.clientX - rect.left, y: ev.clientY - rect.top }
      const delta = -ev.deltaY
      const factor = delta > 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR
      setViewport(prev => {
        const newZoom = Math.max(VIEWPORT_ZOOM_MIN, Math.min(VIEWPORT_ZOOM_MAX, prev.zoom * factor))
        const worldX = (centerScreen.x - prev.panX) / prev.zoom
        const worldY = (centerScreen.y - prev.panY) / prev.zoom
        return { zoom: newZoom, panX: centerScreen.x - worldX * newZoom, panY: centerScreen.y - worldY * newZoom }
      })
    }
    el.addEventListener('wheel', handler, { passive: false })
    return () => el.removeEventListener('wheel', handler)
  }, [canvasRef, panBy])

  const handlePanEnd = useCallback(() => {
    panStartRef.current = null
    isPanningRef.current = false
    setIsPanning(false)
  }, [])

  const handlePanStart = useCallback((event: React.MouseEvent): boolean => {
    const isMiddle = event.button === 1
    const isLeft = event.button === 0
    // Middle-click (scroll wheel) always pans; left-drag pans only with Shift
    if (isMiddle) {
      // allow pan
    } else if (isLeft && shiftKeyRef.current) {
      // allow pan
    } else {
      return false
    }
    event.preventDefault()
    if (!canvasRef.current) return false
    const rect = canvasRef.current.getBoundingClientRect()
    panStartRef.current = {
      panX: viewport.panX,
      panY: viewport.panY,
      screenX: event.clientX - rect.left,
      screenY: event.clientY - rect.top
    }
    isPanningRef.current = true
    setIsPanning(true)
    return true
  }, [canvasRef, viewport.panX, viewport.panY])

  const handlePanMove = useCallback((event: React.MouseEvent) => {
    if (!panStartRef.current) return
    if (!canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const screenX = event.clientX - rect.left
    const screenY = event.clientY - rect.top
    const dx = screenX - panStartRef.current.screenX
    const dy = screenY - panStartRef.current.screenY
    setViewport({
      zoom: viewport.zoom,
      panX: panStartRef.current.panX + dx,
      panY: panStartRef.current.panY + dy
    })
  }, [canvasRef, viewport.zoom])

  return {
    viewport,
    setViewport,
    setZoom,
    panBy,
    resetView,
    handleWheel,
    handlePanStart,
    handlePanMove,
    handlePanEnd,
    isPanningRef,
    isPanning
  }
}

export default useViewportState
