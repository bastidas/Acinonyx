/**
 * Canvas Interaction Hook
 *
 * Provides utilities for canvas coordinate conversion and element detection.
 * Extracted from BuilderTab.tsx for better organization and reusability.
 */

import { useCallback, RefObject } from 'react'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface CanvasCoordinates {
  /** Pixel coordinates relative to canvas */
  pixelX: number
  pixelY: number
  /** Unit coordinates (world space) */
  x: number
  y: number
  /** Point tuple for convenience */
  point: [number, number]
}

export interface JointWithPosition {
  name: string
  position: [number, number]
}

export interface LinkWithPosition {
  name: string
  start: [number, number]
  end: [number, number]
}

export interface NearestResult<T> {
  item: T
  distance: number
}

export interface UseCanvasInteractionConfig {
  /** Reference to the canvas element */
  canvasRef: RefObject<HTMLDivElement | SVGSVGElement | null>
  /** Pixels per unit for coordinate conversion */
  pixelsPerUnit: number
}

export interface UseCanvasInteractionReturn {
  /** Convert pixel coordinates to unit coordinates */
  pixelsToUnits: (pixels: number) => number
  /** Convert unit coordinates to pixel coordinates */
  unitsToPixels: (units: number) => number
  /** Get canvas coordinates from a mouse event */
  getCanvasCoordinates: (event: React.MouseEvent) => CanvasCoordinates | null
  /** Calculate distance between two points */
  calculateDistance: (p1: [number, number], p2: [number, number]) => number
  /** Find the nearest joint to a point */
  findNearestJoint: (
    point: [number, number],
    joints: JointWithPosition[],
    maxDistance?: number
  ) => (JointWithPosition & { distance: number }) | null
  /** Find the nearest link to a point */
  findNearestLink: (
    point: [number, number],
    links: LinkWithPosition[],
    maxDistance?: number
  ) => (LinkWithPosition & { distance: number }) | null
  /** Check if a point is within a bounding box */
  isPointInBounds: (
    point: [number, number],
    bounds: { minX: number; minY: number; maxX: number; maxY: number }
  ) => boolean
  /** Get bounding box from corner points */
  getBoundsFromCorners: (
    corner1: [number, number],
    corner2: [number, number]
  ) => { minX: number; minY: number; maxX: number; maxY: number }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Hook providing canvas interaction utilities
 */
export function useCanvasInteraction(
  config: UseCanvasInteractionConfig
): UseCanvasInteractionReturn {
  const { canvasRef, pixelsPerUnit } = config

  // ─────────────────────────────────────────────────────────────────────────
  // COORDINATE CONVERSION
  // ─────────────────────────────────────────────────────────────────────────

  const pixelsToUnits = useCallback((pixels: number) => {
    return pixels / pixelsPerUnit
  }, [pixelsPerUnit])

  const unitsToPixels = useCallback((units: number) => {
    return units * pixelsPerUnit
  }, [pixelsPerUnit])

  const getCanvasCoordinates = useCallback((event: React.MouseEvent): CanvasCoordinates | null => {
    if (!canvasRef.current) return null

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelX / pixelsPerUnit
    const y = pixelY / pixelsPerUnit

    return {
      pixelX,
      pixelY,
      x,
      y,
      point: [x, y]
    }
  }, [canvasRef, pixelsPerUnit])

  // ─────────────────────────────────────────────────────────────────────────
  // DISTANCE CALCULATION
  // ─────────────────────────────────────────────────────────────────────────

  const calculateDistance = useCallback((p1: [number, number], p2: [number, number]): number => {
    const dx = p2[0] - p1[0]
    const dy = p2[1] - p1[1]
    return Math.sqrt(dx * dx + dy * dy)
  }, [])

  // ─────────────────────────────────────────────────────────────────────────
  // NEAREST ELEMENT FINDING
  // ─────────────────────────────────────────────────────────────────────────

  const findNearestJoint = useCallback((
    point: [number, number],
    joints: JointWithPosition[],
    maxDistance: number = 2 // Default 2 units threshold
  ): (JointWithPosition & { distance: number }) | null => {
    let nearest: (JointWithPosition & { distance: number }) | null = null
    let nearestDist = maxDistance

    for (const joint of joints) {
      const dist = calculateDistance(point, joint.position)
      if (dist < nearestDist) {
        nearestDist = dist
        nearest = { ...joint, distance: dist }
      }
    }

    return nearest
  }, [calculateDistance])

  const findNearestLink = useCallback((
    point: [number, number],
    links: LinkWithPosition[],
    maxDistance: number = 1 // Default 1 unit threshold
  ): (LinkWithPosition & { distance: number }) | null => {
    let nearest: (LinkWithPosition & { distance: number }) | null = null
    let nearestDist = maxDistance

    for (const link of links) {
      // Calculate distance from point to line segment
      const dist = pointToLineDistance(point, link.start, link.end)
      if (dist < nearestDist) {
        nearestDist = dist
        nearest = { ...link, distance: dist }
      }
    }

    return nearest
  }, [])

  // ─────────────────────────────────────────────────────────────────────────
  // BOUNDS CHECKING
  // ─────────────────────────────────────────────────────────────────────────

  const isPointInBounds = useCallback((
    point: [number, number],
    bounds: { minX: number; minY: number; maxX: number; maxY: number }
  ): boolean => {
    return point[0] >= bounds.minX && point[0] <= bounds.maxX &&
           point[1] >= bounds.minY && point[1] <= bounds.maxY
  }, [])

  const getBoundsFromCorners = useCallback((
    corner1: [number, number],
    corner2: [number, number]
  ): { minX: number; minY: number; maxX: number; maxY: number } => {
    return {
      minX: Math.min(corner1[0], corner2[0]),
      minY: Math.min(corner1[1], corner2[1]),
      maxX: Math.max(corner1[0], corner2[0]),
      maxY: Math.max(corner1[1], corner2[1])
    }
  }, [])

  return {
    pixelsToUnits,
    unitsToPixels,
    getCanvasCoordinates,
    calculateDistance,
    findNearestJoint,
    findNearestLink,
    isPointInBounds,
    getBoundsFromCorners
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Calculate the distance from a point to a line segment
 */
function pointToLineDistance(
  point: [number, number],
  lineStart: [number, number],
  lineEnd: [number, number]
): number {
  const [px, py] = point
  const [x1, y1] = lineStart
  const [x2, y2] = lineEnd

  const dx = x2 - x1
  const dy = y2 - y1
  const lengthSq = dx * dx + dy * dy

  if (lengthSq === 0) {
    // Line segment is a point
    return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
  }

  // Project point onto line segment
  let t = ((px - x1) * dx + (py - y1) * dy) / lengthSq
  t = Math.max(0, Math.min(1, t))

  // Closest point on line segment
  const closestX = x1 + t * dx
  const closestY = y1 + t * dy

  return Math.sqrt((px - closestX) ** 2 + (py - closestY) ** 2)
}

export default useCanvasInteraction
