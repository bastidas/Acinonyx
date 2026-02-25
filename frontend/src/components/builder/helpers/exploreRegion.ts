/**
 * Explore region helper
 *
 * Samples points in a circle around a center for "explore node trajectories" mode.
 * Pure and testable; coordinates are in mechanism units (same as node positions).
 */

export interface ExploreRegionPoint {
  position: [number, number]
  angleDeg: number
  radius: number
}

export interface ExploreRegionOptions {
  /** Angular step in degrees (default 10). */
  deltaDegrees?: number
  /** Radial step in units (default 5). Ignored when nRadialSamples is set. */
  rDelta?: number
  /** Maximum radius in units (default 20). */
  R?: number
  /**
   * Number of radial sample points including center (e.g. 5 → 0, R/4, R/2, 3R/4, R).
   * When set, overrides rDelta. 2 → sample at 0, R/2, R.
   */
  nRadialSamples?: number
}

const DEFAULT_DELTA_DEGREES = 10
const DEFAULT_R_DELTA = 5
const DEFAULT_R = 20

/** Divisors of 360 (angles possible with integer degree step). */
const ANGLE_DIVISORS = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 60, 72, 90, 120, 180, 360]

export interface ExploreRegionOptionsForMaxPoints {
  deltaDegrees: number
  nRadialSamples: number
}

/**
 * Choose deltaDegrees and nRadialSamples so that total sample count ≤ maxPoints.
 * Uses divisors of 360 for angular steps. Returns options that get close to maxPoints.
 */
export function getExploreRegionOptionsForMaxPoints(
  maxPoints: number,
  _R?: number
): ExploreRegionOptionsForMaxPoints {
  const cap = Math.max(1, Math.min(maxPoints, 360 * 50))
  let bestCount = 0
  let best: ExploreRegionOptionsForMaxPoints = { deltaDegrees: 10, nRadialSamples: 5 }
  for (let nRadial = 1; nRadial <= cap; nRadial++) {
    const maxAngles = Math.min(Math.floor(cap / nRadial), 360)
    const nAngles = ANGLE_DIVISORS.filter(d => d <= maxAngles).pop() ?? 1
    const count = nRadial * nAngles
    if (count <= maxPoints && count > bestCount) {
      bestCount = count
      best = {
        deltaDegrees: 360 / nAngles,
        nRadialSamples: nRadial
      }
    }
  }
  return best
}

/**
 * Choose radial and angular sampling for the second (combinatorial) exploration so that
 * nRadial * nAngles ≤ n2Max. Starts from desired radial and angular counts and reduces
 * angular (to previous divisor of 360) then radial incrementally until under the cap.
 * Use this to get an evenly sampled bounding for the second simulation.
 */
export function getCombinatorialSecondOptions(
  n2Max: number,
  desiredRadial: number,
  desiredAngles: number
): ExploreRegionOptionsForMaxPoints {
  if (n2Max < 1) {
    return { deltaDegrees: 360, nRadialSamples: 1 }
  }
  const maxRadial = Math.min(desiredRadial, n2Max, 360)
  let nRadial = Math.max(1, maxRadial)
  let nAngles = ANGLE_DIVISORS.filter(d => d <= Math.min(desiredAngles, Math.floor(n2Max / nRadial))).pop() ?? 1
  while (nRadial * nAngles > n2Max) {
    const idx = ANGLE_DIVISORS.indexOf(nAngles)
    if (idx > 0) {
      nAngles = ANGLE_DIVISORS[idx - 1]
    } else {
      nRadial = Math.max(1, nRadial - 1)
      nAngles = ANGLE_DIVISORS.filter(d => d <= Math.floor(n2Max / nRadial)).pop() ?? 1
    }
  }
  return {
    deltaDegrees: 360 / nAngles,
    nRadialSamples: nRadial
  }
}

/**
 * Generate sample points in a circle around center.
 * Order: for each angle from 0 to 360 (exclusive of 360) by deltaDegrees,
 * for each radius (either rDelta..R by rDelta, or nRadialSamples points from 0 to R).
 * Position: center + [r*cos(θ), r*sin(θ)] with θ in radians (0° = +x).
 *
 * @param center - [x, y] in mechanism units
 * @param options - deltaDegrees, rDelta, R (defaults: 10, 5, 20), optional nRadialSamples
 * @returns Array of { position, angleDeg, radius } in stable order
 */
export function exploreRegion(
  center: [number, number],
  options?: ExploreRegionOptions
): ExploreRegionPoint[] {
  const deltaDegrees = options?.deltaDegrees ?? DEFAULT_DELTA_DEGREES
  const R = options?.R ?? DEFAULT_R
  const nRadial = options?.nRadialSamples

  let radii: number[] =
    nRadial != null && nRadial >= 1
      ? nRadial === 1
        ? [0]
        : Array.from({ length: nRadial }, (_, i) => (R * i) / (nRadial - 1))
      : (() => {
          const rDelta = options?.rDelta ?? DEFAULT_R_DELTA
          const list: number[] = []
          for (let r = rDelta; r <= R; r += rDelta) list.push(r)
          return list
        })()
  if (radii.length > 0 && radii[0] > 0) {
    radii = [0, ...radii]
  }

  const out: ExploreRegionPoint[] = []

  for (let angleDeg = 0; angleDeg < 360; angleDeg += deltaDegrees) {
    const thetaRad = (angleDeg * Math.PI) / 180
    const cos = Math.cos(thetaRad)
    const sin = Math.sin(thetaRad)

    for (const r of radii) {
      const position: [number, number] = [
        center[0] + r * cos,
        center[1] + r * sin
      ]
      out.push({ position, angleDeg, radius: r })
    }
  }

  return out
}
