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
 * Largest divisor of 360 that is <= maxVal.
 */
function largestAngleDivisor(maxVal: number): number {
  const d = ANGLE_DIVISORS.filter(d => d <= maxVal).pop()
  return d ?? 1
}

/**
 * Previous divisor of 360 (smaller than the given one), or 1.
 */
function prevAngleDivisor(nAngles: number): number {
  const idx = ANGLE_DIVISORS.indexOf(nAngles)
  if (idx > 0) return ANGLE_DIVISORS[idx - 1]
  return 1
}

/**
 * Choose radial and angular sampling for the second (combinatorial) exploration so that
 * nRadial * nAngles ≤ n2Max. Reduces from desired radial/angular by alternating between
 * reducing radial and azimuthal (angular), so both dimensions keep some coverage.
 * We greatly favor radial: we prefer to reduce azimuthal (angles) first so the result
 * tends to have more radial samples and fewer angular (e.g. 33 radial × 3 azimuthal
 * for ~100 samples).
 */
export function getCombinatorialSecondOptions(
  n2Max: number,
  desiredRadial: number,
  desiredAngles: number
): ExploreRegionOptionsForMaxPoints {
  if (n2Max < 1) {
    return { deltaDegrees: 360, nRadialSamples: 1 }
  }
  let nRadial = Math.max(1, Math.min(desiredRadial, n2Max))
  let nAngles = largestAngleDivisor(Math.min(desiredAngles, n2Max))

  // Alternate reducing angles vs radial; greatly favor radial (reduce angles several times per radial reduction)
  const angleReductionsPerRadial = 3
  let angleReductionsSinceRadial = 0
  let preferAngles = true
  while (nRadial * nAngles > n2Max) {
    const canReduceAngles = nAngles > 1
    const canReduceRadial = nRadial > 1
    const shouldReduceAngles =
      canReduceAngles &&
      (preferAngles || !canReduceRadial) &&
      (angleReductionsSinceRadial < angleReductionsPerRadial || !canReduceRadial)

    if (shouldReduceAngles) {
      nAngles = prevAngleDivisor(nAngles)
      angleReductionsSinceRadial += 1
      preferAngles = false
    } else if (canReduceRadial) {
      nRadial -= 1
      angleReductionsSinceRadial = 0
      preferAngles = true
    } else {
      break
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
