/**
 * Tests for exploreRegion
 *
 * Verifies correct count of points, angles in [0, 360), radii in [rDelta, R],
 * and positions at correct polar coordinates.
 */

import { describe, it, expect } from 'vitest'
import { exploreRegion } from './exploreRegion'

const TOL = 1e-10

describe('exploreRegion', () => {
  it('uses default options when none provided', () => {
    const center: [number, number] = [0, 0]
    const points = exploreRegion(center)

    // 36 angles (0, 10, ..., 350), 4 radii (5, 10, 15, 20) => 144 points
    expect(points.length).toBe(36 * 4)

    const angles = [...new Set(points.map((p) => p.angleDeg))].sort((a, b) => a - b)
    expect(angles).toHaveLength(36)
    expect(angles[0]).toBe(0)
    expect(angles[35]).toBe(350)

    const radii = [...new Set(points.map((p) => p.radius))].sort((a, b) => a - b)
    expect(radii).toEqual([5, 10, 15, 20])
  })

  it('orders by angle then radius (angle outer, radius inner)', () => {
    const center: [number, number] = [0, 0]
    const points = exploreRegion(center, { deltaDegrees: 90, rDelta: 5, R: 10 })

    // 4 angles (0, 90, 180, 270), 2 radii (5, 10) => 8 points
    expect(points.length).toBe(8)

    expect(points[0].angleDeg).toBe(0)
    expect(points[0].radius).toBe(5)
    expect(points[1].angleDeg).toBe(0)
    expect(points[1].radius).toBe(10)
    expect(points[2].angleDeg).toBe(90)
    expect(points[2].radius).toBe(5)
  })

  it('computes positions at correct polar coordinates', () => {
    const center: [number, number] = [0, 0]

    // 0° => (r, 0)
    const at0 = exploreRegion(center, { deltaDegrees: 360, rDelta: 10, R: 10 })[0]
    expect(at0.position[0]).toBeCloseTo(10, TOL)
    expect(at0.position[1]).toBeCloseTo(0, TOL)
    expect(at0.angleDeg).toBe(0)
    expect(at0.radius).toBe(10)

    // 90° => (0, r)
    const points90 = exploreRegion(center, { deltaDegrees: 90, rDelta: 10, R: 10 })
    const at90 = points90.find((p) => p.angleDeg === 90 && p.radius === 10)!
    expect(at90.position[0]).toBeCloseTo(0, TOL)
    expect(at90.position[1]).toBeCloseTo(10, TOL)

    // 180° => (-r, 0)
    const at180 = points90.find((p) => p.angleDeg === 180 && p.radius === 10)!
    expect(at180.position[0]).toBeCloseTo(-10, TOL)
    expect(at180.position[1]).toBeCloseTo(0, TOL)

    // 270° => (0, -r)
    const at270 = points90.find((p) => p.angleDeg === 270 && p.radius === 10)!
    expect(at270.position[0]).toBeCloseTo(0, TOL)
    expect(at270.position[1]).toBeCloseTo(-10, TOL)
  })

  it('offsets center correctly', () => {
    const center: [number, number] = [100, -50]
    const points = exploreRegion(center, { deltaDegrees: 360, rDelta: 20, R: 20 })

    expect(points.length).toBe(1)
    expect(points[0].position[0]).toBeCloseTo(100 + 20, TOL)
    expect(points[0].position[1]).toBeCloseTo(-50, TOL)
  })

  it('excludes 360 degrees (stops at 350 for default step)', () => {
    const center: [number, number] = [0, 0]
    const points = exploreRegion(center)

    const maxAngle = Math.max(...points.map((p) => p.angleDeg))
    expect(maxAngle).toBe(350)
    expect(points.some((p) => p.angleDeg === 360)).toBe(false)
  })

  it('respects custom deltaDegrees, rDelta, and R', () => {
    const center: [number, number] = [0, 0]
    const points = exploreRegion(center, {
      deltaDegrees: 45,
      rDelta: 2,
      R: 6
    })

    // 8 angles (0..315), 3 radii (2, 4, 6) => 24 points
    expect(points.length).toBe(24)

    const radii = [...new Set(points.map((p) => p.radius))].sort((a, b) => a - b)
    expect(radii).toEqual([2, 4, 6])

    const angles = [...new Set(points.map((p) => p.angleDeg))].sort((a, b) => a - b)
    expect(angles).toEqual([0, 45, 90, 135, 180, 225, 270, 315])
  })

  it('nRadialSamples: samples at 0, R/(n-1), ..., R (e.g. 3 → 0, 50, 100)', () => {
    const center: [number, number] = [0, 0]
    const points = exploreRegion(center, { deltaDegrees: 360, R: 100, nRadialSamples: 3 })

    expect(points.length).toBe(3) // 1 angle × 3 radii (0, 50, 100)
    const radii = [...new Set(points.map((p) => p.radius))].sort((a, b) => a - b)
    expect(radii).toEqual([0, 50, 100])
  })

  it('nRadialSamples: 5 gives 5 radii from 0 to R', () => {
    const center: [number, number] = [0, 0]
    const points = exploreRegion(center, { deltaDegrees: 90, R: 20, nRadialSamples: 5 })

    const radii = [...new Set(points.map((p) => p.radius))].sort((a, b) => a - b)
    expect(radii).toEqual([0, 5, 10, 15, 20])
    expect(points.length).toBe(4 * 5) // 4 angles × 5 radii
  })
})
