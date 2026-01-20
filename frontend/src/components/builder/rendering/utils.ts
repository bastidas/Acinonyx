/**
 * Pure utility functions for SVG rendering
 *
 * These functions are pure (no side effects) and easily testable.
 */

import { HighlightType, ObjectType } from '../types'
import { JointColors } from './types'

// ═══════════════════════════════════════════════════════════════════════════════
// COORDINATE CONVERSION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create coordinate conversion functions for a given scale
 */
export function createCoordinateConverters(pixelsPerUnit: number) {
  return {
    pixelsToUnits: (pixels: number) => pixels / pixelsPerUnit,
    unitsToPixels: (units: number) => units * pixelsPerUnit
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLOW FILTER MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Maps a color to its corresponding SVG glow filter URL
 *
 * @param color - The color to get a glow filter for
 * @param jointColors - Joint color definitions for matching
 * @returns SVG filter URL string
 */
export function getGlowFilterForColor(color: string, jointColors: JointColors): string {
  const colorLower = color.toLowerCase()

  // Joint type colors
  if (colorLower === '#e74c3c' || colorLower === jointColors.static.toLowerCase()) return 'url(#glow-static)'
  if (colorLower === '#f39c12' || colorLower === jointColors.crank.toLowerCase()) return 'url(#glow-crank)'
  if (colorLower === '#2196f3' || colorLower === jointColors.pivot.toLowerCase()) return 'url(#glow-pivot)'

  // Graph colors (D3 palette)
  if (colorLower === '#1f77b4') return 'url(#glow-blue)'
  if (colorLower === '#ff7f0e') return 'url(#glow-orange)'
  if (colorLower === '#2ca02c') return 'url(#glow-green)'
  if (colorLower === '#d62728') return 'url(#glow-red)'
  if (colorLower === '#9467bd') return 'url(#glow-purple)'
  if (colorLower === '#8c564b') return 'url(#glow-brown)'
  if (colorLower === '#e377c2') return 'url(#glow-pink)'
  if (colorLower === '#7f7f7f') return 'url(#glow-gray)'
  if (colorLower === '#bcbd22') return 'url(#glow-olive)'
  if (colorLower === '#17becf') return 'url(#glow-cyan)'

  // Default to blue glow for unknown colors
  return 'url(#glow-blue)'
}

// ═══════════════════════════════════════════════════════════════════════════════
// HIGHLIGHT STYLING
// ═══════════════════════════════════════════════════════════════════════════════

export interface HighlightStyle {
  stroke: string
  strokeWidth: number
  filter?: string
}

/**
 * Calculates highlight styling for objects (joints, links, polygons)
 * Objects glow in their ORIGINAL color when selected/hovered
 * Move group uses grey glow
 *
 * @param objectType - Type of object being styled
 * @param highlightType - Current highlight state
 * @param baseColor - The object's original color (used for glow)
 * @param baseStrokeWidth - Default stroke width for the object
 * @param jointColors - Joint color definitions
 * @returns Styling object with stroke, strokeWidth, and optional filter
 */
export function getHighlightStyle(
  objectType: ObjectType,
  highlightType: HighlightType,
  baseColor: string,
  baseStrokeWidth: number,
  jointColors: JointColors
): HighlightStyle {
  // No highlight - return base styling
  if (highlightType === 'none') {
    return { stroke: baseColor, strokeWidth: baseStrokeWidth }
  }

  // Different stroke widths based on object type and highlight state
  const glowStrokeWidth = objectType === 'joint'
    ? baseStrokeWidth + 1
    : objectType === 'link'
      ? baseStrokeWidth + 2
      : baseStrokeWidth + 1

  // Move group and merge use special colors, selected/hovered use object's original color
  switch (highlightType) {
    case 'move_group':
      return {
        stroke: jointColors.moveGroup,
        strokeWidth: glowStrokeWidth,
        filter: 'url(#glow-movegroup)'
      }
    case 'merge':
      return {
        stroke: jointColors.mergeHighlight,
        strokeWidth: glowStrokeWidth,
        filter: 'url(#glow-merge)'
      }
    case 'selected':
    case 'hovered':
    default:
      // Glow in the object's original color
      return {
        stroke: baseColor,
        strokeWidth: glowStrokeWidth,
        filter: getGlowFilterForColor(baseColor, jointColors)
      }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GEOMETRY UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Calculate the midpoint between two positions
 */
export function getMidpoint(pos1: [number, number], pos2: [number, number]): [number, number] {
  return [(pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2]
}

/**
 * Calculate distance between two points
 */
export function getDistance(pos1: [number, number], pos2: [number, number]): number {
  const dx = pos2[0] - pos1[0]
  const dy = pos2[1] - pos1[1]
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Check if a position has meaningful movement from a reference position
 */
export function hasMovement(pos: [number, number], reference: [number, number], threshold = 0.001): boolean {
  return Math.abs(pos[0] - reference[0]) > threshold || Math.abs(pos[1] - reference[1]) > threshold
}

/**
 * Validate that a position has valid numeric coordinates
 */
export function isValidPosition(pos: unknown): pos is [number, number] {
  return Array.isArray(pos) &&
    pos.length >= 2 &&
    typeof pos[0] === 'number' &&
    typeof pos[1] === 'number' &&
    isFinite(pos[0]) &&
    isFinite(pos[1])
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATH GENERATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Generate SVG path data from an array of points
 *
 * @param points - Array of [x, y] coordinates
 * @param close - Whether to close the path (add Z command)
 * @param unitsToPixels - Coordinate conversion function
 * @returns SVG path data string
 */
export function generatePathData(
  points: [number, number][],
  close: boolean,
  unitsToPixels: (units: number) => number
): string {
  if (points.length === 0) return ''

  const pathData = points
    .filter(isValidPosition)
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`)
    .join(' ')

  return close ? pathData + ' Z' : pathData
}
