/**
 * Exploration colormap: cyclic hue by angle, radial modulation by radius.
 * Used when "Color-map exploration" is on for the explore node trajectories tool.
 * Radial mapping is modular: e.g. saturation (muted center → vivid edge) or brightness (bright center → dark edge).
 */

import * as d3 from 'd3'
import { Hsluv } from 'hsluv'

export type ExplorationColormapType = 'rainbow' | 'twilight' | 'husl'

/** How radius modulates the base hue color: saturation = muted center → max saturation at edge; brightness = bright center → dark edge. */
export type ExplorationRadialMapping = 'saturation' | 'brightness'

/** Default radial mapping: muted at center, maximally saturated at outer radius. */
export const DEFAULT_EXPLORATION_RADIAL_MAPPING: ExplorationRadialMapping = 'saturation'

/** All radial mapping options (for future UI or programmatic choice). */
export const EXPLORATION_RADIAL_MAPPINGS: readonly ExplorationRadialMapping[] = ['saturation', 'brightness']

/** Very faint grey for invalid exploration samples (barely visible so valid dots stand out) */
export const EXPLORE_INVALID_GREY = '#e0e0e0'

/** Color for the center sample (radius 0) when colormap is on */
export const EXPLORE_CENTER_COLOR = '#333333'

/** Saturation at center (radialT=0). 0 = grey, ~0.1–0.2 = muted, 1 = full color. Tweak to explore. */
export const CENTER_SATURATION = 0.4
/** Saturation at edge (radialT=1). Target 0–1 (1 = full). Base hues from d3 are often < 1, so we use this as target, not multiplier. */
export const EDGE_SATURATION_SCALE = 1
/** Lightness at center for saturation mapping (so center is soft, not grey-white). */
const CENTER_LIGHTNESS = 0.1


/**
 * Twilight-style cyclic key colors (blue → purple → pink → orange → blue).
 * Matches matplotlib's twilight colormap spirit.
 */
const TWILIGHT_KEY_COLORS = [
  '#2d5a8b', // blue
  '#6b2d7b', // purple
  '#b82d5a', // pink/magenta
  '#d46b2d', // orange
  '#2d5a8b'  // blue again (cycle)
]

/**
 * Get cyclic hue color at t in [0, 1) for the given colormap type.
 * Does not apply radial brightness.
 */
function getCyclicHueColor(t: number, type: ExplorationColormapType): string {
  switch (type) {
    case 'rainbow':
      return d3.interpolateRainbow(t)
    case 'twilight': {
      const i = t * (TWILIGHT_KEY_COLORS.length - 1)
      const idx = Math.floor(i)
      const frac = i - idx
      return d3.interpolateRgb(TWILIGHT_KEY_COLORS[idx], TWILIGHT_KEY_COLORS[idx + 1])(frac)
    }
    case 'husl': {
      const conv = new Hsluv()
      conv.hsluv_h = t * 360
      conv.hsluv_s = 80
      conv.hsluv_l = 70
      conv.hsluvToHex()
      return conv.hex
    }
    default:
      return d3.interpolateRainbow(t)
  }
}

// ---- Brightness vs saturation ----
// Brightness mapping: varies only L (lightness). Same hue and saturation everywhere; center = bright/light, edge = dark.
// Saturation mapping: varies S (and optionally L). Same hue; center = muted (low S), edge = full/vivid color (high S).
// So: brightness = light→dark gradient; saturation = muted→vivid gradient.

/**
 * Modulate brightness: center (radialT=0) bright, edge (radialT=1) dark.
 * Only lightness (L) changes; hue and saturation stay the same.
 */
function applyRadialBrightness(hexOrRgb: string, radialT: number): string {
  const clamped = Math.max(0, Math.min(1, radialT))
  const c = d3.color(hexOrRgb)
  if (!c) return hexOrRgb
  const hsl = d3.hsl(c)
  if (!hsl) return hexOrRgb
  const L = 0.95 - 0.2 * clamped
  const out = d3.hsl(hsl.h, hsl.s, L)
  return out.formatHex()
}



/**
 * Modulate by saturation: center uses CENTER_SATURATION (muted), edge uses EDGE_SATURATION_SCALE as target (1 = full).
 */
function applyRadialSaturation(hexOrRgb: string, radialT: number): string {
  const clamped = Math.max(0, Math.min(1, radialT))
  const c = d3.color(hexOrRgb)
  if (!c) return hexOrRgb
  const hsl = d3.hsl(c)
  if (!hsl) return hexOrRgb
  const edgeS = Math.min(1, EDGE_SATURATION_SCALE)
  const S = CENTER_SATURATION + (edgeS - CENTER_SATURATION) * clamped
  const L = CENTER_LIGHTNESS + (hsl.l - CENTER_LIGHTNESS) * clamped
  const out = d3.hsl(hsl.h, S, L)
  return out.formatHex()
}

type RadialMapper = (hex: string, radialT: number) => string

const RADIAL_MAPPERS: Record<ExplorationRadialMapping, RadialMapper> = {
  saturation: applyRadialSaturation,
  brightness: applyRadialBrightness
}

/**
 * Apply the chosen radial mapping to a base hue color.
 */
function applyRadialMapping(
  hexOrRgb: string,
  radialT: number,
  mapping: ExplorationRadialMapping
): string {
  return RADIAL_MAPPERS[mapping](hexOrRgb, radialT)
}

/**
 * Get exploration colormap color for a sample.
 *
 * @param angleT - Angle normalized to [0, 1) (0 = 0°, 1 = 360°)
 * @param radialT - Radius normalized to [0, 1] (0 = center, 1 = max radius)
 * @param type - Colormap type: rainbow, twilight, or husl
 * @param radialMapping - How radius modulates color; default = saturation (muted center → vivid edge)
 * @returns RGB hex string
 */
export function getExplorationColormapColor(
  angleT: number,
  radialT: number,
  type: ExplorationColormapType,
  radialMapping: ExplorationRadialMapping = DEFAULT_EXPLORATION_RADIAL_MAPPING
): string {
  const base = getCyclicHueColor(angleT, type)
  return applyRadialMapping(base, radialT, radialMapping)
}

/**
 * Compute angle and radial t from sample position and explore center.
 * Safe when exploreRadius is 0 (returns radialT = 0).
 */
export function positionToAngleAndRadialT(
  position: [number, number],
  center: [number, number],
  exploreRadius: number
): { angleT: number; radialT: number } {
  const dx = position[0] - center[0]
  const dy = position[1] - center[1]
  const r = Math.sqrt(dx * dx + dy * dy)
  const angle = Math.atan2(dy, dx)
  // angle in [-π, π] → [0, 1)
  const angleT = (angle / (2 * Math.PI) + 0.5) % 1
  const radialT = exploreRadius > 0 ? Math.min(1, r / exploreRadius) : 0
  return { angleT, radialT }
}
