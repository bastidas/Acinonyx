/**
 * User preferences persisted in cookies (minimal payload).
 * - acinonyx-theme: 'light' | 'dark' (missing = use system prefer-color-scheme)
 * - acinonyx-graph: current graph filename (e.g. "4bar_20260202_164210.json")
 * - acinonyx-settings: JSON object for builder settings (grid, sizes, colors, etc.)
 */

import Cookies from 'js-cookie'

const THEME_KEY = 'acinonyx-theme'
const GRAPH_KEY = 'acinonyx-graph'
const SETTINGS_KEY = 'acinonyx-settings'
const TOOLBAR_HEIGHTS_KEY = 'acinonyx-toolbar-heights'
const COOKIE_EXPIRES_DAYS = 365

// ─── Stored settings (builder preferences) ───────────────────────────────────

export type StoredCanvasBgColor = 'default' | 'white' | 'cream' | 'dark'
export type StoredTrajectoryStyle = 'dots' | 'line' | 'both'
export type StoredSelectionHighlightColor = 'blue' | 'orange' | 'green' | 'purple'
export type StoredColorCycleType = 'rainbow' | 'fire' | 'glow' | 'twilight' | 'husl'
export type StoredLinkColorMode = 'various' | 'z-level' | 'single'

export interface StoredSettings {
  showGrid: boolean
  showJointLabels: boolean
  showLinkLabels: boolean
  canvasBgColor: StoredCanvasBgColor
  simulationSteps: number
  autoSimulateDelayMs: number
  trajectoryColorCycle: StoredColorCycleType
  jointMergeRadius: number
  jointSize: number
  linkThickness: number
  /** Link opacity 10–100 (percent). */
  linkTransparency: number
  /** Link color mode: various (per-link), z-level (by meta zlevel), single (one color). */
  linkColorMode: StoredLinkColorMode
  /** Hex color when linkColorMode is 'single'. */
  linkColorSingle: string
  /** Joint outline width in px (0 = none, 10 = max). */
  jointOutline: number
  trajectoryDotSize: number
  trajectoryDotOutline: boolean
  trajectoryDotOpacity: number
  /** Show step numbers (1..N) next to trajectory dots. Default off. */
  showTrajectoryStepNumbers: boolean
  selectionHighlightColor: StoredSelectionHighlightColor
  trajectoryStyle: StoredTrajectoryStyle
  /** Trajectory exploration: max radius in units (5–50). */
  exploreRadius: number
  /** Trajectory exploration: number of radial sample points (8–360). */
  exploreRadialSamples: number
  /** Trajectory exploration: number of angular (azimuthal) samples around the circle (8–45). */
  exploreAzimuthalSamples: number
  /** Trajectory exploration: max N1×N2 for combinatorial run (64–5128). */
  exploreNMaxCombinatorial: number
  /** Trajectory exploration: color-map dots and trajectories by angle/radius (default on). */
  exploreColormapEnabled: boolean
  /** Trajectory exploration: colormap type when exploreColormapEnabled is true. */
  exploreColormapType: 'rainbow' | 'twilight' | 'husl'
}

const MIN_SIMULATION_STEPS = 4
const MAX_SIMULATION_STEPS = 256

export const DEFAULT_STORED_SETTINGS: StoredSettings = {
  showGrid: true,
  showJointLabels: false,
  showLinkLabels: false,
  canvasBgColor: 'default',
  simulationSteps: 64,
  autoSimulateDelayMs: 0,
  trajectoryColorCycle: 'rainbow',
  jointMergeRadius: 2,
  jointSize: 5,
  linkThickness: 6,
  linkTransparency: 90,
  linkColorMode: 'z-level',
  linkColorSingle: '#555555',
  jointOutline: 1,
  trajectoryDotSize: 4,
  trajectoryDotOutline: false,
  trajectoryDotOpacity: 0.85,
  showTrajectoryStepNumbers: false,
  selectionHighlightColor: 'blue',
  trajectoryStyle: 'both',
  exploreRadius: 15,
  exploreRadialSamples: 5,
  exploreAzimuthalSamples: 10,
  exploreNMaxCombinatorial: 2048,
  exploreColormapEnabled: true,
  exploreColormapType: 'rainbow'
}

const VALID_CANVAS_BG: StoredCanvasBgColor[] = ['default', 'white', 'cream', 'dark']
const VALID_TRAJECTORY_STYLE: StoredTrajectoryStyle[] = ['dots', 'line', 'both']
const VALID_SELECTION_COLOR: StoredSelectionHighlightColor[] = ['blue', 'orange', 'green', 'purple']
const VALID_COLOR_CYCLE: StoredColorCycleType[] = ['rainbow', 'fire', 'glow', 'twilight', 'husl']
export type StoredExploreColormapType = 'rainbow' | 'twilight' | 'husl'
const VALID_EXPLORE_COLORMAP_TYPE: StoredExploreColormapType[] = ['rainbow', 'twilight', 'husl']
const VALID_LINK_COLOR_MODE: StoredLinkColorMode[] = ['various', 'z-level', 'single']

function clampNum(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === 'number' && !Number.isNaN(value) ? value : fallback
  return Math.min(max, Math.max(min, n))
}

function oneOf<T>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback
}

export function getStoredSettings(): StoredSettings {
  try {
    const raw = Cookies.get(SETTINGS_KEY)
    if (!raw) return { ...DEFAULT_STORED_SETTINGS }
    const parsed = JSON.parse(raw) as Record<string, unknown>
    return {
      showGrid: typeof parsed.showGrid === 'boolean' ? parsed.showGrid : DEFAULT_STORED_SETTINGS.showGrid,
      showJointLabels: typeof parsed.showJointLabels === 'boolean' ? parsed.showJointLabels : DEFAULT_STORED_SETTINGS.showJointLabels,
      showLinkLabels: typeof parsed.showLinkLabels === 'boolean' ? parsed.showLinkLabels : DEFAULT_STORED_SETTINGS.showLinkLabels,
      canvasBgColor: oneOf(parsed.canvasBgColor, VALID_CANVAS_BG, DEFAULT_STORED_SETTINGS.canvasBgColor),
      simulationSteps: clampNum(parsed.simulationSteps, MIN_SIMULATION_STEPS, MAX_SIMULATION_STEPS, DEFAULT_STORED_SETTINGS.simulationSteps),
      autoSimulateDelayMs: clampNum(parsed.autoSimulateDelayMs, 0, 10000, DEFAULT_STORED_SETTINGS.autoSimulateDelayMs),
      trajectoryColorCycle: oneOf(parsed.trajectoryColorCycle, VALID_COLOR_CYCLE, DEFAULT_STORED_SETTINGS.trajectoryColorCycle),
      jointMergeRadius: clampNum(parsed.jointMergeRadius, 1, 10, DEFAULT_STORED_SETTINGS.jointMergeRadius),
      jointSize: clampNum(parsed.jointSize, 1, 32, DEFAULT_STORED_SETTINGS.jointSize),
      linkThickness: clampNum(parsed.linkThickness, 1, 32, DEFAULT_STORED_SETTINGS.linkThickness),
      linkTransparency: clampNum(parsed.linkTransparency, 10, 100, DEFAULT_STORED_SETTINGS.linkTransparency),
      linkColorMode: oneOf(parsed.linkColorMode, VALID_LINK_COLOR_MODE, DEFAULT_STORED_SETTINGS.linkColorMode),
      linkColorSingle: typeof parsed.linkColorSingle === 'string' && /^#[0-9A-Fa-f]{6}$/.test(parsed.linkColorSingle) ? parsed.linkColorSingle : DEFAULT_STORED_SETTINGS.linkColorSingle,
      jointOutline: clampNum(parsed.jointOutline, 0, 10, DEFAULT_STORED_SETTINGS.jointOutline),
      trajectoryDotSize: clampNum(parsed.trajectoryDotSize, 1, 24, DEFAULT_STORED_SETTINGS.trajectoryDotSize),
      trajectoryDotOutline: typeof parsed.trajectoryDotOutline === 'boolean' ? parsed.trajectoryDotOutline : DEFAULT_STORED_SETTINGS.trajectoryDotOutline,
      trajectoryDotOpacity: clampNum(parsed.trajectoryDotOpacity, 0.1, 1, DEFAULT_STORED_SETTINGS.trajectoryDotOpacity),
      showTrajectoryStepNumbers: typeof parsed.showTrajectoryStepNumbers === 'boolean' ? parsed.showTrajectoryStepNumbers : DEFAULT_STORED_SETTINGS.showTrajectoryStepNumbers,
      selectionHighlightColor: oneOf(parsed.selectionHighlightColor, VALID_SELECTION_COLOR, DEFAULT_STORED_SETTINGS.selectionHighlightColor),
      trajectoryStyle: oneOf(parsed.trajectoryStyle, VALID_TRAJECTORY_STYLE, DEFAULT_STORED_SETTINGS.trajectoryStyle),
      exploreRadius: clampNum(parsed.exploreRadius, 5, 50, DEFAULT_STORED_SETTINGS.exploreRadius),
      exploreRadialSamples: clampNum(parsed.exploreRadialSamples, 8, 360, DEFAULT_STORED_SETTINGS.exploreRadialSamples),
      exploreAzimuthalSamples: clampNum(parsed.exploreAzimuthalSamples, 8, 45, DEFAULT_STORED_SETTINGS.exploreAzimuthalSamples),
      exploreNMaxCombinatorial: clampNum(parsed.exploreNMaxCombinatorial, 64, 5128, DEFAULT_STORED_SETTINGS.exploreNMaxCombinatorial),
      exploreColormapEnabled: typeof parsed.exploreColormapEnabled === 'boolean' ? parsed.exploreColormapEnabled : DEFAULT_STORED_SETTINGS.exploreColormapEnabled,
      exploreColormapType: oneOf(parsed.exploreColormapType, VALID_EXPLORE_COLORMAP_TYPE, DEFAULT_STORED_SETTINGS.exploreColormapType)
    }
  } catch {
    return { ...DEFAULT_STORED_SETTINGS }
  }
}

export function setStoredSettings(settings: StoredSettings): void {
  Cookies.set(SETTINGS_KEY, JSON.stringify(settings), { expires: COOKIE_EXPIRES_DAYS })
}

export type StoredToolbarHeights = Record<string, number>

export function getStoredToolbarHeights(): StoredToolbarHeights {
  try {
    const raw = Cookies.get(TOOLBAR_HEIGHTS_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw) as Record<string, unknown>
    const result: StoredToolbarHeights = {}
    for (const [key, value] of Object.entries(parsed)) {
      if (typeof value !== 'number' || Number.isNaN(value) || !Number.isFinite(value)) continue
      result[key] = Math.max(80, Math.min(5000, value))
    }
    return result
  } catch {
    return {}
  }
}

export function setStoredToolbarHeights(heights: StoredToolbarHeights): void {
  Cookies.set(TOOLBAR_HEIGHTS_KEY, JSON.stringify(heights), { expires: COOKIE_EXPIRES_DAYS })
}

export type ThemePreference = 'light' | 'dark'

/**
 * Effective theme: cookie value if set, otherwise system prefer-color-scheme.
 */
export function getEffectiveTheme(): ThemePreference {
  const stored = Cookies.get(THEME_KEY) as ThemePreference | undefined
  if (stored === 'light' || stored === 'dark') return stored
  if (typeof window !== 'undefined' && window.matchMedia?.('(prefer-color-scheme: dark)').matches) {
    return 'dark'
  }
  return 'light'
}

export function setThemePreference(theme: ThemePreference): void {
  Cookies.set(THEME_KEY, theme, { expires: COOKIE_EXPIRES_DAYS })
}

export function getStoredGraphFilename(): string | undefined {
  const value = Cookies.get(GRAPH_KEY)
  return value && value.trim() !== '' ? value : undefined
}

export function setStoredGraphFilename(filename: string | null): void {
  if (filename === null || filename.trim() === '') {
    Cookies.remove(GRAPH_KEY)
  } else {
    Cookies.set(GRAPH_KEY, filename, { expires: COOKIE_EXPIRES_DAYS })
  }
}

/**
 * Remove all Acinonyx cookies so the app falls back to defaults from this file.
 * Call from browser console: import('/src/prefs.ts').then(m => m.clearStoredPrefs())
 * Or expose on window for one-off use: window.__clearAcinonyxPrefs = clearStoredPrefs
 */
export function clearStoredPrefs(): void {
  Cookies.remove(THEME_KEY)
  Cookies.remove(GRAPH_KEY)
  Cookies.remove(SETTINGS_KEY)
  Cookies.remove(TOOLBAR_HEIGHTS_KEY)
}
