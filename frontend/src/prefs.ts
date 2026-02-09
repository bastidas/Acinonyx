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
const COOKIE_EXPIRES_DAYS = 365

// ─── Stored settings (builder preferences) ───────────────────────────────────

export type StoredCanvasBgColor = 'default' | 'white' | 'cream' | 'dark'
export type StoredTrajectoryStyle = 'dots' | 'line' | 'both'
export type StoredSelectionHighlightColor = 'blue' | 'orange' | 'green' | 'purple'
export type StoredColorCycleType = 'rainbow' | 'fire' | 'glow'

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
  trajectoryDotSize: number
  trajectoryDotOutline: boolean
  trajectoryDotOpacity: number
  selectionHighlightColor: StoredSelectionHighlightColor
  trajectoryStyle: StoredTrajectoryStyle
}

const MIN_SIMULATION_STEPS = 4
const MAX_SIMULATION_STEPS = 256

export const DEFAULT_STORED_SETTINGS: StoredSettings = {
  showGrid: true,
  showJointLabels: false,
  showLinkLabels: false,
  canvasBgColor: 'default',
  simulationSteps: 64,
  autoSimulateDelayMs: 5,
  trajectoryColorCycle: 'rainbow',
  jointMergeRadius: 4,
  jointSize: 8,
  linkThickness: 8,
  trajectoryDotSize: 4,
  trajectoryDotOutline: false,
  trajectoryDotOpacity: 0.85,
  selectionHighlightColor: 'blue',
  trajectoryStyle: 'both'
}

const VALID_CANVAS_BG: StoredCanvasBgColor[] = ['default', 'white', 'cream', 'dark']
const VALID_TRAJECTORY_STYLE: StoredTrajectoryStyle[] = ['dots', 'line', 'both']
const VALID_SELECTION_COLOR: StoredSelectionHighlightColor[] = ['blue', 'orange', 'green', 'purple']
const VALID_COLOR_CYCLE: StoredColorCycleType[] = ['rainbow', 'fire', 'glow']

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
      jointMergeRadius: clampNum(parsed.jointMergeRadius, 0, 50, DEFAULT_STORED_SETTINGS.jointMergeRadius),
      jointSize: clampNum(parsed.jointSize, 1, 32, DEFAULT_STORED_SETTINGS.jointSize),
      linkThickness: clampNum(parsed.linkThickness, 1, 32, DEFAULT_STORED_SETTINGS.linkThickness),
      trajectoryDotSize: clampNum(parsed.trajectoryDotSize, 1, 24, DEFAULT_STORED_SETTINGS.trajectoryDotSize),
      trajectoryDotOutline: typeof parsed.trajectoryDotOutline === 'boolean' ? parsed.trajectoryDotOutline : DEFAULT_STORED_SETTINGS.trajectoryDotOutline,
      trajectoryDotOpacity: clampNum(parsed.trajectoryDotOpacity, 0, 1, DEFAULT_STORED_SETTINGS.trajectoryDotOpacity),
      selectionHighlightColor: oneOf(parsed.selectionHighlightColor, VALID_SELECTION_COLOR, DEFAULT_STORED_SETTINGS.selectionHighlightColor),
      trajectoryStyle: oneOf(parsed.trajectoryStyle, VALID_TRAJECTORY_STYLE, DEFAULT_STORED_SETTINGS.trajectoryStyle)
    }
  } catch {
    return { ...DEFAULT_STORED_SETTINGS }
  }
}

export function setStoredSettings(settings: StoredSettings): void {
  Cookies.set(SETTINGS_KEY, JSON.stringify(settings), { expires: COOKIE_EXPIRES_DAYS })
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
