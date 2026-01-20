/**
 * Configuration constants for the Builder component
 */

import { MERGE_THRESHOLD } from '../BuilderTools'
import type { ColorCycleType } from '../../theme'

// ═══════════════════════════════════════════════════════════════════════════════
// CANVAS CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/** Pixels per unit for canvas scaling (6 pixels = 1 unit) */
export const PIXELS_PER_UNIT = 6

/** Minimum canvas dimensions in pixels */
export const CANVAS_MIN_WIDTH_PX = 800
export const CANVAS_MIN_HEIGHT_PX = 500

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION SETTINGS
// ═══════════════════════════════════════════════════════════════════════════════

/** Range for simulation steps input */
export const MIN_SIMULATION_STEPS = 4
export const MAX_SIMULATION_STEPS = 256

/** Default simulation step count */
export const DEFAULT_SIMULATION_STEPS = 64

/** Default delay before auto-simulation triggers (ms) */
export const DEFAULT_AUTO_SIMULATE_DELAY_MS = 5

// ═══════════════════════════════════════════════════════════════════════════════
// INTERACTION SETTINGS
// ═══════════════════════════════════════════════════════════════════════════════

/** Default radius for joint merge detection (in units) */
export const DEFAULT_JOINT_MERGE_RADIUS = MERGE_THRESHOLD

/** Default color cycle for trajectories */
export const DEFAULT_TRAJECTORY_COLOR_CYCLE: ColorCycleType = 'rainbow'

// ═══════════════════════════════════════════════════════════════════════════════
// TOOLBAR LAYOUT
// ═══════════════════════════════════════════════════════════════════════════════

/** Padding inside the tools toolbar */
export const TOOLS_PADDING = 8

/** Width of the tools box */
export const TOOLS_BOX_WIDTH = 200

/** Tool button dimensions */
export const TOOL_BUTTON_SIZE = 40
export const TOOL_BUTTON_FONT_SIZE = 16
export const TOOL_BUTTON_BORDER_RADIUS = 8

// ═══════════════════════════════════════════════════════════════════════════════
// VISUAL DEFAULTS
// ═══════════════════════════════════════════════════════════════════════════════

/** Default joint visual size (px) */
export const DEFAULT_JOINT_SIZE = 8

/** Default link thickness (px) */
export const DEFAULT_LINK_THICKNESS = 8

/** Default trajectory dot size (px) */
export const DEFAULT_TRAJECTORY_DOT_SIZE = 4

/** Default selection highlight color */
export const DEFAULT_SELECTION_COLOR = 'blue' as const
