/* =============================================================================
   ACINONYX THEME - TypeScript Theme Constants
   =============================================================================
   
   Use these constants in React/TypeScript code where CSS variables aren't ideal.
   These values mirror the CSS custom properties in theme.css
   
   USAGE:
   import { theme, colors, statusColors } from './theme'
   
   <Box sx={{ backgroundColor: colors.primary }}>
   ============================================================================= */

import * as d3 from 'd3'

// Primary Palette
export const colors = {
  // Main brand colors
  primary: '#FA8112',
  primaryLight: '#FFB347',
  primaryDark: '#D96A00',
  
  // Background colors
  bgTan: '#FAF3E1',
  bgTanAlt: '#F5E7C6',
  bgTanAccent: '#EAC381',
  bgDark: '#222222',
  bgDarkAlt: '#231A32',
  
  // Text colors
  textPrimary: '#222222',
  textSecondary: '#555555',
  textMuted: '#888888',
  textOnDark: '#FAF3E1',
  textOnPrimary: '#FFFFFF',
  
  // Surface colors
  surface: '#FFFFFF',
  surfaceHover: '#F5F5F5',
  border: '#E0E0E0',
  borderLight: '#EEEEEE',
  divider: '#E0E0E0',
} as const

// Status & Feedback Colors
export const statusColors = {
  success: '#B4D6A9',
  successDark: '#4CAF50',
  nominal: '#A2C8E7',
  nominalDark: '#1976D2',
  warning: '#E6A743',
  warningDark: '#ED6C02',
  error: '#D32F2F',
  errorLight: '#FFEBEE',
} as const

// Graph/Visualization Palette (D3-compatible)
export const graphColors = [
  '#1F77B4', // blue
  '#FF7F0E', // orange
  '#2CA02C', // green
  '#D62728', // red
  '#9467BD', // purple
  '#8C564B', // brown
  '#E377C2', // pink
  '#7F7F7F', // gray
  '#BCBD22', // olive
  '#17BECF', // cyan
] as const

// Named graph colors for specific use
export const graphColorNames = {
  blue: '#1F77B4',
  orange: '#FF7F0E',
  green: '#2CA02C',
  red: '#D62728',
  purple: '#9467BD',
  brown: '#8C564B',
  pink: '#E377C2',
  gray: '#7F7F7F',
  olive: '#BCBD22',
  cyan: '#17BECF',
} as const

// Joint/Linkage specific colors
export const jointColors = {
  static: '#E74C3C',
  crank: '#F39C12',
  pivot: '#2196F3',
  ground: '#888888',
  selected: '#1976D2',
  moveGroup: '#9E9E9E',
  mergeHighlight: '#00BCD4',
} as const

// Helper function to get graph color by index (cycles through palette)
export function getGraphColor(index: number): string {
  return graphColors[index % graphColors.length]
}

// MUI Theme configuration - use this in createTheme()
export const muiThemeConfig = {
  palette: {
    mode: 'light' as const,
    primary: {
      main: colors.primary,
      light: colors.primaryLight,
      dark: colors.primaryDark,
      contrastText: colors.textOnPrimary,
    },
    secondary: {
      main: colors.bgDark,
      light: '#4A4A4A',
      dark: '#000000',
      contrastText: colors.textOnDark,
    },
    success: {
      main: statusColors.successDark,
      light: statusColors.success,
    },
    warning: {
      main: statusColors.warningDark,
      light: statusColors.warning,
    },
    error: {
      main: statusColors.error,
      light: statusColors.errorLight,
    },
    info: {
      main: statusColors.nominalDark,
      light: statusColors.nominal,
    },
    background: {
      default: colors.surface,
      paper: colors.surface,
    },
    text: {
      primary: colors.textPrimary,
      secondary: colors.textSecondary,
      disabled: colors.textMuted,
    },
    divider: colors.divider,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none' as const,
        },
      },
    },
  },
}

// Dark mode MUI theme config
export const muiThemeConfigDark = {
  ...muiThemeConfig,
  palette: {
    ...muiThemeConfig.palette,
    mode: 'dark' as const,
    background: {
      default: colors.bgDark,
      paper: colors.bgDarkAlt,
    },
    text: {
      primary: colors.textOnDark,
      secondary: '#CCCCCC',
      disabled: colors.textMuted,
    },
  },
}

/* =============================================================================
   CYCLIC COLOR GRADIENTS FOR TRAJECTORY VISUALIZATION
   =============================================================================
   
   These functions generate colors that cycle back to the starting color at the
   end of the trajectory. This creates a smooth visual loop where:
   - At t=0 (start): Starting color
   - At t=0.5 (middle): Opposite/contrasting color  
   - At t=1 (end): Returns to starting color (or very close to it)
   
   USAGE:
   const color = getCyclicColor(stepIndex, totalSteps, 'rainbow')
   
   Available cycle types:
   - 'rainbow': Full hue rotation (red → yellow → green → cyan → blue → magenta → red)
   - 'fire': Orange/red → dark → orange/red (warm, dramatic)
   - 'glow': Orange/red → light/white → orange/red (bright, ethereal)
   ============================================================================= */

export type ColorCycleType = 'rainbow' | 'fire' | 'glow'

/**
 * Get a color from a cyclic gradient that returns to its starting color.
 * 
 * @param stepIndex - Current step in the simulation (0 to totalSteps-1)
 * @param totalSteps - Total number of simulation steps
 * @param cycleType - Type of color cycle: 'rainbow', 'fire', or 'glow'
 * @returns RGB color string like 'rgb(255, 128, 0)'
 * 
 * The color at stepIndex=0 will be nearly identical to the color at stepIndex=totalSteps-1,
 * creating a smooth visual loop for cyclic animations.
 */
export function getCyclicColor(
  stepIndex: number, 
  totalSteps: number, 
  cycleType: ColorCycleType = 'rainbow'
): string {
  // Normalize t to [0, 1) - note we don't reach 1 to avoid exact duplicate at end
  // This ensures step 0 and step (totalSteps-1) are very close but not identical
  const t = stepIndex / Math.max(1, totalSteps)
  
  switch (cycleType) {
    case 'rainbow':
      return getRainbowCycleColor(t)
    case 'fire':
      return getFireCycleColor(t)
    case 'glow':
      return getGlowCycleColor(t)
    default:
      return getRainbowCycleColor(t)
  }
}

/**
 * RAINBOW CYCLE: Full hue rotation through the color wheel
 * 
 * Uses HSL color space with constant saturation and lightness.
 * Hue rotates 360° so the color returns to the starting point.
 * 
 * t=0.00: Red (hue=0°)
 * t=0.17: Yellow (hue=60°)
 * t=0.33: Green (hue=120°)
 * t=0.50: Cyan (hue=180°)
 * t=0.67: Blue (hue=240°)
 * t=0.83: Magenta (hue=300°)
 * t=1.00: Red again (hue=360°=0°)
 */
function getRainbowCycleColor(t: number): string {
  // d3.interpolateRainbow uses a perceptually-uniform rainbow
  // It naturally cycles back to the starting color at t=1
  return d3.interpolateRainbow(t)
}

/**
 * FIRE CYCLE: Orange/red → dark black → orange/red
 * 
 * Creates a warm, dramatic effect like embers glowing and fading.
 * Uses HSL interpolation through a dark midpoint.
 * 
 * t=0.00: Bright orange (#FA8112 - our primary color)
 * t=0.25: Deep red/brown
 * t=0.50: Near black (#1A0A00)
 * t=0.75: Deep red/brown
 * t=1.00: Bright orange (returns to start)
 */
function getFireCycleColor(t: number): string {
  // Use a "bounce" function: 0→1→0 as t goes 0→0.5→1
  // This makes the color go: bright → dark → bright
  const bounce = 1 - Math.abs(2 * t - 1)
  
  // Interpolate from bright orange to near-black
  const startColor = d3.hsl(colors.primary)  // #FA8112 - bright orange
  const darkColor = d3.hsl(15, 0.8, 0.06)    // Very dark reddish-brown
  
  // Interpolate in HSL space for smoother color transitions
  const interpolator = d3.interpolateHsl(startColor.formatHsl(), darkColor.formatHsl())
  return interpolator(bounce)
}

/**
 * GLOW CYCLE: Orange/red → light/white → orange/red
 * 
 * Creates a bright, ethereal effect like a pulsing glow.
 * The midpoint is a warm white/cream color.
 * 
 * t=0.00: Bright orange (#FA8112 - our primary color)
 * t=0.25: Light peach
 * t=0.50: Near white/cream (#FFF8E8)
 * t=0.75: Light peach  
 * t=1.00: Bright orange (returns to start)
 */
function getGlowCycleColor(t: number): string {
  // Use a "bounce" function: 0→1→0 as t goes 0→0.5→1
  const bounce = 1 - Math.abs(2 * t - 1)
  
  // Interpolate from bright orange to warm white
  const startColor = d3.hsl(colors.primary)       // #FA8112 - bright orange
  const lightColor = d3.hsl(40, 1.0, 0.95)        // Warm cream/white
  
  // Interpolate in HSL space
  const interpolator = d3.interpolateHsl(startColor.formatHsl(), lightColor.formatHsl())
  return interpolator(bounce)
}

/**
 * Legacy function: Original trajectory color (non-cyclic, for backward compatibility)
 * Blue (start) → Cyan → Green → Yellow → Red (end)
 * 
 * @deprecated Use getCyclicColor() instead for cyclic trajectories
 */
export function getLegacyTrajectoryColor(stepIndex: number, totalSteps: number): string {
  const t = stepIndex / Math.max(1, totalSteps - 1)
  const r = Math.round(255 * t)
  const g = Math.round(255 * (1 - Math.abs(t - 0.5) * 2))
  const b = Math.round(255 * (1 - t))
  return `rgb(${r}, ${g}, ${b})`
}

/**
 * Legacy function: Spectral colormap approximation (non-cyclic)
 * Similar to matplotlib's Spectral colormap.
 * 
 * @deprecated Use getCyclicColor() instead for cyclic trajectories
 */
export function getSpectralColor(t: number): string {
  let r: number, g: number, b: number
  if (t < 0.25) {
    r = 158 + Math.floor((255-158) * t * 4)
    g = 1 + Math.floor((116-1) * t * 4)
    b = 5 + Math.floor((9-5) * t * 4)
  } else if (t < 0.5) {
    r = 255 - Math.floor((255-255) * (t-0.25) * 4)
    g = 116 + Math.floor((217-116) * (t-0.25) * 4)
    b = 9 + Math.floor((54-9) * (t-0.25) * 4)
  } else if (t < 0.75) {
    r = 255 - Math.floor((255-171) * (t-0.5) * 4)
    g = 217 + Math.floor((221-217) * (t-0.5) * 4)
    b = 54 + Math.floor((164-54) * (t-0.5) * 4)
  } else {
    r = 171 - Math.floor((171-94) * (t-0.75) * 4)
    g = 221 - Math.floor((221-79) * (t-0.75) * 4)
    b = 164 - Math.floor((164-162) * (t-0.75) * 4)
  }
  return `rgb(${r}, ${g}, ${b})`
}

// Complete theme object for convenience
export const theme = {
  colors,
  statusColors,
  graphColors,
  graphColorNames,
  jointColors,
  getGraphColor,
  getCyclicColor,
  getLegacyTrajectoryColor,
  getSpectralColor,
  muiThemeConfig,
  muiThemeConfigDark,
} as const

export default theme

