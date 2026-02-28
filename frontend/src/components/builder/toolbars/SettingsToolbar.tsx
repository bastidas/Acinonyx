/**
 * Settings Toolbar - Comprehensive settings panel for Builder
 */
import React, { useState, useRef, useEffect } from 'react'
import {
  Box, Typography, FormControlLabel, Switch, TextField, Select,
  MenuItem, FormControl, Divider, Slider
} from '@mui/material'
import type { SliderProps } from '@mui/material/Slider'
import type { ColorCycleType } from '../../../theme'

/**
 * Wrapper that keeps local state while dragging so the parent doesn't re-render on every
 * mouse move. Commits to parent only on release (onChangeCommitted), giving smooth thumb drag
 * like the MUI docs: https://mui.com/material-ui/react-slider/
 */
function SmoothSlider({
  value,
  onChange,
  ...rest
}: SliderProps & { value: number; onChange: (value: number) => void }) {
  const [local, setLocal] = useState(value)
  const isDragging = useRef(false)
  useEffect(() => {
    if (!isDragging.current) setLocal(value)
  }, [value])
  return (
    <Slider
      {...rest}
      value={local}
      onChange={(_, v) => {
        isDragging.current = true
        setLocal(v as number)
      }}
      onChangeCommitted={(_, v) => {
        isDragging.current = false
        const n = v as number
        setLocal(n)
        onChange(n)
      }}
    />
  )
}
import {
  MIN_SIMULATION_STEPS,
  MAX_SIMULATION_STEPS,
  EXPLORE_RADIUS_MIN,
  EXPLORE_RADIUS_MAX,
  EXPLORE_RADIAL_SAMPLES_MIN,
  EXPLORE_RADIAL_SAMPLES_MAX,
  EXPLORE_AZIMUTHAL_SAMPLES_MIN,
  EXPLORE_AZIMUTHAL_SAMPLES_MAX,
  EXPLORE_NMAX_COMBINATORIAL_MIN,
  EXPLORE_NMAX_COMBINATORIAL_MAX
} from '../constants'

export type CanvasBgColor = 'default' | 'white' | 'cream' | 'dark'
export type TrajectoryStyle = 'dots' | 'line' | 'both'
export type SelectionHighlightColor = 'blue' | 'orange' | 'green' | 'purple'
export type LinkColorMode = 'various' | 'z-level' | 'single'

export interface SettingsToolbarProps {
  // Appearance
  darkMode: boolean
  setDarkMode: (dark: boolean) => void
  showGrid: boolean
  setShowGrid: (show: boolean) => void
  showJointLabels: boolean
  setShowJointLabels: (show: boolean) => void
  showLinkLabels: boolean
  setShowLinkLabels: (show: boolean) => void

  // Simulation
  simulationStepsInput: string
  setSimulationStepsInput: (value: string) => void
  autoSimulateDelayMs: number
  setAutoSimulateDelayMs: (delay: number) => void
  trajectoryColorCycle: ColorCycleType
  setTrajectoryColorCycle: (cycle: ColorCycleType) => void
  trajectoryData: unknown
  autoSimulateEnabled: boolean
  setAutoSimulateEnabled: (enabled: boolean) => void
  triggerMechanismChange: () => void

  // Interaction
  jointMergeRadius: number
  setJointMergeRadius: (radius: number) => void

  // Canvas/Grid
  canvasBgColor: CanvasBgColor
  setCanvasBgColor: (color: CanvasBgColor) => void

  // Mechanism visualization
  jointSize: number
  setJointSize: (size: number) => void
  jointOutline: number
  setJointOutline: (size: number) => void
  linkThickness: number
  setLinkThickness: (thickness: number) => void
  linkTransparency: number
  setLinkTransparency: (pct: number) => void
  linkColorMode: LinkColorMode
  setLinkColorMode: (mode: LinkColorMode) => void
  linkColorSingle: string
  setLinkColorSingle: (color: string) => void

  // Trajectory visualization
  trajectoryDotSize: number
  setTrajectoryDotSize: (size: number) => void
  trajectoryDotOutline: boolean
  setTrajectoryDotOutline: (show: boolean) => void
  trajectoryDotOpacity: number
  setTrajectoryDotOpacity: (opacity: number) => void
  showTrajectoryStepNumbers: boolean
  setShowTrajectoryStepNumbers: (show: boolean) => void
  trajectoryStyle: TrajectoryStyle
  setTrajectoryStyle: (style: TrajectoryStyle) => void

  // Trajectory exploration (explore node trajectories tool)
  exploreRadius: number
  setExploreRadius: (radius: number) => void
  exploreRadialSamples: number
  setExploreRadialSamples: (n: number) => void
  exploreAzimuthalSamples: number
  setExploreAzimuthalSamples: (n: number) => void
  exploreNMaxCombinatorial: number
  setExploreNMaxCombinatorial: (n: number) => void
  exploreColormapEnabled: boolean
  setExploreColormapEnabled: (enabled: boolean) => void
  exploreColormapType: 'rainbow' | 'twilight' | 'husl'
  setExploreColormapType: (type: 'rainbow' | 'twilight' | 'husl') => void
}

export const SettingsToolbar: React.FC<SettingsToolbarProps> = ({
  darkMode, setDarkMode,
  showGrid, setShowGrid,
  showJointLabels, setShowJointLabels,
  showLinkLabels, setShowLinkLabels,
  simulationStepsInput, setSimulationStepsInput,
  autoSimulateDelayMs, setAutoSimulateDelayMs,
  trajectoryColorCycle, setTrajectoryColorCycle,
  trajectoryData, autoSimulateEnabled, setAutoSimulateEnabled, triggerMechanismChange,
  jointMergeRadius, setJointMergeRadius,
  canvasBgColor, setCanvasBgColor,
  jointSize, setJointSize,
  jointOutline, setJointOutline,
  linkThickness, setLinkThickness,
  linkTransparency, setLinkTransparency,
  linkColorMode, setLinkColorMode,
  linkColorSingle, setLinkColorSingle,
  trajectoryDotSize, setTrajectoryDotSize,
  trajectoryDotOutline, setTrajectoryDotOutline,
  trajectoryDotOpacity, setTrajectoryDotOpacity,
  showTrajectoryStepNumbers, setShowTrajectoryStepNumbers,
  trajectoryStyle, setTrajectoryStyle,
  exploreRadius, setExploreRadius,
  exploreRadialSamples, setExploreRadialSamples,
  exploreAzimuthalSamples, setExploreAzimuthalSamples,
  exploreNMaxCombinatorial, setExploreNMaxCombinatorial,
  exploreColormapEnabled, setExploreColormapEnabled,
  exploreColormapType, setExploreColormapType
}) => {
  const sectionTitle = { fontWeight: 600, color: 'text.secondary', fontSize: '0.7rem', mb: 0.35 }
  const row = { display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }
  const label = { color: 'text.secondary', fontSize: '0.75rem', flexShrink: 0, minWidth: 115 }
  const control = { minWidth: 0, flex: 1 }
  const divider = { my: 0.75 }
  // Shared slider layout; Slider color="primary" uses theme (cheetah orange)
  const sliderSx = { ...control }
  // Compact dropdown: minimal list and item padding
  const menuItemSx = { minHeight: 28, py: 0, px: 1, fontSize: '0.8rem', lineHeight: 1.2 }
  const selectMenuProps = {
    PaperProps: {
      sx: {
        '& .MuiList-root': { py: 0 },
        '& .MuiMenuItem-root': { minHeight: 28, py: 0, px: 1, fontSize: '0.8rem', lineHeight: 1.2 }
      }
    }
  }
  const selectTriggerSx = { fontSize: '0.8rem', '& .MuiSelect-select': { py: 0.25 } }

  // Number fields: commit only on Enter or blur (no live parent updates while typing)
  const [localSteps, setLocalSteps] = useState(simulationStepsInput)
  const [localDelay, setLocalDelay] = useState(String(autoSimulateDelayMs))
  const stepsFocused = useRef(false)
  const delayFocused = useRef(false)

  useEffect(() => {
    if (!stepsFocused.current) setLocalSteps(simulationStepsInput)
  }, [simulationStepsInput])
  useEffect(() => {
    if (!delayFocused.current) setLocalDelay(String(autoSimulateDelayMs))
  }, [autoSimulateDelayMs])

  const commitSteps = () => {
    const val = parseInt(localSteps, 10)
    if (!isNaN(val)) {
      const clamped = Math.max(MIN_SIMULATION_STEPS, Math.min(MAX_SIMULATION_STEPS, val))
      setSimulationStepsInput(String(clamped))
    }
    stepsFocused.current = false
  }
  const commitDelay = () => {
    const val = parseInt(localDelay, 10)
    if (!isNaN(val)) setAutoSimulateDelayMs(Math.max(0, Math.min(1000, val)))
    delayFocused.current = false
  }

  const inputSx = { '& .MuiInputBase-input': { fontSize: '0.8rem', py: 0.5 } }

  return (
    <Box sx={{ p: 1, minWidth: 260 }}>
      {/* SIMULATION (top) */}
      <Typography variant="caption" sx={sectionTitle}>Simulation</Typography>
      <Box sx={row}>
        <Typography sx={label}>Steps (N)</Typography>
        <TextField
          size="small"
          type="number"
          value={localSteps}
          onChange={(e) => setLocalSteps(e.target.value)}
          onFocus={() => { stepsFocused.current = true }}
          onBlur={commitSteps}
          onKeyDown={(e) => { if (e.key === 'Enter') commitSteps() }}
          inputProps={{ step: 4 }}
          sx={{ ...control, ...inputSx }}
          helperText={`${MIN_SIMULATION_STEPS}-${MAX_SIMULATION_STEPS}. Enter or blur to apply.`}
          FormHelperTextProps={{ sx: { fontSize: '0.65rem', mt: 0.15 } }}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Auto-sim delay (ms)</Typography>
        <TextField
          size="small"
          type="number"
          value={localDelay}
          onChange={(e) => setLocalDelay(e.target.value)}
          onFocus={() => { delayFocused.current = true }}
          onBlur={commitDelay}
          onKeyDown={(e) => { if (e.key === 'Enter') commitDelay() }}
          inputProps={{ min: 0, max: 1000, step: 5 }}
          sx={{ ...control, ...inputSx }}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Continuous simulation</Typography>
        <Box sx={control}>
          <Switch
            size="small"
            checked={autoSimulateEnabled}
            onChange={(e) => {
              const enabled = e.target.checked
              setAutoSimulateEnabled(enabled)
              if (enabled) triggerMechanismChange()
            }}
          />
        </Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Trajectory color</Typography>
        <FormControl size="small" sx={control}>
          <Select
            value={trajectoryColorCycle}
            onChange={(e) => {
              setTrajectoryColorCycle(e.target.value as ColorCycleType)
              if (trajectoryData && autoSimulateEnabled) triggerMechanismChange()
            }}
            sx={selectTriggerSx}
            MenuProps={selectMenuProps}
          >
            <MenuItem value="rainbow" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: '50%', background: 'linear-gradient(90deg, red, orange, yellow, green, blue, violet)' }} /> Rainbow</Box></MenuItem>
            <MenuItem value="fire" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: '50%', background: 'linear-gradient(90deg, #FA8112, #1A0A00, #FA8112)' }} /> Fire</Box></MenuItem>
            <MenuItem value="glow" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: '50%', background: 'linear-gradient(90deg, #FA8112, #FFF8E8, #FA8112)' }} /> Glow</Box></MenuItem>
            <MenuItem value="twilight" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: '50%', background: 'linear-gradient(90deg, #2d5a8b, #6b2d7b, #b82d5a, #d46b2d, #2d5a8b)' }} /> Twilight</Box></MenuItem>
            <MenuItem value="husl" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: '50%', background: 'linear-gradient(90deg, #e6194b, #3cb44b, #ffe119, #4363d8, #f58231, #911eb4, #e6194b)' }} /> HUSL</Box></MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Divider sx={divider} />

      {/* APPEARANCE */}
      <Typography variant="caption" sx={sectionTitle}>Appearance</Typography>
      <Box sx={row}>
        <Typography sx={label}>{darkMode ? '🌙' : '☀️'} Dark Mode</Typography>
        <Box sx={control}><Switch checked={darkMode} onChange={(e) => setDarkMode(e.target.checked)} size="small" /></Box>
      </Box>

      <Divider sx={divider} />

      {/* INTERACTION */}
      <Typography variant="caption" sx={sectionTitle}>Interaction</Typography>
      <Box sx={row}>
        <Typography sx={label}>Merge radius</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Merge radius"
          value={jointMergeRadius}
          onChange={setJointMergeRadius}
          min={1}
          max={10}
          step={1}
          shiftStep={2}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => String(v)}
          sx={sliderSx}
        />
      </Box>

      <Divider sx={divider} />

      {/* CANVAS / GRID */}
      <Typography variant="caption" sx={sectionTitle}>Canvas / Grid</Typography>
      <Box sx={row}>
        <Typography sx={label}>Show Grid</Typography>
        <Box sx={control}><Switch checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Background</Typography>
        <FormControl size="small" sx={control}>
          <Select value={canvasBgColor} onChange={(e) => setCanvasBgColor(e.target.value as CanvasBgColor)} sx={selectTriggerSx} MenuProps={selectMenuProps}>
            <MenuItem value="default" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: darkMode ? '#1a1a1a' : '#fafafa', border: '1px solid #ccc' }} /> Default</Box></MenuItem>
            <MenuItem value="white" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: '#fff', border: '1px solid #ccc' }} /> White</Box></MenuItem>
            <MenuItem value="cream" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: '#FAF3E1', border: '1px solid #ccc' }} /> Cream</Box></MenuItem>
            <MenuItem value="dark" sx={menuItemSx}><Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}><Box sx={{ width: 12, height: 12, borderRadius: 1, bgcolor: '#1a1a1a', border: '1px solid #ccc' }} /> Dark</Box></MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Divider sx={divider} />

      {/* MECHANISM VISUALIZATION */}
      <Typography variant="caption" sx={sectionTitle}>Mechanism visualization</Typography>
      <Box sx={row}>
        <Typography sx={label}>Link labels</Typography>
        <Box sx={control}><Switch checked={showLinkLabels} onChange={(e) => setShowLinkLabels(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Link thickness</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Link thickness (pixels)"
          value={linkThickness}
          onChange={setLinkThickness}
          min={1}
          max={16}
          step={1}
          shiftStep={5}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => `${v}px`}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Link transparency</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Link transparency"
          value={linkTransparency}
          onChange={setLinkTransparency}
          min={10}
          max={100}
          step={5}
          shiftStep={25}
          valueLabelDisplay="auto"
          valueLabelFormat={(v) => `${v}%`}
          getAriaValueText={(v) => `${v}%`}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Link colors</Typography>
        <FormControl size="small" sx={control}>
          <Select value={linkColorMode} onChange={(e) => setLinkColorMode(e.target.value as LinkColorMode)} sx={selectTriggerSx} MenuProps={selectMenuProps}>
            <MenuItem value="various" sx={menuItemSx}>Various (per link)</MenuItem>
            <MenuItem value="z-level" sx={menuItemSx}>Z-level</MenuItem>
            <MenuItem value="single" sx={menuItemSx}>Single color</MenuItem>
          </Select>
        </FormControl>
      </Box>
      {linkColorMode === 'single' && (
        <Box sx={row}>
          <Typography sx={label}>Single color</Typography>
          <Box sx={{ ...control, display: 'flex', alignItems: 'center', gap: 1 }}>
            <input
              type="color"
              value={linkColorSingle}
              onChange={(e) => setLinkColorSingle(e.target.value)}
              style={{ width: 28, height: 28, padding: 0, border: '1px solid var(--color-border)', borderRadius: 4, cursor: 'pointer' }}
              aria-label="Link color"
            />
            <Typography component="span" sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>{linkColorSingle}</Typography>
          </Box>
        </Box>
      )}
      <Box sx={row}>
        <Typography sx={label}>Joint labels</Typography>
        <Box sx={control}><Switch checked={showJointLabels} onChange={(e) => setShowJointLabels(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Joint size</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Joint size (pixels)"
          value={jointSize}
          onChange={setJointSize}
          min={1}
          max={16}
          step={1}
          shiftStep={5}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => `${v}px`}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Joint outline</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Joint outline (px)"
          value={jointOutline}
          onChange={setJointOutline}
          min={0}
          max={10}
          step={1}
          shiftStep={2}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => `${v}px`}
          sx={sliderSx}
        />
      </Box>

      <Divider sx={divider} />

      {/* TRAJECTORY VISUALIZATION */}
      <Typography variant="caption" sx={sectionTitle}>Trajectory visualization</Typography>
      <Box sx={row}>
        <Typography sx={label}>Trajectory style</Typography>
        <FormControl size="small" sx={control}>
          <Select value={trajectoryStyle} onChange={(e) => setTrajectoryStyle(e.target.value as TrajectoryStyle)} sx={selectTriggerSx} MenuProps={selectMenuProps}>
            <MenuItem value="dots" sx={menuItemSx}>Dots only</MenuItem>
            <MenuItem value="line" sx={menuItemSx}>Line only</MenuItem>
            <MenuItem value="both" sx={menuItemSx}>Dots + Line</MenuItem>
          </Select>
        </FormControl>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Dot size</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Trajectory dot size (pixels)"
          value={trajectoryDotSize}
          onChange={setTrajectoryDotSize}
          min={2}
          max={8}
          step={1}
          shiftStep={3}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => `${v}px`}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Dot outline</Typography>
        <Box sx={control}><Switch checked={trajectoryDotOutline} onChange={(e) => setTrajectoryDotOutline(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Dot opacity</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Trajectory dot opacity"
          value={Math.round(trajectoryDotOpacity * 100)}
          onChange={(v) => setTrajectoryDotOpacity(v / 100)}
          min={10}
          max={100}
          step={5}
          shiftStep={25}
          valueLabelDisplay="auto"
          valueLabelFormat={(v) => `${v}%`}
          getAriaValueText={(v) => `${v}%`}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Show simulation #</Typography>
        <Box sx={control}><Switch checked={showTrajectoryStepNumbers} onChange={(e) => setShowTrajectoryStepNumbers(e.target.checked)} size="small" /></Box>
      </Box>

      <Divider sx={divider} />

      {/* TRAJECTORY EXPLORATION */}
      <Typography variant="caption" sx={sectionTitle}>Trajectory exploration</Typography>
      <Box sx={row}>
        <Typography sx={label}>Explore radius</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Explore radius"
          value={exploreRadius}
          onChange={setExploreRadius}
          min={EXPLORE_RADIUS_MIN}
          max={EXPLORE_RADIUS_MAX}
          step={1}
          shiftStep={5}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => String(v)}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Radial samples</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Radial samples"
          value={exploreRadialSamples}
          onChange={setExploreRadialSamples}
          min={EXPLORE_RADIAL_SAMPLES_MIN}
          max={EXPLORE_RADIAL_SAMPLES_MAX}
          step={1}
          shiftStep={5}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => String(v)}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Azimuthal samples</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Azimuthal samples"
          value={exploreAzimuthalSamples}
          onChange={setExploreAzimuthalSamples}
          min={EXPLORE_AZIMUTHAL_SAMPLES_MIN}
          max={EXPLORE_AZIMUTHAL_SAMPLES_MAX}
          step={1}
          shiftStep={5}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => String(v)}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Max combinatorics (N₁×N₂)</Typography>
        <SmoothSlider
          color="primary"
          size="small"
          aria-label="Max combinatorics"
          value={exploreNMaxCombinatorial}
          onChange={setExploreNMaxCombinatorial}
          min={EXPLORE_NMAX_COMBINATORIAL_MIN}
          max={EXPLORE_NMAX_COMBINATORIAL_MAX}
          step={64}
          shiftStep={320}
          valueLabelDisplay="auto"
          getAriaValueText={(v) => String(v)}
          sx={sliderSx}
        />
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Color-map exploration</Typography>
        <Box sx={control}><Switch checked={exploreColormapEnabled} onChange={(e) => setExploreColormapEnabled(e.target.checked)} size="small" /></Box>
      </Box>
      {exploreColormapEnabled && (
        <Box sx={row}>
          <Typography sx={label}>Colormap</Typography>
          <FormControl size="small" sx={control}>
            <Select value={exploreColormapType} onChange={(e) => setExploreColormapType(e.target.value as 'rainbow' | 'twilight' | 'husl')} sx={selectTriggerSx} MenuProps={selectMenuProps}>
              <MenuItem value="rainbow" sx={menuItemSx}>Rainbow</MenuItem>
              <MenuItem value="twilight" sx={menuItemSx}>Twilight</MenuItem>
              <MenuItem value="husl" sx={menuItemSx}>HUSL</MenuItem>
            </Select>
          </FormControl>
        </Box>
      )}
    </Box>
  )
}

export default SettingsToolbar
