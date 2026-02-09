/**
 * Settings Toolbar - Comprehensive settings panel for Builder
 */
import React, { useState, useRef, useEffect } from 'react'
import {
  Box, Typography, FormControlLabel, Switch, TextField, Select,
  MenuItem, FormControl, Divider, Slider
} from '@mui/material'
import type { ColorCycleType } from '../../../theme'
import {
  MIN_SIMULATION_STEPS,
  MAX_SIMULATION_STEPS,
  DEFAULT_AUTO_SIMULATE_DELAY_MS,
  DEFAULT_JOINT_MERGE_RADIUS
} from '../constants'

export type CanvasBgColor = 'default' | 'white' | 'cream' | 'dark'
export type TrajectoryStyle = 'dots' | 'line' | 'both'
export type SelectionHighlightColor = 'blue' | 'orange' | 'green' | 'purple'

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
  triggerMechanismChange: () => void

  // Interaction
  jointMergeRadius: number
  setJointMergeRadius: (radius: number) => void

  // Canvas/Grid
  canvasBgColor: CanvasBgColor
  setCanvasBgColor: (color: CanvasBgColor) => void

  // Visualization
  jointSize: number
  setJointSize: (size: number) => void
  linkThickness: number
  setLinkThickness: (thickness: number) => void
  trajectoryDotSize: number
  setTrajectoryDotSize: (size: number) => void
  trajectoryDotOutline: boolean
  setTrajectoryDotOutline: (show: boolean) => void
  trajectoryDotOpacity: number
  setTrajectoryDotOpacity: (opacity: number) => void

  // Animation
  trajectoryStyle: TrajectoryStyle
  setTrajectoryStyle: (style: TrajectoryStyle) => void
}

export const SettingsToolbar: React.FC<SettingsToolbarProps> = ({
  darkMode, setDarkMode,
  showGrid, setShowGrid,
  showJointLabels, setShowJointLabels,
  showLinkLabels, setShowLinkLabels,
  simulationStepsInput, setSimulationStepsInput,
  autoSimulateDelayMs, setAutoSimulateDelayMs,
  trajectoryColorCycle, setTrajectoryColorCycle,
  trajectoryData, autoSimulateEnabled, triggerMechanismChange,
  jointMergeRadius, setJointMergeRadius,
  canvasBgColor, setCanvasBgColor,
  jointSize, setJointSize,
  linkThickness, setLinkThickness,
  trajectoryDotSize, setTrajectoryDotSize,
  trajectoryDotOutline, setTrajectoryDotOutline,
  trajectoryDotOpacity, setTrajectoryDotOpacity,
  trajectoryStyle, setTrajectoryStyle
}) => {
  const sectionTitle = { fontWeight: 600, color: 'text.secondary', fontSize: '0.7rem', mb: 0.35 }
  const row = { display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }
  const label = { color: 'text.secondary', fontSize: '0.75rem', flexShrink: 0, minWidth: 115 }
  const control = { minWidth: 0, flex: 1 }
  const divider = { my: 0.75 }
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
  const [localMergeRadius, setLocalMergeRadius] = useState(String(jointMergeRadius))
  const stepsFocused = useRef(false)
  const delayFocused = useRef(false)
  const mergeFocused = useRef(false)

  useEffect(() => {
    if (!stepsFocused.current) setLocalSteps(simulationStepsInput)
  }, [simulationStepsInput])
  useEffect(() => {
    if (!delayFocused.current) setLocalDelay(String(autoSimulateDelayMs))
  }, [autoSimulateDelayMs])
  useEffect(() => {
    if (!mergeFocused.current) setLocalMergeRadius(String(jointMergeRadius))
  }, [jointMergeRadius])

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
  const commitMergeRadius = () => {
    const val = parseFloat(localMergeRadius)
    if (!isNaN(val)) setJointMergeRadius(Math.max(0.5, Math.min(20, val)))
    mergeFocused.current = false
  }

  const inputSx = { '& .MuiInputBase-input': { fontSize: '0.8rem', py: 0.5 } }

  return (
    <Box sx={{ p: 1, minWidth: 300 }}>
      {/* APPEARANCE */}
      <Typography variant="caption" sx={sectionTitle}>Appearance</Typography>
      <Box sx={row}>
        <Typography sx={label}>{darkMode ? '🌙' : '☀️'} Dark Mode</Typography>
        <Box sx={control}><Switch checked={darkMode} onChange={(e) => setDarkMode(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Joint Labels</Typography>
        <Box sx={control}><Switch checked={showJointLabels} onChange={(e) => setShowJointLabels(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Link Labels</Typography>
        <Box sx={control}><Switch checked={showLinkLabels} onChange={(e) => setShowLinkLabels(e.target.checked)} size="small" /></Box>
      </Box>

      <Divider sx={divider} />

      {/* SIMULATION */}
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
          </Select>
        </FormControl>
      </Box>

      <Divider sx={divider} />

      {/* INTERACTION */}
      <Typography variant="caption" sx={sectionTitle}>Interaction</Typography>
      <Box sx={row}>
        <Typography sx={label}>Merge radius</Typography>
        <TextField
          size="small"
          type="number"
          value={localMergeRadius}
          onChange={(e) => setLocalMergeRadius(e.target.value)}
          onFocus={() => { mergeFocused.current = true }}
          onBlur={commitMergeRadius}
          onKeyDown={(e) => { if (e.key === 'Enter') commitMergeRadius() }}
          inputProps={{ min: 0.5, max: 20, step: 0.5 }}
          sx={{ ...control, ...inputSx }}
        />
      </Box>

      <Divider sx={divider} />

      {/* CANVAS/GRID */}
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
      <Box sx={{ ...row, opacity: 0.5 }}>
        <Typography sx={{ ...label, color: 'text.disabled' }}>Grid spacing (TODO)</Typography>
        <FormControl size="small" sx={control} disabled>
          <Select value={20} sx={selectTriggerSx} MenuProps={selectMenuProps}><MenuItem value={20} sx={menuItemSx}>20</MenuItem></Select>
        </FormControl>
      </Box>
      <Box sx={{ ...row, opacity: 0.5 }}>
        <Typography sx={{ ...label, color: 'text.disabled' }}>Snap to Grid (TODO)</Typography>
        <Box sx={control}><Switch size="small" disabled /></Box>
      </Box>

      <Divider sx={divider} />

      {/* VISUALIZATION */}
      <Typography variant="caption" sx={sectionTitle}>Visualization</Typography>
      <Box sx={row}>
        <Typography sx={label}>Joint {jointSize}px</Typography>
        <Slider size="small" value={jointSize} onChange={(_, v) => setJointSize(v as number)} min={3} max={16} step={1} valueLabelDisplay="auto" sx={{ ...control, color: '#FA8112' }}/>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Link {linkThickness}px</Typography>
        <Slider size="small" value={linkThickness} onChange={(_, v) => setLinkThickness(v as number)} min={1} max={16} step={1} valueLabelDisplay="auto" sx={{ ...control, color: '#FA8112' }}/>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Dot size {trajectoryDotSize}px</Typography>
        <Slider size="small" value={trajectoryDotSize} onChange={(_, v) => setTrajectoryDotSize(v as number)} min={2} max={8} step={1} valueLabelDisplay="auto" sx={{ ...control, color: '#FA8112' }}/>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Dot outline</Typography>
        <Box sx={control}><Switch checked={trajectoryDotOutline} onChange={(e) => setTrajectoryDotOutline(e.target.checked)} size="small" /></Box>
      </Box>
      <Box sx={row}>
        <Typography sx={label}>Dot opacity {Math.round(trajectoryDotOpacity * 100)}%</Typography>
        <Slider size="small" value={trajectoryDotOpacity * 100} onChange={(_, v) => setTrajectoryDotOpacity((v as number) / 100)} min={50} max={100} step={1} valueLabelDisplay="auto" valueLabelFormat={(v) => `${v}%`} sx={{ ...control, color: '#FA8112' }}/>
      </Box>

      <Divider sx={divider} />

      {/* ANIMATION */}
      <Typography variant="caption" sx={sectionTitle}>Animation</Typography>
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
    </Box>
  )
}

export default SettingsToolbar
