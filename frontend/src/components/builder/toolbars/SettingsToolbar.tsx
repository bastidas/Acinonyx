/**
 * Settings Toolbar - Comprehensive settings panel for Builder
 */
import React from 'react'
import {
  Box, Typography, FormControlLabel, Switch, TextField, Select,
  MenuItem, FormControl, Divider, Chip
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
  selectionHighlightColor: SelectionHighlightColor
  setSelectionHighlightColor: (color: SelectionHighlightColor) => void
  showMeasurementUnits: boolean
  setShowMeasurementUnits: (show: boolean) => void

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
  selectionHighlightColor, setSelectionHighlightColor,
  showMeasurementUnits, setShowMeasurementUnits,
  trajectoryStyle, setTrajectoryStyle
}) => {
  return (
    <Box sx={{ p: 1.5, minWidth: 240 }}>
      {/* APPEARANCE */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Appearance
      </Typography>

      <FormControlLabel
        control={<Switch checked={darkMode} onChange={(e) => setDarkMode(e.target.checked)} size="small" />}
        label={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <span>{darkMode ? '🌙' : '☀️'}</span>
            <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Dark Mode</Typography>
          </Box>
        }
        sx={{ mt: 1, mb: 0.5, ml: 0 }}
      />

      <FormControlLabel
        control={<Switch checked={showGrid} onChange={(e) => setShowGrid(e.target.checked)} size="small" />}
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Show Grid</Typography>}
        sx={{ mb: 0.5, ml: 0 }}
      />

      <FormControlLabel
        control={<Switch checked={showJointLabels} onChange={(e) => setShowJointLabels(e.target.checked)} size="small" />}
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Joint Labels</Typography>}
        sx={{ mb: 0.5, ml: 0 }}
      />

      <FormControlLabel
        control={<Switch checked={showLinkLabels} onChange={(e) => setShowLinkLabels(e.target.checked)} size="small" />}
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Link Labels</Typography>}
        sx={{ mb: 1, ml: 0 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* SIMULATION */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Simulation
      </Typography>

      <Box sx={{ mt: 1.5, mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Simulation Steps (N)
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          value={simulationStepsInput}
          onChange={(e) => setSimulationStepsInput(e.target.value)}
          inputProps={{ step: 4 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
          helperText={`Range: ${MIN_SIMULATION_STEPS}-${MAX_SIMULATION_STEPS} (auto-validates)`}
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Auto-Simulate Delay (ms)
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          value={autoSimulateDelayMs}
          onChange={(e) => {
            const val = parseInt(e.target.value) || DEFAULT_AUTO_SIMULATE_DELAY_MS
            setAutoSimulateDelayMs(Math.max(0, Math.min(1000, val)))
          }}
          inputProps={{ min: 0, max: 1000, step: 5 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
          helperText="Delay before auto-simulation triggers"
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Trajectory Color Cycle
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={trajectoryColorCycle}
            onChange={(e) => {
              setTrajectoryColorCycle(e.target.value as ColorCycleType)
              if (trajectoryData && autoSimulateEnabled) {
                triggerMechanismChange()
              }
            }}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="rainbow">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%',
                  background: 'linear-gradient(90deg, red, orange, yellow, green, blue, violet)' }} />
                Rainbow
              </Box>
            </MenuItem>
            <MenuItem value="fire">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%',
                  background: 'linear-gradient(90deg, #FA8112, #1A0A00, #FA8112)' }} />
                Fire
              </Box>
            </MenuItem>
            <MenuItem value="glow">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%',
                  background: 'linear-gradient(90deg, #FA8112, #FFF8E8, #FA8112)' }} />
                Glow
              </Box>
            </MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Divider sx={{ my: 1.5 }} />

      {/* INTERACTION */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Interaction
      </Typography>

      <Box sx={{ mt: 1.5, mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Joint Merge Radius (units)
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          value={jointMergeRadius}
          onChange={(e) => {
            const val = parseFloat(e.target.value) || DEFAULT_JOINT_MERGE_RADIUS
            setJointMergeRadius(Math.max(0.5, Math.min(20, val)))
          }}
          inputProps={{ min: 0.5, max: 20, step: 0.5 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
          helperText="Snap distance for merging joints"
        />
      </Box>

      <Divider sx={{ my: 1.5 }} />

      {/* CANVAS/GRID */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Canvas / Grid
      </Typography>

      <Box sx={{ mt: 1.5, mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Canvas Background
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={canvasBgColor}
            onChange={(e) => setCanvasBgColor(e.target.value as CanvasBgColor)}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="default">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: darkMode ? '#1a1a1a' : '#fafafa', border: '1px solid #ccc' }} />
                Default
              </Box>
            </MenuItem>
            <MenuItem value="white">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: '#ffffff', border: '1px solid #ccc' }} />
                White
              </Box>
            </MenuItem>
            <MenuItem value="cream">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: '#FAF3E1', border: '1px solid #ccc' }} />
                Cream
              </Box>
            </MenuItem>
            <MenuItem value="dark">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: '#1a1a1a', border: '1px solid #ccc' }} />
                Dark
              </Box>
            </MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box sx={{ mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Grid Spacing (units) <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value={20} sx={{ fontSize: '0.85rem' }}>
            <MenuItem value={5}>5 units</MenuItem>
            <MenuItem value={10}>10 units</MenuItem>
            <MenuItem value={20}>20 units</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <FormControlLabel
        control={<Switch size="small" disabled />}
        label={
          <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>
            Snap to Grid <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
          </Typography>
        }
        sx={{ mb: 1, ml: 0, opacity: 0.5 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* VISUALIZATION */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Visualization
      </Typography>

      <Box sx={{ mt: 1.5, mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Joint Size: {jointSize}px
        </Typography>
        <Box sx={{ px: 1 }}>
          <input
            type="range"
            min={3}
            max={16}
            value={jointSize}
            onChange={(e) => setJointSize(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: '#FA8112' }}
          />
        </Box>
      </Box>

      <Box sx={{ mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Link Thickness: {linkThickness}px
        </Typography>
        <Box sx={{ px: 1 }}>
          <input
            type="range"
            min={1}
            max={16}
            value={linkThickness}
            onChange={(e) => setLinkThickness(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: '#FA8112' }}
          />
        </Box>
      </Box>

      <Box sx={{ mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Trajectory Dot Size: {trajectoryDotSize}px
        </Typography>
        <Box sx={{ px: 1 }}>
          <input
            type="range"
            min={2}
            max={8}
            value={trajectoryDotSize}
            onChange={(e) => setTrajectoryDotSize(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: '#FA8112' }}
          />
        </Box>
      </Box>

      <Box sx={{ mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Selection Highlight
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {(['blue', 'orange', 'green', 'purple'] as const).map((color) => {
            const colorMap = { blue: '#1976d2', orange: '#FA8112', green: '#2e7d32', purple: '#9c27b0' }
            return (
              <Box
                key={color}
                onClick={() => setSelectionHighlightColor(color)}
                sx={{
                  width: 28, height: 28, borderRadius: 1,
                  bgcolor: colorMap[color],
                  cursor: 'pointer',
                  border: selectionHighlightColor === color ? '3px solid #fff' : '1px solid #ccc',
                  boxShadow: selectionHighlightColor === color ? `0 0 0 2px ${colorMap[color]}` : 'none',
                  '&:hover': { transform: 'scale(1.1)' },
                  transition: 'all 0.15s ease'
                }}
              />
            )
          })}
        </Box>
      </Box>

      <FormControlLabel
        control={<Switch checked={showMeasurementUnits} onChange={(e) => setShowMeasurementUnits(e.target.checked)} size="small" />}
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Show Measurement Units</Typography>}
        sx={{ mb: 1, ml: 0 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* ANIMATION */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Animation
      </Typography>

      <Box sx={{ mt: 1.5, mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Trajectory Style
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={trajectoryStyle}
            onChange={(e) => setTrajectoryStyle(e.target.value as TrajectoryStyle)}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="dots">Dots only</MenuItem>
            <MenuItem value="line">Line only</MenuItem>
            <MenuItem value="both">Dots + Line</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Disabled/TODO items */}
      <Box sx={{ mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Playback Speed <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value="1x" sx={{ fontSize: '0.85rem' }}>
            <MenuItem value="0.5x">0.5x</MenuItem>
            <MenuItem value="1x">1x</MenuItem>
            <MenuItem value="2x">2x</MenuItem>
          </Select>
        </FormControl>
      </Box>
    </Box>
  )
}

export default SettingsToolbar
