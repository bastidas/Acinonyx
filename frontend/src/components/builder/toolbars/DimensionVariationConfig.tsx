/**
 * DimensionVariationConfig Component
 *
 * Shared component for configuring dimension variations.
 * Compact layout: single row per dimension with toggles and range slider.
 */

import React, { useMemo, useEffect, useRef } from 'react'
import {
  Box, Typography, Switch, FormControlLabel, Slider, Stack, Button
} from '@mui/material'

export interface DimensionInfo {
  names: string[]
  initial_values: number[]
  bounds: [number, number][]
  n_dimensions: number
}

export interface DimensionOverride {
  enabled: boolean
  minPct: number
  maxPct: number
  useDefaultRange?: boolean // Track if using default range
}

export interface DimensionVariationConfigProps {
  dimensionInfo: DimensionInfo | null
  dimensionOverrides: Record<string, [boolean, number, number]>
  dimensionSyncStates?: Record<string, boolean> // Track which dimensions are synced to default
  defaultEnabled: boolean
  defaultVariationRange: number
  onOverrideChange: (name: string, override: [boolean, number, number]) => void
  onSyncStateChange?: (name: string, synced: boolean) => void // Callback when sync state changes
  onDefaultEnabledChange?: (enabled: boolean) => void
  onDefaultRangeChange?: (range: number) => void
  compact?: boolean
  showTopControls?: boolean // Show "Vary link lengths" toggle and default range slider at top
}

export const DimensionVariationConfig: React.FC<DimensionVariationConfigProps> = ({
  dimensionInfo,
  dimensionOverrides,
  dimensionSyncStates = {},
  defaultEnabled,
  defaultVariationRange,
  onOverrideChange,
  onSyncStateChange,
  onDefaultEnabledChange,
  onDefaultRangeChange,
  compact = false,
  showTopControls = false
}) => {
  // Calculate default min/max percentages from default variation range
  const defaultMinPct = -defaultVariationRange
  const defaultMaxPct = defaultVariationRange

  // When default range changes, update all synced dimensions (only their current values, not bounds)
  // Use a ref to track the previous value to avoid unnecessary updates during slider dragging
  const prevDefaultRangeRef = useRef(defaultVariationRange)
  const syncTimeoutRef = useRef<number | null>(null)

  useEffect(() => {
    // Only sync if the value actually changed (not just a re-render)
    if (Math.abs(prevDefaultRangeRef.current - defaultVariationRange) < 0.001) {
      return
    }
    prevDefaultRangeRef.current = defaultVariationRange

    // Debounce the sync to avoid cascading updates during slider dragging
    if (syncTimeoutRef.current) {
      clearTimeout(syncTimeoutRef.current)
    }

    syncTimeoutRef.current = window.setTimeout(() => {
      if (dimensionInfo && onOverrideChange) {
        dimensionInfo.names.forEach(name => {
          if (dimensionSyncStates[name] !== false) { // Default to true if not explicitly false
            const override = dimensionOverrides[name]
            const [enabled] = override !== undefined ? override : [defaultEnabled, defaultMinPct, defaultMaxPct]
            // Sync to default range (with min link length clamp, but within fixed slider bounds)
            const idx = dimensionInfo.names.indexOf(name)
            const initial = dimensionInfo.initial_values[idx]
            const MIN_LINK_LENGTH = 2
            const minPctForMinLength = (MIN_LINK_LENGTH - initial) / initial
            const sliderMin = Math.max(-5.0, minPctForMinLength)
            const sliderMax = 5.0
            // Clamp default range to slider bounds
            const syncedMinPct = Math.max(Math.min(defaultMinPct, sliderMax), sliderMin)
            const syncedMaxPct = Math.max(Math.min(defaultMaxPct, sliderMax), sliderMin)
            onOverrideChange(name, [enabled, syncedMinPct, syncedMaxPct])
          }
        })
      }
    }, 150) // Debounce by 150ms to batch updates during slider dragging

    return () => {
      if (syncTimeoutRef.current) {
        clearTimeout(syncTimeoutRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultVariationRange]) // Only trigger when default range changes


  if (!dimensionInfo || dimensionInfo.names.length === 0) {
    return (
      <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        No dimensions available
      </Typography>
    )
  }

  return (
    <Stack spacing={compact ? 1 : 1.5}>
      {/* Top row: "Vary link lengths" toggle + "Default link variation range" slider */}
      {showTopControls && (
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          p: 1.5,
          bgcolor: 'rgba(0,0,0,0.02)',
          borderRadius: 1,
          border: '1px solid rgba(0,0,0,0.1)'
        }}>
          <FormControlLabel
            control={
              <Switch
                checked={defaultEnabled}
                onChange={(e) => onDefaultEnabledChange?.(e.target.checked)}
                size="small"
              />
            }
            label={
              <Typography variant="body2" sx={{ fontSize: compact ? '0.75rem' : '0.8rem' }}>
                Vary link lengths
              </Typography>
            }
            sx={{ m: 0, minWidth: 120 }}
          />
          <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="caption" sx={{ minWidth: 140, fontSize: compact ? '0.7rem' : '0.75rem' }}>
              Default link variation range: ±{(defaultVariationRange * 100).toFixed(0)}%
            </Typography>
            <Slider
              value={defaultVariationRange}
              onChange={(_, value) => onDefaultRangeChange?.(value as number)}
              min={0.05}
              max={5.0}
              step={0.05}
              size="small"
              disabled={!defaultEnabled}
              sx={{ flex: 1, maxWidth: 300 }}
            />
          </Box>
        </Box>
      )}

      {/* Dimension rows - all in one green box */}
      <Box sx={{
        maxHeight: compact ? 300 : 400,
        overflowY: 'auto',
        p: compact ? 1 : 1.5,
        borderRadius: 1,
        border: '1px solid rgba(0,0,0,0.1)',
        bgcolor: 'rgba(46, 125, 50, 0.05)'
      }}>
        <Stack spacing={compact ? 0.75 : 1}>
          {dimensionInfo.names.map((name, idx) => {
            const initial = dimensionInfo.initial_values[idx]
            const [minBound, maxBound] = dimensionInfo.bounds[idx]
            const override = dimensionOverrides[name]

            // Fixed slider bounds: always +/- 500% with min link length clamp
            const MIN_LINK_LENGTH = 2
            const minPctForMinLength = (MIN_LINK_LENGTH - initial) / initial
            const sliderMin = Math.max(-5.0, minPctForMinLength) // -500% or min link length, whichever is more restrictive
            const sliderMax = 5.0 // +500%

            // Only use default values if there's no explicit override
            // This prevents default range changes from affecting dimensions with overrides
            const [enabled, minPct, maxPct] = override !== undefined
              ? override
              : [defaultEnabled, defaultMinPct, defaultMaxPct]

            // Current values (clamped to slider bounds)
            const finalMinPct = Math.max(Math.min(minPct, sliderMax), sliderMin)
            const finalMaxPct = Math.max(Math.min(maxPct, sliderMax), sliderMin)

            // Check if this dimension is synced to default (defaults to true if not explicitly false)
            const isSynced = dimensionSyncStates[name] !== false

            // Calculate actual min/max values
            const minValue = (finalMinPct * initial) + initial
            const maxValue = (finalMaxPct * initial) + initial

            return (
              <Box
                key={name}
                sx={{
                  opacity: defaultEnabled ? 1 : 0.5,
                  transition: 'opacity 0.2s'
                }}
              >
                {/* Single row: Name, Variable toggle, Use default range toggle, Range slider */}
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {/* Dimension name */}
                  <Typography
                    variant="caption"
                    sx={{
                      fontFamily: 'monospace',
                      fontWeight: 500,
                      minWidth: compact ? 100 : 120,
                      fontSize: compact ? '0.7rem' : '0.75rem'
                    }}
                  >
                    {name}
                  </Typography>

                  {/* Variable toggle */}
                  <FormControlLabel
                    control={
                      <Switch
                        size="small"
                        checked={enabled}
                        onChange={(e) => {
                          const newEnabled = e.target.checked
                          onOverrideChange(name, [newEnabled, minPct, maxPct])
                        }}
                        disabled={!defaultEnabled}
                      />
                    }
                    label={
                      <Typography variant="caption" sx={{ fontSize: compact ? '0.65rem' : '0.7rem' }}>
                        Variable
                      </Typography>
                    }
                    sx={{ m: 0, minWidth: compact ? 70 : 80 }}
                  />

                  {/* Sync to default toggle */}
                  <FormControlLabel
                    control={
                      <Switch
                        size="small"
                        checked={isSynced}
                        onChange={(e) => {
                          const newSynced = e.target.checked
                          onSyncStateChange?.(name, newSynced)
                          if (newSynced) {
                            // Sync current value to default range (within fixed slider bounds)
                            const MIN_LINK_LENGTH = 2
                            const minPctForMinLength = (MIN_LINK_LENGTH - initial) / initial
                            const sliderMin = Math.max(-5.0, minPctForMinLength)
                            const sliderMax = 5.0
                            const syncedMinPct = Math.max(Math.min(defaultMinPct, sliderMax), sliderMin)
                            const syncedMaxPct = Math.max(Math.min(defaultMaxPct, sliderMax), sliderMin)
                            onOverrideChange(name, [enabled, syncedMinPct, syncedMaxPct])
                          }
                        }}
                        disabled={!enabled || !defaultEnabled}
                      />
                    }
                    label={
                      <Typography variant="caption" sx={{ fontSize: compact ? '0.65rem' : '0.7rem' }}>
                        Sync to default
                      </Typography>
                    }
                    sx={{ m: 0, minWidth: compact ? 100 : 110 }}
                  />

                  {/* Range slider */}
                  {enabled && defaultEnabled ? (
                    <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 0.75 }}>
                      {/* Min value display */}
                      <Typography
                        variant="caption"
                        sx={{
                          minWidth: compact ? 40 : 50,
                          fontSize: compact ? '0.65rem' : '0.7rem',
                          fontFamily: 'monospace',
                          color: 'text.secondary'
                        }}
                      >
                        {minValue.toFixed(1)}
                      </Typography>

                      {/* Range slider with dual thumbs */}
                      <Box sx={{ flex: 1, px: 0.5, position: 'relative' }}>
                        <Slider
                          value={[finalMinPct, finalMaxPct]}
                          onChange={(_, value) => {
                            const [newMinPct, newMaxPct] = value as number[]
                            // Clamp to fixed slider bounds
                            let finalMin = Math.max(Math.min(newMinPct, newMaxPct), sliderMin)
                            let finalMax = Math.min(Math.max(newMinPct, newMaxPct), sliderMax)

                            // If user slides the slider, turn off sync to default
                            if (isSynced) {
                              onSyncStateChange?.(name, false)
                            }

                            onOverrideChange(name, [enabled, finalMin, finalMax])
                          }}
                          min={sliderMin}
                          max={sliderMax}
                          step={0.01}
                          size="small"
                          sx={{
                            '& .MuiSlider-thumb': {
                              width: compact ? 12 : 14,
                              height: compact ? 12 : 14
                            },
                            '& .MuiSlider-track': {
                              height: compact ? 2 : 3
                            },
                            '& .MuiSlider-rail': {
                              height: compact ? 2 : 3
                            }
                          }}
                        />
                        {/* Show marker for current link length (initial value = 0% on slider) as a low opacity diamond/square */}
                        <Box
                          sx={{
                            position: 'absolute',
                            top: '50%',
                            left: `${((0 - sliderMin) / (sliderMax - sliderMin)) * 100}%`,
                            width: compact ? 8 : 10,
                            height: compact ? 8 : 10,
                            bgcolor: 'rgba(102, 102, 102, 0.3)',
                            transform: 'translate(-50%, -50%) rotate(45deg)',
                            pointerEvents: 'none',
                            zIndex: 1,
                            border: '1px solid rgba(102, 102, 102, 0.5)'
                          }}
                        />
                      </Box>

                      {/* Max value display */}
                      <Typography
                        variant="caption"
                        sx={{
                          minWidth: compact ? 40 : 50,
                          fontSize: compact ? '0.65rem' : '0.7rem',
                          fontFamily: 'monospace',
                          color: 'text.secondary',
                          textAlign: 'right'
                        }}
                      >
                        {maxValue.toFixed(1)}
                      </Typography>
                    </Box>
                  ) : (
                    <Box sx={{ flex: 1 }} /> // Spacer when disabled
                  )}
                </Box>
              </Box>
            )
          })}
        </Stack>
      </Box>
    </Stack>
  )
}

export default DimensionVariationConfig
