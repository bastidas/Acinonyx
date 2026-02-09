/**
 * StaticJointMovementConfig Component
 *
 * Simple component for configuring static joint movement.
 * Each joint has a variable toggle and X/Y range sliders.
 */

import React, { useEffect } from 'react'
import {
  Box, Typography, Switch, FormControlLabel, Slider, Stack
} from '@mui/material'

export interface StaticJoint {
  name: string
  type: string
}

export interface StaticJointMovementConfigProps {
  joints: StaticJoint[]
  enabled: boolean
  defaultMaxX: number
  defaultMaxY: number
  jointOverrides: Record<string, [boolean, number, number]>
  onEnabledChange: (enabled: boolean) => void
  onDefaultMaxXChange: (value: number) => void
  onDefaultMaxYChange: (value: number) => void
  onJointOverrideChange: (jointName: string, override: [boolean, number, number]) => void
  compact?: boolean
}

export const StaticJointMovementConfig: React.FC<StaticJointMovementConfigProps> = ({
  joints,
  enabled,
  defaultMaxX,
  defaultMaxY,
  jointOverrides,
  onEnabledChange,
  onDefaultMaxXChange,
  onDefaultMaxYChange,
  onJointOverrideChange,
  compact = false
}) => {
  // Filter to only static joints
  const staticJoints = joints.filter(j => j.type === 'Static')

  if (staticJoints.length === 0) {
    return (
      <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
        No static joints available
      </Typography>
    )
  }

  // Slider bounds
  const SLIDER_MIN = -100
  const SLIDER_MAX = 100

  // DEBUG: Log when jointOverrides prop changes
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.log('[DEBUG] StaticJointMovementConfig: jointOverrides prop changed', {
        jointOverrides,
        keys: Object.keys(jointOverrides),
        entries: Object.entries(jointOverrides)
      })
    }
  }, [jointOverrides])

  return (
    <Stack spacing={compact ? 1 : 1.5}>
      {/* Enable/Disable toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={enabled}
            onChange={(e) => onEnabledChange(e.target.checked)}
            size="small"
          />
        }
        label={
          <Typography variant="body2" sx={{ fontSize: compact ? '0.75rem' : '0.8rem' }}>
            Enable Static Joint Movement
          </Typography>
        }
        sx={{ m: 0 }}
      />

      {/* Joint rows - all in one green box */}
      {enabled && (
        <Box sx={{
          maxHeight: compact ? 300 : 400,
          overflowY: 'auto',
          p: compact ? 1 : 1.5,
          borderRadius: 1,
          border: '1px solid rgba(0,0,0,0.1)',
          bgcolor: 'rgba(46, 125, 50, 0.05)'
        }}>
          <Stack spacing={compact ? 0.75 : 1}>
            {staticJoints.map((joint) => {
              // Get current override or use defaults
              // Backend stores [enabled, maxX, maxY] where maxX/maxY are absolute values
              // e.g., maxX=10 means range is -10 to 10
              const override = jointOverrides[joint.name]

              const jointEnabled = override !== undefined ? override[0] : enabled
              // Always read fresh from override or default - never use closure variables
              const maxX = override !== undefined ? override[1] : defaultMaxX
              const maxY = override !== undefined ? override[2] : defaultMaxY

              // Store old values for comparison in onChange
              const oldMaxX = maxX
              const oldMaxY = maxY

              // Convert max values to ranges for display: [-max, max]
              // Always recalculate from latest maxX/maxY values - these are fresh from props
              // Create fresh arrays each render (React will handle optimization)
              const minX = -maxX
              const maxXDisplay = maxX
              const minY = -maxY
              const maxYDisplay = maxY

              // DEBUG: Log current values on each render (after all variables are initialized)
              if (process.env.NODE_ENV === 'development') {
                console.log(`[DEBUG] StaticJoint ${joint.name} render:`, {
                  override,
                  storedValues: { maxX, maxY },
                  displayRange: { minX, maxX: maxXDisplay, minY, maxY: maxYDisplay },
                  jointEnabled,
                  jointOverridesKey: Object.keys(jointOverrides)
                })
              }
              const xRange: [number, number] = [minX, maxXDisplay]
              const yRange: [number, number] = [minY, maxYDisplay]

              return (
                <Box
                  key={`${joint.name}-${maxX}-${maxY}`}
                  sx={{
                    opacity: enabled ? 1 : 0.5,
                    transition: 'opacity 0.2s'
                  }}
                >
                  {/* Single row: Name, Variable toggle, X range slider, Y range slider */}
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {/* Joint name */}
                    <Typography
                      variant="caption"
                      sx={{
                        fontFamily: 'monospace',
                        fontWeight: 500,
                        minWidth: compact ? 100 : 120,
                        fontSize: compact ? '0.7rem' : '0.75rem'
                      }}
                    >
                      {joint.name}
                    </Typography>

                    {/* Variable toggle */}
                    <FormControlLabel
                      control={
                        <Switch
                          size="small"
                          checked={jointEnabled}
                          onChange={(e) => {
                            const newEnabled = e.target.checked
                            // Preserve current max values
                            onJointOverrideChange(joint.name, [newEnabled, maxX, maxY])
                          }}
                          disabled={!enabled}
                        />
                      }
                      label={
                        <Typography variant="caption" sx={{ fontSize: compact ? '0.65rem' : '0.7rem' }}>
                          Variable
                        </Typography>
                      }
                      sx={{ m: 0, minWidth: compact ? 70 : 80 }}
                    />

                    {/* X range slider */}
                    {jointEnabled ? (
                      <>
                        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Typography
                            variant="caption"
                            sx={{
                              minWidth: compact ? 25 : 30,
                              fontSize: compact ? '0.65rem' : '0.7rem',
                              fontFamily: 'monospace',
                              color: 'text.secondary'
                            }}
                          >
                            X:
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              minWidth: compact ? 35 : 40,
                              fontSize: compact ? '0.65rem' : '0.7rem',
                              fontFamily: 'monospace',
                              color: 'text.secondary'
                            }}
                          >
                            {minX.toFixed(0)}
                          </Typography>
                          <Slider
                            key={`${joint.name}-x-${maxX}`}
                            value={xRange}
                            onChange={(_, value) => {
                              const [newMin, newMax] = value as number[]

                              // Ensure min <= max and clamp to slider bounds
                              let finalMin = Math.max(Math.min(newMin, newMax), SLIDER_MIN)
                              let finalMax = Math.min(Math.max(newMin, newMax), SLIDER_MAX)

                              // For symmetric ranges, ensure they're truly symmetric
                              // Take the smaller absolute value to allow making ranges smaller
                              // This ensures if user drags to [-10, 10] from [-11, 11], we get maxX=10
                              const absMin = Math.abs(finalMin)
                              const absMax = Math.abs(finalMax)
                              // Use the smaller absolute value to allow shrinking
                              const newMaxXValue = Math.min(absMin, absMax)

                              // DEBUG: Log onChange event with actual values
                              if (process.env.NODE_ENV === 'development') {
                                console.log(`[DEBUG] StaticJoint ${joint.name} X onChange:`, {
                                  userInput: { newMin, newMax },
                                  afterClamp: { finalMin, finalMax },
                                  absValues: { absMin, absMax },
                                  oldState: { minX, maxX: maxXDisplay, storedMaxX: oldMaxX },
                                  calculatedNewMaxX: newMaxXValue,
                                  willUpdate: newMaxXValue !== oldMaxX
                                })
                              }

                              // Read current Y value fresh from jointOverrides to avoid stale closure
                              const currentOverride = jointOverrides[joint.name]
                              const currentYValue = currentOverride !== undefined ? currentOverride[2] : defaultMaxY

                              // Always update (don't skip) - let React handle optimization
                              // DEBUG: Log before update
                              if (process.env.NODE_ENV === 'development') {
                                console.log(`[DEBUG] StaticJoint ${joint.name} X updating:`, {
                                  newOverride: [jointEnabled, newMaxXValue, currentYValue],
                                  currentOverride,
                                  change: `${oldMaxX} → ${newMaxXValue}`
                                })
                              }

                              // Update with new X value, preserving current Y value
                              onJointOverrideChange(joint.name, [jointEnabled, newMaxXValue, currentYValue])
                            }}
                            min={SLIDER_MIN}
                            max={SLIDER_MAX}
                            step={1}
                            size="small"
                            sx={{
                              flex: 1,
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
                          <Typography
                            variant="caption"
                            sx={{
                              minWidth: compact ? 35 : 40,
                              fontSize: compact ? '0.65rem' : '0.7rem',
                              fontFamily: 'monospace',
                              color: 'text.secondary',
                              textAlign: 'right'
                            }}
                          >
                            {maxXDisplay.toFixed(0)}
                          </Typography>
                        </Box>

                        {/* Y range slider */}
                        <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Typography
                            variant="caption"
                            sx={{
                              minWidth: compact ? 25 : 30,
                              fontSize: compact ? '0.65rem' : '0.7rem',
                              fontFamily: 'monospace',
                              color: 'text.secondary'
                            }}
                          >
                            Y:
                          </Typography>
                          <Typography
                            variant="caption"
                            sx={{
                              minWidth: compact ? 35 : 40,
                              fontSize: compact ? '0.65rem' : '0.7rem',
                              fontFamily: 'monospace',
                              color: 'text.secondary'
                            }}
                          >
                            {minY.toFixed(0)}
                          </Typography>
                          <Slider
                            key={`${joint.name}-y-${maxY}`}
                            value={yRange}
                            onChange={(_, value) => {
                              const [newMin, newMax] = value as number[]

                              // Ensure min <= max and clamp to slider bounds
                              let finalMin = Math.max(Math.min(newMin, newMax), SLIDER_MIN)
                              let finalMax = Math.min(Math.max(newMin, newMax), SLIDER_MAX)

                              // For symmetric ranges, ensure they're truly symmetric
                              // Take the smaller absolute value to allow making ranges smaller
                              // This ensures if user drags to [-10, 10] from [-11, 11], we get maxY=10
                              const absMin = Math.abs(finalMin)
                              const absMax = Math.abs(finalMax)
                              // Use the smaller absolute value to allow shrinking
                              const newMaxYValue = Math.min(absMin, absMax)

                              // DEBUG: Log onChange event with actual values
                              if (process.env.NODE_ENV === 'development') {
                                console.log(`[DEBUG] StaticJoint ${joint.name} Y onChange:`, {
                                  userInput: { newMin, newMax },
                                  afterClamp: { finalMin, finalMax },
                                  absValues: { absMin, absMax },
                                  oldState: { minY, maxY: maxYDisplay, storedMaxY: oldMaxY },
                                  calculatedNewMaxY: newMaxYValue,
                                  willUpdate: newMaxYValue !== oldMaxY
                                })
                              }

                              // Read current X value fresh from jointOverrides to avoid stale closure
                              const currentOverride = jointOverrides[joint.name]
                              const currentXValue = currentOverride !== undefined ? currentOverride[1] : defaultMaxX

                              // Always update (don't skip) - let React handle optimization
                              // DEBUG: Log before update
                              if (process.env.NODE_ENV === 'development') {
                                console.log(`[DEBUG] StaticJoint ${joint.name} Y updating:`, {
                                  newOverride: [jointEnabled, currentXValue, newMaxYValue],
                                  currentOverride,
                                  change: `${oldMaxY} → ${newMaxYValue}`
                                })
                              }

                              // Update with new Y value, preserving current X value
                              onJointOverrideChange(joint.name, [jointEnabled, currentXValue, newMaxYValue])
                            }}
                            min={SLIDER_MIN}
                            max={SLIDER_MAX}
                            step={1}
                            size="small"
                            sx={{
                              flex: 1,
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
                          <Typography
                            variant="caption"
                            sx={{
                              minWidth: compact ? 35 : 40,
                              fontSize: compact ? '0.65rem' : '0.7rem',
                              fontFamily: 'monospace',
                              color: 'text.secondary',
                              textAlign: 'right'
                            }}
                          >
                            {maxYDisplay.toFixed(0)}
                          </Typography>
                        </Box>
                      </>
                    ) : (
                      <Box sx={{ flex: 1 }} /> // Spacer when disabled
                    )}
                  </Box>
                </Box>
              )
            })}
          </Stack>
        </Box>
      )}
    </Stack>
  )
}

export default StaticJointMovementConfig
