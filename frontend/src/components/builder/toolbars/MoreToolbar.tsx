/**
 * More Toolbar - Animation, Simulation, Demos, File Operations, Validation
 */
import React from 'react'
import { Box, Typography, Button, Tooltip, Divider } from '@mui/material'
import { canSimulate, type TrajectoryData } from '../../AnimateSimulate'
import type { PylinkJoint } from '../types'

export interface AnimationState {
  isAnimating: boolean
  currentFrame: number
  totalFrames: number
  playbackSpeed: number
}

// Re-export TrajectoryData for convenience
export type { TrajectoryData }

export interface MoreToolbarProps {
  // Joint data for simulation checks
  joints: PylinkJoint[]

  // Animation state & controls
  animationState: AnimationState
  playAnimation: () => void
  pauseAnimation: () => void
  stopAnimation: () => void
  setPlaybackSpeed: (speed: number) => void
  setAnimatedPositions: (positions: null) => void

  // Simulation state & controls
  isSimulating: boolean
  trajectoryData: TrajectoryData | null
  autoSimulateEnabled: boolean
  setAutoSimulateEnabled: (enabled: boolean) => void
  autoSimulateDelayMs: number
  runSimulation: () => Promise<void>
  triggerMechanismChange: () => void

  // Trajectory display
  showTrajectory: boolean
  setShowTrajectory: (show: boolean) => void

  // Stretching links (constraint violations)
  stretchingLinks: string[]

  // File & demo operations
  loadDemo4Bar: () => void
  loadPylinkGraph: () => void
  savePylinkGraph: () => void
  validateMechanism: () => void

  // Status display
  showStatus: (message: string, type: 'info' | 'success' | 'error' | 'action' | 'warning', duration?: number) => void
}

export const MoreToolbar: React.FC<MoreToolbarProps> = ({
  joints,
  animationState,
  playAnimation,
  pauseAnimation,
  stopAnimation,
  setPlaybackSpeed,
  setAnimatedPositions,
  isSimulating,
  trajectoryData,
  autoSimulateEnabled,
  setAutoSimulateEnabled,
  autoSimulateDelayMs,
  runSimulation,
  triggerMechanismChange,
  showTrajectory,
  setShowTrajectory,
  stretchingLinks,
  loadDemo4Bar,
  loadPylinkGraph,
  savePylinkGraph,
  validateMechanism,
  showStatus
}) => {
  const hasCrank = canSimulate(joints)
  const canAnimate = trajectoryData !== null && trajectoryData.nSteps > 0 && stretchingLinks.length === 0
  const hasStretchingLinks = stretchingLinks.length > 0

  return (
    <Box sx={{ p: 1.5 }}>
      {/* ═══════════════════════════════════════════════════════════════════════
          ANIMATION - Plays through simulation frames to animate linkage motion
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Animation
      </Typography>
      <Box sx={{ display: 'flex', gap: 0.5, mt: 1, mb: 1 }}>
        <Tooltip
          title={
            <Box>
              <Typography variant="body2">
                {animationState.isAnimating ? 'Pause animation' : 'Play animation'}
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8 }}>
                Shortcut: Spacebar
              </Typography>
            </Box>
          }
          enterDelay={400}
          leaveDelay={100}
          placement="top"
          arrow
        >
          <span style={{ flex: 1 }}>
            <Button
              variant="contained"
              fullWidth
              size="small"
              onMouseDown={(e) => {
                e.preventDefault()
                if (hasStretchingLinks) {
                  showStatus(
                    `Cannot animate: ${stretchingLinks.join(', ')} would stretch. Fix kinematic constraints first.`,
                    'error',
                    3000
                  )
                  return
                }
                if (animationState.isAnimating) {
                  pauseAnimation()
                } else {
                  if (!canAnimate) {
                    runSimulation().then(() => {
                      setTimeout(() => playAnimation(), 100)
                    })
                  } else {
                    playAnimation()
                  }
                }
              }}
              disabled={!hasCrank || isSimulating || hasStretchingLinks}
              sx={{
                textTransform: 'none', fontSize: '0.75rem',
                backgroundColor: animationState.isAnimating ? '#ff9800' : '#4caf50',
                '&:hover': { backgroundColor: animationState.isAnimating ? '#f57c00' : '#388e3c' },
                '&.Mui-disabled': { backgroundColor: '#e0e0e0' },
                pointerEvents: 'auto'
              }}
            >
              {animationState.isAnimating ? '⏸ Pause' : '▶ Play'}
            </Button>
          </span>
        </Tooltip>
        <Tooltip
          title="Reset: Returns the mechanism to its starting position (frame 0)"
          enterDelay={400}
          leaveDelay={100}
          placement="top"
          arrow
        >
          <span>
            <Button
              variant="outlined"
              size="small"
              onClick={() => {
                stopAnimation()
                setAnimatedPositions(null)
              }}
              disabled={!canAnimate && animationState.currentFrame === 0}
              sx={{
                textTransform: 'none', fontSize: '0.75rem', minWidth: 40,
                borderColor: '#666', color: '#666'
              }}
            >
              ↺
            </Button>
          </span>
        </Tooltip>
      </Box>

      {/* Animation info & speed control */}
      {canAnimate && (
        <Box sx={{ mb: 1.5 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
            Frame: {animationState.currentFrame + 1} / {animationState.totalFrames}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
              Speed:
            </Typography>
            {[0.5, 1, 2].map(speed => (
              <Box
                key={speed}
                onClick={() => setPlaybackSpeed(speed)}
                sx={{
                  px: 0.75, py: 0.25, borderRadius: 1,
                  fontSize: '0.65rem', cursor: 'pointer',
                  bgcolor: animationState.playbackSpeed === speed ? 'primary.main' : 'grey.100',
                  color: animationState.playbackSpeed === speed ? '#fff' : 'text.secondary',
                  '&:hover': { bgcolor: animationState.playbackSpeed === speed ? 'primary.dark' : 'grey.200' }
                }}
              >
                {speed}x
              </Box>
            ))}
          </Box>
        </Box>
      )}

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          SIMULATION - Computes and displays trajectory dots
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Trajectory Simulation
      </Typography>
      <Box sx={{ display: 'flex', gap: 0.5, mt: 1, mb: 1 }}>
        <Tooltip
          title={
            <Box sx={{ p: 0.5, maxWidth: 280 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                {autoSimulateEnabled ? 'Disable Continuous Simulation' : 'Enable Continuous Simulation'}
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontSize: '0.65rem' }}>
                When enabled, the trajectory is automatically recomputed whenever you modify the mechanism.
                Delay: {autoSimulateDelayMs}ms (configurable in Settings).
              </Typography>
            </Box>
          }
          enterDelay={500}
          leaveDelay={100}
          placement="top"
          arrow
        >
          <span style={{ flex: 1 }}>
            <Button
              variant="contained"
              fullWidth
              size="small"
              onClick={() => {
                if (autoSimulateEnabled) {
                  setAutoSimulateEnabled(false)
                } else {
                  setAutoSimulateEnabled(true)
                  triggerMechanismChange()
                }
              }}
              disabled={isSimulating || !hasCrank}
              sx={{
                textTransform: 'none', fontSize: '0.7rem',
                backgroundColor: autoSimulateEnabled ? '#2196f3' : '#666',
                '&:hover': { backgroundColor: autoSimulateEnabled ? '#1976d2' : '#555' }
              }}
            >
              {autoSimulateEnabled ? '◉ Continuous Simulation' : '○ Continuous Simulation'}
            </Button>
          </span>
        </Tooltip>
      </Box>

      {/* Show/Hide trajectory toggle when we have data */}
      {trajectoryData && (
        <Box sx={{ mb: 1 }}>
          <Button
            variant={showTrajectory ? 'contained' : 'outlined'}
            fullWidth
            size="small"
            onClick={() => setShowTrajectory(!showTrajectory)}
            sx={{
              textTransform: 'none', fontSize: '0.7rem',
              backgroundColor: showTrajectory ? '#9c27b0' : 'transparent',
              borderColor: '#9c27b0', color: showTrajectory ? '#fff' : '#9c27b0',
              '&:hover': { backgroundColor: showTrajectory ? '#7b1fa2' : 'rgba(156, 39, 176, 0.1)' }
            }}
          >
            {showTrajectory ? 'Hide Paths' : 'Show All Paths'}
          </Button>
        </Box>
      )}

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          DEMOS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Demos
      </Typography>
      <Button
        variant="outlined"
        fullWidth
        size="small"
        onClick={loadDemo4Bar}
        sx={{ mt: 1, mb: 1.5, textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
      >
        ◇ Four Bar Demo
      </Button>

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          FILE OPERATIONS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        File Operations
      </Typography>
      <Button
        variant="outlined"
        fullWidth
        size="small"
        onClick={loadPylinkGraph}
        sx={{ mt: 1, mb: 1, textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
      >
        ↑ Load
      </Button>
      <Button
        variant="outlined"
        fullWidth
        size="small"
        onClick={savePylinkGraph}
        sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
      >
        ↓ Save
      </Button>

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          VALIDATION
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Validation
      </Typography>
      <Tooltip
        title={
          <Box sx={{ p: 0.5, maxWidth: 240 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
              Validate Mechanism
            </Typography>
            <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontSize: '0.65rem' }}>
              Checks if the mechanism can be simulated. Requires links, a Crank driver, and Static ground.
            </Typography>
          </Box>
        }
        enterDelay={500}
        leaveDelay={100}
        placement="top"
        arrow
      >
        <span style={{ display: 'block', marginTop: 8 }}>
          <Button
            variant="outlined"
            fullWidth
            size="small"
            onClick={validateMechanism}
            sx={{
              textTransform: 'none',
              justifyContent: 'flex-start',
              fontSize: '0.75rem',
              borderColor: '#1976d2',
              color: '#1976d2',
              '&:hover': {
                backgroundColor: 'rgba(25, 118, 210, 0.08)',
                borderColor: '#1565c0'
              }
            }}
          >
            ✓ Validate
          </Button>
        </span>
      </Tooltip>
    </Box>
  )
}

export default MoreToolbar
