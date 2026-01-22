/**
 * Animate Toolbar - Horizontal animation controls at bottom of screen
 * Features: Play/Pause with cheetah animation, step controls, speed slider, reset
 */
import React from 'react'
import { Box, IconButton, Slider, Tooltip, Typography } from '@mui/material'
import { canSimulate, type TrajectoryData } from '../../AnimateSimulate'
import cheetahGif from '../../../assets/cheetah_run.gif'

export interface AnimationState {
  isAnimating: boolean
  currentFrame: number
  totalFrames: number
  playbackSpeed: number
  loop: boolean
}

export interface AnimateToolbarProps {
  // Joint data for simulation checks
  joints: { type: string; name: string }[]

  // Animation state & controls
  animationState: AnimationState
  playAnimation: () => void
  pauseAnimation: () => void
  stopAnimation: () => void
  setPlaybackSpeed: (speed: number) => void
  setAnimatedPositions: (positions: null) => void
  setFrame: (frame: number) => void

  // Simulation state & controls
  isSimulating: boolean
  trajectoryData: TrajectoryData | null
  autoSimulateEnabled: boolean
  setAutoSimulateEnabled: (enabled: boolean) => void
  runSimulation: () => Promise<void>
  triggerMechanismChange: () => void

  // Trajectory display
  showTrajectory: boolean
  setShowTrajectory: (show: boolean) => void

  // Stretching links (constraint violations)
  stretchingLinks: string[]

  // Status display
  showStatus: (message: string, type: 'info' | 'success' | 'error' | 'action' | 'warning', duration?: number) => void

  // Dark mode (from app state, not MUI theme)
  darkMode?: boolean
}

export const AnimateToolbar: React.FC<AnimateToolbarProps> = ({
  joints,
  animationState,
  playAnimation,
  pauseAnimation,
  stopAnimation,
  setPlaybackSpeed,
  setAnimatedPositions,
  setFrame,
  isSimulating,
  trajectoryData,
  autoSimulateEnabled,
  setAutoSimulateEnabled,
  runSimulation,
  triggerMechanismChange,
  showTrajectory,
  setShowTrajectory,
  stretchingLinks,
  showStatus,
  darkMode = false
}) => {
  // Use darkMode prop directly (app uses body class, not MUI theme)
  const isDark = darkMode

  const hasCrank = canSimulate(joints)
  const canAnimate = trajectoryData !== null && trajectoryData.nSteps > 0 && stretchingLinks.length === 0
  const hasStretchingLinks = stretchingLinks.length > 0

  // Theme-aware colors - distinct palettes for light and dark modes
  const colors = isDark ? {
    // Dark theme - dark background, light elements
    bg: 'rgba(25, 25, 30, 0.96)',
    border: 'rgba(255, 255, 255, 0.12)',
    text: 'rgba(255, 255, 255, 0.85)',
    textMuted: 'rgba(255, 255, 255, 0.55)',
    buttonBg: 'rgba(255, 255, 255, 0.08)',
    buttonHover: 'rgba(255, 255, 255, 0.15)',
    disabled: 'rgba(255, 255, 255, 0.25)',
    shadow: '0 8px 32px rgba(0, 0, 0, 0.5), 0 0 1px rgba(255, 255, 255, 0.15) inset',
    sliderRail: 'rgba(255, 255, 255, 0.25)',
    sliderThumb: '#fff',
    sliderColor: '#90caf9',
    pathActive: { bg: 'rgba(156, 39, 176, 0.25)', color: '#ce93d8' },
    simActive: { bg: 'rgba(33, 150, 243, 0.25)', color: '#90caf9' },
  } : {
    // Light theme - light background, dark elements
    bg: 'rgba(250, 250, 252, 0.97)',
    border: 'rgba(0, 0, 0, 0.1)',
    text: 'rgba(0, 0, 0, 0.75)',
    textMuted: 'rgba(0, 0, 0, 0.5)',
    buttonBg: 'rgba(0, 0, 0, 0.05)',
    buttonHover: 'rgba(0, 0, 0, 0.1)',
    disabled: 'rgba(0, 0, 0, 0.2)',
    shadow: '0 4px 24px rgba(0, 0, 0, 0.12), 0 0 1px rgba(0, 0, 0, 0.08)',
    sliderRail: 'rgba(0, 0, 0, 0.2)',
    sliderThumb: '#FA8112', // Primary/cheetah orange
    sliderColor: '#FA8112',
    pathActive: { bg: 'rgba(156, 39, 176, 0.15)', color: '#9c27b0' },
    simActive: { bg: 'rgba(33, 150, 243, 0.15)', color: '#1976d2' },
  }

  // Step frame forward
  const stepForward = () => {
    if (!canAnimate) return
    const nextFrame = (animationState.currentFrame + 1) % animationState.totalFrames
    setFrame(nextFrame)
  }

  // Step frame backward
  const stepBackward = () => {
    if (!canAnimate) return
    const prevFrame = animationState.currentFrame === 0
      ? animationState.totalFrames - 1
      : animationState.currentFrame - 1
    setFrame(prevFrame)
  }

  // Handle play button click
  const handlePlayClick = () => {
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
  }

  // Handle reset
  const handleReset = () => {
    stopAnimation()
    setAnimatedPositions(null)
  }

  // Handle speed slider change
  const handleSpeedChange = (_event: Event, value: number | number[]) => {
    setPlaybackSpeed(value as number)
  }

  // Toggle continuous simulation
  const toggleContinuousSimulation = () => {
    if (autoSimulateEnabled) {
      setAutoSimulateEnabled(false)
    } else {
      setAutoSimulateEnabled(true)
      triggerMechanismChange()
    }
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 20,  // Moved up to prevent clipping
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2.2,  // 10% bigger gaps
        px: 3.3,   // 10% bigger padding
        py: 1.8,   // More vertical padding
        backgroundColor: colors.bg,
        backdropFilter: 'blur(12px)',
        borderRadius: 3.3,  // 10% bigger radius
        boxShadow: colors.shadow,
        border: `1px solid ${colors.border}`,
        zIndex: 1000,
        minWidth: 638,  // 10% bigger (580 * 1.1)
        height: 88,     // Taller to prevent clipping (72 * 1.1 + extra)
      }}
    >
      {/* ═══════════════════════════════════════════════════════════════════════
          FURTHEST LEFT: Hide Paths Button
          ═══════════════════════════════════════════════════════════════════════ */}
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {showTrajectory ? 'Hide Trajectory Paths' : 'Show Trajectory Paths'}
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
              Toggle visibility of all joint trajectory paths on the canvas.
              Individual paths can be controlled per-joint in the Edit Joint modal.
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <span>
          <IconButton
            onClick={() => setShowTrajectory(!showTrajectory)}
            disabled={!trajectoryData}
            sx={{
              width: 44,  // 10% bigger
              height: 44,
              backgroundColor: showTrajectory ? colors.pathActive.bg : colors.buttonBg,
              color: showTrajectory ? colors.pathActive.color : colors.textMuted,
              '&:hover': {
                backgroundColor: showTrajectory
                  ? (isDark ? 'rgba(156, 39, 176, 0.35)' : 'rgba(156, 39, 176, 0.25)')
                  : colors.buttonHover,
              },
              '&.Mui-disabled': {
                color: colors.disabled,
              },
              transition: 'all 0.2s ease',
            }}
          >
            <Typography sx={{ fontSize: '1.2rem' }}>{showTrajectory ? '◉' : '○'}</Typography>
          </IconButton>
        </span>
      </Tooltip>

      {/* ═══════════════════════════════════════════════════════════════════════
          FURTHER LEFT: Continuous Simulation Button
          ═══════════════════════════════════════════════════════════════════════ */}
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              {autoSimulateEnabled ? 'Disable' : 'Enable'} Continuous Simulation
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
              When enabled, the trajectory is automatically recomputed whenever
              you modify the mechanism (move joints, change links, etc).
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.6, display: 'block', mt: 0.25 }}>
              Disable for complex mechanisms to improve editing performance.
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <span>
          <IconButton
            onClick={toggleContinuousSimulation}
            disabled={isSimulating || !hasCrank}
            sx={{
              width: 44,  // 10% bigger
              height: 44,
              backgroundColor: autoSimulateEnabled ? colors.simActive.bg : colors.buttonBg,
              color: autoSimulateEnabled ? colors.simActive.color : colors.textMuted,
              '&:hover': {
                backgroundColor: autoSimulateEnabled
                  ? (isDark ? 'rgba(33, 150, 243, 0.35)' : 'rgba(33, 150, 243, 0.25)')
                  : colors.buttonHover,
              },
              '&.Mui-disabled': {
                color: colors.disabled,
              },
              transition: 'all 0.2s ease',
            }}
          >
            <Typography sx={{ fontSize: '1.1rem' }}>∞</Typography>
          </IconButton>
        </span>
      </Tooltip>

      {/* ═══════════════════════════════════════════════════════════════════════
          LEFT: Step Back Button
          ═══════════════════════════════════════════════════════════════════════ */}
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>Previous Frame</Typography>
            <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
              Step back one frame in the animation sequence.
              Loops to the last frame when at the beginning.
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <span>
          <IconButton
            onClick={stepBackward}
            disabled={!canAnimate}
            sx={{
              width: 48,  // 10% bigger
              height: 48,
              backgroundColor: colors.buttonBg,
              color: colors.text,
              '&:hover': {
                backgroundColor: colors.buttonHover,
              },
              '&.Mui-disabled': {
                color: colors.disabled,
              },
            }}
          >
            <Typography sx={{ fontSize: '1.4rem', fontWeight: 300 }}>⏮</Typography>
          </IconButton>
        </span>
      </Tooltip>

      {/* ═══════════════════════════════════════════════════════════════════════
          CENTER: Play Button with Cheetah Animation
          ═══════════════════════════════════════════════════════════════════════ */}
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Tooltip
          title={
            <Box sx={{ p: 0.5 }}>
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {animationState.isAnimating ? 'Pause Animation' : 'Play Animation'}
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
                {animationState.isAnimating
                  ? 'Pause the animation at the current frame.'
                  : 'Start animating the mechanism through its trajectory cycle.'
                }
              </Typography>
              <Typography variant="caption" sx={{
                opacity: 0.6,
                display: 'block',
                mt: 0.5,
                borderTop: '1px solid rgba(255,255,255,0.2)',
                pt: 0.5
              }}>
                Keyboard shortcut: Spacebar
              </Typography>
            </Box>
          }
          placement="top"
          arrow
        >
          <span>
            <IconButton
              onClick={handlePlayClick}
              disabled={!hasCrank || isSimulating || hasStretchingLinks}
              sx={{
                width: 62,   // 10% bigger
                height: 62,
                backgroundColor: animationState.isAnimating
                  ? 'rgba(255, 152, 0, 0.2)'
                  : 'rgba(76, 175, 80, 0.2)',
                border: `2px solid ${animationState.isAnimating ? '#ff9800' : '#4caf50'}`,
                overflow: 'hidden',
                position: 'relative',
                '&:hover': {
                  backgroundColor: animationState.isAnimating
                    ? 'rgba(255, 152, 0, 0.3)'
                    : 'rgba(76, 175, 80, 0.3)',
                },
                '&.Mui-disabled': {
                  backgroundColor: colors.buttonBg,
                  borderColor: colors.disabled,
                },
                transition: 'all 0.3s ease',
              }}
            >
              {animationState.isAnimating ? (
                // Show running cheetah gif when playing - shifted down and right for centering
                <Box
                  component="img"
                  src={cheetahGif}
                  alt="Running cheetah"
                  sx={{
                    width: 54,
                    height: 54,
                    objectFit: 'cover',
                    filter: 'brightness(1.2) contrast(1.1)',
                    transform: 'translate(2px, 2px)',  // Shift down and right ~20%
                  }}
                />
              ) : (
                // Show play icon when paused - shifted down and right for centering
                <Typography
                  sx={{
                    fontSize: '2rem',
                    color: '#4caf50',
                    textShadow: '0 0 8px rgba(76, 175, 80, 0.5)',
                    transform: 'translate(2px, 1px)',  // Shift down and right ~20%
                  }}
                >
                  ▶
                </Typography>
              )}
            </IconButton>
          </span>
        </Tooltip>

        {/* Frame Counter */}
        <Typography
          variant="caption"
          sx={{
            color: colors.textMuted,
            mt: 0.3,
            fontFamily: 'monospace',
            fontSize: '0.75rem',  // Slightly bigger
            letterSpacing: '0.05em',
          }}
        >
          {canAnimate
            ? `${String(animationState.currentFrame + 1).padStart(2, '0')}/${animationState.totalFrames}`
            : '--/--'
          }
        </Typography>
      </Box>

      {/* ═══════════════════════════════════════════════════════════════════════
          RIGHT: Step Forward Button
          ═══════════════════════════════════════════════════════════════════════ */}
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>Next Frame</Typography>
            <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
              Advance one frame in the animation sequence.
              Loops back to the first frame when at the end.
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <span>
          <IconButton
            onClick={stepForward}
            disabled={!canAnimate}
            sx={{
              width: 48,  // 10% bigger
              height: 48,
              backgroundColor: colors.buttonBg,
              color: colors.text,
              '&:hover': {
                backgroundColor: colors.buttonHover,
              },
              '&.Mui-disabled': {
                color: colors.disabled,
              },
            }}
          >
            <Typography sx={{ fontSize: '1.4rem', fontWeight: 300 }}>⏭</Typography>
          </IconButton>
        </span>
      </Tooltip>

      {/* ═══════════════════════════════════════════════════════════════════════
          FURTHER RIGHT: Reset Button
          ═══════════════════════════════════════════════════════════════════════ */}
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>Reset Animation</Typography>
            <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
              Stop playback and return the mechanism to its initial position (frame 0).
              Joint positions will be restored to their starting configuration.
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <span>
          <IconButton
            onClick={handleReset}
            disabled={!canAnimate && animationState.currentFrame === 0}
            sx={{
              width: 44,  // 10% bigger
              height: 44,
              backgroundColor: colors.buttonBg,
              color: colors.text,
              '&:hover': {
                backgroundColor: colors.buttonHover,
              },
              '&.Mui-disabled': {
                color: colors.disabled,
              },
            }}
          >
            <Typography sx={{ fontSize: '1.3rem' }}>↺</Typography>
          </IconButton>
        </span>
      </Tooltip>

      {/* ═══════════════════════════════════════════════════════════════════════
          SPEED SLIDER - Compact version
          ═══════════════════════════════════════════════════════════════════════ */}
      <Tooltip
        title={
          <Box sx={{ p: 0.5 }}>
            <Typography variant="body2" sx={{ fontWeight: 600 }}>
              Playback Speed: {animationState.playbackSpeed}×
            </Typography>
            <Typography variant="caption" sx={{ opacity: 0.8, display: 'block', mt: 0.5 }}>
              Adjust how fast the animation plays.
              Range: 0.1× (slow-mo) to 10× (fast-forward).
            </Typography>
          </Box>
        }
        placement="top"
        arrow
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            width: 77,  // 10% bigger
          }}
        >
          <Slider
            value={animationState.playbackSpeed}
            onChange={handleSpeedChange}
            min={0.1}
            max={10}
            step={0.1}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}×`}
            sx={{
              width: '100%',
              color: colors.sliderColor,
              '& .MuiSlider-thumb': {
                width: 15,  // Slightly bigger
                height: 15,
                backgroundColor: colors.sliderThumb,
                '&:hover': {
                  boxShadow: `0 0 0 6px ${isDark ? 'rgba(144, 202, 249, 0.16)' : 'rgba(25, 118, 210, 0.16)'}`,
                },
              },
              '& .MuiSlider-track': {
                height: 4,
              },
              '& .MuiSlider-rail': {
                height: 4,
                backgroundColor: colors.sliderRail,
              },
              '& .MuiSlider-valueLabel': {
                backgroundColor: isDark ? 'rgba(30, 30, 35, 0.95)' : 'rgba(50, 50, 55, 0.95)',
                fontSize: '0.75rem',
                padding: '3px 7px',
              },
            }}
          />
        </Box>
      </Tooltip>
    </Box>
  )
}

export default AnimateToolbar
