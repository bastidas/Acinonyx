/**
 * AnimateSimulate.tsx
 *
 * Helper module for animation and simulation functionality.
 * Handles:
 * - Animation state management (play/pause frame-by-frame linkage motion)
 * - Simulation (computing trajectories from the backend)
 * - Auto-simulation with configurable delay
 */

import { useState, useRef, useCallback, useEffect, useLayoutEffect } from 'react'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/** Trajectory data returned from simulation */
export interface TrajectoryData {
  trajectories: Record<string, [number, number][]>
  nSteps: number
  jointTypes: Record<string, string>
}

/** Playback direction: 1 = forward, -1 = reverse */
export type PlaybackDirection = 1 | -1

/** Animation state */
export interface AnimationState {
  isAnimating: boolean
  currentFrame: number
  totalFrames: number
  /** Frames (simulation steps) per second. 1–400, default 10. */
  playbackFps: number
  loop: boolean
  playbackDirection: PlaybackDirection  // 1 = forward, -1 = reverse
}

/** Default playback FPS when there is no trajectory. */
export const DEFAULT_PLAYBACK_FPS = 10

/**
 * FPS range is proportional to simulation steps.
 * - ~1 to 100 for 10 steps (default 10)
 * - ~10 to 1000 for 100 steps (default 200)
 */
export function getMinPlaybackFps(totalFrames: number): number {
  if (totalFrames <= 0) return 1
  return Math.max(1, Math.floor(totalFrames / 10))
}

const MAX_PLAYBACK_SPEED = 500

export function getMaxPlaybackFps(totalFrames: number): number {
  if (totalFrames <= 0) return 100
  return Math.min(MAX_PLAYBACK_SPEED, totalFrames * 10)
}

export function getDefaultPlaybackFps(totalFrames: number): number {
  if (totalFrames <= 0) return DEFAULT_PLAYBACK_FPS
  if (totalFrames <= 10) return totalFrames
  return Math.min(getMaxPlaybackFps(totalFrames), totalFrames * 2)
}

/** Initial animation state */
export const initialAnimationState: AnimationState = {
  isAnimating: false,
  currentFrame: 0,
  totalFrames: 0,
  playbackFps: DEFAULT_PLAYBACK_FPS,
  loop: true,
  playbackDirection: 1
}

/** Simulation state */
export interface SimulationState {
  isSimulating: boolean
  autoSimulateEnabled: boolean
  simulationSteps: number
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION HOOK
// ═══════════════════════════════════════════════════════════════════════════════

interface UseAnimationProps {
  trajectoryData: TrajectoryData | null
  onFrameChange: (frame: number) => void
}

interface UseAnimationReturn {
  animationState: AnimationState
  play: () => void
  pause: () => void
  stop: () => void
  reset: () => void
  setFrame: (frame: number) => void
  setPlaybackFps: (fps: number) => void
  setLoop: (loop: boolean) => void
  setPlaybackDirection: (direction: PlaybackDirection) => void
  /** Get interpolated joint positions for the current frame */
  getAnimatedPositions: () => Record<string, [number, number]> | null
}

/**
 * Hook for managing linkage animation playback.
 * Plays through simulation frames to animate the mechanism.
 */
export function useAnimation({
  trajectoryData,
  onFrameChange
}: UseAnimationProps): UseAnimationReturn {
  const [animationState, setAnimationState] = useState<AnimationState>({
    ...initialAnimationState,
    totalFrames: trajectoryData?.nSteps ?? 0
  })

  const animationFrameRef = useRef<number | null>(null)
  const lastFrameTimeRef = useRef<number>(0)
  const prevTrajectoryDataRef = useRef<TrajectoryData | null>(null)

  // When trajectoryData reference changes, show frame 0 in the SAME render so we never paint N/N.
  const trajectoryDataChanged = trajectoryData !== prevTrajectoryDataRef.current
  const effectiveFrame = trajectoryDataChanged ? 0 : animationState.currentFrame
  const effectiveAnimationState = trajectoryDataChanged ? { ...animationState, currentFrame: 0 } : animationState
  const effectiveFrameRef = useRef(effectiveFrame)
  effectiveFrameRef.current = effectiveFrame

  // Use refs to track current values without causing re-renders in animation loop
  const isAnimatingRef = useRef(animationState.isAnimating)
  const playbackFpsRef = useRef(animationState.playbackFps)
  const loopRef = useRef(animationState.loop)
  const playbackDirectionRef = useRef(animationState.playbackDirection)
  const currentFrameRef = useRef(animationState.currentFrame)
  const totalFramesRef = useRef(animationState.totalFrames)
  const onFrameChangeRef = useRef(onFrameChange)

  // Keep refs in sync with state
  useEffect(() => {
    isAnimatingRef.current = animationState.isAnimating
    playbackFpsRef.current = animationState.playbackFps
    loopRef.current = animationState.loop
    playbackDirectionRef.current = animationState.playbackDirection
    currentFrameRef.current = animationState.currentFrame
    totalFramesRef.current = animationState.totalFrames
  }, [animationState])

  useEffect(() => {
    onFrameChangeRef.current = onFrameChange
  }, [onFrameChange])

  // Update total frames when trajectory data changes; always start at first frame (1/N)
  // and set playback FPS to the step-proportional default (e.g. 10 for 10 steps, 200 for 100 steps).
  useLayoutEffect(() => {
    prevTrajectoryDataRef.current = trajectoryData
    if (trajectoryData) {
      const newTotalFrames = trajectoryData.nSteps
      totalFramesRef.current = newTotalFrames
      const defaultFps = getDefaultPlaybackFps(newTotalFrames)
      setAnimationState(prev => ({
        ...prev,
        totalFrames: newTotalFrames,
        currentFrame: 0,
        playbackFps: defaultFps
      }))
      playbackFpsRef.current = defaultFps
    } else {
      totalFramesRef.current = 0
      isAnimatingRef.current = false
      currentFrameRef.current = 0
      setAnimationState(prev => ({
        ...prev,
        totalFrames: 0,
        currentFrame: 0,
        isAnimating: false,
        playbackFps: DEFAULT_PLAYBACK_FPS
      }))
      playbackFpsRef.current = DEFAULT_PLAYBACK_FPS
    }
  }, [trajectoryData])

  // Animation loop - only depends on isAnimating state to start/stop
  useEffect(() => {
    if (!animationState.isAnimating || !trajectoryData) {
      // Cancel any running animation when we stop
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
      return
    }

    const animate = (timestamp: number) => {
      // Check ref to see if we should stop (allows immediate pause)
      if (!isAnimatingRef.current) {
        animationFrameRef.current = null
        return
      }

      const elapsed = timestamp - lastFrameTimeRef.current
      const fps = playbackFpsRef.current
      const adjustedInterval = fps > 0 ? 1000 / fps : 1000

      if (elapsed >= adjustedInterval) {
        lastFrameTimeRef.current = timestamp

        const direction = playbackDirectionRef.current
        let nextFrame = currentFrameRef.current + direction

        // Forward: handle end of timeline
        if (nextFrame >= totalFramesRef.current) {
          if (loopRef.current) {
            nextFrame = 0
          } else {
            isAnimatingRef.current = false
            setAnimationState(prev => ({
              ...prev,
              isAnimating: false,
              currentFrame: prev.totalFrames - 1
            }))
            animationFrameRef.current = null
            return
          }
        }

        // Reverse: handle start of timeline
        if (nextFrame < 0) {
          if (loopRef.current) {
            nextFrame = totalFramesRef.current - 1
          } else {
            isAnimatingRef.current = false
            setAnimationState(prev => ({
              ...prev,
              isAnimating: false,
              currentFrame: 0
            }))
            animationFrameRef.current = null
            return
          }
        }

        // Update refs immediately
        currentFrameRef.current = nextFrame

        // Notify parent of frame change
        onFrameChangeRef.current(nextFrame)

        // Batch state update
        setAnimationState(prev => ({ ...prev, currentFrame: nextFrame }))
      }

      // Continue animation loop
      animationFrameRef.current = requestAnimationFrame(animate)
    }

    lastFrameTimeRef.current = performance.now()
    animationFrameRef.current = requestAnimationFrame(animate)

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }
    }
  }, [animationState.isAnimating, trajectoryData])

  const play = useCallback(() => {
    if (!trajectoryData || trajectoryData.nSteps === 0) return
    // Update ref immediately for responsive animation start
    isAnimatingRef.current = true
    setAnimationState(prev => ({ ...prev, isAnimating: true }))
  }, [trajectoryData])

  const pause = useCallback(() => {
    // Update ref immediately so animation loop stops on next check
    isAnimatingRef.current = false
    // Cancel any pending animation frame
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    setAnimationState(prev => ({ ...prev, isAnimating: false }))
  }, [])

  const stop = useCallback(() => {
    // Update refs immediately
    isAnimatingRef.current = false
    currentFrameRef.current = 0
    // Cancel any pending animation frame
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
    setAnimationState(prev => ({ ...prev, isAnimating: false, currentFrame: 0 }))
    onFrameChangeRef.current(0)
  }, [])

  const reset = useCallback(() => {
    currentFrameRef.current = 0
    setAnimationState(prev => ({ ...prev, currentFrame: 0 }))
    onFrameChangeRef.current(0)
  }, [])

  const setFrame = useCallback((frame: number) => {
    const clampedFrame = Math.max(0, Math.min(frame, totalFramesRef.current - 1))
    currentFrameRef.current = clampedFrame
    setAnimationState(prev => ({
      ...prev,
      currentFrame: clampedFrame
    }))
    onFrameChangeRef.current(clampedFrame)
  }, [])

  const setPlaybackFps = useCallback((fps: number) => {
    const min = getMinPlaybackFps(totalFramesRef.current)
    const max = getMaxPlaybackFps(totalFramesRef.current)
    const clamped = Math.max(min, Math.min(max, fps))
    playbackFpsRef.current = clamped
    setAnimationState(prev => ({ ...prev, playbackFps: clamped }))
  }, [])

  const setLoop = useCallback((loop: boolean) => {
    loopRef.current = loop
    setAnimationState(prev => ({ ...prev, loop }))
  }, [])

  const setPlaybackDirection = useCallback((direction: PlaybackDirection) => {
    playbackDirectionRef.current = direction
    setAnimationState(prev => ({ ...prev, playbackDirection: direction }))
  }, [])

  /** Get joint positions for the current animation frame (uses effective frame when trajectoryData just changed). */
  const getAnimatedPositions = useCallback((): Record<string, [number, number]> | null => {
    if (!trajectoryData) return null

    const positions: Record<string, [number, number]> = {}
    const frame = effectiveFrameRef.current

    for (const [jointName, trajectory] of Object.entries(trajectoryData.trajectories)) {
      if (trajectory && trajectory[frame]) {
        positions[jointName] = trajectory[frame]
      }
    }

    return positions
  }, [trajectoryData])

  return {
    animationState: effectiveAnimationState,
    play,
    pause,
    stop,
    reset,
    setFrame,
    setPlaybackFps,
    setLoop,
    setPlaybackDirection,
    getAnimatedPositions
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION HOOK
// ═══════════════════════════════════════════════════════════════════════════════

// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface UseSimulationProps {
  /**
   * The linkage document to simulate.
   * Supports both formats:
   * - New hypergraph format: { linkage: { nodes, edges }, meta }
   * - Legacy format: { pylinkage: { joints }, meta }
   */
  linkageDoc: any
  simulationSteps: number
  autoSimulateDelayMs: number
  autoSimulateEnabled: boolean
  mechanismVersion: number
  /** When true, do not schedule auto-sim (avoids running with inconsistent doc during drag). */
  isDragging?: boolean
  onSimulationStart?: () => void
  onSimulationComplete?: (data: TrajectoryData) => void
  onSimulationError?: (error: string) => void
  showStatus?: (message: string, type: 'info' | 'success' | 'error' | 'action' | 'warning', duration?: number) => void
}

interface UseSimulationReturn {
  isSimulating: boolean
  trajectoryData: TrajectoryData | null
  /** Manually trigger a simulation. Pass docOverride to use that doc instead of current state (e.g. drag-end synced doc). */
  runSimulation: (docOverride?: any) => Promise<void>
  /** Clear trajectory data */
  clearTrajectory: () => void
  /** Enable/disable auto-simulation */
  setAutoSimulateEnabled: (enabled: boolean) => void
  autoSimulateEnabled: boolean
}

/**
 * Check if a document has a crank joint (required for simulation).
 * Supports both hypergraph and legacy formats.
 */
function hasCrankJoint(doc: any): boolean {
  // Hypergraph format: check nodes for role === 'crank'
  if (doc.linkage?.nodes) {
    return Object.values(doc.linkage.nodes).some(
      (node: any) => node.role === 'crank' || node.role === 'driven'
    )
  }
  // Legacy format: check joints for type === 'Crank'
  if (doc.pylinkage?.joints) {
    return doc.pylinkage.joints.some((j: any) => j.type === 'Crank')
  }
  return false
}

/**
 * Hook for managing linkage simulation.
 * Handles both manual simulation and auto-simulation with delay.
 *
 * The backend now supports both hypergraph (v2.0.0) and legacy formats,
 * so we send the document directly without conversion.
 */
export function useSimulation({
  linkageDoc,
  simulationSteps,
  autoSimulateDelayMs,
  autoSimulateEnabled: initialAutoSimulate,
  mechanismVersion,
  isDragging = false,
  onSimulationStart,
  onSimulationComplete,
  onSimulationError,
  showStatus
}: UseSimulationProps): UseSimulationReturn {
  const [isSimulating, setIsSimulating] = useState(false)
  const [trajectoryData, setTrajectoryData] = useState<TrajectoryData | null>(null)
  const [autoSimulateEnabled, setAutoSimulateEnabled] = useState(initialAutoSimulate)
  const autoSimulateTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  /** Run simulation against the backend. Use docOverride when provided (e.g. drag-end synced doc).
   * TRAJECTORY HOP FIX: When BuilderTab passes syncedDoc after a drag, we send that exact doc so the
   * backend returns trajectory[0] = that state; otherwise we'd get wrong frame 0 and a cyclic shift. */
  const runSimulation = useCallback(async (docOverride?: any) => {
    const docToSend = docOverride != null ? docOverride : linkageDoc
    if (!hasCrankJoint(docToSend)) {
      console.warn('[Simulation] Cannot simulate: no Crank joint defined in document')
      showStatus?.('Cannot simulate: no Crank joint defined', 'warning', 2000)
      return
    }

    try {
      setIsSimulating(true)
      onSimulationStart?.()
      showStatus?.(`Simulating ${simulationSteps} steps...`, 'action')

      // Send document directly - backend handles both formats
      const response = await fetch('/api/compute-pylink-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...docToSend,
          n_steps: simulationSteps
        })
      })

      if (!response.ok) {
        console.error('[Simulation] HTTP error:', response.status, response.statusText)
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const result = await response.json()

      if (result.status === 'success') {
        const data: TrajectoryData = {
          trajectories: result.trajectories,
          nSteps: result.n_steps,
          jointTypes: result.joint_types || {}
        }
        setTrajectoryData(data)
        onSimulationComplete?.(data)
        showStatus?.(`Simulation complete: ${result.n_steps} steps`, 'success', 1500)
        if (result.trajectory_did_not_close) {
          showStatus?.('Trajectory did not close after one cycle.', 'error', 3000)
        }
      } else {
        const errorMsg = result.message || 'Simulation failed'
        console.error('[Simulation] Backend returned error:', errorMsg, result)
        setTrajectoryData(null)
        onSimulationError?.(errorMsg)
        showStatus?.(errorMsg, 'error', 3000)
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : 'Simulation error'
      console.error('[Simulation] Exception during trajectory computation:', error)
      setTrajectoryData(null)
      onSimulationError?.(errorMsg)
      showStatus?.(`Simulation error: ${errorMsg}`, 'error', 3000)
    } finally {
      setIsSimulating(false)
    }
  }, [linkageDoc, simulationSteps, showStatus, onSimulationStart, onSimulationComplete, onSimulationError])

  /** Clear trajectory data */
  const clearTrajectory = useCallback(() => {
    setTrajectoryData(null)
    showStatus?.('Trajectory cleared', 'info', 1500)
  }, [showStatus])

  // Auto-simulation: run after autoSimulateDelayMs from last dependency change (linkageDoc / mechanismVersion).
  // During drag, linkageDoc is updated by moveJoint/bake so we can run sim with current doc and update
  // trajectories while the user drags (unless "disable continuous simulation" is on).
  useEffect(() => {
    if (!autoSimulateEnabled) return

    if (!hasCrankJoint(linkageDoc)) return

    if (autoSimulateTimerRef.current) {
      clearTimeout(autoSimulateTimerRef.current)
    }

    autoSimulateTimerRef.current = setTimeout(() => {
      runSimulation()
    }, autoSimulateDelayMs)

    return () => {
      if (autoSimulateTimerRef.current) {
        clearTimeout(autoSimulateTimerRef.current)
      }
    }
  }, [mechanismVersion, autoSimulateEnabled, linkageDoc, autoSimulateDelayMs, isDragging, runSimulation])

  return {
    isSimulating,
    trajectoryData,
    runSimulation,
    clearTrajectory,
    setAutoSimulateEnabled,
    autoSimulateEnabled
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION CONTROLS COMPONENT (Optional - for standalone use)
// ═══════════════════════════════════════════════════════════════════════════════

import { Box, IconButton, Tooltip, Typography, Slider } from '@mui/material'

interface AnimationControlsProps {
  animationState: AnimationState
  onPlay: () => void
  onPause: () => void
  onStop: () => void
  onFrameChange: (frame: number) => void
  onSpeedChange: (fps: number) => void
  disabled?: boolean
  compact?: boolean
}

/**
 * Standalone animation controls component.
 * Can be used in toolbars or as a floating control panel.
 */
export const AnimationControls: React.FC<AnimationControlsProps> = ({
  animationState,
  onPlay,
  onPause,
  onStop,
  onFrameChange,
  onSpeedChange,
  disabled = false,
  compact = false
}) => {
  const { isAnimating, currentFrame, totalFrames, playbackFps } = animationState

  return (
    <Box sx={{
      display: 'flex',
      flexDirection: compact ? 'row' : 'column',
      alignItems: 'center',
      gap: compact ? 1 : 0.5,
      p: compact ? 0 : 1
    }}>
      {/* Play/Pause Button */}
      <Box sx={{ display: 'flex', gap: 0.5 }}>
        <Tooltip title={isAnimating ? 'Pause' : 'Play Animation'}>
          <span>
            <IconButton
              size="small"
              onClick={isAnimating ? onPause : onPlay}
              disabled={disabled || totalFrames === 0}
              sx={{
                bgcolor: isAnimating ? 'warning.main' : 'success.main',
                color: '#fff',
                '&:hover': { bgcolor: isAnimating ? 'warning.dark' : 'success.dark' },
                '&.Mui-disabled': { bgcolor: 'grey.300' }
              }}
            >
              {isAnimating ? '⏸' : '▶'}
            </IconButton>
          </span>
        </Tooltip>

        <Tooltip title="Stop & Reset">
          <span>
            <IconButton
              size="small"
              onClick={onStop}
              disabled={disabled || totalFrames === 0}
              sx={{ bgcolor: 'grey.200', '&:hover': { bgcolor: 'grey.300' } }}
            >
              ⏹
            </IconButton>
          </span>
        </Tooltip>
      </Box>

      {/* Frame Counter */}
      <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 60, textAlign: 'center' }}>
        {currentFrame + 1} / {totalFrames || '-'}
      </Typography>

      {/* Timeline Slider (only if not compact) */}
      {!compact && totalFrames > 0 && (
        <Slider
          size="small"
          value={currentFrame}
          min={0}
          max={Math.max(0, totalFrames - 1)}
          onChange={(_, value) => onFrameChange(value as number)}
          disabled={disabled}
          sx={{ width: '100%', mt: 0.5 }}
        />
      )}

      {/* Speed Control (only if not compact) - presets */}
      {!compact && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
            Speed:
          </Typography>
          {[5, 10, 20].map(fps => (
            <Box
              key={fps}
              onClick={() => !disabled && onSpeedChange(fps)}
              sx={{
                px: 0.75,
                py: 0.25,
                borderRadius: 1,
                fontSize: '0.65rem',
                cursor: disabled ? 'default' : 'pointer',
                bgcolor: playbackFps === fps ? 'primary.main' : 'grey.100',
                color: playbackFps === fps ? '#fff' : 'text.secondary',
                '&:hover': disabled ? {} : { bgcolor: playbackFps === fps ? 'primary.dark' : 'grey.200' }
              }}
            >
              {fps}
            </Box>
          ))}
        </Box>
      )}
    </Box>
  )
}


// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Interpolate between two positions for smoother animation.
 * Can be used for sub-frame interpolation if needed.
 */
export function interpolatePositions(
  pos1: [number, number],
  pos2: [number, number],
  t: number  // 0 to 1
): [number, number] {
  return [
    pos1[0] + (pos2[0] - pos1[0]) * t,
    pos1[1] + (pos2[1] - pos1[1]) * t
  ]
}

/**
 * Get the duration of a full animation cycle in milliseconds.
 * playbackFps = simulation steps per second (internal).
 */
export function getAnimationDuration(totalFrames: number, playbackFps: number): number {
  return playbackFps > 0 ? (totalFrames / playbackFps) * 1000 : 0
}

/**
 * Check if a mechanism is valid for simulation (has at least one crank).
 */
export function canSimulate(joints: Array<{ type: string }>): boolean {
  return joints.some(j => j.type === 'Crank')
}
