import { useState, useEffect, useCallback } from 'react'
import type { TrajectoryData, AnimationState } from '../AnimateSimulate'
import { useAnimation, useSimulation } from '../AnimateSimulate'
import type { LinkageDocument } from './types'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface AnimationControllerContext {
  // Simulation dependencies
  linkageDoc: LinkageDocument
  simulationSteps: number
  autoSimulateDelayMs: number
  mechanismVersion: number
  /** TRAJECTORY HOP FIX: When true, auto-sim skips scheduling so we never run with an inconsistent doc
   * (only dragged node updated; crank/others stale). After drag end this is false and the delay timer runs. */
  isDragging?: boolean
  onSimulationComplete: (data: TrajectoryData) => void
  showStatus: (text: string, type?: 'info' | 'success' | 'warning' | 'error' | 'action', duration?: number) => void

  // Animation dependencies
  onFrameChange: (frame: number) => void
}

export interface UseAnimationControllerReturn {
  // Animation state (from useAnimation hook)
  animationState: AnimationState

  // Simulation state (from useSimulation hook)
  isSimulating: boolean
  trajectoryData: TrajectoryData | null
  autoSimulateEnabled: boolean

  // Additional animation-related state
  animatedPositions: Record<string, [number, number]> | null
  stretchingLinks: string[]

  // Animation functions (from useAnimation hook)
  playAnimation: () => void
  pauseAnimation: () => void
  stopAnimation: () => void
  resetAnimation: () => void
  setAnimationFrame: (frame: number) => void
  setPlaybackFps: (fps: number) => void
  setLoop: (loop: boolean) => void
  setPlaybackDirection: (direction: 1 | -1) => void
  getAnimatedPositions: () => Record<string, [number, number]> | null

  // Simulation functions (from useSimulation hook)
  runSimulation: (docOverride?: any) => Promise<void>
  clearTrajectory: () => void
  setAutoSimulateEnabled: (enabled: boolean) => void

  // Setters (for additional state)
  setAnimatedPositions: React.Dispatch<React.SetStateAction<Record<string, [number, number]> | null>>
  setStretchingLinks: React.Dispatch<React.SetStateAction<string[]>>
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing animation and simulation state and functions
 * Step 2.2: Wraps useAnimation and useSimulation hooks
 *
 * Convention: When trajectory data exists, frame index 0 is the true first step of the
 * trajectory (trajectory[0]). Code that uses "base" or "first" step (e.g. optimization,
 * validation) should rely on frame index 0 for the initial configuration.
 */
export function useAnimationController(
  context: AnimationControllerContext
): UseAnimationControllerReturn {
  // Additional animation-related state (not managed by hooks)
  const [animatedPositions, setAnimatedPositions] = useState<Record<string, [number, number]> | null>(null)
  const [stretchingLinks, setStretchingLinks] = useState<string[]>([])

  // Simulation hook - handles trajectory computation and auto-simulate
  const {
    isSimulating,
    trajectoryData,
    runSimulation,
    clearTrajectory,
    setAutoSimulateEnabled,
    autoSimulateEnabled
  } = useSimulation({
    linkageDoc: context.linkageDoc,
    simulationSteps: context.simulationSteps,
    autoSimulateDelayMs: context.autoSimulateDelayMs,
    autoSimulateEnabled: true,  // Start with continuous simulation ON by default
    mechanismVersion: context.mechanismVersion,
    isDragging: context.isDragging,
    onSimulationComplete: context.onSimulationComplete,
    showStatus: context.showStatus
  })

  // Animation hook - handles playback of simulation frames
  const {
    animationState,
    play: playAnimation,
    pause: pauseAnimation,
    stop: stopAnimation,
    reset: resetAnimation,
    setFrame: setAnimationFrame,
    setPlaybackFps,
    setLoop,
    setPlaybackDirection,
    getAnimatedPositions
  } = useAnimation({
    trajectoryData,
    onFrameChange: context.onFrameChange
  })

  // Update animated positions when animation frame changes.
  // Frame 0 is the true first step of the trajectory (trajectory[0]) when trajectory data exists.
  useEffect(() => {
    if (animationState.isAnimating || animationState.currentFrame > 0) {
      // Update positions when animating OR when stepping through frames while paused
      const positions = getAnimatedPositions()
      setAnimatedPositions(positions)
    } else if (!animationState.isAnimating && animationState.currentFrame === 0) {
      // At frame 0: show trajectory[0] when we have trajectory data, else clear (fall back to document positions)
      const positions = trajectoryData ? getAnimatedPositions() : null
      setAnimatedPositions(positions)
    }
  }, [animationState.isAnimating, animationState.currentFrame, getAnimatedPositions, trajectoryData])

  return {
    // State
    animationState,
    isSimulating,
    trajectoryData,
    autoSimulateEnabled,
    animatedPositions,
    stretchingLinks,

    // Animation functions
    playAnimation,
    pauseAnimation,
    stopAnimation,
    resetAnimation,
    setAnimationFrame,
    setPlaybackFps,
    setLoop,
    setPlaybackDirection,
    getAnimatedPositions,

    // Simulation functions
    runSimulation,
    clearTrajectory,
    setAutoSimulateEnabled,

    // Setters (for additional state)
    setAnimatedPositions,
    setStretchingLinks,
  }
}
