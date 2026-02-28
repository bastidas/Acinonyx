/**
 * useDisplayFrame
 *
 * Single owner for "display frame" and "edit mode" in the builder.
 * Explicit state machine: editing | paused | playing. Rule: in editing, display frame = 0.
 *
 * When editing (e.g. dragging a node or group), display frame is forced to 0
 * so mechanism and step label stay at 1/N. When not editing, display frame
 * follows animation currentFrame.
 *
 * TRAJECTORY HOP FIX: enterEditMode sets frame 0 so the user always edits at 1/N; drag-end sync
 * in BuilderTab uses this convention to build the synced doc from trajectory[0]. Do not change
 * to "current frame" on drag start or the doc sync and backend would be out of sync.
 *
 * Both single-node drag and group drag call enterEditMode on drag start and
 * rely on the parent to call exitEditMode when any drag ends (e.g. via effect).
 *
 * INVARIANT (first paint 1/N): The consumer (BuilderTab) computes effectiveDisplayFrame as
 * displayFrameOverrideRef.current ?? displayFrame. For the first paint after drag start to
 * show 1/N (frame 0), enterEditMode must run synchronously before the next paint so the ref
 * is already 0 when that paint reads it. The canvas event handlers therefore call
 * flushSync(enterEditMode) on drag start—do not call enterEditMode asynchronously or the
 * first frame could still show N/N until the next re-render.
 */

import { useState, useCallback, useRef } from 'react'
import type { MutableRefObject } from 'react'

/** Explicit mode for builder animation: editing (dragging) forces frame 0; otherwise playback frame. */
export type BuilderAnimationMode = 'editing' | 'paused' | 'playing'

export interface UseDisplayFrameParams {
  /** Current frame from animation state (playback). */
  animationCurrentFrame: number
  /** Set animation to a frame (used when entering edit mode to sync to 0). */
  setAnimationFrame: (frame: number) => void
  /** Pause playback so the animation loop does not overwrite frame 0 during edit. */
  pauseAnimation: () => void
  /** True when animation is playing (for mode derivation). */
  isAnimating: boolean
}

export interface UseDisplayFrameReturn {
  /** The frame to use for mechanism pose and step label (0 when editing, else animation currentFrame). */
  displayFrame: number
  /**
   * Ref set to 0 at start of enterEditMode() and null in exitEditMode().
   * Used to compute effective display frame synchronously so first paint after drag start shows 1/N, not N/N.
   * Caller must invoke enterEditMode synchronously (e.g. via flushSync) on drag start so this ref is 0
   * before the next paint; otherwise the first paint may still use the previous frame.
   */
  displayFrameOverrideRef: MutableRefObject<number | null>
  /** True from drag start until drag end (single-node or group). */
  isEditMode: boolean
  /** Explicit mode: editing => display frame 0; playing/paused => display frame = animation currentFrame. */
  mode: BuilderAnimationMode
  /** Call when starting a drag (single-node or group). Pauses animation, sets frame 0, enters edit mode. */
  enterEditMode: () => void
  /** Call when drag ends (or when anyDragging becomes false). Exits edit mode so display frame follows animation again. */
  exitEditMode: () => void
}

export function useDisplayFrame({
  animationCurrentFrame,
  setAnimationFrame,
  pauseAnimation,
  isAnimating
}: UseDisplayFrameParams): UseDisplayFrameReturn {
  const [isEditMode, setIsEditMode] = useState(false)
  const displayFrameOverrideRef = useRef<number | null>(null)

  const enterEditMode = useCallback(() => {
    displayFrameOverrideRef.current = 0
    pauseAnimation()
    setAnimationFrame(0)
    setIsEditMode(true)
  }, [pauseAnimation, setAnimationFrame])

  const exitEditMode = useCallback(() => {
    displayFrameOverrideRef.current = null
    setIsEditMode(false)
  }, [])

  // Single rule: in editing, display frame = 0; otherwise use playback frame.
  const displayFrame = isEditMode ? 0 : animationCurrentFrame
  const mode: BuilderAnimationMode = isEditMode ? 'editing' : isAnimating ? 'playing' : 'paused'

  return {
    displayFrame,
    displayFrameOverrideRef,
    isEditMode,
    mode,
    enterEditMode,
    exitEditMode
  }
}
