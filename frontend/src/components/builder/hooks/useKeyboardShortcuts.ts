/**
 * Keyboard Shortcuts Hook
 *
 * Handles keyboard event listeners for the Builder component.
 * Extracted from BuilderTab.tsx for better organization and reusability.
 */

import { useEffect, useCallback } from 'react'
import { ToolMode, ToolInfo as Tool } from '../../BuilderTools'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface KeyboardShortcutsConfig {
  /** Available tools with their shortcuts */
  tools: Tool[]

  /** Current state flags */
  state: {
    isLinkDrawing: boolean
    isGroupSelecting: boolean
    isPolygonDrawing: boolean
    isMeasuring: boolean
    isPathDrawing: boolean
    isMerging: boolean
    isAnimating: boolean
    hasSelectedItems: boolean
    hasTrajectory: boolean
    canSimulate: boolean
  }

  /** Callbacks for actions */
  actions: {
    cancelAction: () => void
    completePathDrawing: () => void
    playAnimation: () => void
    pauseAnimation: () => void
    runSimulation: () => Promise<void>
    handleDeleteSelected: () => void
    setToolMode: (mode: ToolMode) => void
    showStatus: (message: string, type: 'info' | 'success' | 'error' | 'action', duration?: number) => void
  }

  /** State setters for resetting modes when switching tools */
  resetters: {
    resetLinkCreation: () => void
    resetGroupSelection: () => void
    resetPolygonDrawing: () => void
    resetMeasureState: () => void
    resetMergeState: () => void
    resetPathDrawing: () => void
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Hook to handle keyboard shortcuts for the Builder
 *
 * Shortcuts:
 * - Escape: Cancel current action
 * - Enter: Complete path drawing (when in path draw mode)
 * - Space: Play/pause animation, or run simulation if no trajectory
 * - Delete/Backspace/X: Delete selected items
 * - Tool shortcuts: S (select), L (link), G (group), P (polygon), M (measure), etc.
 */
export function useKeyboardShortcuts(config: KeyboardShortcutsConfig) {
  const { tools, state, actions, resetters } = config

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    // Ignore keyboard events when typing in input fields
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
      return
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ESCAPE - Cancel current action
    // ─────────────────────────────────────────────────────────────────────────
    if (event.key === 'Escape') {
      actions.cancelAction()
      return
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ENTER - Complete path drawing
    // ─────────────────────────────────────────────────────────────────────────
    if (event.key === 'Enter' && state.isPathDrawing) {
      actions.completePathDrawing()
      event.preventDefault()
      return
    }

    // ─────────────────────────────────────────────────────────────────────────
    // SPACE - Play/pause animation or run simulation
    // ─────────────────────────────────────────────────────────────────────────
    if (event.key === ' ' || event.code === 'Space') {
      event.preventDefault()  // Prevent page scroll

      if (state.isAnimating) {
        actions.pauseAnimation()
        actions.showStatus('Animation paused', 'info', 1500)
      } else if (state.hasTrajectory) {
        actions.playAnimation()
        actions.showStatus('Animation playing', 'info', 1500)
      } else if (state.canSimulate) {
        // No trajectory yet, run simulation first then play
        actions.runSimulation().then(() => {
          setTimeout(() => actions.playAnimation(), 100)
        })
        actions.showStatus('Running simulation...', 'action')
      }
      return
    }

    // ─────────────────────────────────────────────────────────────────────────
    // DELETE/BACKSPACE/X - Delete selected items
    // ─────────────────────────────────────────────────────────────────────────
    if ((event.key === 'Delete' || event.key === 'Backspace' || event.key === 'x' || event.key === 'X') && state.hasSelectedItems) {
      // X is also a tool shortcut, only delete if items are selected
      if (event.key === 'x' || event.key === 'X') {
        if (state.hasSelectedItems) {
          actions.handleDeleteSelected()
          event.preventDefault()
          return
        }
        // Otherwise fall through to tool selection
      } else {
        actions.handleDeleteSelected()
        event.preventDefault()
        return
      }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TOOL SHORTCUTS
    // ─────────────────────────────────────────────────────────────────────────
    const key = event.key.toUpperCase()
    const tool = tools.find(t => t.shortcut === key)

    if (tool) {
      // Cancel ongoing actions if switching tools
      if (state.isLinkDrawing && tool.id !== 'draw_link') {
        resetters.resetLinkCreation()
      }
      if (state.isGroupSelecting && tool.id !== 'group_select') {
        resetters.resetGroupSelection()
      }
      if (state.isPolygonDrawing && tool.id !== 'draw_polygon') {
        resetters.resetPolygonDrawing()
      }
      if (state.isMeasuring && tool.id !== 'measure') {
        resetters.resetMeasureState()
      }
      if (state.isMerging && tool.id !== 'merge') {
        resetters.resetMergeState()
      }
      if (state.isPathDrawing && tool.id !== 'draw_path') {
        resetters.resetPathDrawing()
      }

      actions.setToolMode(tool.id)

      // Show appropriate message for special modes
      if (tool.id === 'merge') {
        actions.showStatus('Select a link or a polygon to begin merge', 'action')
      } else if (tool.id === 'draw_path') {
        actions.showStatus('Click to start drawing target path', 'action')
      } else {
        actions.showStatus(`${tool.label} mode`, 'info', 1500)
      }

      event.preventDefault()
    }
  }, [tools, state, actions, resetters])

  // Set up event listener
  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])
}

export default useKeyboardShortcuts
