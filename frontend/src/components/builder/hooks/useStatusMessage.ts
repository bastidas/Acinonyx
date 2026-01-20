/**
 * Status Message Hook
 *
 * Manages the status message display system for the Builder component.
 * Provides show/clear functionality with auto-dismiss timers.
 */

import { useState, useCallback, useRef } from 'react'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export type StatusType = 'info' | 'success' | 'error' | 'action'

export interface StatusMessage {
  text: string
  type: StatusType
}

export interface UseStatusMessageReturn {
  /** Current status message (or null if none) */
  statusMessage: StatusMessage | null
  /** Show a status message */
  showStatus: (text: string, type: StatusType, duration?: number) => void
  /** Clear the current status message */
  clearStatus: () => void
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Hook to manage status messages with auto-dismiss
 *
 * @example
 * const { statusMessage, showStatus, clearStatus } = useStatusMessage()
 *
 * // Show a temporary success message
 * showStatus('Operation completed', 'success', 2000)
 *
 * // Show a persistent action message
 * showStatus('Drag to select', 'action')
 *
 * // Clear manually
 * clearStatus()
 */
export function useStatusMessage(): UseStatusMessageReturn {
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null)
  const timerRef = useRef<number | null>(null)

  const clearStatus = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }
    setStatusMessage(null)
  }, [])

  const showStatus = useCallback((text: string, type: StatusType, duration?: number) => {
    // Clear any existing timer
    if (timerRef.current) {
      clearTimeout(timerRef.current)
      timerRef.current = null
    }

    // Set the new message
    setStatusMessage({ text, type })

    // Set auto-dismiss timer if duration specified
    if (duration) {
      timerRef.current = window.setTimeout(() => {
        setStatusMessage(null)
        timerRef.current = null
      }, duration)
    }
  }, [])

  return {
    statusMessage,
    showStatus,
    clearStatus
  }
}

export default useStatusMessage
