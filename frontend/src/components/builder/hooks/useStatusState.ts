/**
 * useStatusState
 *
 * Consolidates status message state for BuilderTab (Phase 6.3 — group 7).
 * Provides statusMessage, showStatus (with timestamp and optional duration), and clearStatus.
 * Matches BuilderTools StatusMessage shape (text, type, timestamp).
 */

import { useState, useCallback } from 'react'
import type { StatusMessage, StatusType } from '../../BuilderTools'

export interface UseStatusStateReturn {
  statusMessage: StatusMessage | null
  showStatus: (text: string, type?: StatusType, duration?: number) => void
  clearStatus: () => void
}

export function useStatusState(): UseStatusStateReturn {
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null)

  const showStatus = useCallback((text: string, type: StatusType = 'info', duration?: number) => {
    setStatusMessage({ text, type, timestamp: Date.now() })

    if (duration && type !== 'action') {
      setTimeout(() => {
        setStatusMessage(prev => {
          if (prev && prev.text === text) return null
          return prev
        })
      }, duration)
    }
  }, [])

  const clearStatus = useCallback(() => {
    setStatusMessage(null)
  }, [])

  return {
    statusMessage,
    showStatus,
    clearStatus
  }
}
