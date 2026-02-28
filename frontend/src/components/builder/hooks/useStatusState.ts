/**
 * useStatusState
 *
 * Consolidates status message state for BuilderTab (Phase 6.3 — group 7).
 * Provides statusMessage, showStatus (with timestamp and optional duration), clearStatus,
 * and statusHistory (last 10 error/warning messages) for the issue history popout.
 */

import { useState, useCallback } from 'react'
import type { StatusMessage, StatusType } from '../../BuilderTools'

const HISTORY_MAX = 10
const DEFAULT_DURATION_MS: Record<StatusType, number> = {
  error: 8000,
  warning: 6000,
  success: 3000,
  action: 4000,
  info: 2500
}

export interface UseStatusStateReturn {
  statusMessage: StatusMessage | null
  showStatus: (text: string, type?: StatusType, duration?: number) => void
  clearStatus: () => void
  /** Last 10 error/warning messages (newest first) for issue history popout */
  statusHistory: StatusMessage[]
  clearStatusHistory: () => void
}

export function useStatusState(): UseStatusStateReturn {
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null)
  const [statusHistory, setStatusHistory] = useState<StatusMessage[]>([])

  const showStatus = useCallback((text: string, type: StatusType = 'info', duration?: number) => {
    const msg: StatusMessage = { text, type, timestamp: Date.now() }
    setStatusMessage(msg)

    if (type === 'error' || type === 'warning') {
      setStatusHistory(prev => [msg, ...prev].slice(0, HISTORY_MAX))
    }

    const durationMs = duration ?? DEFAULT_DURATION_MS[type]
    if (durationMs > 0 && type !== 'action') {
      setTimeout(() => {
        setStatusMessage(prev => {
          if (prev && prev.text === text) return null
          return prev
        })
      }, durationMs)
    }
  }, [])

  const clearStatus = useCallback(() => {
    setStatusMessage(null)
  }, [])

  const clearStatusHistory = useCallback(() => {
    setStatusHistory([])
  }, [])

  return {
    statusMessage,
    showStatus,
    clearStatus,
    statusHistory,
    clearStatusHistory
  }
}
