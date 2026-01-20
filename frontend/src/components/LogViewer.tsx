/**
 * LogViewer Component
 *
 * A modal overlay that displays the backend.log file content.
 * Triggered by pressing the `~` (backtick/tilde) key.
 */

import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Box,
  IconButton,
  Typography,
  CircularProgress,
  Button,
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import RefreshIcon from '@mui/icons-material/Refresh'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import VerticalAlignBottomIcon from '@mui/icons-material/VerticalAlignBottom'

interface LogViewerProps {
  open: boolean
  onClose: () => void
}

const LogViewer: React.FC<LogViewerProps> = ({
  open,
  onClose,
}) => {
  const [logContent, setLogContent] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [totalLines, setTotalLines] = useState(0)
  const [autoScroll, setAutoScroll] = useState(true)
  const contentRef = useRef<HTMLPreElement>(null)

  const fetchLogs = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch('/api/logs/backend?lines=1000')
      const data = await response.json()

      if (data.status === 'success') {
        setLogContent(data.content || '(Empty log)')
        setTotalLines(data.total_lines || 0)
      } else {
        setError(data.message || 'Failed to fetch logs')
      }
    } catch (err) {
      setError(`Connection error: ${err instanceof Error ? err.message : 'Unknown error'}`)
    } finally {
      setLoading(false)
    }
  }, [])

  const clearLogs = useCallback(async () => {
    try {
      const response = await fetch('/api/logs/backend', {
        method: 'DELETE',
      })
      const data = await response.json()

      if (data.status === 'success') {
        setLogContent('(Log cleared)')
        setTotalLines(0)
      } else {
        setError(data.message || 'Failed to clear logs')
      }
    } catch (err) {
      setError(`Connection error: ${err instanceof Error ? err.message : 'Unknown error'}`)
    }
  }, [])

  // Fetch logs when opened
  useEffect(() => {
    if (open) {
      fetchLogs()
    }
  }, [open, fetchLogs])

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [logContent, autoScroll])

  // Handle Escape key to close
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) {
        onClose()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, onClose])

  if (!open) return null

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        zIndex: 9999,
        display: 'flex',
        flexDirection: 'column',
        p: 2,
      }}
      onClick={(e) => {
        // Close if clicking the backdrop
        if (e.target === e.currentTarget) {
          onClose()
        }
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          mb: 1,
          pb: 1,
          borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography
            variant="h6"
            sx={{
              color: '#4fc3f7',
              fontFamily: 'monospace',
              fontWeight: 600,
              letterSpacing: '0.05em',
            }}
          >
            📋 backend.log
          </Typography>
          <Typography
            variant="body2"
            sx={{
              color: 'rgba(255, 255, 255, 0.5)',
              fontFamily: 'monospace',
            }}
          >
            {totalLines} lines total
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Button
            size="small"
            startIcon={<VerticalAlignBottomIcon />}
            onClick={() => {
              setAutoScroll(!autoScroll)
              if (!autoScroll && contentRef.current) {
                contentRef.current.scrollTop = contentRef.current.scrollHeight
              }
            }}
            sx={{
              color: autoScroll ? '#4caf50' : 'rgba(255, 255, 255, 0.5)',
              textTransform: 'none',
              fontFamily: 'monospace',
              fontSize: '0.75rem',
            }}
          >
            Auto-scroll {autoScroll ? 'ON' : 'OFF'}
          </Button>

          <IconButton
            size="small"
            onClick={fetchLogs}
            disabled={loading}
            sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
            title="Refresh logs"
          >
            <RefreshIcon fontSize="small" />
          </IconButton>

          <IconButton
            size="small"
            onClick={clearLogs}
            sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
            title="Clear logs"
          >
            <DeleteOutlineIcon fontSize="small" />
          </IconButton>

          <IconButton
            size="small"
            onClick={onClose}
            sx={{ color: 'rgba(255, 255, 255, 0.7)' }}
            title="Close (Esc)"
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* Content */}
      <Box
        sx={{
          flex: 1,
          overflow: 'hidden',
          borderRadius: 1,
          backgroundColor: '#0d1117',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        }}
      >
        {loading ? (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
            }}
          >
            <CircularProgress size={40} sx={{ color: '#4fc3f7' }} />
          </Box>
        ) : error ? (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              flexDirection: 'column',
              gap: 2,
            }}
          >
            <Typography sx={{ color: '#f44336', fontFamily: 'monospace' }}>
              ⚠️ {error}
            </Typography>
            <Button
              variant="outlined"
              size="small"
              onClick={fetchLogs}
              sx={{
                color: '#4fc3f7',
                borderColor: '#4fc3f7',
                textTransform: 'none',
              }}
            >
              Retry
            </Button>
          </Box>
        ) : (
          <pre
            ref={contentRef}
            style={{
              margin: 0,
              padding: '12px 16px',
              height: '100%',
              overflow: 'auto',
              fontFamily: '"JetBrains Mono", "Fira Code", "SF Mono", Consolas, monospace',
              fontSize: '12px',
              lineHeight: 1.5,
              color: '#c9d1d9',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {logContent || '(No log content)'}
          </pre>
        )}
      </Box>

      {/* Footer hint */}
      <Box
        sx={{
          mt: 1,
          display: 'flex',
          justifyContent: 'center',
        }}
      >
        <Typography
          variant="caption"
          sx={{
            color: 'rgba(255, 255, 255, 0.4)',
            fontFamily: 'monospace',
          }}
        >
          Press <kbd style={{
            padding: '2px 6px',
            backgroundColor: 'rgba(255,255,255,0.1)',
            borderRadius: '3px',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>~</kbd> or <kbd style={{
            padding: '2px 6px',
            backgroundColor: 'rgba(255,255,255,0.1)',
            borderRadius: '3px',
            border: '1px solid rgba(255,255,255,0.2)'
          }}>Esc</kbd> to close
        </Typography>
      </Box>
    </Box>
  )
}

export default LogViewer
