import React, { useState, useEffect, useCallback } from 'react'
import {
  Box,
  Container,
  Typography,
  ThemeProvider,
  createTheme,
  CssBaseline,
  Tabs,
  Tab
} from '@mui/material'
import acinonyxLogo from './assets/acinonyx_logo.png'
import BuilderTab from './components/BuilderTab'
import StatusAboutTab from './components/StatusAboutTab'
import LogViewer from './components/LogViewer'
import { muiThemeConfig } from './theme'
import './theme.css'

const theme = createTheme(muiThemeConfig)

function App() {
  const [currentTab, setCurrentTab] = useState(0)
  const [logViewerOpen, setLogViewerOpen] = useState(false)

  // Global keyboard shortcut for ~ to open log viewer
  const handleGlobalKeyDown = useCallback((event: KeyboardEvent) => {
    // Ignore when typing in input fields
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
      return
    }

    // Toggle log viewer with ~ (backtick/tilde key)
    if (event.key === '`' || event.key === '~') {
      event.preventDefault()
      setLogViewerOpen(prev => !prev)
    }
  }, [])

  useEffect(() => {
    document.addEventListener('keydown', handleGlobalKeyDown)
    return () => document.removeEventListener('keydown', handleGlobalKeyDown)
  }, [handleGlobalKeyDown])

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth={false} sx={{
        py: 0,
        px: currentTab === 0 ? 0 : 2,
        maxWidth: currentTab === 0 ? '100%' : '1600px',
        backgroundColor: 'var(--color-surface)',
        minHeight: '100vh',
        transition: 'background-color 0.25s ease'
      }}>
        {/* Compact Header - Logo, Title, and Subtitle on single line */}
        <Box
          className="app-header"
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 2,
            py: 1,
            px: 2,
            backgroundColor: 'var(--color-bg-tan)',
            borderBottom: '1px solid var(--color-border-light)',
            transition: 'all 0.25s ease'
          }}
        >
          {/* Logo wrapper - clips to middle third and floats above header */}
          <Box
            sx={{
              position: 'relative',
              width: '140px',
              height: '46px', // Keep header compact
              overflow: 'hidden',
              marginTop: '-10px', // Float above header (10px from top)
              marginBottom: '-12px',
              flexShrink: 0,
            }}
          >
            <img
              src={acinonyxLogo}
              alt="Acinonyx"
              style={{
                width: '140px',
                height: '140px', // Full size for the content
                objectFit: 'cover',
                objectPosition: 'center 45%', // Show middle-ish third (cheetah body)
                position: 'absolute',
                top: '-47px', // Pull up to show middle third
                left: '0',
              }}
            />
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
            <Typography
              variant="h6"
              component="h1"
              className="app-title"
              sx={{
                fontWeight: 700,
                fontSize: '1.2rem',
                color: 'var(--color-text-primary)',
                letterSpacing: '0.5px',
                transition: 'color 0.25s ease'
              }}
            >
              Acinonyx
            </Typography>
            <Typography
              variant="body2"
              className="app-subtitle"
              sx={{
                color: 'var(--color-text-muted)',
                fontSize: '0.85rem',
                fontWeight: 400,
                transition: 'color 0.25s ease'
              }}
            >
              Mechanical Linkage Simulation
            </Typography>
          </Box>
        </Box>

        {/* Tabs */}
        <Box sx={{
          borderBottom: '1px solid var(--color-border-light)',
          backgroundColor: 'var(--color-surface)',
          transition: 'all 0.25s ease'
        }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            centered
            sx={{
              minHeight: 40,
              '& .MuiTab-root': {
                minHeight: 40,
                py: 1,
                fontSize: '0.8rem',
                fontWeight: 500,
                color: 'var(--color-text-secondary)',
                transition: 'color 0.2s ease',
                '&.Mui-selected': {
                  color: 'var(--color-primary)',
                  fontWeight: 600
                }
              },
              '& .MuiTabs-indicator': {
                backgroundColor: 'var(--color-primary)',
                height: 3,
                borderRadius: '3px 3px 0 0'
              }
            }}
          >
            <Tab label="Pylink Builder" />
            <Tab label="Status & About" />
          </Tabs>
        </Box>

        {currentTab === 0 && <BuilderTab />}
        {currentTab === 1 && <StatusAboutTab />}
      </Container>

      {/* Log Viewer - toggled with ~ key */}
      <LogViewer
        open={logViewerOpen}
        onClose={() => setLogViewerOpen(false)}
      />
    </ThemeProvider>
  )
}

export default App
