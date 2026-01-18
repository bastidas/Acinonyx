import React, { useState } from 'react'
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
import cheetahGif from './assets/cheetah_run.gif'
import PylinkBuilderTab from './components/PylinkBuilderTab'
import GraphBuilderTab from './components/GraphBuilderTab'
import ForceGraphViewTab from './components/ForceGraphViewTab'
import StatusAboutTab from './components/StatusAboutTab'
import { muiThemeConfig } from './theme'
import './theme.css'

const theme = createTheme(muiThemeConfig)

function App() {
  const [currentTab, setCurrentTab] = useState(0)

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
        <Box sx={{ textAlign: 'center', mb: 0, mt: -1 }}>
          <img 
            src={cheetahGif} 
            alt="Acinonyx Cheetah" 
            style={{ 
              maxWidth: '140px', 
              height: 'auto', 
              marginBottom: '-45px',
              borderRadius: '0px'
            }} 
          />
          <Typography variant="h6" component="h1" sx={{ mb: -0.5, fontWeight: 600 }}>
            Acinonyx
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
          Mechanical Linkage Simulation
          </Typography>
        </Box>

        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 1 }}>
          <Tabs value={currentTab} onChange={handleTabChange} centered>
            <Tab label="Pylink Builder" />
            <Tab label="Graph Builder" />
            <Tab label="Graph View" />
            <Tab label="Status & About" />
          </Tabs>
        </Box>

        {currentTab === 0 && <PylinkBuilderTab />}
        {currentTab === 1 && <GraphBuilderTab />}
        {currentTab === 2 && <ForceGraphViewTab />}
        {currentTab === 3 && <StatusAboutTab />}
      </Container>
    </ThemeProvider>
  )
}

export default App
