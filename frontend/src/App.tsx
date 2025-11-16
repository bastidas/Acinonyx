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
import GraphBuilderTab from './components/GraphBuilderTab'
import StatusAboutTab from './components/StatusAboutTab'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#ff8c00', // Orange like a cheetah
    },
    secondary: {
      main: '#2e2e2e', // Dark like cheetah spots
    },
  },
})

function App() {
  const [currentTab, setCurrentTab] = useState(0)

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue)
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 0 }}>
        <Box sx={{ textAlign: 'center', mb: 0 }}>
          <img 
            src={cheetahGif} 
            alt="Acinonyx Cheetah" 
            style={{ 
              maxWidth: '200px', 
              height: 'auto', 
              marginBottom: '-60px',
              borderRadius: '0px'
            }} 
          />
          <Typography variant="h5" component="h1" gutterBottom>
            Acinonyx - Mechanical Linkage Simulation
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Fast and agile like a cheetah
          </Typography>
        </Box>

        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={currentTab} onChange={handleTabChange} centered>
            <Tab label="Graph Builder" />
            <Tab label="Status & About" />
          </Tabs>
        </Box>

        {currentTab === 0 && <GraphBuilderTab />}
        {currentTab === 1 && <StatusAboutTab />}
      </Container>
    </ThemeProvider>
  )
}

export default App
