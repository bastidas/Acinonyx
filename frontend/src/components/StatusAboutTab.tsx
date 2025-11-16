import React, { useState } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress
} from '@mui/material'
import { StatusResponse } from '../response_models'

const StatusAboutTab: React.FC = () => {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const callStatusEndpoint = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/status')
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const data = await response.json()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Status & About
      </Typography>
      
      <Card sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            Backend Status
          </Typography>
          
          <Box sx={{ mb: 2 }}>
            <Button 
              variant="contained" 
              onClick={callStatusEndpoint}
              disabled={loading}
              sx={{ mr: 2 }}
            >
              {loading ? <CircularProgress size={20} /> : 'Check Status'}
            </Button>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Error: {error}
            </Alert>
          )}
          
          {status && (
            <Alert severity="success">
              <Typography variant="body1">
                <strong>Status:</strong> {status.status}
              </Typography>
              {status.message && (
                <Typography variant="body2">
                  <strong>Message:</strong> {status.message}
                </Typography>
              )}
            </Alert>
          )}
        </CardContent>
      </Card>

      <Card sx={{ mt: 4 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            About Acinonyx
          </Typography>
          <Typography variant="body1" paragraph>
            Acinonyx is a high-speed mechanical linkage simulation platform, designed to be as fast and agile as a cheetah.
          </Typography>
          <Typography variant="body1" paragraph>
            Features:
          </Typography>
          <Typography variant="body2" component="ul" sx={{ pl: 2 }}>
            <li>High-performance mechanical linkage simulation</li>
            <li>Interactive graph-based system design</li>
            <li>Real-time visualization and animation</li>
            <li>FastAPI backend for efficient processing</li>
          </Typography>
        </CardContent>
      </Card>
      
      <Typography variant="body2" align="center" sx={{ mt: 4, color: 'text.secondary' }}>
        Acinonyx connects to the FastAPI backend to deliver high-speed mechanical linkage simulation capabilities.
      </Typography>
    </Box>
  )
}

export default StatusAboutTab