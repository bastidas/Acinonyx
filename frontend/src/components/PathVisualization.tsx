import React, { useState, useRef, useEffect } from 'react'
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Slider,
  FormControlLabel,
  Switch,
  Grid
} from '@mui/material'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PauseIcon from '@mui/icons-material/Pause'
import StopIcon from '@mui/icons-material/Stop'
import { getCyclicColor, jointColors, ColorCycleType } from '../theme'

interface PathVisualizationProps {
  pathData: {
    bounds: {
      xmin: number
      xmax: number
      ymin: number
      ymax: number
    } | null
    links: Array<{
      name: string
      is_driven: boolean
      is_ground?: boolean  // Ground/frame link (dashed style)
      has_fixed: boolean
      has_constraint: boolean
      pos1: number[][]
      pos2: number[][]
    }>
    history_data: Array<Array<{
      link_name: string
      positions: number[][]
      colors: Array<{
        color: string
        alpha: number
      }>
    }>>
    n_iterations: number
  } | null
}

const PathVisualization: React.FC<PathVisualizationProps> = ({ pathData }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [showHistory, setShowHistory] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  // Slider position: 70 (left) = 500ms, 500 (right) = 70ms
  // Transform: actualSpeed = 570 - sliderPosition
  const [sliderPosition, setSliderPosition] = useState(440) // 570 - 440 = 130ms default
  const playbackSpeed = 570 - sliderPosition
  const animationRef = useRef<number>()

  useEffect(() => {
    if (pathData) {
      drawFrame(currentFrame)
    }
  }, [pathData, currentFrame, showHistory, showLabels])

  useEffect(() => {
    if (isPlaying && pathData) {
      animationRef.current = setInterval(() => {
        setCurrentFrame(prev => (prev + 1) % pathData.n_iterations)
      }, playbackSpeed)
    } else {
      if (animationRef.current) {
        clearInterval(animationRef.current)
      }
    }

    return () => {
      if (animationRef.current) {
        clearInterval(animationRef.current)
      }
    }
  }, [isPlaying, playbackSpeed, pathData])

  const drawFrame = (frame: number) => {
    if (!pathData || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx || !pathData.bounds) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const { bounds, links, history_data } = pathData
    const canvasWidth = canvas.width
    const canvasHeight = canvas.height

    // Transform coordinates from data space to canvas space
    const transformX = (x: number) => 
      ((x - bounds.xmin) / (bounds.xmax - bounds.xmin)) * canvasWidth
    const transformY = (y: number) => 
      canvasHeight - ((y - bounds.ymin) / (bounds.ymax - bounds.ymin)) * canvasHeight

    // Draw historical trails if enabled (30% bigger: 3 -> 4)
    if (showHistory && history_data[frame]) {
      for (const historyItem of history_data[frame]) {
        for (let i = 0; i < historyItem.positions.length; i++) {
          const pos = historyItem.positions[i]
          const colorInfo = historyItem.colors[i]
          
          ctx.globalAlpha = colorInfo.alpha
          ctx.fillStyle = colorInfo.color
          ctx.beginPath()
          ctx.arc(transformX(pos[0]), transformY(pos[1]), 4, 0, 2 * Math.PI)
          ctx.fill()
        }
      }
    }

    // Reset alpha for main elements
    ctx.globalAlpha = 1.0

    // ─────────────────────────────────────────────────────────────────────────
    // FRAME COLOR - Cyclic color for current animation frame
    // ─────────────────────────────────────────────────────────────────────────
    // Uses a cyclic color gradient that returns to the starting color at the
    // end of the animation, creating smooth visual continuity for looping.
    //
    // Available options: 'rainbow', 'fire', 'glow'
    // ─────────────────────────────────────────────────────────────────────────
    const frameColorCycle: ColorCycleType = 'rainbow'
    const frameColor = getCyclicColor(frame, pathData.n_iterations, frameColorCycle)

    // First pass: Draw current frame links and nodes (30% bigger: lines 3->4, 2->3; nodes 4->5)
    const labelPositions: Array<{name: string, x: number, y: number}> = []
    
    for (const link of links) {
      if (frame < link.pos1.length && frame < link.pos2.length) {
        const pos1 = link.pos1[frame]
        const pos2 = link.pos2[frame]
        
        const x1 = transformX(pos1[0])
        const y1 = transformY(pos1[1])
        const x2 = transformX(pos2[0])
        const y2 = transformY(pos2[1])

        // Draw link line
        ctx.strokeStyle = link.is_ground ? jointColors.ground : frameColor  // Gray for ground
        ctx.lineWidth = link.is_driven ? 4 : (link.is_ground ? 2 : 3)
        
        // Use dashed line for ground links
        if (link.is_ground) {
          ctx.setLineDash([8, 4])  // 8px dash, 4px gap
        } else {
          ctx.setLineDash([])  // Solid line
        }
        
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
        
        // Reset line dash for other drawing
        ctx.setLineDash([])

        // Draw nodes (pos1 and pos2)
        ctx.fillStyle = frameColor
        
        // pos1 node
        ctx.beginPath()
        ctx.arc(x1, y1, 5, 0, 2 * Math.PI)
        ctx.fill()
        
        // pos2 node
        ctx.beginPath()
        ctx.arc(x2, y2, 5, 0, 2 * Math.PI)
        ctx.fill()

        // Store label position for second pass
        if (showLabels) {
          const midX = (x1 + x2) / 2
          const midY = (y1 + y2) / 2
          labelPositions.push({ name: link.name, x: midX, y: midY })
        }
      }
    }
    
    // Second pass: Draw labels on top of all lines
    if (showLabels) {
      ctx.fillStyle = '#333'
      ctx.font = '11px Arial'
      ctx.textAlign = 'center'
      for (const label of labelPositions) {
        ctx.fillText(label.name, label.x, label.y - 5)
      }
    }

    // Draw frame info
    ctx.fillStyle = '#333'
    ctx.font = '14px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(`Frame: ${frame + 1}/${pathData.n_iterations}`, 10, 20)
  }

  const handlePlay = () => {
    setIsPlaying(!isPlaying)
  }

  const handleStop = () => {
    setIsPlaying(false)
    setCurrentFrame(0)
  }

  const handleFrameChange = (_: Event, value: number | number[]) => {
    // Pause playback when manually scrubbing to prevent jumping
    setIsPlaying(false)
    if (typeof value === 'number') {
      setCurrentFrame(value)
    }
  }

  if (!pathData || !pathData.bounds) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6">Path Visualization</Typography>
          <Typography color="text.secondary">
            No path data available. Run compute graph to generate visualization data.
          </Typography>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Path Visualization
        </Typography>
        
        <Box sx={{ mb: 2 }}>
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            style={{
              border: '1px solid #ccc',
              borderRadius: '4px',
              backgroundColor: '#fff',
              display: 'block',
              margin: '0 auto',
              imageRendering: 'crisp-edges'
            }}
          />
        </Box>

        <Grid container spacing={2} alignItems="center">
          <Grid item>
            <Button
              variant="contained"
              onClick={handlePlay}
              startIcon={isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
          </Grid>
          
          <Grid item>
            <Button
              variant="outlined"
              onClick={handleStop}
              startIcon={<StopIcon />}
            >
              Reset
            </Button>
          </Grid>

          {/* Circular Frame Dial */}
          <Grid item>
            <Box sx={{ position: 'relative', width: 80, height: 80 }}>
              <svg width="80" height="80" viewBox="0 0 80 80">
                {/* Outer circle */}
                <circle cx="40" cy="40" r="35" fill="none" stroke="#ddd" strokeWidth="2" />
                
                {/* Tick marks for each frame */}
                {Array.from({ length: pathData.n_iterations }, (_, i) => {
                  const angle = (i / pathData.n_iterations) * 2 * Math.PI - Math.PI / 2
                  const innerR = 28
                  const outerR = 34
                  const x1 = 40 + innerR * Math.cos(angle)
                  const y1 = 40 + innerR * Math.sin(angle)
                  const x2 = 40 + outerR * Math.cos(angle)
                  const y2 = 40 + outerR * Math.sin(angle)
                  return (
                    <line
                      key={i}
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={i === currentFrame ? getCyclicColor(i, pathData.n_iterations, 'rainbow') : '#999'}
                      strokeWidth={i === currentFrame ? 3 : 1}
                    />
                  )
                })}
                
                {/* Pointer/hand */}
                {(() => {
                  const angle = (currentFrame / pathData.n_iterations) * 2 * Math.PI - Math.PI / 2
                  const pointerLength = 24
                  const x = 40 + pointerLength * Math.cos(angle)
                  const y = 40 + pointerLength * Math.sin(angle)
                  const dialColor = getCyclicColor(currentFrame, pathData.n_iterations, 'rainbow')
                  return (
                    <>
                      <line
                        x1="40"
                        y1="40"
                        x2={x}
                        y2={y}
                        stroke={dialColor}
                        strokeWidth="3"
                        strokeLinecap="round"
                      />
                      <circle cx="40" cy="40" r="4" fill={dialColor} />
                      <circle cx={x} cy={y} r="3" fill={dialColor} />
                    </>
                  )
                })()}
                
                {/* Frame number in center */}
                <text
                  x="40"
                  y="58"
                  textAnchor="middle"
                  fontSize="10"
                  fill="#666"
                >
                  {currentFrame + 1}/{pathData.n_iterations}
                </text>
              </svg>
            </Box>
          </Grid>

          <Grid item xs>
            <Typography gutterBottom>Frame ({currentFrame + 1}/{pathData.n_iterations})</Typography>
            <Slider
              value={currentFrame}
              onChange={handleFrameChange}
              min={0}
              max={pathData.n_iterations - 1}
              step={1}
              marks={Array.from({ length: pathData.n_iterations }, (_, i) => ({ value: i }))}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => v + 1}
            />
          </Grid>
        </Grid>

        <Grid container spacing={2} sx={{ mt: 1 }}>
          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={showHistory}
                  onChange={(e) => setShowHistory(e.target.checked)}
                />
              }
              label="Show History Trail"
            />
          </Grid>
          
          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={showLabels}
                  onChange={(e) => setShowLabels(e.target.checked)}
                />
              }
              label="Show Link Labels"
            />
          </Grid>
          
          <Grid item xs>
            <Typography gutterBottom>Playback Speed (ms/frame)</Typography>
            <Slider
              value={sliderPosition}
              onChange={(_, value) => setSliderPosition(value as number)}
              min={70}
              max={500}
              step={10}
              valueLabelDisplay="auto"
              valueLabelFormat={(v) => 570 - v}
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  )
}

export default PathVisualization