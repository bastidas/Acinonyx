import React, { useState, useRef, useCallback } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  List,
  ListItem,
  ListItemText,
  Paper,
  Alert,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControlLabel,
  Switch,
  Grid,
  Divider,
  Chip,
  IconButton,
  Tabs,
  Tab,
  DialogContentText
} from '@mui/material'
import DeleteIcon from '@mui/icons-material/Delete'
import { Link } from '../response_models'
import { useGraphManager } from './GraphManager'

// Seaborn-style "tab10" color palette equivalent
const TAB10_COLORS = [
  '#1f77b4', // blue
  '#ff7f0e', // orange
  '#2ca02c', // green
  '#d62728', // red
  '#9467bd', // purple
  '#8c564b', // brown
  '#e377c2', // pink
  '#7f7f7f', // gray
  '#bcbd22', // olive
  '#17becf'  // cyan
]

interface ClickPoint {
  x: number
  y: number
  timestamp: number
}

const GraphBuilderTab: React.FC = () => {
  const graphManager = useGraphManager()
  const [currentClick, setCurrentClick] = useState<ClickPoint | null>(null)
  const [linkCounter, setLinkCounter] = useState(0)
  const [nodeCounter, setNodeCounter] = useState(0)
  const [selectedLink, setSelectedLink] = useState<Link | null>(null)
  const [editDialog, setEditDialog] = useState(false)
  const [editForm, setEditForm] = useState({
    length: '',
    name: '',
    has_fixed: false,
    has_constraint: false,
    is_driven: false,
    flip: false,
    color: '#1976d2',
    zlevel: 0
  })
  const [colorMode, setColorMode] = useState<'default' | 'zlevel'>('default')
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'links' | 'nodes'>('links')
  const [deleteDialog, setDeleteDialog] = useState(false)
  const [itemToDelete, setItemToDelete] = useState<{type: 'link' | 'node', id: string, name: string} | null>(null)
  const [hoveredItem, setHoveredItem] = useState<{type: 'link' | 'node', id: string} | null>(null)
  const [dragging, setDragging] = useState<{nodeId: string, offset: {x: number, y: number}} | null>(null)
  const [justFinishedDragging, setJustFinishedDragging] = useState(false)
  const [mouseDownOnNode, setMouseDownOnNode] = useState<{nodeId: string, startTime: number, startPos: {x: number, y: number}} | null>(null)
  const [previewLine, setPreviewLine] = useState<{start: {x: number, y: number}, end: {x: number, y: number}} | null>(null)
  const [statusMessage, setStatusMessage] = useState<string>('')

  // Helper function to get color by default palette
  const getDefaultColor = (index: number): string => {
    return TAB10_COLORS[index % TAB10_COLORS.length]
  }

  // Helper function to get color by z-level
  const getZLevelColor = (zlevel: number): string => {
    const normalizedLevel = Math.abs(zlevel) % TAB10_COLORS.length
    return TAB10_COLORS[normalizedLevel]
  }

  // Get link display color based on mode
  const getLinkColor = (link: Link, index: number): string => {
    if (colorMode === 'zlevel') {
      return getZLevelColor(link.zlevel || 0)
    }
    return link.color || getDefaultColor(index)
  }

  // Get unique z-levels from current links
  const getUniqueZLevels = graphManager.getUniqueZLevels
  const canvasRef = useRef<HTMLDivElement>(null)

  const createLink = async (linkData: any) => {
    try {
      const response = await fetch('/api/links', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(linkData)
      })
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      if (result.status === 'error') {
        throw new Error(result.message)
      }
      
      return result
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create link')
      return null
    }
  }

  const modifyLink = async (id: string, property: string, value: any) => {
    try {
      const response = await fetch('/api/links/modify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, property, value })
      })
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      return await response.json()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to modify link')
      return null
    }
  }

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current) return
    
    const rect = canvasRef.current.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Check if clicking on a node
    const nodeAtClick = graphManager.findNodeAt(x, y)
    if (nodeAtClick && event.button === 0) { // Left mouse button
      // Don't allow interaction with fixed nodes for dragging
      if (nodeAtClick.fixed) {
        setStatusMessage(`Node ${nodeAtClick.id} is fixed and cannot be moved`)
        setTimeout(() => setStatusMessage(''), 2000)
        return
      }
      // Store mouse down info to distinguish between drag and link creation
      setMouseDownOnNode({
        nodeId: nodeAtClick.id,
        startTime: Date.now(),
        startPos: { x, y }
      })
      event.preventDefault()
      return
    }
  }, [graphManager])

  const handleMouseMove = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top

    // Handle dragging
    if (dragging) {
      const newX = x - dragging.offset.x
      const newY = y - dragging.offset.y
      graphManager.updateNodePosition(dragging.nodeId, newX, newY)
      return
    }

    // Handle mouse down on node - check if we should start dragging
    if (mouseDownOnNode) {
      const distance = Math.sqrt(
        Math.pow(x - mouseDownOnNode.startPos.x, 2) + 
        Math.pow(y - mouseDownOnNode.startPos.y, 2)
      )
      const timeHeld = Date.now() - mouseDownOnNode.startTime
      
      // If moved more than 8 pixels OR held for more than 200ms, start dragging
      if (distance > 8 || timeHeld > 200) {
        const node = graphManager.findNodeAt(mouseDownOnNode.startPos.x, mouseDownOnNode.startPos.y)
        if (node && !node.fixed) { // Double-check node is not fixed
          setDragging({
            nodeId: node.id,
            offset: { 
              x: mouseDownOnNode.startPos.x - node.pos[0],
              y: mouseDownOnNode.startPos.y - node.pos[1]
            }
          })
          setMouseDownOnNode(null)
          setStatusMessage('Dragging node')
        } else if (node && node.fixed) {
          setMouseDownOnNode(null)
          setStatusMessage(`Node ${node.id} is fixed and cannot be moved`)
          setTimeout(() => setStatusMessage(''), 2000)
        }
      }
      return
    }

    // Show preview line if we have a first click and are hovering
    if (currentClick && !dragging) {
      setPreviewLine({
        start: { x: currentClick.x, y: currentClick.y },
        end: { x, y }
      })
    } else {
      setPreviewLine(null)
    }

    // Handle hover highlighting
    const nodeAtHover = graphManager.findNodeAt(x, y)
    if (nodeAtHover) {
      setHoveredItem({ type: 'node', id: nodeAtHover.id })
      return
    }

    // Check for link hover (simplified - check if close to any link line)
    const hoveredLink = graphManager.graphState.links.find(link => {
      if (!link.start_point || !link.end_point) return false
      
      // Simple distance to line calculation
      const A = y - link.start_point[1]
      const B = link.start_point[0] - x
      const C = x * link.start_point[1] - link.start_point[0] * y
      const distance = Math.abs(A * link.end_point[0] + B * link.end_point[1] + C) / Math.sqrt(A * A + B * B)
      
      return distance < 10 // 10px tolerance
    })

    if (hoveredLink) {
      setHoveredItem({ type: 'link', id: hoveredLink.id })
    } else {
      setHoveredItem(null)
    }
  }, [graphManager, dragging, mouseDownOnNode, currentClick])

  const handleMouseUp = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current) return
    
    const rect = canvasRef.current.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    
    // Handle dragging completion
    if (dragging) {
      // Check if we're dropping on another node
      const targetNode = graphManager.findNodeAt(x, y)
      if (targetNode && targetNode.id !== dragging.nodeId) {
        // Don't allow merging nodes as it breaks link connections
        // Instead, just keep the node where it is
        setStatusMessage(`Cannot merge nodes - links would be broken`)
      } else {
        setStatusMessage(`Moved node: ${dragging.nodeId}`)
      }
      
      // Set flag to prevent accidental link creation
      setJustFinishedDragging(true)
      setTimeout(() => setJustFinishedDragging(false), 300) // 300ms delay
      setDragging(null)
      setMouseDownOnNode(null)
      setTimeout(() => setStatusMessage(''), 2000)
      return
    }
    
    // Handle node click without dragging
    if (mouseDownOnNode) {
      const clickedNode = graphManager.findNodeAt(x, y)
      if (clickedNode && clickedNode.id === mouseDownOnNode.nodeId) {
        const clickPoint: ClickPoint = { x: clickedNode.pos[0], y: clickedNode.pos[1], timestamp: Date.now() }
        
        if (!currentClick) {
          // Start a new link from this node
          setCurrentClick(clickPoint)
          setError(null)
          setStatusMessage(`Creating link from node ${clickedNode.id}`)
        } else {
          // End the link on this node
          const startPoint: [number, number] = [currentClick.x, currentClick.y]
          const endPoint: [number, number] = [clickedNode.pos[0], clickedNode.pos[1]]
          
          // Check if clicking near existing nodes for connection
          const startNode = graphManager.findNodeAt(startPoint[0], startPoint[1])
          const endNode = clickedNode
          
          if (startNode && endNode && startNode.id !== endNode.id) {
            // Create connection between existing nodes using backend API
            const length = Math.sqrt(
              Math.pow(endPoint[0] - startPoint[0], 2) + Math.pow(endPoint[1] - startPoint[1], 2)
            ) / 50 // Scale down for reasonable units

            const isFirstLink = graphManager.graphState.links.length === 0
            const linkName = isFirstLink ? 'drive_link' : `link${linkCounter + 1}`
            const defaultColor = getDefaultColor(graphManager.graphState.links.length)
            
            const newLinkData = {
              length: Math.max(0.1, length), // Minimum length
              name: linkName,
              n_iterations: 100,
              is_driven: isFirstLink,
              flip: false,
              zlevel: 0 // Will be updated after successful creation
            }

            createLink(newLinkData).then(response => {
              if (response && response.status === 'success' && response.link) {
                const newLink: Link = {
                  ...response.link,
                  length: typeof response.link.length === 'number' ? response.link.length : parseFloat(response.link.length) || 0,
                  start_point: startPoint,
                  end_point: endPoint,
                  color: defaultColor,
                  zlevel: response.link.zlevel || 0
                }
                
                // Add connection between existing nodes with this link
                graphManager.addConnection(startNode.id, endNode.id, newLink)
                setLinkCounter(prev => prev + 1)
                setStatusMessage(`Link created between ${startNode.id} and ${endNode.id}`)
                setTimeout(() => setStatusMessage(''), 2000)
              }
            })
          }
          
          setCurrentClick(null)
          setPreviewLine(null)
          if (!startNode || !endNode || startNode.id === endNode.id) {
            setStatusMessage('')
          }
        }
      }
      setMouseDownOnNode(null)
      return
    }
    
    // Clear any preview line
    setPreviewLine(null)
  }, [dragging, mouseDownOnNode, graphManager, currentClick, editForm, getUniqueZLevels, linkCounter])

  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current || dragging || justFinishedDragging || mouseDownOnNode) return // Don't create links while dragging or handling node interactions

    const rect = canvasRef.current.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    const clickPoint: ClickPoint = { x, y, timestamp: Date.now() }

    // Check if clicking on an existing node
    const nodeAtClick = graphManager.findNodeAt(x, y)
    if (nodeAtClick) {
      // Node clicks are handled by mouseDown/mouseUp
      return
    }

    if (!currentClick) {
      // First click - start a new link
      setCurrentClick(clickPoint)
      setError(null)
      setStatusMessage('Creating link - click second point')
    } else {
      // Second click - complete the link
      const startPoint: [number, number] = [currentClick.x, currentClick.y]
      const endPoint: [number, number] = [x, y]
      
      // Check if clicking near existing nodes for connection
      const startNode = graphManager.findNodeAt(startPoint[0], startPoint[1])
      const endNode = graphManager.findNodeAt(endPoint[0], endPoint[1])
      
      // Check if clicking near existing connections for z-level inheritance
      const nearbyConnections = graphManager.findConnectionsAt(endPoint[0], endPoint[1])
      const startConnections = graphManager.findConnectionsAt(startPoint[0], startPoint[1])
      
      let inheritedZLevel = 0
      // Prioritize start point connections, then end point connections
      const sourceConnections = startConnections.length > 0 ? startConnections : nearbyConnections
      if (sourceConnections.length > 0) {
        console.log('Inheriting z-level from existing connection:', sourceConnections[0].link.name)
        // Inherit z-level + 1 from the existing connection
        inheritedZLevel = (sourceConnections[0].link.zlevel || 0) + 1
      }

      const length = Math.sqrt(
        Math.pow(endPoint[0] - startPoint[0], 2) + Math.pow(endPoint[1] - startPoint[1], 2)
      ) / 50 // Scale down for reasonable units

      const isFirstLink = graphManager.graphState.links.length === 0
      const linkName = isFirstLink ? 'drive_link' : `link${linkCounter + 1}`
      const defaultColor = getDefaultColor(graphManager.graphState.links.length)
      
      const newLinkData = {
        length: Math.max(0.1, length), // Minimum length
        name: linkName,
        n_iterations: 100,
        is_driven: isFirstLink,
        flip: false,
        zlevel: inheritedZLevel // Use inherited z-level from nearby connection
      }

      createLink(newLinkData).then(response => {
        if (response && response.status === 'success' && response.link) {
          const newLink: Link = {
            ...response.link,
            length: typeof response.link.length === 'number' ? response.link.length : parseFloat(response.link.length) || 0,
            start_point: startPoint,
            end_point: endPoint,
            color: defaultColor,
            zlevel: inheritedZLevel
          }
          
          // Ensure every link has exactly two nodes - create nodes if they don't exist
          let finalStartNode = startNode
          let currentNodeCounter = nodeCounter
          
          if (!finalStartNode) {
            currentNodeCounter += 1
            finalStartNode = graphManager.addNode(startPoint[0], startPoint[1], `node${currentNodeCounter}`)
            setNodeCounter(currentNodeCounter)
          }
          
          let finalEndNode = endNode
          if (!finalEndNode) {
            currentNodeCounter += 1
            finalEndNode = graphManager.addNode(endPoint[0], endPoint[1], `node${currentNodeCounter}`)
            setNodeCounter(currentNodeCounter)
          }     if (!finalEndNode) {
                  currentNodeCounter += 1
                  finalEndNode = graphManager.addNode(endPoint[0], endPoint[1], `node${currentNodeCounter}`)
                  setNodeCounter(currentNodeCounter)
                }          // Add connection between nodes with this link
          graphManager.addConnection(finalStartNode.id, finalEndNode.id, newLink)
          setLinkCounter(prev => prev + 1)
          setStatusMessage(`Link created: ${newLink.name}`)
          setTimeout(() => setStatusMessage(''), 2000)
        }
      })

      setCurrentClick(null)
      setStatusMessage('')
    }
  }, [currentClick, graphManager, linkCounter, justFinishedDragging, mouseDownOnNode])

  const handleLinkClick = (link: Link) => {
    setSelectedLink(link)
    setEditForm({
      length: link.length.toString(),
      name: link.name || '',
      has_fixed: link.has_fixed,
      has_constraint: link.has_constraint,
      is_driven: link.is_driven,
      flip: link.flip,
      color: link.color || '#1976d2',
      zlevel: link.zlevel || 0
    })
    setEditDialog(true)
  }

  const handleSaveLink = async () => {
    if (!selectedLink) return
    
    const lengthValue = parseFloat(editForm.length)
    if (isNaN(lengthValue) || lengthValue <= 0) {
      setError('Length must be a positive number')
      return
    }

    try {
      // Update multiple properties
      const updates = [
        { property: 'length', value: lengthValue },
        { property: 'name', value: editForm.name || null },
        { property: 'has_fixed', value: editForm.has_fixed },
        { property: 'has_constraint', value: editForm.has_constraint },
        { property: 'is_driven', value: editForm.is_driven },
        { property: 'flip', value: editForm.flip },
        { property: 'zlevel', value: parseInt(editForm.zlevel.toString()) || 0 }
      ]

      for (const update of updates) {
        const result = await modifyLink(selectedLink.id, update.property, update.value)
        if (!result) return
      }

      // Update the link using GraphManager
      graphManager.updateLink(selectedLink.id, {
        length: lengthValue,
        name: editForm.name || undefined,
        has_fixed: editForm.has_fixed,
        has_constraint: editForm.has_constraint,
        is_driven: editForm.is_driven,
        flip: editForm.flip,
        color: editForm.color,
        zlevel: parseInt(editForm.zlevel.toString()) || 0
      })
      
      setEditDialog(false)
      setSelectedLink(null)
      setError(null)
    } catch (err) {
      setError('Failed to update link properties')
    }
  }

  const clearCanvas = () => {
    graphManager.clearGraph()
    setCurrentClick(null)
    setLinkCounter(0)
    setNodeCounter(0)
    setError(null)
  }

  const handleToggleNodeFixed = (nodeId: string, fixed: boolean) => {
    graphManager.toggleNodeFixed(nodeId, fixed)
  }

  const handleDeleteClick = (type: 'link' | 'node', id: string, name: string) => {
    setItemToDelete({ type, id, name })
    setDeleteDialog(true)
  }

  const confirmDelete = () => {
    if (!itemToDelete) return
    
    if (itemToDelete.type === 'link') {
      graphManager.deleteLink(itemToDelete.id)
    } else if (itemToDelete.type === 'node') {
      graphManager.deleteNode(itemToDelete.id)
    }
    
    setDeleteDialog(false)
    setItemToDelete(null)
  }

  const computeGraph = async () => {
    try {
      const graphStructure = graphManager.getGraphStructure()

      const response = await fetch('/api/compute-graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphStructure)
      })
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      if (result.status === 'error') {
        throw new Error(result.message)
      }
      
      setError(null)
      console.log('Graph computation result:', result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compute graph')
    }
  }

  return (
    <Box sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom align="center">
        Graph Builder
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ display: 'flex', gap: 2, mb: 2, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button variant="outlined" onClick={clearCanvas}>
          Clear Canvas
        </Button>
        <Button variant="contained" onClick={computeGraph} color="success">
          Compute Graph
        </Button>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button 
            variant={colorMode === 'default' ? 'contained' : 'outlined'}
            onClick={() => setColorMode('default')}
            size="small"
          >
            Default Colors
          </Button>
          <Button 
            variant={colorMode === 'zlevel' ? 'contained' : 'outlined'}
            onClick={() => setColorMode('zlevel')}
            size="small"
          >
            Z-Level Colors
          </Button>
        </Box>
        <Typography variant="body2" sx={{ alignSelf: 'center' }}>
          Click twice to create a link. First link will be the drive link.
        </Typography>
      </Box>

      <Box sx={{ display: 'flex', gap: 2 }}>
        {/* Canvas */}
        <Box sx={{ flex: 1 }}>
          <Paper 
            ref={canvasRef}
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            sx={{ 
              width: '100%', 
              height: 600, 
              cursor: dragging ? 'grabbing' : (hoveredItem?.type === 'node' ? 'grab' : 'crosshair'),
              position: 'relative',
              backgroundColor: '#f5f5f5',
              border: '1px solid #ccc'
            }}
          >
          {/* Render current click point */}
          {currentClick && (
            <Box
              sx={{
                position: 'absolute',
                left: currentClick.x - 4,
                top: currentClick.y - 4,
                width: 8,
                height: 8,
                backgroundColor: '#ff0000',
                borderRadius: '50%'
              }}
            />
          )}

          {/* Preview line while creating a link */}
          {previewLine && (
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 2
              }}
            >
              <line
                x1={previewLine.start.x}
                y1={previewLine.start.y}
                x2={previewLine.end.x}
                y2={previewLine.end.y}
                stroke="#999"
                strokeWidth="2"
                strokeDasharray="5,5"
                opacity="0.6"
              />
            </svg>
          )}
          
          {/* Render nodes */}
          {graphManager.graphState.nodes.map((node) => {
            const isHovered = hoveredItem?.type === 'node' && hoveredItem.id === node.id
            const isDragging = dragging?.nodeId === node.id
            const isFixed = node.fixed || false
            return (
              <Box
                key={node.id}
                sx={{
                  position: 'absolute',
                  left: node.pos[0] - (isHovered || isDragging ? 8 : 6),
                  top: node.pos[1] - (isHovered || isDragging ? 8 : 6),
                  width: isHovered || isDragging ? 16 : 12,
                  height: isHovered || isDragging ? 16 : 12,
                  backgroundColor: isFixed ? '#ff4444' : '#4444ff',
                  borderRadius: isFixed ? '10%' : '50%', // Square for fixed, circle for movable
                  border: `${isHovered || isDragging ? 3 : 2}px solid ${isFixed ? '#ffaa00' : '#fff'}`,
                  boxShadow: isHovered || isDragging ? '0 4px 8px rgba(0,0,0,0.3)' : '0 2px 4px rgba(0,0,0,0.2)',
                  zIndex: isDragging ? 20 : (isHovered ? 15 : 10),
                  transform: isHovered || isDragging ? 'scale(1.1)' : 'scale(1)',
                  transition: isDragging ? 'none' : 'all 0.2s ease',
                  cursor: isFixed ? 'not-allowed' : 'grab'
                }}
                title={`${node.id}${isFixed ? ' (FIXED)' : ' (movable)'}`}
              />
            )
          })}
          
          {/* Render links with arrows */}
          {graphManager.graphState.links.map((link, index) => {
            if (!link.start_point || !link.end_point) return null
            
            const isHovered = hoveredItem?.type === 'link' && hoveredItem.id === link.id
            
            // Calculate arrow direction
            const dx = link.end_point[0] - link.start_point[0]
            const dy = link.end_point[1] - link.start_point[1]
            const length = Math.sqrt(dx * dx + dy * dy)
            const unitX = dx / length
            const unitY = dy / length
            
            // Arrow head position (slightly before end point)
            const arrowX = link.end_point[0] - unitX * 15
            const arrowY = link.end_point[1] - unitY * 15
            
            return (
              <svg
                key={index}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                  zIndex: isHovered ? 5 : 1
                }}
              >
                {/* Link line */}
                <line
                  x1={link.start_point[0]}
                  y1={link.start_point[1]}
                  x2={link.end_point[0]}
                  y2={link.end_point[1]}
                  stroke={getLinkColor(link, index)}
                  strokeWidth={isHovered ? (link.is_driven ? 6 : 4) : (link.is_driven ? 4 : 2)}
                  style={{ 
                    pointerEvents: 'auto', 
                    cursor: 'pointer',
                    filter: isHovered ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))' : 'none'
                  }}
                  onDoubleClick={(e) => {
                    e.stopPropagation()
                    handleLinkClick(link)
                  }}
                />
                {/* Arrow head */}
                <polygon
                  points={`${link.end_point[0]},${link.end_point[1]} ${arrowX - unitY * (isHovered ? 7 : 5)},${arrowY + unitX * (isHovered ? 7 : 5)} ${arrowX + unitY * (isHovered ? 7 : 5)},${arrowY - unitX * (isHovered ? 7 : 5)}`}
                  fill={getLinkColor(link, index)}
                  stroke={getLinkColor(link, index)}
                  strokeWidth={isHovered ? "2" : "1"}
                  style={{
                    filter: isHovered ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))' : 'none'
                  }}
                />
              </svg>
            )
          })}
        </Paper>
        
        {/* Status Indicator */}
        {(statusMessage || currentClick || dragging) && (
          <Box
            sx={{
              position: 'fixed',
              bottom: 4,
              left: '50%',
              transform: 'translateX(-50%)',
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              color: 'white',
              padding: '2px 8px',
              borderRadius: '12px',
              fontSize: '0.65rem',
              textAlign: 'center',
              zIndex: 1000,
              opacity: 0.7,
              maxWidth: '300px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}
          >
            {statusMessage || 
             (dragging ? `Dragging ${dragging.nodeId}` : '') ||
             (currentClick ? 'Link creation mode - click to complete' : '')}
          </Box>
        )}
        
        {/* Z-Level Color Legend - moved below canvas */}
        {colorMode === 'zlevel' && graphManager.graphState.links.length > 0 && (
          <Card sx={{ mt: 2, p: 1.5 }}>
            <Typography variant="subtitle2" gutterBottom sx={{ fontSize: '0.875rem' }}>
              Z-Level Colors
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {getUniqueZLevels().map(zlevel => {
                const color = getZLevelColor(zlevel)
                const linksAtLevel = graphManager.graphState.links.filter(link => (link.zlevel || 0) === zlevel)
                return (
                  <Box
                    key={zlevel}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 0.5,
                      px: 1,
                      py: 0.25,
                      border: '1px solid #ddd',
                      borderRadius: 0.5,
                      backgroundColor: '#f9f9f9'
                    }}
                  >
                    <Box
                      sx={{
                        width: 12,
                        height: 8,
                        backgroundColor: color,
                        borderRadius: 0.25,
                        border: '1px solid #ccc'
                      }}
                    />
                    <Typography variant="caption" sx={{ fontSize: '0.75rem' }}>
                      Z{zlevel} ({linksAtLevel.length})
                    </Typography>
                  </Box>
                )
              })}
            </Box>
          </Card>
        )}
        </Box>

        {/* Right Sidebar - Links/Nodes View */}
        <Card sx={{ width: 240, flexShrink: 0 }}>
          <CardContent sx={{ p: 1.5 }}>
            <Tabs 
              value={viewMode} 
              onChange={(_, newValue) => setViewMode(newValue)}
              variant="fullWidth"
              sx={{ mb: 1.5, minHeight: 'auto', '& .MuiTab-root': { fontSize: '0.8rem', minHeight: 'auto', py: 1 } }}
            >
              <Tab label={`Links (${graphManager.graphState.links.length})`} value="links" />
              <Tab label={`Nodes (${graphManager.graphState.nodes.length})`} value="nodes" />
            </Tabs>
            
            {viewMode === 'links' && (
              <List dense sx={{ maxHeight: 420, overflow: 'auto' }}>
                {graphManager.graphState.links.map((link, index) => {
                  const isHovered = hoveredItem?.type === 'link' && hoveredItem.id === link.id
                  return (
                    <ListItem 
                      key={index}
                      onMouseEnter={() => setHoveredItem({ type: 'link', id: link.id })}
                      onMouseLeave={() => setHoveredItem(null)}
                      sx={{ 
                        backgroundColor: isHovered ? '#f0f0f0' : (link.is_driven ? '#fff3e0' : 'transparent'),
                        mb: 0.5,
                        borderRadius: 1,
                        border: `${isHovered ? 2 : 1}px solid ${getLinkColor(link, index)}`,
                        borderLeft: `${isHovered ? 5 : 4}px solid ${getLinkColor(link, index)}`,
                        pr: 0.5,
                        minWidth: 0,
                        maxWidth: '100%',
                        transform: isHovered ? 'scale(0.98)' : 'scale(1)',
                        transition: 'all 0.2s ease',
                        boxShadow: isHovered ? '0 1px 4px rgba(0,0,0,0.1)' : 'none'
                      }}
                    >
                    <ListItemText
                      onClick={() => handleLinkClick(link)}
                      sx={{ cursor: 'pointer' }}
                      primary={
                        <span style={{ display: 'flex', alignItems: 'center', gap: '4px', minWidth: 0 }}>
                          <span style={{ fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                            {link.name || `Link ${index + 1}`}
                          </span>
                          {link.is_driven && (
                            <Chip label="D" size="small" color="warning" sx={{ height: 16, fontSize: '0.6rem', minWidth: 'auto', '& .MuiChip-label': { px: 0.5 } }} />
                          )}
                        </span>
                      }
                      secondary={
                        <span style={{ fontSize: '0.65rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block', color: '#666' }}>
                          L: {typeof link.length === 'number' ? link.length.toFixed(1) : 'N/A'} | Z: {link.zlevel || 0} {link.has_fixed ? '| F' : ''}{link.has_constraint ? '| C' : ''}{link.flip ? '| Fl' : ''}
                        </span>
                      }
                    />
                    <IconButton 
                      size="small" 
                      onClick={() => handleDeleteClick('link', link.id, link.name || `Link ${index + 1}`)}
                      sx={{ ml: 0.5, p: 0.25, minWidth: 'auto' }}
                    >
                      <DeleteIcon sx={{ fontSize: '0.9rem' }} />
                    </IconButton>
                  </ListItem>
                  )
                })}
                {graphManager.graphState.links.length === 0 && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
                    No links created yet.
                  </Typography>
                )}
              </List>
            )}
            
            {viewMode === 'nodes' && (
              <List dense sx={{ maxHeight: 420, overflow: 'auto' }}>
                {graphManager.graphState.nodes.map((node, index) => {
                  const isHovered = hoveredItem?.type === 'node' && hoveredItem.id === node.id
                  return (
                    <ListItem 
                      key={index}
                      onMouseEnter={() => setHoveredItem({ type: 'node', id: node.id })}
                      onMouseLeave={() => setHoveredItem(null)}
                      sx={{ 
                        backgroundColor: isHovered ? '#f0f0f0' : (node.fixed ? '#ffebee' : 'transparent'),
                        mb: 0.5,
                        borderRadius: 1,
                        border: `${isHovered ? 2 : 1}px solid #ddd`,
                        borderLeft: `${isHovered ? 5 : 4}px solid ${node.fixed ? '#f44336' : '#2196f3'}`,
                        pr: 0.5,
                        minWidth: 0,
                        maxWidth: '100%',
                        transform: isHovered ? 'scale(0.98)' : 'scale(1)',
                        transition: 'all 0.2s ease',
                        boxShadow: isHovered ? '0 1px 4px rgba(0,0,0,0.1)' : 'none',
                        flexDirection: 'column',
                        alignItems: 'stretch'
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <ListItemText
                          primary={
                            <span style={{ fontSize: '0.75rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {node.id}
                            </span>
                          }
                          secondary={
                            <span style={{ fontSize: '0.65rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'block', color: '#666' }}>
                              pos: ({node.pos[0].toFixed(0)}, {node.pos[1].toFixed(0)}) | C:{node.connections.length}
                            </span>
                          }
                        />
                        <IconButton 
                          size="small" 
                          onClick={() => handleDeleteClick('node', node.id, node.id)}
                          sx={{ ml: 0.5, p: 0.25, minWidth: 'auto' }}
                        >
                          <DeleteIcon sx={{ fontSize: '0.9rem' }} />
                        </IconButton>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5, ml: 2 }}>
                        <FormControlLabel
                          control={
                            <Switch
                              size="small"
                              checked={node.fixed || false}
                              onChange={(e) => handleToggleNodeFixed(node.id, e.target.checked)}
                              sx={{ mr: 0.5 }}
                            />
                          }
                          label={
                            <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
                              Fixed {node.fixed && node.fixed_loc ? `at (${node.fixed_loc[0].toFixed(0)}, ${node.fixed_loc[1].toFixed(0)})` : ''}
                            </Typography>
                          }
                          sx={{ margin: 0 }}
                        />
                      </Box>
                    </ListItem>
                  )
                })}
                {graphManager.graphState.nodes.length === 0 && (
                  <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', mt: 2 }}>
                    No nodes created yet.
                  </Typography>
                )}
              </List>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* Comprehensive Edit Dialog */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Edit Link Properties
          {selectedLink && (
            <Typography variant="subtitle2" color="text.secondary">
              Link ID: {selectedLink.id}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ pt: 1 }}>
            {/* Immutable Properties */}
            <Grid item xs={10}>
              <Typography variant="h6" gutterBottom>
                Immutable Properties
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                <Chip 
                  label={`ID: ${selectedLink?.id || 'N/A'}`} 
                  variant="outlined" 
                  size="small"
                />
                <Chip 
                  label={`Iterations: ${selectedLink?.n_iterations || 100}`} 
                  variant="outlined" 
                  size="small"
                />
              </Box>
              <Divider sx={{ mb: 1 }} />
            </Grid>

            {/* Editable Properties */}
            <Grid item xs={10}>
              <Typography variant="h6" gutterBottom>
                Editable Properties
              </Typography>
            </Grid>
            
            <Grid item xs={12} sm={3}>
              <TextField
                label="Length"
                type="number"
                value={editForm.length}
                onChange={(e) => setEditForm(prev => ({ ...prev, length: e.target.value }))}
                fullWidth
                inputProps={{ min: 0.1, step: 0.1 }}
                helperText="Length of the link in inches"
              />
            </Grid>
            
            <Grid item xs={12} sm={3}>
              <TextField
                label="Name"
                type="text"
                value={editForm.name}
                onChange={(e) => setEditForm(prev => ({ ...prev, name: e.target.value }))}
                fullWidth
                helperText="Optional name for the link"
              />
            </Grid>
            
            <Grid item xs={12} sm={3}>
              <TextField
                label="Color"
                type="color"
                value={editForm.color}
                onChange={(e) => setEditForm(prev => ({ ...prev, color: e.target.value }))}
                fullWidth
                helperText="Visual color for the link (default mode)"
              />
            </Grid>
            
            <Grid item xs={12} sm={3}>
              <TextField
                label="Z-Level"
                type="number"
                value={editForm.zlevel}
                onChange={(e) => setEditForm(prev => ({ ...prev, zlevel: parseInt(e.target.value) || 0 }))}
                fullWidth
                inputProps={{ step: 1 }}
                helperText="Z-level for depth ordering and coloring"
              />
            </Grid>
            
            <Grid item xs={10}>
              <Typography variant="subtitle1" gutterBottom>
                Boolean Properties
              </Typography>
            </Grid>
            
            <Grid item xs={10} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editForm.has_fixed}
                    onChange={(e) => setEditForm(prev => ({ ...prev, has_fixed: e.target.checked }))}
                  />
                }
                label="Has Fixed Location"
              />
            </Grid>
            
            <Grid item xs={10} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editForm.has_constraint}
                    onChange={(e) => setEditForm(prev => ({ ...prev, has_constraint: e.target.checked }))}
                  />
                }
                label="Has Constraint"
              />
            </Grid>
            
            <Grid item xs={10} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editForm.is_driven}
                    onChange={(e) => setEditForm(prev => ({ ...prev, is_driven: e.target.checked }))}
                  />
                }
                label="Is Driven Link"
              />
            </Grid>
            
            <Grid item xs={10} sm={4}>
              <FormControlLabel
                control={
                  <Switch
                    checked={editForm.flip}
                    onChange={(e) => setEditForm(prev => ({ ...prev, flip: e.target.checked }))}
                  />
                }
                label="Flip Orientation"
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setEditDialog(false)}>Cancel</Button>
          <Button onClick={handleSaveLink} variant="contained" color="primary">
            Save Changes
          </Button>
        </DialogActions>
      </Dialog>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialog} onClose={() => setDeleteDialog(false)}>
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete {itemToDelete?.type} "{itemToDelete?.name}"? This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>Cancel</Button>
          <Button onClick={confirmDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default GraphBuilderTab