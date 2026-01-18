import React, { useState, useRef, useCallback, useEffect } from 'react'
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
import { useGraphManager, calculateLength } from './GraphManager'
import {
  createInitialLinkData,
  constructFrontendLink,
  createDeleteItem,
  DeleteItem
} from './GraphManagerHelpers'
import PathVisualization from './PathVisualization'
import { graphColors } from '../theme'

// Use centralized graph color palette from theme
const TAB10_COLORS = graphColors

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
  
  // Simple scaling system: 6 pixels = 1 "inch" unit
  // Canvas is 600px high = 100 units, proportional width
  const PIXELS_PER_UNIT = 6
  const CANVAS_HEIGHT_PX = 600
  const MAX_UNITS = CANVAS_HEIGHT_PX / PIXELS_PER_UNIT // = 100 units
  
  // Convert between pixel coordinates and work units (inches)
  const pixelsToUnits = (pixels: number) => pixels / PIXELS_PER_UNIT
  const unitsToPixels = (units: number) => units * PIXELS_PER_UNIT
  
  const [selectedLink, setSelectedLink] = useState<Link | null>(null)
  const [editDialog, setEditDialog] = useState(false)
  const [editForm, setEditForm] = useState({
    // Required fields with defaults
    name: '',
    n_iterations: 24,
    has_fixed: false,
    // Optional fields with defaults
    has_constraint: false,
    is_driven: false,
    flip: false,
    zlevel: 0,
    // Frontend-specific
    color: '#474747ff'
  })
  const [colorMode, setColorMode] = useState<'default' | 'zlevel'>('default')
  const [error, setError] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<'links' | 'nodes'>('links')
  const [deleteDialog, setDeleteDialog] = useState(false)
  const [itemToDelete, setItemToDelete] = useState<DeleteItem | null>(null)
  const [hoveredItem, setHoveredItem] = useState<{type: 'link' | 'node', id: string} | null>(null)
  const [dragging, setDragging] = useState<{nodeId: string, offset: {x: number, y: number}} | null>(null)
  const [justFinishedDragging, setJustFinishedDragging] = useState(false)
  const [mouseDownOnNode, setMouseDownOnNode] = useState<{nodeId: string, startTime: number, startPos: {x: number, y: number}} | null>(null)
  const [previewLine, setPreviewLine] = useState<{start: {x: number, y: number}, end: {x: number, y: number}} | null>(null)
  const [statusMessage, setStatusMessage] = useState<string>('')
  const [pathData, setPathData] = useState<any>(null)
  const [linkCreationMode, setLinkCreationMode] = useState<'idle' | 'waiting-for-second-click'>('idle')
  const [justStartedLinkFromNode, setJustStartedLinkFromNode] = useState(false)
  
  // Pylinkage integration state
  const [pylinkageLoading, setPylinkageLoading] = useState(false)
  const [pylinkageResult, setPylinkageResult] = useState<any>(null)
  const [pylinkageDialogOpen, setPylinkageDialogOpen] = useState(false)

  // Unified function to enter link creation mode from a given start point (units)
  const enterLinkCreationMode = useCallback(
    (
      startX: number,
      startY: number,
      fromNodeId?: string,
      initialEndPixel?: { x: number; y: number }
    ) => {
      const clickPoint: ClickPoint = { x: startX, y: startY, timestamp: Date.now() }
      setCurrentClick(clickPoint)
      setLinkCreationMode('waiting-for-second-click')
      setError(null)

      // Grey preview line behavior:
      // - Normally, the preview line is updated in handleMouseMove when `currentClick` is set.
      // - To provide immediate visual feedback, we also initialize `previewLine` here
      //   using the current mouse pixel position if provided, else a slight offset
      //   so the dashed line is visible even before the user moves the mouse.
      const startPx = { x: unitsToPixels(startX), y: unitsToPixels(startY) }
      const endPx = initialEndPixel
        ? initialEndPixel
        : { x: startPx.x + 16, y: startPx.y } // small offset to make the line visible immediately
      setPreviewLine({ start: startPx, end: endPx })

      if (fromNodeId) {
        setStatusMessage(`Creating link from node ${fromNodeId} - click second point`)
        setJustStartedLinkFromNode(true)
        setTimeout(() => setJustStartedLinkFromNode(false), 100)
      } else {
        setStatusMessage('Creating link - click second point (press Escape to cancel)')
        setJustStartedLinkFromNode(false)
      }
    },
    []
  )

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
    return link.meta.color || getDefaultColor(index)
  }

  // Get unique z-levels from current links
  const getUniqueZLevels = graphManager?.getUniqueZLevels || (() => [])
  const canvasRef = useRef<HTMLDivElement>(null)
  const skipCanvasClickRef = useRef(false)

  // Add keyboard event handling for escape key
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && currentClick) {
        // Cancel link creation
        setCurrentClick(null)
        setLinkCreationMode('idle')
        setJustStartedLinkFromNode(false)
        setPreviewLine(null)
        setStatusMessage('Link creation cancelled')
        setTimeout(() => setStatusMessage(''), 3000) // 3 seconds for cancelled message
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [currentClick])

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
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)

    // Check if clicking on a node
    //console.log('MouseDown - pixel coords:', { pixelX, pixelY })
    console.log('MouseDown - unit coords:', { x, y })
    const nodeAtClick = graphManager.findNodeAt(x, y)
    console.log('MouseDown - found node:', nodeAtClick)
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
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    
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
      
      // Only start dragging if:
      // - mouse moved significantly (8+ units = 48px)
      // - mouse button is actively held (event.buttons === 1)
      // - user held for at least 250ms to distinguish from brief clicks
      // This ensures clicks go to link creation; only deliberate drags trigger drag mode.
      if (event.buttons === 1 && distance > 8.0 && timeHeld > 250) {
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
        start: { x: unitsToPixels(currentClick.x), y: unitsToPixels(currentClick.y) },
        end: { x: pixelX, y: pixelY }
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
      if (!link.meta.start_point || !link.meta.end_point) return false
      
      // Simple distance to line calculation (all in unit coordinates)
      const A = y - link.meta.start_point[1]
      const B = link.meta.start_point[0] - x
      const C = x * link.meta.start_point[1] - link.meta.start_point[0] * y
      const distance = Math.abs(A * link.meta.end_point[0] + B * link.meta.end_point[1] + C) / Math.sqrt(A * A + B * B)
      
      return distance < 1.7 // ~10px tolerance in unit coordinates (10/6 ≈ 1.7)
    })

    if (hoveredLink) {
      setHoveredItem({ type: 'link', id: hoveredLink.meta.id })
    } else {
      setHoveredItem(null)
    }
  }, [graphManager, dragging, mouseDownOnNode, currentClick])

  const handleMouseUp = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    if (!canvasRef.current) return
    
    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    
    // Handle dragging completion
    if (dragging) {
      // Check if we're dropping on another node (use unit coordinates for findNodeAt)
      const targetNode = graphManager.findNodeAt(x, y)
      if (targetNode && targetNode.id !== dragging.nodeId) {
        // Try to merge nodes by connecting their links
        const draggedNode = graphManager.graphState.nodes.find(n => n.id === dragging.nodeId)
        if (draggedNode) {
          // Find connections involving the dragged node
          const draggedConnections = graphManager.graphState.connections.filter(
            conn => conn.from_node === dragging.nodeId || conn.to_node === dragging.nodeId
          )
          
          // If the dragged node has connections, try to merge by reconnecting them to the target node
          if (draggedConnections.length > 0) {
            try {
              // Update connections to point to target node instead of dragged node
              draggedConnections.forEach(conn => {
                const newFromNode = conn.from_node === dragging.nodeId ? targetNode.id : conn.from_node
                const newToNode = conn.to_node === dragging.nodeId ? targetNode.id : conn.to_node
                
                // Update the connection
                graphManager.updateConnection(conn.id, newFromNode, newToNode)
              })
              
              // Delete the dragged node
              graphManager.deleteNode(dragging.nodeId)
              setStatusMessage(`Merged node ${dragging.nodeId} into ${targetNode.id}`)
            } catch (error) {
              setStatusMessage(`Cannot merge nodes - operation failed`)
            }
          } else {
            // No connections, just delete the dragged node
            graphManager.deleteNode(dragging.nodeId)
            setStatusMessage(`Deleted isolated node ${dragging.nodeId}`)
          }
        }
      } else {
        setStatusMessage(`Moved node: ${dragging.nodeId}`)
      }
      
      // Set flag to prevent accidental link creation (but not if we're in link creation mode)
      if (linkCreationMode === 'idle') {
        setJustFinishedDragging(true)
        setTimeout(() => setJustFinishedDragging(false), 300) // 300ms delay
      }
      setDragging(null)
      setMouseDownOnNode(null)
      setTimeout(() => setStatusMessage(''), 3000)
      return
    }
    
    // Handle node click without dragging
    if (mouseDownOnNode) {
      const clickedNode = graphManager.findNodeAt(x, y)
      console.log('MouseUp - clickedNode:', clickedNode)
      if (clickedNode && clickedNode.id === mouseDownOnNode.nodeId) {
        if (!currentClick) {
          // Start a new link from this node
          console.log('MouseUp - Starting link from existing node:', clickedNode.id)
          // Ignore the subsequent click event fired after this mouseup to prevent accidental cancellation
          skipCanvasClickRef.current = true
          enterLinkCreationMode(
            clickedNode.pos[0],
            clickedNode.pos[1],
            clickedNode.id,
            { x: pixelX, y: pixelY }
          )
          // Prevent canvas click from also handling this event
          event.preventDefault()
          event.stopPropagation()
          return // Exit early to avoid clearing currentClick
        } else {
          // End the link on this node
          const startPoint: [number, number] = [currentClick.x, currentClick.y]
          const endPoint: [number, number] = [clickedNode.pos[0], clickedNode.pos[1]]
          
          // Check if clicking near existing nodes for connection
          const startNode = graphManager.findNodeAt(startPoint[0], startPoint[1])
          const endNode = clickedNode
          
          if (startNode && endNode && startNode.id !== endNode.id) {
            // Create connection between existing nodes using backend API
            const length = calculateLength(startPoint, endPoint)

            const isFirstLink = graphManager.graphState.links.length === 0
            const linkName = isFirstLink ? 'drive_link' : `link${linkCounter + 1}`
            const defaultColor = getDefaultColor(graphManager.graphState.links.length)
            
            const newLinkData = createInitialLinkData(linkName, length, isFirstLink, 0)

            createLink(newLinkData).then(response => {
              if (response && response.status === 'success' && response.link) {
                const newLink = constructFrontendLink(
                  response.link,
                  startPoint,
                  endPoint,
                  defaultColor,
                  0
                )
                
                // Add connection between existing nodes with this link
                graphManager.addConnection(startNode.id, endNode.id, newLink)
                setLinkCounter(prev => prev + 1)
                setStatusMessage(`Link created between ${startNode.id} and ${endNode.id}`)
                setTimeout(() => setStatusMessage(''), 2000)
                
                // Clear state only after successful link creation
                console.log('MouseUp - Clearing currentClick after completing link')
                setCurrentClick(null)
                setLinkCreationMode('idle')
                setPreviewLine(null)
              }
            })
            // Prevent canvas click from also handling this event
            event.stopPropagation()
          } else {
            // If we can't create a link between existing nodes, don't clear the state
            console.log('MouseUp - Cannot create link, keeping link creation state')
          }
        }
      }
      setMouseDownOnNode(null)
      return
    }
    
    // Clear any preview line
    setPreviewLine(null)
  }, [dragging, mouseDownOnNode, graphManager, currentClick, editForm, getUniqueZLevels, linkCounter, linkCreationMode])

  const handleCanvasClick = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    console.log('CanvasClick - entry conditions:', { dragging, justFinishedDragging, mouseDownOnNode, justStartedLinkFromNode })
    if (!canvasRef.current || dragging) return // Don't create links while dragging

    // Some interactions (e.g., starting a link from a node) should ignore the next click event
    if (skipCanvasClickRef.current) {
      skipCanvasClickRef.current = false
      console.log('CanvasClick - Ignored due to skip flag')
      return
    }
    
    // Don't process canvas click if we just started a link from a node
    if (justStartedLinkFromNode) {
      console.log('CanvasClick - Ignoring click, just started link from node')
      return
    }
    
    // Allow canvas click if we're in link creation mode, even if we just finished dragging
    if (justFinishedDragging && linkCreationMode === 'idle') return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const clickPoint: ClickPoint = { x, y, timestamp: Date.now() }

    // Check if clicking on an existing node
    //console.log('CanvasClick - pixel coords:', { pixelX, pixelY })
    console.log('CanvasClick - unit coords:', { x, y })
    const nodeAtClick = graphManager.findNodeAt(x, y)
    console.log('CanvasClick - found node:', nodeAtClick)
    


    console.log('CanvasClick - currentClick state:', currentClick)
    console.log('CanvasClick - linkCreationMode:', linkCreationMode)
    
    // If we're not in link creation mode and clicked on a node, let mouseDown/mouseUp handle it
    if (nodeAtClick && linkCreationMode === 'idle') {
      // Node clicks are handled by mouseDown/mouseUp when not creating links
      console.log('CanvasClick - node click ignored, handled by mouseDown/mouseUp')
      return
    }

    if (!currentClick || linkCreationMode === 'idle') {
      // First click - start a new link
      console.log('CanvasClick - Starting new link at:', clickPoint)
      enterLinkCreationMode(clickPoint.x, clickPoint.y, undefined, { x: pixelX, y: pixelY })
    } else {
      // Second click - complete the link
      const startPoint: [number, number] = [currentClick.x, currentClick.y]
      const endPoint: [number, number] = [x, y]
      
      // If we clicked on an existing node, use its exact position for the end point
      if (nodeAtClick) {
        endPoint[0] = nodeAtClick.pos[0]
        endPoint[1] = nodeAtClick.pos[1]
        console.log('CanvasClick - Using existing node position for end point:', endPoint)
      }
      
      // Check if clicking near existing nodes for connection
      console.log('Link creation - checking for nodes:')
      console.log('  Start point:', startPoint, 'End point:', endPoint)
      console.log('  Available nodes:', graphManager.graphState.nodes.map(n => ({ id: n.id, pos: n.pos })))
      let startNode = graphManager.findNodeAt(startPoint[0], startPoint[1])
      let endNode = graphManager.findNodeAt(endPoint[0], endPoint[1])
      console.log('  Found start node:', startNode?.id, 'at position:', startNode?.pos)
      console.log('  Found end node:', endNode?.id, 'at position:', endNode?.pos)
      
      // Prevent creating links from a node to itself
      if (startNode && endNode && startNode.id === endNode.id) {
        console.log('CanvasClick - Cannot create link from node to itself:', startNode.id, 'ignoring')
        console.log('  Start point actual:', startPoint, 'End point actual:', endPoint)
        setCurrentClick(null)
        setLinkCreationMode('idle')
        setStatusMessage('Cannot create link from node to itself')
        setTimeout(() => setStatusMessage(''), 2000)
        return
      }
      
      // If first point doesn't have a node but second point does, create node at first point
      if (!startNode && endNode) {
        const newNodeId = `node${nodeCounter + 1}`
        startNode = graphManager.addNode(startPoint[0], startPoint[1], newNodeId)
        setNodeCounter(nodeCounter + 1)
      }
      
      // If second point doesn't have a node but first point does, create node at second point  
      if (startNode && !endNode) {
        const newNodeId = `node${nodeCounter + 1}`
        endNode = graphManager.addNode(endPoint[0], endPoint[1], newNodeId)
        setNodeCounter(nodeCounter + 1)
      }
      
      // Check if clicking near existing connections for z-level inheritance
      const nearbyConnections = graphManager.findConnectionsAt(endPoint[0], endPoint[1])
      const startConnections = graphManager.findConnectionsAt(startPoint[0], startPoint[1])
      
      let inheritedZLevel = 0
      // Prioritize start point connections, then end point connections
      const sourceConnections = startConnections.length > 0 ? startConnections : nearbyConnections
      if (sourceConnections.length > 0) {
        // Look up the link for this connection
        const sourceLink = graphManager.getLinkForConnection(sourceConnections[0])
        if (sourceLink) {
          console.log('Inheriting z-level from existing connection:', sourceLink.name)
          // Inherit z-level + 1 from the existing connection
          inheritedZLevel = (sourceLink.zlevel || 0) + 1
        }
      }

      const length = calculateLength(startPoint, endPoint)

      const isFirstLink = graphManager.graphState.links.length === 0
      const linkName = isFirstLink ? 'drive_link' : `link${linkCounter + 1}`
      const defaultColor = getDefaultColor(graphManager.graphState.links.length)
      
      const newLinkData = createInitialLinkData(linkName, length, isFirstLink, inheritedZLevel)

      createLink(newLinkData).then(response => {
        if (response && response.status === 'success' && response.link) {
          const newLink = constructFrontendLink(
            response.link,
            startPoint,
            endPoint,
            defaultColor,
            inheritedZLevel
          )
          
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
          }          // Add connection between nodes with this link
          graphManager.addConnection(finalStartNode.id, finalEndNode.id, newLink)
          setLinkCounter(prev => prev + 1)
          setStatusMessage(`Link created: ${newLink.name}`)
          setTimeout(() => setStatusMessage(''), 2000)
        }
      })

      setCurrentClick(null)
      setLinkCreationMode('idle')
      setJustStartedLinkFromNode(false)
      setStatusMessage('')
    }
  }, [currentClick, linkCreationMode, graphManager, linkCounter, justFinishedDragging, mouseDownOnNode, nodeCounter, justStartedLinkFromNode])

  const handleLinkClick = (link: Link) => {
    setSelectedLink(link)
    setEditForm({
      // Required fields
      name: link.name,
      n_iterations: link.n_iterations,
      has_fixed: link.has_fixed,
      // Optional fields with defaults
      has_constraint: link.has_constraint ?? false,
      is_driven: link.is_driven ?? false,
      flip: link.flip ?? false,
      zlevel: link.zlevel ?? 0,
      // Frontend fields from meta
      color: link.meta.color || '#1976d2'
    })
    setEditDialog(true)
  }

  const handleSaveLink = async () => {
    if (!selectedLink) return

    try {
      // Update multiple properties (excluding length - it's immutable and set via canvas)
      const updates = [
        { property: 'name', value: editForm.name || null },
        { property: 'has_fixed', value: editForm.has_fixed },
        { property: 'has_constraint', value: editForm.has_constraint },
        { property: 'is_driven', value: editForm.is_driven },
        { property: 'flip', value: editForm.flip },
        { property: 'zlevel', value: parseInt(editForm.zlevel.toString()) || 0 }
      ]

      for (const update of updates) {
        const result = await modifyLink(selectedLink.meta.id, update.property, update.value)
        if (!result) return
      }

      // Update the link using GraphManager (excluding length)
      // Now we need to update meta separately
      graphManager.updateLink(selectedLink.meta.id, {
        name: editForm.name || undefined,
        has_fixed: editForm.has_fixed,
        has_constraint: editForm.has_constraint,
        is_driven: editForm.is_driven,
        flip: editForm.flip,
        zlevel: parseInt(editForm.zlevel.toString()) || 0,
        meta: {
          ...selectedLink.meta,
          color: editForm.color
        }
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
    setLinkCreationMode('idle')
    setJustStartedLinkFromNode(false)
    setLinkCounter(0)
    setNodeCounter(0)
    setError(null)
  }

  const handleToggleNodeFixed = (nodeId: string, fixed: boolean) => {
    graphManager.toggleNodeFixed(nodeId, fixed)
  }

  const handleDeleteClick = (type: 'link' | 'node', id: string, name: string) => {
    setItemToDelete(createDeleteItem(type, id, name))
    setDeleteDialog(true)
  }

  const confirmDelete = () => {
    if (!itemToDelete) return
    
    if (itemToDelete.type === 'link') {
      graphManager.deleteLink(itemToDelete.id)
    } else if (itemToDelete.type === 'node') {
      // Collect all links connected to this node and delete them first
      const connectedConns = graphManager.graphState.connections.filter(
        conn => conn.from_node === itemToDelete.id || conn.to_node === itemToDelete.id
      )
      const linkIds = connectedConns.map(conn => conn.link_id)
      linkIds.forEach(linkId => graphManager.deleteLink(linkId))
      // Finally delete the node itself
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
        // Extract detailed error information if available
        let errorMessage = result.message || 'Failed to compute graph'
        
        // If there are specific errors or traceback, add them to the console for debugging
        if (result.errors) {
          console.error('Graph computation errors:', result.errors)
          
          // Look for the most informative error in the traceback
          if (result.errors.traceback && Array.isArray(result.errors.traceback)) {
            const relevantErrors = result.errors.traceback.filter(
              (line: string) => line.includes('ValueError') || line.includes('No viable solution')
            )
            if (relevantErrors.length > 0) {
              // Extract the actual error message (usually after "ValueError: ")
              const lastError = relevantErrors[relevantErrors.length - 1]
              const match = lastError.match(/ValueError:\s*(.+)$/)
              if (match) {
                errorMessage = match[1].trim()
              }
            }
          }
          
          // Also check for computation_error
          if (result.errors.computation_error && result.errors.computation_error.length > 0) {
            errorMessage = result.errors.computation_error[0]
          }
        }
        
        throw new Error(errorMessage)
      }
      
      // Update path data for visualization
      if (result.path_data) {
        setPathData(result.path_data)
      }
      
      setError(null)
      setStatusMessage('Graph computed and saved successfully. Path visualization updated.')
      setTimeout(() => setStatusMessage(''), 3000)
      console.log('Graph computation result:', result)
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to compute graph'
      setError(errorMsg)
      console.error('Graph computation failed:', errorMsg)
    }
  }

  const saveGraph = async () => {
    try {
      const graphStructure = graphManager.getGraphStructure()

      const response = await fetch('/api/save-graph', {
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
      setStatusMessage(`Graph saved: ${result.filename}`)
      setTimeout(() => setStatusMessage(''), 3000)
      console.log('Graph saved:', result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save graph')
    }
  }

  const loadLastSavedGraph = async () => {
    try {
      const response = await fetch('/api/load-last-saved-graph')
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      if (result.status === 'error') {
        throw new Error(result.message)
      }
      
      if (result.status === 'success' && result.graph_data) {
        // Clear current graph
        graphManager.clearGraph()
        
        // Load the saved graph data
        const { nodes, connections, links } = result.graph_data
        
        // Add nodes first  
        if (nodes && Array.isArray(nodes)) {
          nodes.forEach(node => {
            // Convert node positions from old pixel system to new unit system if needed
            let x = node.pos[0]
            let y = node.pos[1]
            
            // If coordinates are much larger than expected (likely old pixel coordinates), convert them
            if (x > MAX_UNITS || y > MAX_UNITS) {
              x = pixelsToUnits(x)
              y = pixelsToUnits(y)
            }
            
            graphManager.addNode(x, y, node.id)
            // Set fixed state if needed
            if (node.fixed) {
              graphManager.toggleNodeFixed(node.id, true)
            }
          })
        }
        
        // Add links and connections
        // Handle both old format (connections with embedded link) and new format (link_id reference)
        if (links && Array.isArray(links)) {
          links.forEach((link, index) => {
            // Find the connection for this link
            let conn = null
            if (connections && Array.isArray(connections)) {
              // Try new format first (link_id)
              conn = connections.find(c => c.link_id === link.meta?.id)
              // Fall back to old format (embedded link with name)
              if (!conn) {
                conn = connections.find(c => c.link?.name === link.name)
              }
            }
            
            if (conn) {
              // Reconstruct the link with proper structure
              const reconstructedLink: Link = {
                name: link.name,
                length: link.length,
                n_iterations: link.n_iterations ?? 24,
                has_fixed: link.has_fixed ?? false,
                target_length: link.target_length ?? null,
                target_cost_func: link.target_cost_func ?? null,
                fixed_loc: link.fixed_loc ?? null,
                has_constraint: link.has_constraint ?? false,
                is_driven: link.is_driven ?? false,
                flip: link.flip ?? false,
                zlevel: link.zlevel ?? 0,
                meta: link.meta || {
                  id: link.id || `link_${Date.now()}_${index}`,
                  start_point: link.start_point || [0, 0],
                  end_point: link.end_point || [0, 0],
                  color: link.color || getDefaultColor(index)
                }
              }
              graphManager.addConnection(conn.from_node, conn.to_node, reconstructedLink)
            }
          })
        }
        
        // Update counters
        if (nodes) setNodeCounter(nodes.length)
        if (links) setLinkCounter(links.length)
        
        setError(null)
        setStatusMessage(`Loaded graph: ${result.filename}`)
        setTimeout(() => setStatusMessage(''), 3000)
        console.log('Loaded saved graph:', result.filename)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load last saved graph')
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // PYLINKAGE INTEGRATION FUNCTIONS
  // ═══════════════════════════════════════════════════════════════════════════════

  const convertToPylinkage = async () => {
    try {
      setPylinkageLoading(true)
      setError(null)
      
      const graphStructure = graphManager.getGraphStructure()
      
      const response = await fetch('/api/convert-to-pylinkage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(graphStructure)
      })
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      setPylinkageResult(result)
      setPylinkageDialogOpen(true)
      
      if (result.status === 'success') {
        setStatusMessage('Pylinkage conversion successful!')
      } else {
        setStatusMessage('Pylinkage conversion failed - see details')
      }
      setTimeout(() => setStatusMessage(''), 3000)
      
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to convert to pylinkage'
      setError(errorMsg)
      setPylinkageResult({ status: 'error', message: errorMsg })
      setPylinkageDialogOpen(true)
    } finally {
      setPylinkageLoading(false)
    }
  }

  const simulateWithPylinkage = async () => {
    try {
      setPylinkageLoading(true)
      setError(null)
      
      const graphStructure = graphManager.getGraphStructure()
      
      const response = await fetch('/api/simulate-pylinkage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...graphStructure,
          n_iterations: 24
        })
      })
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      setPylinkageResult(result)
      setPylinkageDialogOpen(true)
      
      if (result.status === 'success' && result.path_data) {
        setPathData(result.path_data)
        setStatusMessage(`Pylinkage simulation completed in ${result.execution_time_ms?.toFixed(1) || '?'}ms`)
      } else {
        setStatusMessage('Pylinkage simulation failed - see details')
      }
      setTimeout(() => setStatusMessage(''), 3000)
      
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to simulate with pylinkage'
      setError(errorMsg)
      setPylinkageResult({ status: 'error', message: errorMsg })
      setPylinkageDialogOpen(true)
    } finally {
      setPylinkageLoading(false)
    }
  }

  const compareSolvers = async () => {
    try {
      setPylinkageLoading(true)
      setError(null)
      
      const graphStructure = graphManager.getGraphStructure()
      
      const response = await fetch('/api/compare-solvers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...graphStructure,
          n_iterations: 24
        })
      })
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      setPylinkageResult(result)
      setPylinkageDialogOpen(true)
      
      if (result.comparison) {
        const speedup = result.comparison.speedup_factor?.toFixed(2) || 'N/A'
        setStatusMessage(`Solver comparison complete - speedup: ${speedup}x`)
      }
      setTimeout(() => setStatusMessage(''), 3000)
      
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to compare solvers'
      setError(errorMsg)
      setPylinkageResult({ status: 'error', message: errorMsg })
      setPylinkageDialogOpen(true)
    } finally {
      setPylinkageLoading(false)
    }
  }

  const runDemo4Bar = async () => {
    try {
      setPylinkageLoading(true)
      setError(null)
      
      const response = await fetch('/api/demo-4bar-pylinkage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ground_length: 30.0,
          crank_length: 10.0,
          coupler_length: 25.0,
          rocker_length: 20.0,
          crank_anchor: [20.0, 30.0],
          n_iterations: 24,
          include_ui_format: true
        })
      })
      
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()
      
      setPylinkageResult(result)
      setPylinkageDialogOpen(true)
      
      if (result.status === 'success' && result.path_data) {
        setPathData(result.path_data)
        setStatusMessage(`Demo 4-bar completed in ${result.execution_time_ms?.toFixed(1) || '?'}ms`)
      } else {
        setStatusMessage('Demo 4-bar failed - see details')
      }
      setTimeout(() => setStatusMessage(''), 3000)
      
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : 'Failed to run demo 4-bar'
      setError(errorMsg)
      setPylinkageResult({ status: 'error', message: errorMsg })
      setPylinkageDialogOpen(true)
    } finally {
      setPylinkageLoading(false)
    }
  }

  const loadUiGraphIntoBuilder = (uiGraph: any) => {
    if (!uiGraph) return
    
    // Clear current graph
    graphManager.clearGraph()
    
    const { nodes, links, connections } = uiGraph
    
    // Add nodes first
    if (nodes && Array.isArray(nodes)) {
      nodes.forEach((node: any) => {
        graphManager.addNode(node.pos[0], node.pos[1], node.id)
        if (node.fixed) {
          graphManager.toggleNodeFixed(node.id, true)
        }
      })
    }
    
    // Add links and connections
    if (links && Array.isArray(links) && connections && Array.isArray(connections)) {
      links.forEach((link: any, index: number) => {
        // Find the connection for this link
        const conn = connections.find((c: any) => c.link_id === link.meta?.id)
        
        if (conn) {
          // Reconstruct the link with proper structure
          const reconstructedLink: Link = {
            name: link.name,
            length: link.length,
            n_iterations: link.n_iterations ?? 24,
            has_fixed: link.has_fixed ?? false,
            target_length: link.target_length ?? null,
            target_cost_func: link.target_cost_func ?? null,
            fixed_loc: link.fixed_loc ?? null,
            has_constraint: link.has_constraint ?? false,
            is_driven: link.is_driven ?? false,
            is_ground: link.is_ground ?? false,  // Ground link support
            flip: link.flip ?? false,
            zlevel: link.zlevel ?? 0,
            meta: link.meta || {
              id: link.id || `link_${Date.now()}_${index}`,
              start_point: link.start_point || [0, 0],
              end_point: link.end_point || [0, 0],
              color: link.color || getDefaultColor(index)
            }
          }
          graphManager.addConnection(conn.from_node, conn.to_node, reconstructedLink)
        }
      })
    }
    
    // Update counters
    setNodeCounter(nodes?.length || 0)
    setLinkCounter(links?.length || 0)
    
    // Close dialog
    setPylinkageDialogOpen(false)
    
    setStatusMessage('Demo 4-bar loaded into builder!')
    setTimeout(() => setStatusMessage(''), 3000)
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
        <Button variant="contained" onClick={saveGraph} color="primary">
          Save Graph
        </Button>
        <Button variant="contained" onClick={computeGraph} color="success">
          Compute Graph
        </Button>
        <Button variant="outlined" onClick={loadLastSavedGraph} color="primary">
          Load Last Saved
        </Button>
        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
        {/* Pylinkage Integration Buttons */}
        <Button 
          variant="contained" 
          onClick={runDemo4Bar}
          disabled={pylinkageLoading}
          color="info"
          size="small"
          title="Run a demo 4-bar linkage using native pylinkage API"
        >
          {pylinkageLoading ? '...' : '🎯 Demo 4-Bar'}
        </Button>
        <Button 
          variant="outlined" 
          onClick={convertToPylinkage}
          disabled={pylinkageLoading || graphManager.graphState.links.length === 0}
          color="secondary"
          size="small"
          title="Test conversion to pylinkage format"
        >
          {pylinkageLoading ? '...' : '🔄 Test Pylinkage'}
        </Button>
        <Button 
          variant="contained" 
          onClick={simulateWithPylinkage}
          disabled={pylinkageLoading || graphManager.graphState.links.length === 0}
          color="secondary"
          size="small"
          title="Run simulation using pylinkage solver"
        >
          {pylinkageLoading ? '...' : '⚡ Pylinkage Sim'}
        </Button>
        <Button 
          variant="outlined" 
          onClick={compareSolvers}
          disabled={pylinkageLoading || graphManager.graphState.links.length === 0}
          color="secondary"
          size="small"
          title="Compare automata vs pylinkage solvers"
        >
          {pylinkageLoading ? '...' : '📊 Compare'}
        </Button>
        <Divider orientation="vertical" flexItem sx={{ mx: 1 }} />
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
          {/* Grid lines to show 100\" x 100\" workspace */}
          <svg
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 0
            }}
          >
            {/* Major grid lines every 10 units */}
            {Array.from({ length: 11 }, (_, i) => i * 10).map(unit => (
              <g key={`grid-${unit}`}>
                {/* Vertical lines */}
                <line
                  x1={unitsToPixels(unit)}
                  y1={0}
                  x2={unitsToPixels(unit)}
                  y2={CANVAS_HEIGHT_PX}
                  stroke="#ddd"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
                {/* Horizontal lines */}
                <line
                  x1={0}
                  y1={unitsToPixels(unit)}
                  x2="100%"
                  y2={unitsToPixels(unit)}
                  stroke="#ddd"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
                {/* Labels */}
                {unit > 0 && (
                  <>
                    <text
                      x={unitsToPixels(unit) + 2}
                      y={12}
                      fontSize="10"
                      fill="#999"
                    >
                      {unit}"
                    </text>
                    <text
                      x={2}
                      y={unitsToPixels(unit) - 2}
                      fontSize="10"
                      fill="#999"
                    >
                      {unit}"
                    </text>
                  </>
                )}
              </g>
            ))}
          </svg>
          
          {/* Render current click point */}
          {currentClick && (
            <Box
              sx={{
                position: 'absolute',
                left: unitsToPixels(currentClick.x) - 4,
                top: unitsToPixels(currentClick.y) - 4,
                width: 8,
                height: 8,
                backgroundColor: '#ff0000',
                borderRadius: '50%'
              }}
            />
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
                  left: unitsToPixels(node.pos[0]) - (isHovered || isDragging ? 8 : 6),
                  top: unitsToPixels(node.pos[1]) - (isHovered || isDragging ? 8 : 6),
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
            if (!link.meta.start_point || !link.meta.end_point) return null
            
            const isHovered = hoveredItem?.type === 'link' && hoveredItem.id === link.meta.id
            
            // Convert link points from units to pixels for rendering
            const startPx = [unitsToPixels(link.meta.start_point[0]), unitsToPixels(link.meta.start_point[1])]
            const endPx = [unitsToPixels(link.meta.end_point[0]), unitsToPixels(link.meta.end_point[1])]
            
            // Calculate arrow direction
            const dx = endPx[0] - startPx[0]
            const dy = endPx[1] - startPx[1]
            const length = Math.sqrt(dx * dx + dy * dy)
            
            // Skip rendering if link has zero length to avoid division by zero
            if (length < 0.1) return null
            
            const unitX = dx / length
            const unitY = dy / length
            
            // Arrow head position (slightly before end point)
            const arrowX = endPx[0] - unitX * 15
            const arrowY = endPx[1] - unitY * 15
            
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
                  x1={startPx[0]}
                  y1={startPx[1]}
                  x2={endPx[0]}
                  y2={endPx[1]}
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
                  points={`${endPx[0]},${endPx[1]} ${arrowX - unitY * (isHovered ? 7 : 5)},${arrowY + unitX * (isHovered ? 7 : 5)} ${arrowX + unitY * (isHovered ? 7 : 5)},${arrowY - unitX * (isHovered ? 7 : 5)}`}
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
          
          {/* Preview line while creating a link - rendered last so it appears on top */}
          {previewLine && (
            <svg
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 50
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
                  const isHovered = hoveredItem?.type === 'link' && hoveredItem.id === link.meta.id
                  return (
                    <ListItem 
                      key={index}
                      onMouseEnter={() => setHoveredItem({ type: 'link', id: link.meta.id })}
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
                          L: {typeof link.length === 'number' ? link.length.toFixed(1) : 'N/A'}" | Z: {link.zlevel || 0} {link.has_fixed ? '| F' : ''}{link.has_constraint ? '| C' : ''}{link.flip ? '| Fl' : ''}
                        </span>
                      }
                    />
                    <IconButton 
                      size="small" 
                      onClick={() => handleDeleteClick('link', link.meta.id, link.name || `Link ${index + 1}`)}
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
                              pos: ({node.pos[0].toFixed(1)}", {node.pos[1].toFixed(1)}") | C:{node.connections.length}
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

      {/* Path Visualization Component */}
      {pathData && (
        <Box sx={{ mt: 3 }}>
          <PathVisualization pathData={pathData} />
        </Box>
      )}

      {/* Comprehensive Edit Dialog */}
      <Dialog open={editDialog} onClose={() => setEditDialog(false)} maxWidth="md" fullWidth>
        <DialogTitle>
          Edit Link Properties
          {selectedLink && (
            <Typography variant="subtitle2" component="span" color="text.secondary" sx={{ display: 'block' }}>
              Link ID: {selectedLink.meta.id}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={3} sx={{ pt: 1 }}>
            {/* Current State from Canvas */}
            <Grid item xs={10}>
              <Typography variant="h6" gutterBottom sx={{ color: '#1976d2', fontWeight: 'bold' }}>
                Current State (from Canvas)
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                <Chip 
                  label={`Length: ${graphManager.graphState.links.find(l => l.meta.id === selectedLink?.meta.id)?.length?.toFixed(2) || selectedLink?.length?.toFixed(2) || 'N/A'}" `} 
                  variant="filled" 
                  size="small"
                  color="info"
                  sx={{ fontWeight: 'bold' }}
                />
              </Box>
              <Typography variant="caption" sx={{ color: '#666', display: 'block', mb: 1 }}>
                ℹ️ Length is calculated from the link's node positions. Drag nodes on the canvas to change it.
              </Typography>
              <Divider sx={{ mb: 2 }} />
            </Grid>

            {/* Immutable Properties */}
            <Grid item xs={10}>
              <Typography variant="h6" gutterBottom>
                Immutable Properties
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mb: 1, flexWrap: 'wrap' }}>
                <Chip 
                  label={`ID: ${selectedLink?.meta.id || 'N/A'}`} 
                  variant="outlined" 
                  size="small"
                />
                <Chip 
                  label={`Iterations: ${selectedLink?.n_iterations || 24}`} 
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
      <Dialog
        open={deleteDialog}
        onClose={() => setDeleteDialog(false)}
        PaperProps={{
          sx: {
            bgcolor: 'rgba(255,255,255,0.9)',
            backdropFilter: 'blur(2px)',
            boxShadow: '0 8px 24px rgba(0,0,0,0.25)'
          }
        }}
        slotProps={{
          backdrop: {
            sx: {
              bgcolor: 'rgba(0,0,0,0.2)'
            }
          }
        }}
      >
        <DialogTitle>Confirm Delete</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete {itemToDelete?.type} "{itemToDelete?.name}"? This action cannot be undone.
          </DialogContentText>
          {itemToDelete?.type === 'node' && (() => {
            const connectedConns = graphManager.graphState.connections.filter(
              conn => conn.from_node === itemToDelete.id || conn.to_node === itemToDelete.id
            )
            if (connectedConns.length === 0) return null
            return (
              <Box sx={{ mt: 2 }}>
                <Alert
                  severity="warning"
                  sx={{
                    mb: 1,
                    bgcolor: 'rgba(255, 193, 7, 0.15)',
                    color: '#8a6d1f',
                    border: '1px solid rgba(255, 193, 7, 0.35)',
                    '& .MuiAlert-icon': { color: '#8a6d1f' }
                  }}
                >
                  This node is connected to {connectedConns.length} link(s). Deleting it will also remove these links:
                </Alert>
                <List dense>
                  {connectedConns.map(conn => {
                    const link = graphManager.getLinkForConnection(conn)
                    return (
                      <ListItem key={conn.id} sx={{ py: 0.25 }}>
                        <ListItemText
                          primary={link?.name || conn.link_id}
                          secondary={`${conn.from_node} → ${conn.to_node}`}
                          primaryTypographyProps={{ fontSize: '0.85rem' }}
                          secondaryTypographyProps={{ fontSize: '0.75rem' }}
                        />
                      </ListItem>
                    )
                  })}
                </List>
              </Box>
            )
          })()}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(false)}>Cancel</Button>
          <Button onClick={confirmDelete} color="error" variant="contained">
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Pylinkage Results Dialog */}
      <Dialog
        open={pylinkageDialogOpen}
        onClose={() => setPylinkageDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ 
          bgcolor: pylinkageResult?.status === 'success' ? '#e8f5e9' : '#ffebee',
          borderBottom: '1px solid #ddd'
        }}>
          {pylinkageResult?.status === 'success' ? '✓ ' : '✗ '}
          Pylinkage {pylinkageResult?.comparison ? 'Comparison' : pylinkageResult?.path_data ? 'Simulation' : 'Conversion'} Results
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          {pylinkageResult && (
            <Box>
              {/* Status */}
              <Alert 
                severity={pylinkageResult.status === 'success' ? 'success' : 'error'}
                sx={{ mb: 2 }}
              >
                {pylinkageResult.message || (pylinkageResult.status === 'success' ? 'Operation completed successfully' : 'Operation failed')}
              </Alert>

              {/* Conversion Stats */}
              {pylinkageResult.conversion_result?.stats && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Conversion Statistics:</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {Object.entries(pylinkageResult.conversion_result.stats).map(([key, value]) => (
                      <Chip 
                        key={key} 
                        label={`${key.replace(/_/g, ' ')}: ${value}`}
                        size="small"
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Simulation Stats */}
              {pylinkageResult.execution_time_ms !== undefined && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Simulation Performance:</Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip 
                      label={`Execution: ${pylinkageResult.execution_time_ms?.toFixed(2)}ms`}
                      size="small"
                      color="primary"
                    />
                    <Chip 
                      label={`Solver: ${pylinkageResult.solver || 'pylinkage'}`}
                      size="small"
                      variant="outlined"
                    />
                    <Chip 
                      label={`Iterations: ${pylinkageResult.n_iterations || 24}`}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </Box>
              )}

              {/* Comparison Results */}
              {pylinkageResult.comparison && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Solver Comparison:</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1 }}>
                          <Typography variant="caption" color="text.secondary">Automata Solver</Typography>
                          <Typography variant="h6">
                            {pylinkageResult.automata_result?.success ? '✓' : '✗'} {pylinkageResult.automata_result?.time_ms?.toFixed(2) || 'N/A'}ms
                          </Typography>
                          {pylinkageResult.automata_result?.error && (
                            <Typography variant="caption" color="error">
                              {String(pylinkageResult.automata_result.error).substring(0, 100)}
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                    <Grid item xs={6}>
                      <Card variant="outlined">
                        <CardContent sx={{ py: 1 }}>
                          <Typography variant="caption" color="text.secondary">Pylinkage Solver</Typography>
                          <Typography variant="h6">
                            {pylinkageResult.pylinkage_result?.success ? '✓' : '✗'} {pylinkageResult.pylinkage_result?.time_ms?.toFixed(2) || 'N/A'}ms
                          </Typography>
                          {pylinkageResult.pylinkage_result?.error && (
                            <Typography variant="caption" color="error">
                              {String(pylinkageResult.pylinkage_result.error).substring(0, 100)}
                            </Typography>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                  {pylinkageResult.comparison.speedup_factor && (
                    <Box sx={{ mt: 2, textAlign: 'center' }}>
                      <Chip 
                        label={`Speedup: ${pylinkageResult.comparison.speedup_factor?.toFixed(2)}x`}
                        color={pylinkageResult.comparison.speedup_factor > 1 ? 'success' : 'warning'}
                      />
                      {pylinkageResult.comparison.max_position_error !== undefined && (
                        <Chip 
                          label={`Max Error: ${pylinkageResult.comparison.max_position_error?.toFixed(4)}`}
                          sx={{ ml: 1 }}
                          variant="outlined"
                        />
                      )}
                    </Box>
                  )}
                </Box>
              )}

              {/* 4-Bar Metadata (from demo) */}
              {pylinkageResult.metadata && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="primary">
                    Pylinkage Structure Explanation:
                  </Typography>
                  <Alert severity="info" sx={{ mb: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
                      Key Insight: A 4-bar in pylinkage uses only 2 joints!
                    </Typography>
                    <Typography variant="body2">
                      • <strong>Crank</strong>: Rotating driver (angle + distance from ground)
                    </Typography>
                    <Typography variant="body2">
                      • <strong>Revolute</strong>: The coupler-rocker <em>connection point</em>, not a link!
                    </Typography>
                    <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
                      Revolute.distance0 = coupler length, Revolute.distance1 = rocker length
                    </Typography>
                  </Alert>
                  
                  {pylinkageResult.metadata.parameters && (
                    <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 1 }}>
                      {Object.entries(pylinkageResult.metadata.parameters).map(([key, value]) => (
                        <Chip 
                          key={key} 
                          label={`${key}: ${Array.isArray(value) ? `[${value.join(', ')}]` : value}`}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  )}
                  
                  {pylinkageResult.metadata.joints && (
                    <Box sx={{ bgcolor: '#f5f5f5', p: 1, borderRadius: 1, fontFamily: 'monospace', fontSize: '0.75rem' }}>
                      <Typography variant="caption" sx={{ fontWeight: 'bold' }}>Joint Structure:</Typography>
                      {Object.entries(pylinkageResult.metadata.joints).map(([name, joint]: [string, any]) => (
                        <Box key={name} sx={{ ml: 1 }}>
                          <strong>{name}</strong> ({joint.type}): {joint.represents}
                          {joint.length && <span> | length={joint.length}</span>}
                          {joint.distance0 && <span> | dist0={joint.distance0}</span>}
                          {joint.distance1 && <span> | dist1={joint.distance1}</span>}
                        </Box>
                      ))}
                    </Box>
                  )}
                </Box>
              )}

              {/* Warnings */}
              {(pylinkageResult.conversion_result?.warnings?.length > 0 || pylinkageResult.conversion_warnings?.length > 0) && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="warning.main">Warnings:</Typography>
                  <List dense>
                    {(pylinkageResult.conversion_result?.warnings || pylinkageResult.conversion_warnings || []).map((warning: string, i: number) => (
                      <ListItem key={i} sx={{ py: 0 }}>
                        <ListItemText 
                          primary={`⚠️ ${warning}`}
                          primaryTypographyProps={{ fontSize: '0.85rem', color: 'warning.main' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}

              {/* Errors */}
              {(pylinkageResult.conversion_result?.errors?.length > 0 || pylinkageResult.errors) && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom color="error">Errors:</Typography>
                  <List dense>
                    {(pylinkageResult.conversion_result?.errors || 
                      (Array.isArray(pylinkageResult.errors) ? pylinkageResult.errors : [pylinkageResult.errors]) || 
                      []).map((error: string, i: number) => (
                      <ListItem key={i} sx={{ py: 0 }}>
                        <ListItemText 
                          primary={`❌ ${error}`}
                          primaryTypographyProps={{ fontSize: '0.85rem', color: 'error' }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}

              {/* Joint Mapping */}
              {pylinkageResult.conversion_result?.joint_mapping && Object.keys(pylinkageResult.conversion_result.joint_mapping).length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>Joint Mapping:</Typography>
                  <Box sx={{ 
                    maxHeight: 150, 
                    overflow: 'auto', 
                    bgcolor: '#f5f5f5', 
                    p: 1, 
                    borderRadius: 1,
                    fontFamily: 'monospace',
                    fontSize: '0.75rem'
                  }}>
                    {Object.entries(pylinkageResult.conversion_result.joint_mapping).map(([key, value]) => (
                      <Box key={key}>
                        {key} → {String(value)}
                      </Box>
                    ))}
                  </Box>
                </Box>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          {pylinkageResult?.ui_graph && (
            <Button 
              onClick={() => loadUiGraphIntoBuilder(pylinkageResult.ui_graph)}
              variant="contained"
              color="primary"
              sx={{ mr: 'auto' }}
            >
              📥 Load into Builder
            </Button>
          )}
          <Button onClick={() => setPylinkageDialogOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default GraphBuilderTab
