import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  Box, Typography, Paper, IconButton, Modal, TextField,
  FormControl, InputLabel, Select, MenuItem, Button, Chip, Divider,
  FormControlLabel, Switch, Tooltip
} from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import { graphColors, statusColors, colors, jointColors } from '../theme'
import acinonyxLogo from '../assets/acinonyx_logo.png'
import { Edge, EdgeId, NodeId } from '../types'


// Threshold distance for merge detection (in units)
export const MERGE_THRESHOLD = 4.0

// ═══════════════════════════════════════════════════════════════════════════════
// DRAGGABLE TOOLBAR COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export interface ToolbarPosition {
  x: number
  y: number
}

export interface DraggableToolbarProps {
  id: string
  title: string
  icon: string
  children: React.ReactNode
  initialPosition?: ToolbarPosition
  onClose: () => void
  onPositionChange?: (id: string, position: ToolbarPosition) => void
  onInteract?: () => void  // Called when user clicks/interacts with toolbar
  minWidth?: number
  maxHeight?: number
}

export const DraggableToolbar: React.FC<DraggableToolbarProps> = ({
  id,
  title,
  icon,
  children,
  initialPosition = { x: 100, y: 100 },
  onClose,
  onPositionChange,
  onInteract,
  minWidth = 200,
  maxHeight = 400
}) => {
  const [position, setPosition] = useState<ToolbarPosition>(initialPosition)
  const [isDragging, setIsDragging] = useState(false)
  const dragOffset = useRef<{ x: number; y: number }>({ x: 0, y: 0 })
  const toolbarRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('.toolbar-close')) return
    e.preventDefault()
    setIsDragging(true)
    dragOffset.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    }
  }, [position])

  useEffect(() => {
    if (!isDragging) return

    const handleMouseMove = (e: MouseEvent) => {
      const newX = e.clientX - dragOffset.current.x
      const newY = e.clientY - dragOffset.current.y
      // Keep within reasonable bounds
      const boundedX = Math.max(0, newX)
      const boundedY = Math.max(0, newY)
      setPosition({ x: boundedX, y: boundedY })
    }

    const handleMouseUp = () => {
      setIsDragging(false)
      if (onPositionChange) {
        onPositionChange(id, position)
      }
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)

    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isDragging, id, position, onPositionChange])

  return (
    <Paper
      ref={toolbarRef}
      elevation={6}
      onMouseDownCapture={() => onInteract?.()}
      sx={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        minWidth,
        maxHeight,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        backgroundColor: 'rgba(255, 255, 255, 0.98)',
        backdropFilter: 'blur(12px)',
        borderRadius: 3,
        border: '1px solid rgba(0,0,0,0.1)',
        boxShadow: isDragging
          ? '0 12px 40px rgba(0,0,0,0.2)'
          : '0 4px 20px rgba(0,0,0,0.12)',
        transition: isDragging ? 'none' : 'box-shadow 0.2s ease',
        zIndex: isDragging ? 1400 : 1300,
        userSelect: 'none'
      }}
    >
      {/* Title bar - draggable */}
      <Box
        onMouseDown={handleMouseDown}
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1.5,
          py: 1,
          backgroundColor: 'rgba(0,0,0,0.03)',
          borderBottom: '1px solid rgba(0,0,0,0.08)',
          cursor: isDragging ? 'grabbing' : 'grab'
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography sx={{ fontSize: '1rem' }}>{icon}</Typography>
          <Typography sx={{ fontSize: '0.85rem', fontWeight: 600, color: 'text.primary' }}>
            {title}
          </Typography>
        </Box>
        <IconButton
          className="toolbar-close"
          size="small"
          onClick={onClose}
          sx={{
            width: 24,
            height: 24,
            color: 'text.secondary',
            '&:hover': {
              backgroundColor: 'rgba(211, 47, 47, 0.1)',
              color: '#d32f2f'
            }
          }}
        >
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Box>

      {/* Content area - userSelect unset so range/inputs work; sliders stop propagation in their own wrappers */}
      <Box sx={{ overflow: 'auto', flex: 1, userSelect: 'unset' }}>
        {children}
      </Box>
    </Paper>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOOLBAR TOGGLE BUTTONS (sidebar)
// ═══════════════════════════════════════════════════════════════════════════════

export interface ToolbarConfig {
  id: string
  title: string
  icon: string
  defaultPosition: ToolbarPosition
}

// Default toolbar positions - positioned for optimal workspace layout
// Tools & More: FULL LEFT ALIGNED, Tools below toggle buttons, More well below Tools
// Links & Nodes: far right edge (negative x = offset from right), stacked vertically
// Settings: gear icon, opens settings panel
// Optimize: dedicated optimization panel
export const TOOLBAR_CONFIGS: ToolbarConfig[] = [
  { id: 'tools', title: 'Tools', icon: '⚒', defaultPosition: { x: 8, y: 60 } },        // Full left, below toggle buttons
  { id: 'more', title: 'More', icon: '≡', defaultPosition: { x: 8, y: 370 } },         // Full left, well below Tools
  { id: 'optimize', title: 'Optimize', icon: '✦', defaultPosition: { x: 8, y: -630 } }, // Bottom left (negative y = from bottom)
  { id: 'links', title: 'Links', icon: '—', defaultPosition: { x: -220, y: 8 } },      // Far right edge (negative = from right)
  { id: 'nodes', title: 'Nodes', icon: '○', defaultPosition: { x: -220, y: 500 } },    // Below Links on far right
  { id: 'settings', title: 'Settings', icon: '⚙', defaultPosition: { x: -280, y: 12 } } // Upper right, near top of canvas
]

export interface ToolbarToggleButtonsProps {
  openToolbars: Set<string>
  onToggleToolbar: (id: string) => void
  darkMode?: boolean
  onInteract?: () => void  // Called when user clicks/interacts with toolbar buttons
}

/**
 * ToolbarToggleButtonsContainer - The horizontal button bar for toggling toolbars
 *
 * Contains: Tools, Links, Nodes, More buttons in a horizontal row
 * Position: Upper left of the canvas
 */
export const ToolbarToggleButtons: React.FC<ToolbarToggleButtonsProps> = ({
  openToolbars,
  onToggleToolbar,
  darkMode = false,
  onInteract
}) => {
  return (
    <Box
      id="toolbar-toggle-buttons-container"
      className="toolbar-toggle-buttons-container"
      onMouseDownCapture={() => onInteract?.()}
      sx={{
        position: 'absolute',
        left: 8,
        top: 8,
        display: 'flex',
        flexDirection: 'row',  // Horizontal layout
        gap: 0.75,
        zIndex: 1200,
        // Subtle container styling - dark mode aware
        backgroundColor: darkMode ? 'rgba(40, 40, 40, 0.9)' : 'rgba(255, 255, 255, 0.7)',
        backdropFilter: 'blur(8px)',
        borderRadius: 2,
        padding: '6px 8px',
        border: darkMode ? '1px solid rgba(255, 255, 255, 0.1)' : '1px solid rgba(0, 0, 0, 0.08)',
        boxShadow: darkMode ? '0 2px 12px rgba(0, 0, 0, 0.3)' : '0 2px 12px rgba(0, 0, 0, 0.06)'
      }}
    >
      {TOOLBAR_CONFIGS.map(config => {
        const isOpen = openToolbars.has(config.id)
        return (
          <IconButton
            key={config.id}
            onClick={() => onToggleToolbar(config.id)}
            sx={{
              width: 36,
              height: 36,
              borderRadius: 1.5,
              fontSize: '1.1rem',
              backgroundColor: isOpen
                ? 'primary.main'
                : darkMode ? 'rgba(60, 60, 60, 0.9)' : 'rgba(255,255,255,0.9)',
              color: isOpen
                ? 'white'
                : darkMode ? '#e0e0e0' : 'text.primary',
              border: '1px solid',
              borderColor: isOpen
                ? 'primary.main'
                : darkMode ? 'rgba(255, 255, 255, 0.15)' : 'rgba(0,0,0,0.1)',
              boxShadow: isOpen ? '0 2px 6px rgba(250, 129, 18, 0.3)' : 'none',
              transition: 'all 0.15s ease',
              '&:hover': {
                backgroundColor: isOpen
                  ? 'primary.dark'
                  : darkMode ? 'rgba(250, 129, 18, 0.2)' : 'rgba(250, 129, 18, 0.1)',
                borderColor: 'primary.main',
                transform: 'translateY(-1px)'
              }
            }}
            title={config.title}
          >
            {config.icon}
          </IconButton>
        )
      })}
    </Box>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export type ToolMode =
  | 'select'
  | 'group_select'
  | 'mechanism_select'
  | 'draw_link'
  | 'add_joint'
  | 'draw_polygon'
  | 'merge'
  | 'measure'
  | 'delete'
  | 'draw_path'

export interface ToolInfo {
  id: ToolMode
  label: string
  icon: string
  description: string
  shortcut?: string
}

export const TOOLS: ToolInfo[] = [
  // Row 1: Core editing tools
  {
    id: 'draw_link',
    label: 'Create Link',
    icon: '╱',
    description: 'Click two points to create a new link between them.',
    shortcut: 'C'
  },
  {
    id: 'select',
    label: 'Select',
    icon: '⎚',
    description: 'Click to select individual joints or links. Drag to move.',
    shortcut: 'S'
  },
  {
    id: 'delete',
    label: 'Delete',
    icon: '⌫',
    description: 'Click a joint or link to delete it. Deleting a link removes orphan nodes. Deleting a node removes connected links.',
    shortcut: 'X'
  },
  // Row 2: Selection modes
  {
    id: 'group_select',
    label: 'Group Select',
    icon: '⊡',
    description: 'Drag a box to select multiple elements at once.',
    shortcut: 'G'
  },
  {
    id: 'mechanism_select',
    label: 'Mechanism Select',
    icon: '⚙',
    description: 'Click any element to select its entire connected mechanism.',
    shortcut: 'M'
  },
  {
    id: 'measure',
    label: 'Measure',
    icon: '⌗',
    description: 'Click two points to measure the distance between them.',
    shortcut: 'R'
  },
  // Row 3: Drawing tools
  {
    id: 'draw_polygon',
    label: 'Draw Polygon',
    icon: '⬡',
    description: 'Click multiple points to create a polygon shape. Double-click to close.',
    shortcut: 'P'
  },
  {
    id: 'merge',
    label: 'Merge Polygon',
    icon: '⋒',
    description: 'Merge a polygon with an enclosed link, or click a merged polygon to unmerge it',
    shortcut: 'E'
  },
  {
    id: 'draw_path',
    label: 'Draw Trajectory',
    icon: '⌇',
    description: 'Draw a target trajectory for optimization. Click to add points, double-click or press Enter to finish.',
    shortcut: 'T'
  }
  // TODO: Re-enable Add Joint feature later
  // {
  //   id: 'add_joint',
  //   label: 'Add Joint',
  //   icon: '⊕',
  //   description: 'Click on an existing link to add a joint at that position.',
  //   shortcut: 'J'
  // },
]

// ═══════════════════════════════════════════════════════════════════════════════
// STATUS MESSAGE TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export type StatusType = 'info' | 'action' | 'success' | 'warning' | 'error'

export interface StatusMessage {
  text: string
  type: StatusType
  timestamp: number
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINK CREATION STATE
// ═══════════════════════════════════════════════════════════════════════════════

export interface LinkCreationState {
  isDrawing: boolean
  startPoint: [number, number] | null
  startJointName: string | null  // If started from an existing joint
  endPoint: [number, number] | null
}

export const initialLinkCreationState: LinkCreationState = {
  isDrawing: false,
  startPoint: null,
  startJointName: null,
  endPoint: null
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAG STATE - For drag/drop/move/merge functionality
// ═══════════════════════════════════════════════════════════════════════════════

export interface DragState {
  isDragging: boolean
  draggedJoint: string | null          // Name of joint being dragged
  dragStartPosition: [number, number] | null  // Original position before drag
  currentPosition: [number, number] | null    // Current drag position
  mergeTarget: string | null           // Joint we're hovering over for merge
  mergeProximity: number               // Distance to merge target (for visual feedback)
}

export const initialDragState: DragState = {
  isDragging: false,
  draggedJoint: null,
  dragStartPosition: null,
  currentPosition: null,
  mergeTarget: null,
  mergeProximity: Infinity
}

// ═══════════════════════════════════════════════════════════════════════════════
// GROUP SELECTION STATE
// ═══════════════════════════════════════════════════════════════════════════════

export interface GroupSelectionState {
  isSelecting: boolean
  startPoint: [number, number] | null
  currentPoint: [number, number] | null
}

export const initialGroupSelectionState: GroupSelectionState = {
  isSelecting: false,
  startPoint: null,
  currentPoint: null
}

// ═══════════════════════════════════════════════════════════════════════════════
// POLYGON DRAWING STATE
// ═══════════════════════════════════════════════════════════════════════════════

export interface PolygonDrawState {
  isDrawing: boolean
  points: [number, number][]  // Points for the polygon (not joints, just coordinates)
}

export const initialPolygonDrawState: PolygonDrawState = {
  isDrawing: false,
  points: []
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEASURE TOOL STATE
// ═══════════════════════════════════════════════════════════════════════════════

export interface MeasureState {
  isMeasuring: boolean
  startPoint: [number, number] | null
  endPoint: [number, number] | null
  measurementId: number  // For animating fade out
}

export const initialMeasureState: MeasureState = {
  isMeasuring: false,
  startPoint: null,
  endPoint: null,
  measurementId: 0
}

export interface MeasurementMarker {
  id: number
  point: [number, number]
  timestamp: number
}

// ═══════════════════════════════════════════════════════════════════════════════
// TARGET PATH DRAWING STATE - For trajectory optimization
// ═══════════════════════════════════════════════════════════════════════════════

export interface TargetPath {
  id: string
  name: string
  points: [number, number][]  // User-drawn path points
  targetJoint: string | null  // Which joint should follow this path
  color: string
  isComplete: boolean
  // Optional: Stored TargetTrajectory from preprocessing (for optimizer)
  targetTrajectory?: {
    joint_name: string
    positions: number[][]
    n_steps: number
    weights?: number[]
  }
}

export interface PathDrawState {
  isDrawing: boolean
  points: [number, number][]  // Points being drawn
}

export const initialPathDrawState: PathDrawState = {
  isDrawing: false,
  points: []
}

export const createTargetPath = (
  points: [number, number][],
  existingPaths: TargetPath[]
): TargetPath => {
  // Generate unique ID
  let counter = 1
  let id = `target_path_${counter}`
  while (existingPaths.some(p => p.id === id)) {
    counter++
    id = `target_path_${counter}`
  }

  return {
    id,
    name: `Target Path ${counter}`,
    points,
    targetJoint: null,
    color: '#e91e63',  // Pink for target paths
    isComplete: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAWN OBJECT TYPES - For shapes/polygons that can be attached to links
// ═══════════════════════════════════════════════════════════════════════════════

export type DrawnObjectType = 'polygon' | 'path' | 'rectangle' | 'ellipse'

export interface DrawnObjectAttachment {
  linkName: string              // The link this object is attached to
  parameterT: number            // Position along the link (0-1)
  relativeAngle: number         // Angle relative to the link direction
  offset: [number, number]      // Offset from the attachment point
}

export interface DrawnObject {
  id: string
  type: DrawnObjectType
  name: string
  points: [number, number][]     // Vertices for polygon/path (original positions at creation/merge time)
  fillColor: string
  strokeColor: string
  strokeWidth: number
  fillOpacity: number
  closed: boolean                // Whether the shape is closed (polygon) or open (path)
  attachment?: DrawnObjectAttachment  // If attached to a link, this defines the relationship
  mergedLinkName?: string        // If merged with a link, the link's name
  // Rigid attachment: store link positions at merge time for transformation
  mergedLinkOriginalStart?: [number, number]  // Link start position when merged
  mergedLinkOriginalEnd?: [number, number]    // Link end position when merged
  metadata?: Record<string, unknown>  // For future extensibility
}

// ═══════════════════════════════════════════════════════════════════════════════
// MERGE POLYGON STATE
// ═══════════════════════════════════════════════════════════════════════════════

export interface MergePolygonState {
  step: 'idle' | 'awaiting_selection' | 'polygon_selected' | 'link_selected'
  selectedPolygonId: string | null   // The polygon selected for merging
  selectedLinkName: string | null    // The link selected for merging
}

export const initialMergePolygonState: MergePolygonState = {
  step: 'idle',
  selectedPolygonId: null,
  selectedLinkName: null
}

export const createDrawnObject = (
  type: DrawnObjectType,
  points: [number, number][],
  existingIds: string[]
): DrawnObject => {
  // Generate unique ID
  let counter = 1
  let id = `${type}_${counter}`
  while (existingIds.includes(id)) {
    counter++
    id = `${type}_${counter}`
  }

  return {
    id,
    type,
    name: id,
    points,
    fillColor: 'rgba(128, 128, 128, 0.3)',  // Transparent grey
    strokeColor: '#666',
    strokeWidth: 2,
    fillOpacity: 0.3,
    closed: type === 'polygon' || type === 'rectangle' || type === 'ellipse'
  }
}

export interface DrawnObjectsState {
  objects: DrawnObject[]
  selectedIds: string[]  // Changed to array for multi-select
}

export const initialDrawnObjectsState: DrawnObjectsState = {
  objects: [],
  selectedIds: []
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOVE MODE STATE - For moving groups of selected elements
// ═══════════════════════════════════════════════════════════════════════════════

export interface MoveGroupState {
  isActive: boolean
  isDragging: boolean
  joints: string[]                              // Joint names being moved
  drawnObjectIds: string[]                      // DrawnObject IDs being moved
  startPositions: Record<string, [number, number]>  // Original positions of joints
  drawnObjectStartPositions: Record<string, [number, number][]>  // Original positions of drawn object points
  dragStartPoint: [number, number] | null       // Where the drag started
}

export const initialMoveGroupState: MoveGroupState = {
  isActive: false,
  isDragging: false,
  joints: [],
  drawnObjectIds: [],
  startPositions: {},
  drawnObjectStartPositions: {},
  dragStartPoint: null
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONNECTED GRAPH/MECHANISM HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Find all joints connected to a given joint (directly or indirectly)
 * Uses BFS to traverse the graph through links
 */
export const findConnectedMechanism = (
  startJoint: string,
  links: Record<string, LinkMetaData>
): { joints: string[], links: string[] } => {
  const visitedJoints = new Set<string>()
  const visitedLinks = new Set<string>()
  const queue: string[] = [startJoint]

  while (queue.length > 0) {
    const currentJoint = queue.shift()!
    if (visitedJoints.has(currentJoint)) continue
    visitedJoints.add(currentJoint)

    // Find all links connected to this joint
    for (const [linkName, linkMeta] of Object.entries(links)) {
      if (linkMeta.connects.includes(currentJoint)) {
        visitedLinks.add(linkName)
        // Add the other joint(s) in this link to the queue
        for (const otherJoint of linkMeta.connects) {
          if (!visitedJoints.has(otherJoint)) {
            queue.push(otherJoint)
          }
        }
      }
    }
  }

  return {
    joints: Array.from(visitedJoints),
    links: Array.from(visitedLinks)
  }
}

/**
 * Find all joints and links within a rectangular selection box
 */
export const findElementsInBox = (
  box: { x1: number, y1: number, x2: number, y2: number },
  joints: Array<{ name: string; position: [number, number] | null }>,
  links: Array<{ name: string; start: [number, number] | null; end: [number, number] | null }>
): { joints: string[], links: string[] } => {
  const minX = Math.min(box.x1, box.x2)
  const maxX = Math.max(box.x1, box.x2)
  const minY = Math.min(box.y1, box.y2)
  const maxY = Math.max(box.y1, box.y2)

  const selectedJoints: string[] = []
  const selectedLinks: string[] = []

  // Check joints
  for (const joint of joints) {
    if (!joint.position) continue
    const [x, y] = joint.position
    if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
      selectedJoints.push(joint.name)
    }
  }

  // Check links - select if either endpoint is in the box
  for (const link of links) {
    if (!link.start || !link.end) continue
    const startInBox = link.start[0] >= minX && link.start[0] <= maxX &&
                       link.start[1] >= minY && link.start[1] <= maxY
    const endInBox = link.end[0] >= minX && link.end[0] <= maxX &&
                     link.end[1] >= minY && link.end[1] <= maxY
    if (startInBox || endInBox) {
      selectedLinks.push(link.name)
    }
  }

  return { joints: selectedJoints, links: selectedLinks }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE HELPER FUNCTIONS
// These work with the hypergraph Edge type from types/pylink.ts
// Reference: https://github.com/HugoFara/pylinkage/tree/main/src/pylinkage/hypergraph
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Check if an edge connects to a specific node
 */
export const edgeConnectsNode = (edge: Edge, nodeId: NodeId): boolean => {
  return edge.source === nodeId || edge.target === nodeId
}

/**
 * Check if a node has any connections (is connected to any edges)
 * @param nodeId - The ID of the node to check
 * @param edges - Record of edge IDs to their data
 * @returns true if the node is connected to at least one edge
 */
export const hasConnections = (
  nodeId: NodeId,
  edges: Record<EdgeId, Edge>
): boolean => {
  return Object.values(edges).some(edge => edgeConnectsNode(edge, nodeId))
}

/**
 * Check if a node is orphaned (has no connections to any edges)
 * @param nodeId - The ID of the node to check
 * @param edges - Record of edge IDs to their data
 * @returns true if the node has no connections
 */
export const isOrphan = (
  nodeId: NodeId,
  edges: Record<EdgeId, Edge>
): boolean => {
  return !hasConnections(nodeId, edges)
}

/**
 * Check if a joint is orphaned using legacy LinkMetaData format
 * @param jointName - The name of the joint to check
 * @param links - Record of link names to their metadata (legacy format)
 * @returns true if the joint has no connections
 */
export const isOrphanLegacy = (
  jointName: string,
  links: Record<string, LinkMetaData>
): boolean => {
  return !Object.values(links).some(link => link.connects.includes(jointName))
}

/**
 * Get all edges that connect to a specific node
 * @param nodeId - The ID of the node
 * @param edges - Record of edge IDs to their data
 * @returns Array of edge IDs that connect to this node
 */
export const getEdgesConnectedToNode = (
  nodeId: NodeId,
  edges: Record<EdgeId, Edge>
): EdgeId[] => {
  return Object.entries(edges)
    .filter(([_, edge]) => edgeConnectsNode(edge, nodeId))
    .map(([edgeId]) => edgeId)
}

/**
 * Get all nodes that an edge connects
 * @param edgeId - The ID of the edge
 * @param edges - Record of edge IDs to their data
 * @returns Array of node IDs that this edge connects, or empty array if edge not found
 */
export const getNodesConnectedByEdge = (
  edgeId: EdgeId,
  edges: Record<EdgeId, Edge>
): NodeId[] => {
  const edge = edges[edgeId]
  return edge ? [edge.source, edge.target] : []
}

/**
 * Get the connection count for a node (how many edges connect to it)
 * @param nodeId - The ID of the node
 * @param edges - Record of edge IDs to their data
 * @returns Number of edges connected to this node
 */
export const getConnectionCount = (
  nodeId: NodeId,
  edges: Record<EdgeId, Edge>
): number => {
  return getEdgesConnectedToNode(nodeId, edges).length
}

/**
 * Find all orphan nodes that would result from removing an edge
 * @param edgeId - The ID of the edge to be removed
 * @param edges - Record of edge IDs to their data
 * @returns Array of node IDs that would become orphaned
 */
export const findOrphansAfterEdgeRemoval = (
  edgeId: EdgeId,
  edges: Record<EdgeId, Edge>
): NodeId[] => {
  const edge = edges[edgeId]
  if (!edge) return []

  // Create a copy of edges without the target edge
  const remainingEdges = { ...edges }
  delete remainingEdges[edgeId]

  // Check each node the edge connects to see if it would become orphaned
  const connectedNodes = [edge.source, edge.target]
  return connectedNodes.filter(nodeId => isOrphan(nodeId, remainingEdges))
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY COMPATIBILITY (deprecated - will be removed after migration)
// These maintain compatibility with code still using the old LinkMetaData type
// ═══════════════════════════════════════════════════════════════════════════════

/** @deprecated Use Edge from types/pylink.ts instead */
export interface LinkMetaData {
  color: string
  connects: string[]  // Array of joint names this link connects
  isGround?: boolean  // True if this is a ground/anchored link
}

/** @deprecated Use getEdgesConnectedToNode */
export const getLinksConnectedToJoint = (
  jointName: string,
  links: Record<string, LinkMetaData>
): string[] => {
  return Object.entries(links)
    .filter(([_, linkMeta]) => linkMeta.connects.includes(jointName))
    .map(([linkName]) => linkName)
}

/** @deprecated Use getNodesConnectedByEdge */
export const getJointsConnectedByLink = (
  linkName: string,
  links: Record<string, LinkMetaData>
): string[] => {
  const link = links[linkName]
  return link ? [...link.connects] : []
}

/** @deprecated Use findOrphansAfterEdgeRemoval */
export const findOrphansAfterLinkRemoval = (
  linkName: string,
  links: Record<string, LinkMetaData>
): string[] => {
  const link = links[linkName]
  if (!link) return []

  // Create a copy of links without the target link
  const remainingLinks = { ...links }
  delete remainingLinks[linkName]

  // Check each joint the link connects to see if it would become orphaned
  return link.connects.filter(jointName =>
    !Object.values(remainingLinks).some(l => l.connects.includes(jointName))
  )
}

/**
 * Find all links that would be removed when deleting a joint (one degree out)
 * @param jointName - The name of the joint to be deleted
 * @param links - Record of link names to their metadata
 * @returns Array of link names that should be deleted
 */
export const findLinksToDeleteWithJoint = (
  jointName: string,
  links: Record<string, LinkMetaData>
): string[] => {
  return getLinksConnectedToJoint(jointName, links)
}

/**
 * Calculate the full deletion result for removing a link
 * Returns all elements that should be removed
 * @param linkName - The name of the link to delete
 * @param links - Record of link names to their metadata
 * @returns Object containing arrays of links and joints to delete
 */
export const calculateLinkDeletionResult = (
  linkName: string,
  links: Record<string, LinkMetaData>
): { linksToDelete: string[]; jointsToDelete: string[] } => {
  return {
    linksToDelete: [linkName],
    jointsToDelete: findOrphansAfterLinkRemoval(linkName, links)
  }
}

/**
 * Find all orphans that would result from removing multiple links
 * @param linkNames - Array of link names to be removed
 * @param links - Record of link names to their metadata
 * @param excludeJoints - Joints to exclude from orphan check (e.g., the joint being deleted)
 * @returns Array of joint names that would become orphaned
 */
export const findOrphansAfterMultipleLinkRemovals = (
  linkNames: string[],
  links: Record<string, LinkMetaData>,
  excludeJoints: string[] = []
): string[] => {
  // Create a copy of links without the target links
  const remainingLinks = { ...links }
  linkNames.forEach(linkName => delete remainingLinks[linkName])

  // Collect all joints that were connected to the removed links
  const affectedJoints = new Set<string>()
  linkNames.forEach(linkName => {
    const link = links[linkName]
    if (link) {
      link.connects.forEach(jointName => {
        if (!excludeJoints.includes(jointName)) {
          affectedJoints.add(jointName)
        }
      })
    }
  })

  // Check which affected joints would become orphaned
  return Array.from(affectedJoints).filter(jointName => isOrphanLegacy(jointName, remainingLinks))
}

/**
 * Calculate the full deletion result for removing a joint
 * Deletes the joint and all directly connected links (one degree out)
 * Also deletes any joints that become orphaned as a result of the link deletions
 * @param jointName - The name of the joint to delete
 * @param links - Record of link names to their metadata
 * @returns Object containing arrays of links and joints to delete
 */
export const calculateJointDeletionResult = (
  jointName: string,
  links: Record<string, LinkMetaData>
): { linksToDelete: string[]; jointsToDelete: string[] } => {
  const linksToDelete = findLinksToDeleteWithJoint(jointName, links)

  // Find any orphans created by deleting these links (excluding the joint we're already deleting)
  const orphanedJoints = findOrphansAfterMultipleLinkRemovals(linksToDelete, links, [jointName])

  return {
    linksToDelete,
    jointsToDelete: [jointName, ...orphanedJoints]
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DRAG/DROP/MERGE HELPER FUNCTIONS
// These functions support moving joints and merging them together
// ═══════════════════════════════════════════════════════════════════════════════


/**
 * Check if a position is within merge range of another joint
 * @param position - Current drag position
 * @param targetJoint - The potential merge target joint
 * @param threshold - Distance threshold for merge
 * @returns true if within merge range
 */
export const isWithinMergeRange = (
  position: [number, number],
  targetPosition: [number, number],
  threshold: number = MERGE_THRESHOLD
): boolean => {
  const distance = calculateDistance(position, targetPosition)
  return distance <= threshold
}

/**
 * Find the nearest joint for potential merge (excluding the dragged joint)
 * @param position - Current drag position
 * @param joints - Array of joints with positions
 * @param excludeJoint - Joint name to exclude (the one being dragged)
 * @param threshold - Distance threshold for merge detection
 * @returns Merge target info or null if no valid target
 */
export const findMergeTarget = (
  position: [number, number],
  joints: Array<{ name: string; position: [number, number] | null }>,
  excludeJoint: string,
  threshold: number = MERGE_THRESHOLD
): { name: string; position: [number, number]; distance: number } | null => {
  let nearest: { name: string; position: [number, number]; distance: number } | null = null

  for (const joint of joints) {
    // Skip the joint being dragged
    if (joint.name === excludeJoint || !joint.position) continue

    const distance = calculateDistance(position, joint.position)
    if (distance <= threshold) {
      if (!nearest || distance < nearest.distance) {
        nearest = {
          name: joint.name,
          position: joint.position,
          distance
        }
      }
    }
  }

  return nearest
}

/**
 * Calculate the merge result - what happens when we merge sourceJoint into targetJoint
 * The source joint is deleted and all its links are rewired to the target joint
 * @param sourceJoint - The joint being dragged (will be deleted)
 * @param targetJoint - The joint we're merging into (will absorb connections)
 * @param links - Current link metadata
 * @returns Object describing the merge operation
 */
export const calculateMergeResult = (
  sourceJoint: string,
  targetJoint: string,
  links: Record<string, LinkMetaData>
): {
  jointToDelete: string
  linksToUpdate: Array<{ linkName: string; oldConnects: string[]; newConnects: string[] }>
  linksToDelete: string[]  // Links that would become self-loops
} => {
  const linksToUpdate: Array<{ linkName: string; oldConnects: string[]; newConnects: string[] }> = []
  const linksToDelete: string[] = []

  // Find all links connected to the source joint
  const connectedLinks = getLinksConnectedToJoint(sourceJoint, links)

  for (const linkName of connectedLinks) {
    const link = links[linkName]
    if (!link) continue

    // Create new connects array with source replaced by target
    const newConnects = link.connects.map(j => j === sourceJoint ? targetJoint : j)

    // Check if this creates a self-loop (both ends connect to same joint)
    if (newConnects[0] === newConnects[1]) {
      linksToDelete.push(linkName)
    } else {
      // Check if this link already exists (duplicate link)
      const existingLink = Object.entries(links).find(([name, meta]) => {
        if (name === linkName) return false
        return (meta.connects[0] === newConnects[0] && meta.connects[1] === newConnects[1]) ||
               (meta.connects[0] === newConnects[1] && meta.connects[1] === newConnects[0])
      })

      if (existingLink) {
        // This would create a duplicate link, so delete instead
        linksToDelete.push(linkName)
      } else {
        linksToUpdate.push({
          linkName,
          oldConnects: [...link.connects],
          newConnects
        })
      }
    }
  }

  return {
    jointToDelete: sourceJoint,
    linksToUpdate,
    linksToDelete
  }
}

/**
 * Get a description of what will happen during a merge (for status display)
 * @param sourceJoint - Joint being merged
 * @param targetJoint - Joint being merged into
 * @param links - Current link metadata
 * @returns Human-readable description
 */
export const getMergeDescription = (
  sourceJoint: string,
  targetJoint: string,
  links: Record<string, LinkMetaData>
): string => {
  const result = calculateMergeResult(sourceJoint, targetJoint, links)
  const parts: string[] = [`Merge ${sourceJoint} → ${targetJoint}`]

  if (result.linksToUpdate.length > 0) {
    parts.push(`rewire ${result.linksToUpdate.length} link(s)`)
  }
  if (result.linksToDelete.length > 0) {
    parts.push(`remove ${result.linksToDelete.length} redundant link(s)`)
  }

  return parts.join(', ')
}

// ═══════════════════════════════════════════════════════════════════════════════
// FOOTER TOOLBAR COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface FooterToolbarProps {
  toolMode: ToolMode
  jointCount: number
  linkCount: number
  selectedJoints: string[]
  selectedLinks: string[]
  statusMessage: StatusMessage | null
  linkCreationState: LinkCreationState
  polygonDrawState?: PolygonDrawState
  measureState?: MeasureState
  groupSelectionState?: GroupSelectionState
  mergePolygonState?: MergePolygonState
  pathDrawState?: PathDrawState
  canvasWidth?: number
  onCancelAction?: () => void
  darkMode?: boolean
}

const getStatusColor = (type: StatusType, darkMode: boolean = false): string => {
  if (darkMode) {
    // Brighter colors for dark mode - near white for readability
    switch (type) {
      case 'info': return '#90caf9'      // Light blue
      case 'action': return '#ffb74d'    // Light orange
      case 'success': return '#a5d6a7'   // Light green
      case 'warning': return '#ffcc80'   // Light amber
      case 'error': return '#ef9a9a'     // Light red
      default: return '#e0e0e0'
    }
  }
  // Light mode - darker colors
  switch (type) {
    case 'info': return statusColors.nominalDark
    case 'action': return colors.primary
    case 'success': return statusColors.successDark
    case 'warning': return statusColors.warningDark
    case 'error': return statusColors.error
    default: return '#666'
  }
}

const getStatusBgColor = (type: StatusType, darkMode: boolean = false): string => {
  if (darkMode) {
    // Slightly more opaque backgrounds for dark mode
    switch (type) {
      case 'info': return 'rgba(144, 202, 249, 0.15)'
      case 'action': return 'rgba(255, 183, 77, 0.18)'
      case 'success': return 'rgba(165, 214, 167, 0.15)'
      case 'warning': return 'rgba(255, 204, 128, 0.15)'
      case 'error': return 'rgba(239, 154, 154, 0.15)'
      default: return 'transparent'
    }
  }
  switch (type) {
    case 'info': return 'rgba(25, 118, 210, 0.12)'
    case 'action': return 'rgba(255, 140, 0, 0.15)'
    case 'success': return 'rgba(46, 125, 50, 0.12)'
    case 'warning': return 'rgba(237, 108, 2, 0.12)'
    case 'error': return 'rgba(211, 47, 47, 0.12)'
    default: return 'transparent'
  }
}

// Get tool hint based on current state
const getToolHint = (
  toolMode: ToolMode,
  linkCreationState: LinkCreationState,
  polygonDrawState?: PolygonDrawState,
  measureState?: MeasureState,
  groupSelectionState?: GroupSelectionState,
  selectedJoints?: string[],
  selectedLinks?: string[],
  mergePolygonState?: MergePolygonState,
  pathDrawState?: PathDrawState
): string | null => {
  switch (toolMode) {
    case 'select':
      return 'Click to select • Drag to move joints'
    case 'group_select':
      if (groupSelectionState?.isSelecting) {
        return 'Release to complete selection'
      }
      return 'Click and drag to select multiple elements'
    case 'mechanism_select':
      return 'Click any element to select its connected mechanism'
    case 'draw_link':
      if (linkCreationState.isDrawing) {
        return 'Click second point to complete link'
      }
      return 'Click first point to start drawing'
    case 'draw_polygon':
      if (polygonDrawState?.isDrawing) {
        const sides = polygonDrawState.points.length
        if (sides >= 3) {
          return `${sides} sides • Click near start to close`
        }
        return `${sides} point(s) • Click to add more sides`
      }
      return 'Click to start polygon'
    case 'measure':
      if (measureState?.isMeasuring) {
        return 'Click second point to measure'
      }
      return 'Click first point to start measuring'
    case 'delete':
      const totalSelected = (selectedJoints?.length || 0) + (selectedLinks?.length || 0)
      if (totalSelected > 0) {
        return `Click to delete ${totalSelected} selected item(s)`
      }
      return 'Click on joint or link to delete'
    case 'merge':
      // Merge polygon with link tool
      if (mergePolygonState?.step === 'polygon_selected') {
        return 'Now click a link to merge with polygon'
      }
      if (mergePolygonState?.step === 'link_selected') {
        return 'Now click a polygon to merge with link'
      }
      return 'Click a polygon or link to merge • Click merged polygon to unmerge'
    case 'add_joint':
      return 'Click on a link to add a joint'
    case 'draw_path':
      if (pathDrawState?.isDrawing) {
        const pointCount = pathDrawState.points.length
        return `${pointCount} point(s) • Click to add • Double-click/Enter to finish`
      }
      return 'Click to start drawing target path'
    default:
      return null
  }
}

export const FooterToolbar: React.FC<FooterToolbarProps> = ({
  toolMode,
  jointCount,
  linkCount,
  selectedJoints,
  selectedLinks,
  statusMessage,
  linkCreationState,
  polygonDrawState,
  measureState,
  groupSelectionState,
  mergePolygonState,
  pathDrawState,
  canvasWidth,
  onCancelAction,
  darkMode = false
}) => {
  const activeTool = TOOLS.find(t => t.id === toolMode)
  const toolHint = getToolHint(toolMode, linkCreationState, polygonDrawState, measureState, groupSelectionState, selectedJoints, selectedLinks, mergePolygonState, pathDrawState)

  // Determine if we should show cancel hint
  const showCancelHint = statusMessage?.type === 'action' ||
    linkCreationState.isDrawing ||
    polygonDrawState?.isDrawing ||
    measureState?.isMeasuring ||
    groupSelectionState?.isSelecting ||
    pathDrawState?.isDrawing

  return (
    <Box
      className="footer-toolbar"
      sx={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        width: '100%',
        height: 42,
        backgroundColor: darkMode ? 'rgba(25, 25, 30, 0.85)' : 'rgba(250, 250, 252, 0.85)',
        //backdropFilter: 'blur(8px)',
        borderTop: darkMode ? '1px solid #444' : '1px solid #e0e0e0',
        borderBottomLeftRadius: 8,  // Match canvas border radius (borderRadius: 2 = 8px)
        borderBottomRightRadius: 8,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: 1.5,
        zIndex: 2300,  // Above canvas content but below modals
        transition: 'background-color 0.25s ease, border-color 0.25s ease'
      }}
    >
      {/* LEFT: Tool indicator + Selection */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, minWidth: 200 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          <Typography sx={{ fontSize: '1.1rem', color: darkMode ? '#e0e0e0' : 'inherit' }}>{activeTool?.icon}</Typography>
          <Typography sx={{ fontSize: '0.8rem', fontWeight: 600, color: darkMode ? '#e0e0e0' : 'inherit' }}>
            {activeTool?.label}
          </Typography>
          <Typography sx={{ fontSize: '0.65rem', color: darkMode ? '#888' : 'text.secondary' }}>
            [{activeTool?.shortcut}]
          </Typography>
        </Box>

        {(selectedJoints.length > 0 || selectedLinks.length > 0) && (
          <>
            <Box sx={{ width: '1px', height: 18, backgroundColor: darkMode ? '#555' : '#e0e0e0' }} />
            <Typography sx={{ fontSize: '0.8rem', color: darkMode ? '#ccc' : 'inherit' }}>
              {selectedJoints.length > 1 ? (
                <span>⬤ <strong>{selectedJoints.length} joints</strong></span>
              ) : selectedJoints.length === 1 ? (
                <span>⬤ <strong>{selectedJoints[0]}</strong></span>
              ) : null}
              {selectedJoints.length > 0 && selectedLinks.length > 0 && ' • '}
              {selectedLinks.length > 1 ? (
                <span>— <strong>{selectedLinks.length} links</strong></span>
              ) : selectedLinks.length === 1 ? (
                <span>— <strong>{selectedLinks[0]}</strong></span>
              ) : null}
            </Typography>
          </>
        )}
      </Box>

      {/* CENTER: Status message or tool hint */}
      {statusMessage ? (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 1.5,
            py: 0.25,
            borderRadius: 1,
            // In dark mode: solid dark background for text readability
            backgroundColor: darkMode ? 'rgba(20, 20, 20, 0.95)' : getStatusBgColor(statusMessage.type, darkMode),
            border: darkMode ? '1px solid rgba(255, 255, 255, 0.1)' : 'none'
          }}
        >
          <Box
            sx={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              backgroundColor: getStatusColor(statusMessage.type, darkMode)
            }}
          />
          <Typography
            sx={{
              fontSize: '0.85rem',
              fontWeight: 500,
              // In dark mode: use white text for maximum readability
              color: darkMode ? '#ffffff' : getStatusColor(statusMessage.type, darkMode)
            }}
          >
            {statusMessage.text}
          </Typography>
          {showCancelHint && onCancelAction && (
            <Typography
              component="span"
              onClick={onCancelAction}
              sx={{
                fontSize: '0.65rem',
                color: darkMode ? '#888' : 'text.secondary',
                cursor: 'pointer',
                ml: 0.5,
                '&:hover': { textDecoration: 'underline' }
              }}
            >
              [Esc]
            </Typography>
          )}
        </Box>
      ) : toolHint ? (
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            px: 1.5,
            py: 0.25,
            borderRadius: 1,
            backgroundColor: darkMode ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.04)'
          }}
        >
          <Typography
            component="span"
            sx={{
              fontSize: '0.8rem',
              fontStyle: 'italic',
              // Use !important to override MUI defaults in dark mode
              color: darkMode ? '#ffffff !important' : 'text.secondary'
            }}
          >
            {toolHint}
          </Typography>
        </Box>
      ) : null}

      {/* RIGHT: Counts + Logo */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, minWidth: 180, justifyContent: 'flex-end' }}>
        <Typography sx={{ fontSize: '0.8rem', color: darkMode ? '#bbb' : 'text.secondary' }}>
          <strong style={{ color: darkMode ? '#e0e0e0' : 'inherit' }}>{jointCount}</strong> joints
        </Typography>
        <Typography sx={{ fontSize: '0.8rem', color: darkMode ? '#bbb' : 'text.secondary' }}>
          <strong style={{ color: darkMode ? '#e0e0e0' : 'inherit' }}>{linkCount}</strong> links
        </Typography>
        <Box sx={{ width: '1px', height: 20, backgroundColor: darkMode ? '#444' : '#e0e0e0' }} />
        <img
          src={acinonyxLogo}
          alt="Acinonyx"
          className="footer-logo"
          style={{
            width: '28px',
            height: '28px',
            objectFit: 'contain',
            borderRadius: '4px'
          }}
        />
      </Box>
    </Box>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINK CREATION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

// Distance threshold in units for snapping to existing joints
export const JOINT_SNAP_THRESHOLD = 5.0

// ═══════════════════════════════════════════════════════════════════════════════
// RIGID TRANSFORMATION UTILITIES
// For making merged polygons move with their attached links
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Transform polygon points based on how a link has moved/rotated.
 * This enables merged polygons to move rigidly with their attached links.
 *
 * @param points - Original polygon points (at merge time)
 * @param originalStart - Link start position at merge time
 * @param originalEnd - Link end position at merge time
 * @param currentStart - Link start position now (possibly animated)
 * @param currentEnd - Link end position now (possibly animated)
 * @returns Transformed polygon points
 */
export const transformPolygonPoints = (
  points: [number, number][],
  originalStart: [number, number],
  originalEnd: [number, number],
  currentStart: [number, number],
  currentEnd: [number, number]
): [number, number][] => {
  // Calculate original link center and angle
  const origCenterX = (originalStart[0] + originalEnd[0]) / 2
  const origCenterY = (originalStart[1] + originalEnd[1]) / 2
  const origAngle = Math.atan2(
    originalEnd[1] - originalStart[1],
    originalEnd[0] - originalStart[0]
  )

  // Calculate current link center and angle
  const currCenterX = (currentStart[0] + currentEnd[0]) / 2
  const currCenterY = (currentStart[1] + currentEnd[1]) / 2
  const currAngle = Math.atan2(
    currentEnd[1] - currentStart[1],
    currentEnd[0] - currentStart[0]
  )

  // Compute rotation delta
  const deltaAngle = currAngle - origAngle

  // Transform each point
  return points.map(([px, py]): [number, number] => {
    // 1. Translate to origin (relative to original link center)
    const relX = px - origCenterX
    const relY = py - origCenterY

    // 2. Rotate by delta angle
    const cos = Math.cos(deltaAngle)
    const sin = Math.sin(deltaAngle)
    const rotX = relX * cos - relY * sin
    const rotY = relX * sin + relY * cos

    // 3. Translate to current link center
    return [rotX + currCenterX, rotY + currCenterY]
  })
}

/**
 * Calculate distance between two points
 */
export const calculateDistance = (p1: [number, number], p2: [number, number]): number => {
  return Math.sqrt(Math.pow(p2[0] - p1[0], 2) + Math.pow(p2[1] - p1[1], 2))
}

/**
 * Check if a point is inside a polygon using ray casting algorithm
 * @param point - The point to check [x, y]
 * @param polygon - Array of polygon vertices [[x1,y1], [x2,y2], ...]
 * @returns true if point is inside the polygon
 */
export const isPointInPolygon = (point: [number, number], polygon: [number, number][]): boolean => {
  if (polygon.length < 3) return false

  const [x, y] = point
  let inside = false

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i]
    const [xj, yj] = polygon[j]

    // Ray casting: count intersections with polygon edges
    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside
    }
  }

  return inside
}

/**
 * Check if both endpoints of a link are inside a polygon
 * @param linkStart - Start point of the link [x, y]
 * @param linkEnd - End point of the link [x, y]
 * @param polygon - Array of polygon vertices
 * @returns true if both endpoints are inside the polygon
 */
export const areLinkEndpointsInPolygon = (
  linkStart: [number, number],
  linkEnd: [number, number],
  polygon: [number, number][]
): boolean => {
  return isPointInPolygon(linkStart, polygon) && isPointInPolygon(linkEnd, polygon)
}

/**
 * Find the nearest joint to a given position within the snap threshold
 */
export const findNearestJoint = (
  position: [number, number],
  joints: Array<{ name: string; position: [number, number] | null }>,
  threshold: number = JOINT_SNAP_THRESHOLD
): { name: string; position: [number, number]; distance: number } | null => {
  let nearest: { name: string; position: [number, number]; distance: number } | null = null

  for (const joint of joints) {
    if (!joint.position) continue

    const distance = calculateDistance(position, joint.position)
    if (distance <= threshold) {
      if (!nearest || distance < nearest.distance) {
        nearest = {
          name: joint.name,
          position: joint.position,
          distance
        }
      }
    }
  }

  return nearest
}

/**
 * Find the nearest link to a given position
 * Uses point-to-line-segment distance
 */
export const findNearestLink = (
  position: [number, number],
  links: Array<{ name: string; start: [number, number] | null; end: [number, number] | null }>,
  threshold: number = JOINT_SNAP_THRESHOLD
): { name: string; distance: number } | null => {
  let nearest: { name: string; distance: number } | null = null

  for (const link of links) {
    if (!link.start || !link.end) continue

    // Calculate point-to-line-segment distance
    const distance = pointToLineSegmentDistance(position, link.start, link.end)
    if (distance <= threshold) {
      if (!nearest || distance < nearest.distance) {
        nearest = {
          name: link.name,
          distance
        }
      }
    }
  }

  return nearest
}

/**
 * Calculate the distance from a point to a line segment
 */
export const pointToLineSegmentDistance = (
  point: [number, number],
  lineStart: [number, number],
  lineEnd: [number, number]
): number => {
  const [px, py] = point
  const [x1, y1] = lineStart
  const [x2, y2] = lineEnd

  const dx = x2 - x1
  const dy = y2 - y1
  const lengthSquared = dx * dx + dy * dy

  if (lengthSquared === 0) {
    // Line segment is a point
    return calculateDistance(point, lineStart)
  }

  // Project point onto line, clamped to segment
  let t = ((px - x1) * dx + (py - y1) * dy) / lengthSquared
  t = Math.max(0, Math.min(1, t))

  const closestX = x1 + t * dx
  const closestY = y1 + t * dy

  return calculateDistance(point, [closestX, closestY])
}

/**
 * Generate a unique joint name
 */
export const generateJointName = (existingNames: string[], prefix: string = 'joint'): string => {
  let counter = 1
  let name = `${prefix}_${counter}`
  while (existingNames.includes(name)) {
    counter++
    name = `${prefix}_${counter}`
  }
  return name
}

/**
 * Generate a unique link name
 */
export const generateLinkName = (existingNames: string[], prefix: string = 'link'): string => {
  let counter = 1
  let name = `${prefix}_${counter}`
  while (existingNames.includes(name)) {
    counter++
    name = `${prefix}_${counter}`
  }
  return name
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLOR PALETTE - Uses centralized theme from ../theme.ts
// ═══════════════════════════════════════════════════════════════════════════════

// Re-export graph colors from theme for backward compatibility
export const TAB10_COLORS = graphColors

// Get color by index from the graph palette (cycles through colors)
export const getDefaultColor = (index: number): string => graphColors[index % graphColors.length]


// ═══════════════════════════════════════════════════════════════════════════════
// JOINT EDIT MODAL - Modal dialog for editing joint properties
// ═══════════════════════════════════════════════════════════════════════════════

export interface JointData {
  name: string
  type: 'Static' | 'Crank' | 'Revolute'
  position: [number, number] | null
  // Type-specific data
  parentJoint?: string      // For Crank/Revolute
  parentJoint2?: string     // For Revolute (second parent)
  distance?: number         // For Crank/Revolute
  distance2?: number        // For Revolute (second distance)
  angle?: number            // For Crank
  // Computed/display data
  mechanismGroup?: string
  connectedLinks?: string[]
  showPath?: boolean
}

export interface JointEditModalProps {
  open: boolean
  onClose: () => void
  jointData: JointData | null
  jointTypes: readonly string[]
  onRename: (oldName: string, newName: string) => void
  onTypeChange: (jointName: string, newType: string) => void
  onShowPathChange?: (jointName: string, showPath: boolean) => void
  darkMode?: boolean
}

export const JointEditModal: React.FC<JointEditModalProps> = ({
  open,
  onClose,
  jointData,
  jointTypes,
  onRename,
  onTypeChange,
  onShowPathChange,
  darkMode = false
}) => {
  const [editedName, setEditedName] = useState('')
  const [hasNameError, setHasNameError] = useState(false)

  // Reset edited name when modal opens with new data
  useEffect(() => {
    if (jointData) {
      setEditedName(jointData.name)
      setHasNameError(false)
    }
  }, [jointData])

  if (!jointData) return null

  const typeColor = jointData.type === 'Static'
    ? jointColors.static
    : jointData.type === 'Crank'
      ? jointColors.crank
      : jointColors.pivot

  const handleNameSubmit = () => {
    const trimmedName = editedName.trim()
    if (trimmedName && trimmedName !== jointData.name) {
      onRename(jointData.name, trimmedName)
    }
  }

  const modalStyle = {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 380,
    bgcolor: darkMode ? '#2a2a2a' : '#fff',
    color: darkMode ? '#f5f5f5' : '#333',
    borderRadius: 3,
    boxShadow: 24,
    p: 0,
    outline: 'none'
  }

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={modalStyle}>
        {/* Header */}
        <Box sx={{
          p: 2,
          borderBottom: `3px solid ${typeColor}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          bgcolor: darkMode ? '#333' : '#fafafa'
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box sx={{
              width: 12, height: 12, borderRadius: '50%',
              bgcolor: typeColor,
              boxShadow: `0 0 8px ${typeColor}80`
            }} />
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1rem' }}>
              Edit Joint
            </Typography>
            <Chip
              label={jointData.type}
              size="small"
              sx={{
                height: 20,
                fontSize: '0.65rem',
                bgcolor: `${typeColor}20`,
                color: typeColor,
                fontWeight: 600
              }}
            />
          </Box>
          <IconButton size="small" onClick={onClose} sx={{ color: darkMode ? '#aaa' : '#666' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Content */}
        <Box sx={{ p: 2.5 }}>
          {/* Editable Properties */}
          <Typography variant="overline" sx={{
            color: darkMode ? '#888' : 'text.secondary',
            fontSize: '0.65rem',
            letterSpacing: 1
          }}>
            Editable Properties
          </Typography>

          <Box sx={{ mt: 1.5, mb: 2.5 }}>
            <TextField
              size="small"
              label="Name"
              value={editedName}
              onChange={(e) => {
                setEditedName(e.target.value)
                setHasNameError(!e.target.value.trim())
              }}
              onBlur={handleNameSubmit}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleNameSubmit()
                  ;(e.target as HTMLInputElement).blur()
                }
              }}
              error={hasNameError}
              helperText={hasNameError ? 'Name is required' : ''}
              fullWidth
              sx={{
                mb: 2,
                '& .MuiInputBase-input': { fontSize: '0.85rem' },
                '& .MuiInputLabel-root': { fontSize: '0.8rem' }
              }}
            />

            <FormControl size="small" fullWidth sx={{ mb: 2 }}>
              <InputLabel sx={{ fontSize: '0.8rem' }}>Type</InputLabel>
              <Select
                value={jointData.type}
                label="Type"
                onChange={(e) => onTypeChange(jointData.name, e.target.value)}
                sx={{ fontSize: '0.85rem' }}
              >
                {jointTypes.map(t => (
                  <MenuItem key={t} value={t} sx={{ fontSize: '0.85rem' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{
                        width: 8, height: 8, borderRadius: '50%',
                        bgcolor: t === 'Static' ? jointColors.static : t === 'Crank' ? jointColors.crank : jointColors.pivot
                      }} />
                      {t}
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Show Path toggle for Crank/Revolute */}
            {(jointData.type === 'Crank' || jointData.type === 'Revolute') && onShowPathChange && (
              <Box sx={{
                p: 1.5,
                bgcolor: darkMode ? '#3a3a3a' : '#f5f5f5',
                borderRadius: 1
              }}>
                <Box sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}>
                  <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                    Show Trajectory Path
                  </Typography>
                  <Button
                    size="small"
                    variant={jointData.showPath ? 'contained' : 'outlined'}
                    onClick={() => onShowPathChange(jointData.name, !jointData.showPath)}
                    sx={{
                      minWidth: 60,
                      fontSize: '0.7rem',
                      textTransform: 'none',
                      bgcolor: jointData.showPath ? colors.primary : 'transparent',
                      borderColor: jointData.showPath ? colors.primary : '#999',
                      color: jointData.showPath ? '#fff' : (darkMode ? '#ccc' : '#666'),
                      '&:hover': {
                        bgcolor: jointData.showPath ? '#e67300' : (darkMode ? '#444' : '#eee')
                      }
                    }}
                  >
                    {jointData.showPath ? 'On' : 'Off'}
                  </Button>
                </Box>
                <Typography variant="caption" sx={{
                  display: 'block',
                  mt: 0.75,
                  fontSize: '0.65rem',
                  color: darkMode ? '#777' : '#888',
                  fontStyle: 'italic'
                }}>
                  Note: The path visibility button (◉/○) in the Animate bar at the bottom controls global visibility for all paths
                </Typography>
              </Box>
            )}
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Static Properties */}
          <Typography variant="overline" sx={{
            color: darkMode ? '#888' : 'text.secondary',
            fontSize: '0.65rem',
            letterSpacing: 1
          }}>
            Properties (Read-only)
          </Typography>

          <Box sx={{
            mt: 1.5,
            bgcolor: darkMode ? '#333' : '#f8f8f8',
            borderRadius: 2,
            p: 1.5,
            fontSize: '0.8rem'
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                Position
              </Typography>
              <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                {jointData.position
                  ? `(${jointData.position[0].toFixed(2)}, ${jointData.position[1].toFixed(2)})`
                  : '—'}
              </Typography>
            </Box>

            {jointData.type === 'Crank' && (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Parent Joint
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    {jointData.parentJoint || '—'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Distance
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {jointData.distance?.toFixed(2) || '—'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Angle
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {jointData.angle !== undefined ? `${(jointData.angle * 180 / Math.PI).toFixed(1)}°` : '—'}
                  </Typography>
                </Box>
              </>
            )}

            {jointData.type === 'Revolute' && (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Parent Joint 1
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    {jointData.parentJoint || '—'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Distance 1
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {jointData.distance?.toFixed(2) || '—'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Parent Joint 2
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 500 }}>
                    {jointData.parentJoint2 || '—'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    Distance 2
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {jointData.distance2?.toFixed(2) || '—'}
                  </Typography>
                </Box>
              </>
            )}

            {jointData.connectedLinks && jointData.connectedLinks.length > 0 && (
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                  Connected Links
                </Typography>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>
                  {jointData.connectedLinks.join(', ')}
                </Typography>
              </Box>
            )}

            {jointData.mechanismGroup && (
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                  Mechanism Group
                </Typography>
                <Chip
                  label={jointData.mechanismGroup}
                  size="small"
                  sx={{ height: 18, fontSize: '0.6rem' }}
                />
              </Box>
            )}
          </Box>
        </Box>

        {/* Footer */}
        <Box sx={{
          p: 2,
          borderTop: `1px solid ${darkMode ? '#444' : '#eee'}`,
          display: 'flex',
          justifyContent: 'flex-end',
          gap: 1
        }}>
          <Button
            variant="contained"
            size="small"
            onClick={onClose}
            sx={{
              textTransform: 'none',
              bgcolor: colors.primary,
              '&:hover': { bgcolor: '#e67300' }
            }}
          >
            Done
          </Button>
        </Box>
      </Box>
    </Modal>
  )
}


// ═══════════════════════════════════════════════════════════════════════════════
// LINK EDIT MODAL - Modal dialog for editing link properties
// ═══════════════════════════════════════════════════════════════════════════════

export interface LinkData {
  name: string
  color: string
  connects: [string, string]
  length: number | null
  isGround?: boolean  // True if this is a ground/anchored link
  // Computed/display data
  mechanismGroup?: string
  jointPositions?: [[number, number] | null, [number, number] | null]
}

export interface LinkEditModalProps {
  open: boolean
  onClose: () => void
  linkData: LinkData | null
  onRename: (oldName: string, newName: string) => void
  onColorChange: (linkName: string, color: string) => void
  onGroundChange: (linkName: string, isGround: boolean) => void
  darkMode?: boolean
}

export const LinkEditModal: React.FC<LinkEditModalProps> = ({
  open,
  onClose,
  linkData,
  onRename,
  onColorChange,
  onGroundChange,
  darkMode = false
}) => {
  const [editedName, setEditedName] = useState('')
  const [hasNameError, setHasNameError] = useState(false)

  // Reset edited name when modal opens with new data
  useEffect(() => {
    if (linkData) {
      setEditedName(linkData.name)
      setHasNameError(false)
    }
  }, [linkData])

  if (!linkData) return null

  const handleNameSubmit = () => {
    const trimmedName = editedName.trim()
    if (trimmedName && trimmedName !== linkData.name) {
      onRename(linkData.name, trimmedName)
    }
  }

  const modalStyle = {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 380,
    bgcolor: darkMode ? '#2a2a2a' : '#fff',
    color: darkMode ? '#f5f5f5' : '#333',
    borderRadius: 3,
    boxShadow: 24,
    p: 0,
    outline: 'none'
  }

  return (
    <Modal open={open} onClose={onClose}>
      <Box sx={modalStyle}>
        {/* Header */}
        <Box sx={{
          p: 2,
          borderBottom: `3px solid ${linkData.color}`,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          bgcolor: darkMode ? '#333' : '#fafafa'
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <Box sx={{
              width: 24, height: 4, borderRadius: 2,
              bgcolor: linkData.color,
              boxShadow: `0 0 8px ${linkData.color}80`
            }} />
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1rem' }}>
              Edit Link
            </Typography>
          </Box>
          <IconButton size="small" onClick={onClose} sx={{ color: darkMode ? '#aaa' : '#666' }}>
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>

        {/* Content */}
        <Box sx={{ p: 2.5 }}>
          {/* Editable Properties */}
          <Typography variant="overline" sx={{
            color: darkMode ? '#888' : 'text.secondary',
            fontSize: '0.65rem',
            letterSpacing: 1
          }}>
            Editable Properties
          </Typography>

          <Box sx={{ mt: 1.5, mb: 2.5 }}>
            <TextField
              size="small"
              label="Name"
              value={editedName}
              onChange={(e) => {
                setEditedName(e.target.value)
                setHasNameError(!e.target.value.trim())
              }}
              onBlur={handleNameSubmit}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleNameSubmit()
                  ;(e.target as HTMLInputElement).blur()
                }
              }}
              error={hasNameError}
              helperText={hasNameError ? 'Name is required' : ''}
              fullWidth
              sx={{
                mb: 2,
                '& .MuiInputBase-input': { fontSize: '0.85rem' },
                '& .MuiInputLabel-root': { fontSize: '0.8rem' }
              }}
            />

            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="body2" sx={{ fontSize: '0.8rem', color: darkMode ? '#ccc' : 'text.secondary' }}>
                Color
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <input
                  type="color"
                  value={linkData.color}
                  onChange={(e) => onColorChange(linkData.name, e.target.value)}
                  style={{
                    width: 40,
                    height: 28,
                    border: 'none',
                    borderRadius: 4,
                    cursor: 'pointer',
                    padding: 0
                  }}
                />
                <Typography variant="caption" sx={{ fontFamily: 'monospace', color: darkMode ? '#999' : '#666' }}>
                  {linkData.color.toUpperCase()}
                </Typography>
              </Box>
            </Box>

            {/* Ground Link Toggle */}
            <FormControlLabel
              control={
                <Switch
                  checked={linkData.isGround || false}
                  onChange={(e) => onGroundChange(linkData.name, e.target.checked)}
                  size="small"
                  sx={{
                    '& .MuiSwitch-switchBase.Mui-checked': {
                      color: '#7f7f7f',
                    },
                    '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                      backgroundColor: '#7f7f7f',
                    },
                  }}
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ fontSize: '0.8rem', color: darkMode ? '#ccc' : 'text.secondary' }}>
                    Ground Link
                  </Typography>
                  <Tooltip title="Mark this link as a ground/anchored link. Ground links are rendered with a dashed style and typically connect static (fixed) joints.">
                    <Typography variant="caption" sx={{ color: darkMode ? '#666' : '#999', cursor: 'help' }}>ⓘ</Typography>
                  </Tooltip>
                </Box>
              }
              sx={{ mt: 1.5, ml: 0 }}
            />
          </Box>

          <Divider sx={{ my: 2 }} />

          {/* Static Properties */}
          <Typography variant="overline" sx={{
            color: darkMode ? '#888' : 'text.secondary',
            fontSize: '0.65rem',
            letterSpacing: 1
          }}>
            Properties (Read-only)
          </Typography>

          <Box sx={{
            mt: 1.5,
            bgcolor: darkMode ? '#333' : '#f8f8f8',
            borderRadius: 2,
            p: 1.5,
            fontSize: '0.8rem'
          }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
              <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                Connects
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Chip label={linkData.connects[0]} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
                <Typography variant="caption" sx={{ color: darkMode ? '#666' : '#999' }}>→</Typography>
                <Chip label={linkData.connects[1]} size="small" sx={{ height: 20, fontSize: '0.65rem' }} />
              </Box>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                Length
              </Typography>
              <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                {linkData.length !== null ? linkData.length.toFixed(2) : '—'}
              </Typography>
            </Box>

            {linkData.jointPositions && (
              <>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    {linkData.connects[0]} Position
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {linkData.jointPositions[0]
                      ? `(${linkData.jointPositions[0][0].toFixed(2)}, ${linkData.jointPositions[0][1].toFixed(2)})`
                      : '—'}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                    {linkData.connects[1]} Position
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 500 }}>
                    {linkData.jointPositions[1]
                      ? `(${linkData.jointPositions[1][0].toFixed(2)}, ${linkData.jointPositions[1][1].toFixed(2)})`
                      : '—'}
                  </Typography>
                </Box>
              </>
            )}

            {linkData.mechanismGroup && (
              <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                <Typography variant="caption" sx={{ color: darkMode ? '#999' : 'text.secondary' }}>
                  Mechanism Group
                </Typography>
                <Chip
                  label={linkData.mechanismGroup}
                  size="small"
                  sx={{ height: 18, fontSize: '0.6rem' }}
                />
              </Box>
            )}
          </Box>
        </Box>

        {/* Footer */}
        <Box sx={{
          p: 2,
          borderTop: `1px solid ${darkMode ? '#444' : '#eee'}`,
          display: 'flex',
          justifyContent: 'flex-end',
          gap: 1
        }}>
          <Button
            variant="contained"
            size="small"
            onClick={onClose}
            sx={{
              textTransform: 'none',
              bgcolor: colors.primary,
              '&:hover': { bgcolor: '#e67300' }
            }}
          >
            Done
          </Button>
        </Box>
      </Box>
    </Modal>
  )
}


export default FooterToolbar
