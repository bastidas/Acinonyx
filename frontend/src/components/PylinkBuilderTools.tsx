import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Box, Typography, Paper, IconButton } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import { graphColors, statusColors, colors } from '../theme'


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

      {/* Content area */}
      <Box sx={{ overflow: 'auto', flex: 1 }}>
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
export const TOOLBAR_CONFIGS: ToolbarConfig[] = [
  { id: 'tools', title: 'Tools', icon: '⚒', defaultPosition: { x: 8, y: 60 } },        // Full left, below toggle buttons
  { id: 'more', title: 'More', icon: '≡', defaultPosition: { x: 8, y: 370 } },         // Full left, well below Tools
  { id: 'links', title: 'Links', icon: '—', defaultPosition: { x: -220, y: 8 } },      // Far right edge (negative = from right)
  { id: 'nodes', title: 'Nodes', icon: '○', defaultPosition: { x: -220, y: 500 } },    // Below Links on far right
  { id: 'settings', title: 'Settings', icon: '⚙', defaultPosition: { x: 250, y: 60 } } // Settings panel
]

export interface ToolbarToggleButtonsProps {
  openToolbars: Set<string>
  onToggleToolbar: (id: string) => void
}

/**
 * ToolbarToggleButtonsContainer - The horizontal button bar for toggling toolbars
 * 
 * Contains: Tools, Links, Nodes, More buttons in a horizontal row
 * Position: Upper left of the canvas
 */
export const ToolbarToggleButtons: React.FC<ToolbarToggleButtonsProps> = ({
  openToolbars,
  onToggleToolbar
}) => {
  return (
    <Box
      id="toolbar-toggle-buttons-container"
      className="toolbar-toggle-buttons-container"
      sx={{
        position: 'absolute',
        left: 8,
        top: 8,
        display: 'flex',
        flexDirection: 'row',  // Horizontal layout
        gap: 0.75,
        zIndex: 1200,
        // Subtle container styling
        backgroundColor: 'rgba(255, 255, 255, 0.7)',
        backdropFilter: 'blur(8px)',
        borderRadius: 2,
        padding: '6px 8px',
        border: '1px solid rgba(0, 0, 0, 0.08)',
        boxShadow: '0 2px 12px rgba(0, 0, 0, 0.06)'
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
              backgroundColor: isOpen ? 'primary.main' : 'rgba(255,255,255,0.9)',
              color: isOpen ? 'white' : 'text.primary',
              border: '1px solid',
              borderColor: isOpen ? 'primary.main' : 'rgba(0,0,0,0.1)',
              boxShadow: isOpen ? '0 2px 6px rgba(250, 129, 18, 0.3)' : 'none',
              transition: 'all 0.15s ease',
              '&:hover': {
                backgroundColor: isOpen ? 'primary.dark' : 'rgba(250, 129, 18, 0.1)',
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

export interface ToolInfo {
  id: ToolMode
  label: string
  icon: string
  description: string
  shortcut?: string
}

export const TOOLS: ToolInfo[] = [
  {
    id: 'select',
    label: 'Select',
    icon: '⎚',
    description: 'Click to select individual joints or links. Drag to move.',
    shortcut: 'V'
  },
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
    id: 'draw_link',
    label: 'Draw Link',
    icon: '╱',
    description: 'Click two points to create a new link between them.',
    shortcut: 'L'
  },
  {
    id: 'add_joint',
    label: 'Add Joint',
    icon: '⊕',
    description: 'Click on an existing link to add a joint at that position.',
    shortcut: 'J'
  },
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
    icon: '⊗',
    description: 'Merge a polygon with the enclosed link',
    shortcut: 'E'
  },
  {
    id: 'measure',
    label: 'Measure',
    icon: '⌗',  // Abstract grid/measurement icon
    description: 'Click two points to measure the distance between them.',
    shortcut: 'R'
  },
  {
    id: 'delete',
    label: 'Delete',
    icon: '⌫',  // Delete/backspace icon
    description: 'Click a joint or link to delete it. Deleting a link removes orphan nodes. Deleting a node removes connected links.',
    shortcut: 'X'
  }
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
  points: [number, number][]     // Vertices for polygon/path
  fillColor: string
  strokeColor: string
  strokeWidth: number
  fillOpacity: number
  closed: boolean                // Whether the shape is closed (polygon) or open (path)
  attachment?: DrawnObjectAttachment  // If attached to a link, this defines the relationship
  mergedLinkName?: string        // If merged with a link, the link's name
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
// LINK METADATA TYPE (for helper functions)
// ═══════════════════════════════════════════════════════════════════════════════

export interface LinkMetaData {
  color: string
  connects: string[]  // Array of joint names this link connects
}

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE HELPER FUNCTIONS
// These are small, reusable functions for graph manipulation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Check if a joint has any connections (is connected to any links)
 * @param jointName - The name of the joint to check
 * @param links - Record of link names to their metadata
 * @returns true if the joint is connected to at least one link
 */
export const hasConnections = (
  jointName: string,
  links: Record<string, LinkMetaData>
): boolean => {
  return Object.values(links).some(link => link.connects.includes(jointName))
}

/**
 * Check if a joint is orphaned (has no connections to any links)
 * @param jointName - The name of the joint to check
 * @param links - Record of link names to their metadata
 * @returns true if the joint has no connections
 */
export const isOrphan = (
  jointName: string,
  links: Record<string, LinkMetaData>
): boolean => {
  return !hasConnections(jointName, links)
}

/**
 * Get all links that connect to a specific joint
 * @param jointName - The name of the joint
 * @param links - Record of link names to their metadata
 * @returns Array of link names that connect to this joint
 */
export const getLinksConnectedToJoint = (
  jointName: string,
  links: Record<string, LinkMetaData>
): string[] => {
  return Object.entries(links)
    .filter(([_, linkMeta]) => linkMeta.connects.includes(jointName))
    .map(([linkName, _]) => linkName)
}

/**
 * Get all joints that a link connects
 * @param linkName - The name of the link
 * @param links - Record of link names to their metadata
 * @returns Array of joint names that this link connects, or empty array if link not found
 */
export const getJointsConnectedByLink = (
  linkName: string,
  links: Record<string, LinkMetaData>
): string[] => {
  const link = links[linkName]
  return link ? [...link.connects] : []
}

/**
 * Get the connection count for a joint (how many links connect to it)
 * @param jointName - The name of the joint
 * @param links - Record of link names to their metadata
 * @returns Number of links connected to this joint
 */
export const getConnectionCount = (
  jointName: string,
  links: Record<string, LinkMetaData>
): number => {
  return getLinksConnectedToJoint(jointName, links).length
}

/**
 * Find all orphan joints that would result from removing a link
 * @param linkName - The name of the link to be removed
 * @param links - Record of link names to their metadata
 * @returns Array of joint names that would become orphaned
 */
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
  return link.connects.filter(jointName => isOrphan(jointName, remainingLinks))
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
  return Array.from(affectedJoints).filter(jointName => isOrphan(jointName, remainingLinks))
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
  canvasWidth?: number
  onCancelAction?: () => void
}

const getStatusColor = (type: StatusType): string => {
  switch (type) {
    case 'info': return statusColors.nominalDark
    case 'action': return colors.primary
    case 'success': return statusColors.successDark
    case 'warning': return statusColors.warningDark
    case 'error': return statusColors.error
    default: return '#666'
  }
}

const getStatusBgColor = (type: StatusType): string => {
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
  mergePolygonState?: MergePolygonState
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
      return 'Select a link or a polygon to begin merge'
    case 'add_joint':
      return 'Click on a link to add a joint'
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
  canvasWidth,
  onCancelAction
}) => {
  const activeTool = TOOLS.find(t => t.id === toolMode)
  const toolHint = getToolHint(toolMode, linkCreationState, polygonDrawState, measureState, groupSelectionState, selectedJoints, selectedLinks, mergePolygonState)
  
  // Determine if we should show cancel hint
  const showCancelHint = statusMessage?.type === 'action' || 
    linkCreationState.isDrawing || 
    polygonDrawState?.isDrawing || 
    measureState?.isMeasuring ||
    groupSelectionState?.isSelecting
  
  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 0,
        left: '50%',
        transform: 'translateX(-50%)',
        width: canvasWidth ? `${canvasWidth}px` : '100%',
        maxWidth: canvasWidth ? `${canvasWidth}px` : '1600px',
        height: 44,
        backgroundColor: 'rgba(255, 255, 255, 0.98)',
        backdropFilter: 'blur(8px)',
        borderTop: '1px solid #e0e0e0',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        px: 2,
        zIndex: 1200
      }}
    >
      {/* LEFT: Tool indicator + Selection */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, minWidth: 200 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          <Typography sx={{ fontSize: '1.1rem' }}>{activeTool?.icon}</Typography>
          <Typography sx={{ fontSize: '0.75rem', fontWeight: 600 }}>
            {activeTool?.label}
          </Typography>
          <Typography sx={{ fontSize: '0.6rem', color: 'text.secondary' }}>
            [{activeTool?.shortcut}]
          </Typography>
        </Box>
        
        {(selectedJoints.length > 0 || selectedLinks.length > 0) && (
          <>
            <Box sx={{ width: '1px', height: 20, backgroundColor: '#e0e0e0' }} />
            <Typography sx={{ fontSize: '0.75rem' }}>
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
            px: 2,
            py: 0.5,
            borderRadius: 1,
            backgroundColor: getStatusBgColor(statusMessage.type)
          }}
        >
          <Box
            sx={{
              width: 6,
              height: 6,
              borderRadius: '50%',
              backgroundColor: getStatusColor(statusMessage.type)
            }}
          />
          <Typography
            sx={{
              fontSize: '0.8rem',
              fontWeight: 500,
              color: getStatusColor(statusMessage.type)
            }}
          >
            {statusMessage.text}
          </Typography>
          {showCancelHint && onCancelAction && (
            <Typography
              component="span"
              onClick={onCancelAction}
              sx={{
                fontSize: '0.7rem',
                color: 'text.secondary',
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
            px: 2,
            py: 0.5,
            borderRadius: 1,
            backgroundColor: 'rgba(0, 0, 0, 0.04)'
          }}
        >
          <Typography
            sx={{
              fontSize: '0.75rem',
              color: 'text.secondary',
              fontStyle: 'italic'
            }}
          >
            {toolHint}
          </Typography>
        </Box>
      ) : null}
      
      {/* RIGHT: Counts */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, minWidth: 150, justifyContent: 'flex-end' }}>
        <Typography sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
          <strong>{jointCount}</strong> joints
        </Typography>
        <Typography sx={{ fontSize: '0.75rem', color: 'text.secondary' }}>
          <strong>{linkCount}</strong> links
        </Typography>
      </Box>
    </Box>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINK CREATION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

// Distance threshold in units for snapping to existing joints
export const JOINT_SNAP_THRESHOLD = 5.0

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

export default FooterToolbar
