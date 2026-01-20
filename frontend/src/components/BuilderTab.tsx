import React, { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import {
  Box,
  Typography,
  Paper,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions
} from '@mui/material'
import {
  FooterToolbar,
  TOOLS,
  ToolMode,
  StatusMessage,
  StatusType,
  LinkCreationState,
  initialLinkCreationState,
  DragState,
  initialDragState,
  GroupSelectionState,
  initialGroupSelectionState,
  PolygonDrawState,
  initialPolygonDrawState,
  MeasureState,
  initialMeasureState,
  MeasurementMarker,
  DrawnObjectsState,
  initialDrawnObjectsState,
  createDrawnObject,
  MoveGroupState,
  initialMoveGroupState,
  MergePolygonState,
  initialMergePolygonState,
  PathDrawState,
  initialPathDrawState,
  TargetPath,
  createTargetPath,
  findNearestJoint,
  findNearestLink,
  findMergeTarget,
  calculateDistance,
  getDefaultColor,
  findConnectedMechanism,
  findElementsInBox,
  isPointInPolygon,
  areLinkEndpointsInPolygon,
  MERGE_THRESHOLD,
  DraggableToolbar,
  ToolbarToggleButtons,
  TOOLBAR_CONFIGS,
  ToolbarPosition,
  JointEditModal,
  JointData,
  LinkEditModal,
  LinkData
} from './BuilderTools'
import {
  jointColors,
  getCyclicColor,
  ColorCycleType
} from '../theme'
import {
  useAnimation,
  useSimulation,
  canSimulate
} from './AnimateSimulate'
import { validateLinks, LinkMeta as PylinkLinkMeta } from './Links'

// Import from builder module
import {
  // Types (legacy format for backward compatibility)
  PylinkDocument,
  // Types (hypergraph format)
  LinkageDocument,
  // Conversion utilities (legacy view for toolbars/rendering)
  convertLinkageDocumentToLegacy,
  // Constants
  MIN_SIMULATION_STEPS,
  MAX_SIMULATION_STEPS,
  DEFAULT_SIMULATION_STEPS,
  DEFAULT_AUTO_SIMULATE_DELAY_MS,
  DEFAULT_JOINT_MERGE_RADIUS,
  DEFAULT_TRAJECTORY_COLOR_CYCLE,
  PIXELS_PER_UNIT,
  CANVAS_MIN_WIDTH_PX,
  CANVAS_MIN_HEIGHT_PX,
  // Toolbar components
  ToolsToolbar,
  LinksToolbar,
  NodesToolbar,
  MoreToolbar,
  SettingsToolbar,
  OptimizationToolbar,
  // Toolbar types
  type CanvasBgColor,
  type TrajectoryStyle,
  type SelectionHighlightColor,
  type OptMethod,
  type SmoothMethod,
  type ResampleMethod,
  // Rendering components
  SVGFilters,
  // Hypergraph helpers (new API)
  getNode,
  getConnectedNodes,
  // Hypergraph mutations (new API)
  moveNode as moveNodeMutation,
  syncAllEdgeDistances,
  updateEdgeMeta,
  updateNodeMeta,
  // Hypergraph operations (new API)
  deleteNode as deleteNodeOp,
  deleteEdge as deleteEdgeOp,
  deleteNodes as deleteNodesOp,
  deleteEdges as deleteEdgesOp,
  moveNodesFromOriginal,
  mergeNodesOperation,
  renameEdgeOperation,
  renameNodeOperation,
  // Link creation
  createLinkBetweenPoints,
  changeNodeRole
} from './builder'

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const BuilderTab: React.FC = () => {
  // Canvas scaling helpers
  const pixelsToUnits = (pixels: number) => pixels / PIXELS_PER_UNIT
  const unitsToPixels = (units: number) => units * PIXELS_PER_UNIT

  // State
  const [toolMode, setToolMode] = useState<ToolMode>('select')
  const [openToolbars, setOpenToolbars] = useState<Set<string>>(new Set(['tools', 'more']))
  const [toolbarPositions, setToolbarPositions] = useState<Record<string, ToolbarPosition>>({})
  const [hoveredTool, setHoveredTool] = useState<ToolMode | null>(null)

  // Edit modal state - holds data for the modal dialogs
  const [editingJointData, setEditingJointData] = useState<JointData | null>(null)
  const [editingLinkData, setEditingLinkData] = useState<LinkData | null>(null)

  // Document state - using hypergraph format internally
  // Reference: https://github.com/HugoFara/pylinkage/tree/main/src/pylinkage/hypergraph
  const [linkageDoc, setLinkageDoc] = useState<LinkageDocument>({
    name: 'untitled',
    version: '2.0.0',
    linkage: {
      name: 'untitled',
      nodes: {},
      edges: {},
      hyperedges: {}
    },
    meta: {
      nodes: {},
      edges: {}
    }
  })

  // ═══════════════════════════════════════════════════════════════════════════════
  // LEGACY FORMAT VIEW - Computed from hypergraph state for backward-compatible APIs
  // ═══════════════════════════════════════════════════════════════════════════════

  // Convert new LinkageDocument to legacy PylinkDocument format for backward compatibility
  // Used by: toolbars, rendering, validation, etc. that still use legacy format
  const pylinkDoc: PylinkDocument = useMemo(() => {
    return convertLinkageDocumentToLegacy(linkageDoc)
  }, [linkageDoc])

  // UI state
  const [selectedJoints, setSelectedJoints] = useState<string[]>([])
  const [selectedLinks, setSelectedLinks] = useState<string[]>([])
  const [hoveredJoint, setHoveredJoint] = useState<string | null>(null)
  const [hoveredLink, setHoveredLink] = useState<string | null>(null)

  // Status message system
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null)

  // Link creation state
  const [linkCreationState, setLinkCreationState] = useState<LinkCreationState>(initialLinkCreationState)
  const [previewLine, setPreviewLine] = useState<{ start: [number, number]; end: [number, number] } | null>(null)

  // Drag state for move/merge functionality
  const [dragState, setDragState] = useState<DragState>(initialDragState)

  // Group selection state (for box selection)
  const [groupSelectionState, setGroupSelectionState] = useState<GroupSelectionState>(initialGroupSelectionState)

  // Polygon drawing state
  const [polygonDrawState, setPolygonDrawState] = useState<PolygonDrawState>(initialPolygonDrawState)

  // Measure tool state
  const [measureState, setMeasureState] = useState<MeasureState>(initialMeasureState)
  const [measurementMarkers, setMeasurementMarkers] = useState<MeasurementMarker[]>([])

  // Merge polygon state (for merging polygons with links)
  const [mergePolygonState, setMergePolygonState] = useState<MergePolygonState>(initialMergePolygonState)

  // Target path drawing state (for trajectory optimization)
  const [pathDrawState, setPathDrawState] = useState<PathDrawState>(initialPathDrawState)
  const [targetPaths, setTargetPaths] = useState<TargetPath[]>([])
  const [selectedPathId, setSelectedPathId] = useState<string | null>(null)

  // Optimization state
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [preOptimizationDoc, setPreOptimizationDoc] = useState<LinkageDocument | null>(null)
  const [optimizationResult, setOptimizationResult] = useState<{
    success: boolean
    initialError: number
    finalError: number
    message: string
    iterations?: number
    executionTimeMs?: number
    optimizedDimensions?: Record<string, number>
    originalDimensions?: Record<string, number>
  } | null>(null)

  // Optimization hyperparameters
  const [optMethod, setOptMethod] = useState<'pso' | 'pylinkage' | 'scipy' | 'powell' | 'nelder-mead'>('pylinkage')
  const [optNParticles, setOptNParticles] = useState(32)
  const [optIterations, setOptIterations] = useState(512)
  const [optMaxIterations, setOptMaxIterations] = useState(100)
  const [optTolerance, setOptTolerance] = useState(1e-6)
  const [optBoundsFactor, setOptBoundsFactor] = useState(2.0)
  const [optMinLength, setOptMinLength] = useState(5)
  const [optVerbose, setOptVerbose] = useState(true)

  // Trajectory preprocessing state
  const [prepEnableSmooth, setPrepEnableSmooth] = useState(true)
  const [prepSmoothWindow, setPrepSmoothWindow] = useState(4)
  const [prepSmoothPolyorder, setPrepSmoothPolyorder] = useState(3)
  const [prepSmoothMethod, setPrepSmoothMethod] = useState<'savgol' | 'moving_avg' | 'gaussian'>('savgol')
  const [prepEnableResample, setPrepEnableResample] = useState(true)
  const [prepTargetNSteps, setPrepTargetNSteps] = useState(DEFAULT_SIMULATION_STEPS)
  const [prepResampleMethod, setPrepResampleMethod] = useState<'parametric' | 'cubic' | 'linear'>('parametric')
  const [isPreprocessing, setIsPreprocessing] = useState(false)
  const [preprocessResult, setPreprocessResult] = useState<{
    originalPoints: number
    outputPoints: number
    analysis: Record<string, unknown>
  } | null>(null)

  // Drawn objects state (polygons, shapes that can be attached to links)
  const [drawnObjects, setDrawnObjects] = useState<DrawnObjectsState>(initialDrawnObjectsState)

  // Delete confirmation dialog state
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState<{
    open: boolean
    joints: string[]
    links: string[]
  }>({ open: false, joints: [], links: [] })

  // Move group state (for moving groups of joints and/or DrawnObjects)
  const [moveGroupState, setMoveGroupState] = useState<MoveGroupState>(initialMoveGroupState)

  // Simulation state - using hooks from AnimateSimulate
  const [simulationSteps, setSimulationSteps] = useState(DEFAULT_SIMULATION_STEPS)
  const [simulationStepsInput, setSimulationStepsInput] = useState(String(DEFAULT_SIMULATION_STEPS))  // Local input for debouncing
  const [mechanismVersion, setMechanismVersion] = useState(0)  // Increments on any change
  const [showTrajectory, setShowTrajectory] = useState(true)  // Toggle trajectory visibility

  // Settings state (configurable via Settings panel)
  const [autoSimulateDelayMs, setAutoSimulateDelayMs] = useState(DEFAULT_AUTO_SIMULATE_DELAY_MS)
  const [jointMergeRadius, setJointMergeRadius] = useState(DEFAULT_JOINT_MERGE_RADIUS)
  const [trajectoryColorCycle, setTrajectoryColorCycle] = useState<ColorCycleType>(DEFAULT_TRAJECTORY_COLOR_CYCLE)
  const [darkMode, setDarkMode] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [showJointLabels, setShowJointLabels] = useState(false)
  const [showLinkLabels, setShowLinkLabels] = useState(false)

  // Canvas/Visualization settings
  const [canvasBgColor, setCanvasBgColor] = useState<'default' | 'white' | 'cream' | 'dark'>('default')
  const [jointSize, setJointSize] = useState(8)  // 3-16px
  const [linkThickness, setLinkThickness] = useState(8)  // 1-16px
  const [trajectoryDotSize, setTrajectoryDotSize] = useState(4)  // 2-6px
  const [selectionHighlightColor, setSelectionHighlightColor] = useState<'blue' | 'orange' | 'green' | 'purple'>('blue')
  const [showMeasurementUnits, setShowMeasurementUnits] = useState(true)
  const [trajectoryStyle, setTrajectoryStyle] = useState<'dots' | 'line' | 'both'>('both')

  // Derived selection color
  const selectionColorMap = { blue: '#1976d2', orange: '#FA8112', green: '#2e7d32', purple: '#9c27b0' }
  const selectionColor = selectionColorMap[selectionHighlightColor]

  // ═══════════════════════════════════════════════════════════════════════════════
  // HIGHLIGHT HELPER - Creates consistent glow/outline effects for selected objects
  // ═══════════════════════════════════════════════════════════════════════════════
  type HighlightType = 'none' | 'selected' | 'hovered' | 'move_group' | 'merge'
  type ObjectType = 'joint' | 'link' | 'polygon'

  /**
   * Get glow filter ID based on color - maps colors to pre-defined SVG filters
   * Objects glow in their own original color, not the selection color
   */
  const getGlowFilterForColor = useCallback((color: string): string => {
    const colorLower = color.toLowerCase()

    // Joint type colors
    if (colorLower === '#e74c3c' || colorLower === jointColors.static.toLowerCase()) return 'url(#glow-static)'
    if (colorLower === '#f39c12' || colorLower === jointColors.crank.toLowerCase()) return 'url(#glow-crank)'
    if (colorLower === '#2196f3' || colorLower === jointColors.pivot.toLowerCase()) return 'url(#glow-pivot)'

    // Graph colors (D3 palette)
    if (colorLower === '#1f77b4') return 'url(#glow-blue)'
    if (colorLower === '#ff7f0e') return 'url(#glow-orange)'
    if (colorLower === '#2ca02c') return 'url(#glow-green)'
    if (colorLower === '#d62728') return 'url(#glow-red)'
    if (colorLower === '#9467bd') return 'url(#glow-purple)'
    if (colorLower === '#8c564b') return 'url(#glow-brown)'
    if (colorLower === '#e377c2') return 'url(#glow-pink)'
    if (colorLower === '#7f7f7f') return 'url(#glow-gray)'
    if (colorLower === '#bcbd22') return 'url(#glow-olive)'
    if (colorLower === '#17becf') return 'url(#glow-cyan)'

    // Default to blue glow for unknown colors
    return 'url(#glow-blue)'
  }, [])

  /**
   * Get highlight styling for objects (joints, links, polygons)
   * Objects glow in their ORIGINAL color when selected/hovered
   * Move group uses grey glow
   */
  const getHighlightStyle = useCallback((
    objectType: ObjectType,
    highlightType: HighlightType,
    baseColor: string,  // The object's original color (for glow)
    baseStrokeWidth: number
  ): { stroke: string; strokeWidth: number; filter?: string } => {
    // No highlight - return base styling
    if (highlightType === 'none') {
      return { stroke: baseColor, strokeWidth: baseStrokeWidth }
    }

    // Different stroke widths based on object type and highlight state
    const glowStrokeWidth = objectType === 'joint'
      ? baseStrokeWidth + 1
      : objectType === 'link'
        ? baseStrokeWidth + 2
        : baseStrokeWidth + 1

    // Move group and merge use special colors, selected/hovered use object's original color
    switch (highlightType) {
      case 'move_group':
        return {
          stroke: jointColors.moveGroup,
          strokeWidth: glowStrokeWidth,
          filter: 'url(#glow-movegroup)'
        }
      case 'merge':
        return {
          stroke: jointColors.mergeHighlight,
          strokeWidth: glowStrokeWidth,
          filter: 'url(#glow-merge)'
        }
      case 'selected':
      case 'hovered':
      default:
        // Glow in the object's original color
        return {
          stroke: baseColor,
          strokeWidth: glowStrokeWidth,
          filter: getGlowFilterForColor(baseColor)
        }
    }
  }, [getGlowFilterForColor])

  // Canvas dimensions
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 1200, height: 700 })

  const canvasRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Animation: positions override during playback
  const [animatedPositions, setAnimatedPositions] = useState<Record<string, [number, number]> | null>(null)

  // Update canvas dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setCanvasDimensions({
          width: rect.width,
          height: Math.max(rect.height - 48, CANVAS_MIN_HEIGHT_PX) // Account for footer
        })
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Apply dark mode class to document body
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode')
    } else {
      document.body.classList.remove('dark-mode')
    }
  }, [darkMode])

  // Debounced simulation steps update - wait 500ms after user stops typing
  const simulationStepsInputRef = useRef(simulationStepsInput)
  simulationStepsInputRef.current = simulationStepsInput
  const prevSimulationStepsRef = useRef(simulationSteps)

  useEffect(() => {
    const timer = setTimeout(() => {
      const val = parseInt(simulationStepsInputRef.current)
      if (!isNaN(val)) {
        const clamped = Math.max(MIN_SIMULATION_STEPS, Math.min(MAX_SIMULATION_STEPS, val))
        setSimulationSteps(prev => {
          if (clamped !== prev) {
            // Update input to show clamped value
            setSimulationStepsInput(String(clamped))
            return clamped
          }
          return prev
        })
      }
    }, 500)
    return () => clearTimeout(timer)
  }, [simulationStepsInput])

  // Trigger simulation when simulationSteps changes (after debounce)
  useEffect(() => {
    if (simulationSteps !== prevSimulationStepsRef.current) {
      prevSimulationStepsRef.current = simulationSteps
      // Trigger mechanism change to re-run simulation with new step count
      setMechanismVersion(v => v + 1)
    }
  }, [simulationSteps])

  // Helper to show status messages
  const showStatus = useCallback((text: string, type: StatusType = 'info', duration?: number) => {
    setStatusMessage({ text, type, timestamp: Date.now() })

    if (duration && type !== 'action') {
      setTimeout(() => {
        setStatusMessage(prev => {
          if (prev && prev.text === text) return null
          return prev
        })
      }, duration)
    }
  }, [])

  // Clear status
  const clearStatus = useCallback(() => {
    setStatusMessage(null)
  }, [])

  // Trigger mechanism change (for auto-simulation)
  const triggerMechanismChange = useCallback(() => {
    setMechanismVersion(v => v + 1)
  }, [])

  // Handle animation frame changes - update joint positions visually
  const handleAnimationFrameChange = useCallback((_frame: number) => {
    // This will be called by the animation hook when the frame changes
    // We update animatedPositions in the useEffect below
  }, [])

  // Simulation hook - handles trajectory computation and auto-simulate
  const {
    isSimulating,
    trajectoryData,
    runSimulation,
    clearTrajectory,
    setAutoSimulateEnabled,
    autoSimulateEnabled
  } = useSimulation({
    linkageDoc,  // Send hypergraph format directly - backend handles conversion
    simulationSteps,
    autoSimulateDelayMs,
    autoSimulateEnabled: true,  // Start with continuous simulation ON by default
    mechanismVersion,
    showStatus
  })

  // Animation hook - handles playback of simulation frames
  const {
    animationState,
    play: playAnimation,
    pause: pauseAnimation,
    stop: stopAnimation,
    reset: _resetAnimation,
    setFrame: _setAnimationFrame,
    setPlaybackSpeed,
    getAnimatedPositions
  } = useAnimation({
    trajectoryData,
    onFrameChange: handleAnimationFrameChange,
    frameIntervalMs: 50  // 20fps default
  })

  // Update animated positions when animation frame changes
  useEffect(() => {
    if (animationState.isAnimating) {
      const positions = getAnimatedPositions()
      setAnimatedPositions(positions)
    } else if (!animationState.isAnimating && animationState.currentFrame === 0) {
      // Reset to original positions when stopped
      setAnimatedPositions(null)
    }
  }, [animationState.isAnimating, animationState.currentFrame, getAnimatedPositions])

  // Track invalid links that would stretch during animation
  const [stretchingLinks, setStretchingLinks] = useState<string[]>([])

  // Validate links after simulation completes - detect links that would stretch
  useEffect(() => {
    if (trajectoryData && trajectoryData.trajectories) {
      // Convert local LinkMeta format to PylinkLinkMeta format for validation
      const linksForValidation: Record<string, PylinkLinkMeta> = {}
      for (const [name, link] of Object.entries(pylinkDoc.meta.links)) {
        if (link.connects && link.connects.length === 2) {
          linksForValidation[name] = {
            color: link.color,
            connects: [link.connects[0], link.connects[1]] as [string, string],
            isGround: link.isGround
          }
        }
      }

      const validation = validateLinks(
        linksForValidation,
        trajectoryData.jointTypes,
        trajectoryData.trajectories
      )

      // Use the stretchingLinks directly from validation result
      if (validation.stretchingLinks.length > 0) {
        setStretchingLinks(validation.stretchingLinks)

        // Show warning for stretching links
        const problemLinksStr = validation.stretchingLinks.join(', ')
        showStatus(
          `⚠️ Invalid mechanism: ${problemLinksStr} would stretch during animation. ` +
          `Links must connect joints that are both in the kinematic chain.`,
          'warning',
          5000
        )
        console.warn('Link validation problems:', validation.problems)

        // Stop any running animation - can't animate invalid mechanism
        if (animationState.isAnimating) {
          pauseAnimation()
        }
      } else {
        setStretchingLinks([])
      }
    } else {
      setStretchingLinks([])
    }
  }, [trajectoryData, pylinkDoc.meta.links, showStatus, animationState.isAnimating, pauseAnimation])

  // Cancel current action
  const cancelAction = useCallback(() => {
    if (linkCreationState.isDrawing) {
      setLinkCreationState(initialLinkCreationState)
      setPreviewLine(null)
      showStatus('Link creation cancelled', 'info', 2000)
    }
    if (dragState.isDragging) {
      setDragState(initialDragState)
      showStatus('Drag cancelled', 'info', 2000)
    }
    if (groupSelectionState.isSelecting) {
      setGroupSelectionState(initialGroupSelectionState)
      showStatus('Group selection cancelled', 'info', 2000)
    }
    if (polygonDrawState.isDrawing) {
      setPolygonDrawState(initialPolygonDrawState)
      showStatus('Polygon drawing cancelled', 'info', 2000)
    }
    if (measureState.isMeasuring) {
      setMeasureState(initialMeasureState)
      showStatus('Measurement cancelled', 'info', 2000)
    }
    if (mergePolygonState.step !== 'idle' && mergePolygonState.step !== 'awaiting_selection') {
      setMergePolygonState(initialMergePolygonState)
      setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
      setSelectedLinks([])
      showStatus('Merge cancelled', 'info', 2000)
    }
    if (pathDrawState.isDrawing) {
      setPathDrawState(initialPathDrawState)
      showStatus('Path drawing cancelled', 'info', 2000)
    }
    setToolMode('select')
  }, [linkCreationState.isDrawing, dragState.isDragging, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, mergePolygonState.step, pathDrawState.isDrawing, showStatus])

  // Get visual position for any joint
  // SINGLE SOURCE OF TRUTH:
  // - Static joints: position from pylinkage data (x, y)
  // - Crank/Revolute joints: position from meta.joints (x, y) if available,
  //   otherwise calculate from parent relationships
  const getJointPosition = useCallback((jointName: string): [number, number] | null => {
    // When animating, use animated positions for moving joints
    if (animatedPositions && animatedPositions[jointName]) {
      return animatedPositions[jointName]
    }

    const joint = pylinkDoc.pylinkage.joints.find(j => j.name === jointName)
    if (!joint) return null

    // For Static joints, always use the stored x, y
    if (joint.type === 'Static') {
      return [joint.x, joint.y]
    }

    // For non-Static joints, check meta.joints first (single source of truth for UI position)
    const meta = pylinkDoc.meta.joints[jointName]
    if (meta?.x !== undefined && meta?.y !== undefined) {
      return [meta.x, meta.y]
    }

    // Fallback: calculate position from parent relationships (for initial load/demo)
    if (joint.type === 'Crank') {
      const parent = pylinkDoc.pylinkage.joints.find(j => j.name === joint.joint0.ref)
      if (parent && parent.type === 'Static') {
        const x = parent.x + joint.distance * Math.cos(joint.angle)
        const y = parent.y + joint.distance * Math.sin(joint.angle)
        return [x, y]
      }
    } else if (joint.type === 'Revolute') {
      // For Revolute, use circle-circle intersection or approximate from distances
      const parent0 = pylinkDoc.pylinkage.joints.find(j => j.name === joint.joint0.ref)
      const parent1 = pylinkDoc.pylinkage.joints.find(j => j.name === joint.joint1.ref)
      if (parent0 && parent1) {
        const pos0 = getJointPosition(parent0.name)
        const pos1 = getJointPosition(parent1.name)
        if (pos0 && pos1) {
          // Circle-circle intersection to find the joint position
          const d0 = joint.distance0
          const d1 = joint.distance1
          const dx = pos1[0] - pos0[0]
          const dy = pos1[1] - pos0[1]
          const d = Math.sqrt(dx * dx + dy * dy)

          if (d > 0 && d <= d0 + d1 && d >= Math.abs(d0 - d1)) {
            // Valid triangle, compute intersection point
            const a = (d0 * d0 - d1 * d1 + d * d) / (2 * d)
            const h = Math.sqrt(Math.max(0, d0 * d0 - a * a))
            const px = pos0[0] + (a * dx) / d
            const py = pos0[1] + (a * dy) / d
            // Return one of the two intersection points (above the line)
            const x = px - (h * dy) / d
            const y = py + (h * dx) / d
            return [x, y]
          }
          // Fallback if geometry doesn't work
          return [(pos0[0] + pos1[0]) / 2, (pos0[1] + pos1[1]) / 2 - 20]
        }
      }
    }
    return null
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints, animatedPositions])

  // Get all joints with their positions for snapping
  const getJointsWithPositions = useCallback(() => {
    return pylinkDoc.pylinkage.joints.map(joint => ({
      name: joint.name,
      position: getJointPosition(joint.name)
    }))
  }, [pylinkDoc.pylinkage.joints, getJointPosition])

  // ═══════════════════════════════════════════════════════════════════════════════
  // MODAL DATA BUILDERS - Create data for edit modals
  // ═══════════════════════════════════════════════════════════════════════════════

  /**
   * Build JointData for the joint edit modal
   */
  const buildJointData = useCallback((jointName: string): JointData | null => {
    const joint = pylinkDoc.pylinkage.joints.find(j => j.name === jointName)
    if (!joint) return null

    const position = getJointPosition(jointName)
    const meta = pylinkDoc.meta.joints[jointName]

    // Find connected links
    const connectedLinks = Object.entries(pylinkDoc.meta.links)
      .filter(([_, linkMeta]) => linkMeta.connects.includes(jointName))
      .map(([linkName]) => linkName)

    const baseData: JointData = {
      name: joint.name,
      type: joint.type,
      position,
      connectedLinks,
      showPath: meta?.show_path ?? false
    }

    // Add type-specific data
    if (joint.type === 'Crank') {
      return {
        ...baseData,
        parentJoint: joint.joint0.ref,
        distance: joint.distance,
        angle: joint.angle
      }
    } else if (joint.type === 'Revolute') {
      return {
        ...baseData,
        parentJoint: joint.joint0.ref,
        parentJoint2: joint.joint1.ref,
        distance: joint.distance0,
        distance2: joint.distance1
      }
    }

    return baseData
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints, pylinkDoc.meta.links, getJointPosition])

  /**
   * Build LinkData for the link edit modal
   */
  const buildLinkData = useCallback((linkName: string): LinkData | null => {
    const linkMeta = pylinkDoc.meta.links[linkName]
    if (!linkMeta || linkMeta.connects.length < 2) return null

    const pos0 = getJointPosition(linkMeta.connects[0])
    const pos1 = getJointPosition(linkMeta.connects[1])
    const length = pos0 && pos1 ? calculateDistance(pos0, pos1) : null

    // Get link index for default color
    const linkIndex = Object.keys(pylinkDoc.meta.links).indexOf(linkName)

    return {
      name: linkName,
      color: linkMeta.color || getDefaultColor(linkIndex),
      connects: [linkMeta.connects[0], linkMeta.connects[1]],
      length,
      isGround: linkMeta.isGround || false,
      jointPositions: [pos0, pos1]
    }
  }, [pylinkDoc.meta.links, getJointPosition])

  /**
   * Open joint edit modal
   */
  const openJointEditModal = useCallback((jointName: string) => {
    const data = buildJointData(jointName)
    if (data) {
      setEditingJointData(data)
    }
  }, [buildJointData])

  /**
   * Open link edit modal
   */
  const openLinkEditModal = useCallback((linkName: string) => {
    const data = buildLinkData(linkName)
    if (data) {
      setEditingLinkData(data)
    }
  }, [buildLinkData])

  // Get all links with their positions for snapping
  const getLinksWithPositions = useCallback(() => {
    return Object.entries(pylinkDoc.meta.links).map(([name, meta]) => {
      const start = getJointPosition(meta.connects[0])
      const end = getJointPosition(meta.connects[1])
      return { name, start, end }
    })
  }, [pylinkDoc.meta.links, getJointPosition])

  // Enter move group mode - allows moving selected joints and/or DrawnObjects as a rigid body
  const enterMoveGroupMode = useCallback((
    jointNames: string[],
    drawnObjectIds: string[] = []
  ) => {
    // Store original positions of all joints
    const startPositions: Record<string, [number, number]> = {}
    jointNames.forEach(jointName => {
      const pos = getJointPosition(jointName)
      if (pos) {
        startPositions[jointName] = pos
      }
    })

    // Store original positions of all drawn object points
    const drawnObjectStartPositions: Record<string, [number, number][]> = {}
    drawnObjectIds.forEach(id => {
      const obj = drawnObjects.objects.find(o => o.id === id)
      if (obj) {
        drawnObjectStartPositions[id] = [...obj.points]
      }
    })

    setMoveGroupState({
      isActive: true,
      isDragging: false,
      joints: jointNames,
      drawnObjectIds,
      startPositions,
      drawnObjectStartPositions,
      dragStartPoint: null
    })

    const totalItems = jointNames.length + drawnObjectIds.length
    showStatus(`Move mode: ${totalItems} items selected — click and drag to move`, 'action')
  }, [getJointPosition, drawnObjects.objects, showStatus])

  // Exit move group mode
  const exitMoveGroupMode = useCallback(() => {
    setMoveGroupState(initialMoveGroupState)
    setToolMode('select')
    clearStatus()
  }, [clearStatus])

  // Delete a link (edge) and any orphan nodes
  const deleteLink = useCallback((linkName: string) => {
    // Use new hypergraph operation
    const result = deleteEdgeOp(linkageDoc, linkName)

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    // Update state directly with hypergraph format
    setLinkageDoc(result.doc)

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus])

  // Delete a joint (node) and all connected edges, plus any resulting orphans
  const deleteJoint = useCallback((jointName: string) => {
    // Use new hypergraph operation
    const result = deleteNodeOp(linkageDoc, jointName)

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    // Update state directly with hypergraph format
    setLinkageDoc(result.doc)

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus])

  // Move a joint (node) to a new position
  // In hypergraph format, we just update the node position and sync edge distances
  const moveJoint = useCallback((jointName: string, newPosition: [number, number]) => {
    const node = getNode(linkageDoc, jointName)
    if (!node) return

    // Move the node to new position
    let newDoc = moveNodeMutation(linkageDoc, jointName, newPosition)

    // Sync all edge distances from the new positions
    // This ensures the kinematic constraints match the visual positions
    newDoc = syncAllEdgeDistances(newDoc)

    // Clear trajectory and trigger auto-simulation if enabled
    clearTrajectory()
    triggerMechanismChange()

    setLinkageDoc(newDoc)
  }, [linkageDoc])

  // ═══════════════════════════════════════════════════════════════════════════════
  // RIGID BODY TRANSLATION
  // ═══════════════════════════════════════════════════════════════════════════════
  // Moves a group of nodes to new positions based on original positions + delta.
  // This is a pure translation - edge distances are NOT recalculated.
  // ═══════════════════════════════════════════════════════════════════════════════
  const translateGroupRigid = useCallback((
    jointNames: string[],
    originalPositions: Record<string, [number, number]>,
    dx: number,
    dy: number
  ) => {
    if (jointNames.length === 0) return

    // Use new hypergraph operation for rigid body translation
    const result = moveNodesFromOriginal(linkageDoc, originalPositions, [dx, dy])
    setLinkageDoc(result.doc)
  }, [linkageDoc])

  // Merge two joints (nodes) together (source is absorbed into target)
  const mergeJoints = useCallback((sourceJoint: string, targetJoint: string) => {
    // Use new hypergraph operation
    const result = mergeNodesOperation(linkageDoc, sourceJoint, targetJoint)

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    setLinkageDoc(result.doc)
    setSelectedJoints([targetJoint])
    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, triggerMechanismChange])
  // Create a new link between two points/joints using hypergraph operations
  // If user clicked on an existing joint, use it. Otherwise create a new joint.
  // New joints become 'follower' if connected to a kinematic node, 'fixed' otherwise.
  const createLinkWithRevoluteDefault = useCallback((
    startPoint: [number, number],
    endPoint: [number, number],
    startJointName: string | null,  // Only set if user clicked on an existing joint
    endJointName: string | null      // Only set if user clicked on an existing joint
  ) => {
    // Helper to get connected node IDs for a given node
    const getConnectedNodeIds = (nodeId: string): string[] => {
      return getConnectedNodes(linkageDoc, nodeId)
    }

    // Use the hypergraph link creation function
    const result = createLinkBetweenPoints(
      linkageDoc,
      startPoint,
      endPoint,
      startJointName,
      endJointName,
      getConnectedNodeIds
    )

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    // Update state
    setLinkageDoc(result.doc)

    showStatus(result.message, 'success', 2500)

    return result.edgeId
  }, [linkageDoc, showStatus, triggerMechanismChange])

  // Handle mouse down on canvas (for drag start)
  const handleCanvasMouseDown = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const clickPoint: [number, number] = [x, y]

    // Handle group select mode - start box selection
    if (toolMode === 'group_select') {
      setGroupSelectionState({
        isSelecting: true,
        startPoint: clickPoint,
        currentPoint: clickPoint
      })
      showStatus('Drag to select multiple elements', 'action')
      return
    }

    // Handle move group mode - start dragging
    if (moveGroupState.isActive && moveGroupState.joints.length > 0) {
      // Check multiple ways the user might be clicking on the selection:
      // 1. Directly on a joint in the move group
      // 2. On a link connecting joints in the move group
      // 3. Within the bounding box of the selection (more lenient)

      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)
      const nearestLink = findNearestLink(clickPoint, linksWithPositions)

      // Check 1: Click directly on a joint in the move group
      const clickedOnJoint = nearestJoint && moveGroupState.joints.includes(nearestJoint.name)

      // Check 2: Click on a link where BOTH endpoints are in the move group
      const clickedOnGroupLink = nearestLink && (() => {
        const linkMeta = pylinkDoc.meta.links[nearestLink.name]
        if (!linkMeta) return false
        return linkMeta.connects.every(j => moveGroupState.joints.includes(j))
      })()

      // Check 3: Click within the bounding box of all selected joints (with padding)
      const BBOX_PADDING = 0.5  // Units of padding around the bounding box
      const clickedInBoundingBox = (() => {
        const positions = moveGroupState.joints
          .map(jointName => getJointPosition(jointName))
          .filter((pos): pos is [number, number] => pos !== null)

        if (positions.length === 0) return false

        const minX = Math.min(...positions.map(p => p[0])) - BBOX_PADDING
        const maxX = Math.max(...positions.map(p => p[0])) + BBOX_PADDING
        const minY = Math.min(...positions.map(p => p[1])) - BBOX_PADDING
        const maxY = Math.max(...positions.map(p => p[1])) + BBOX_PADDING

        return clickPoint[0] >= minX && clickPoint[0] <= maxX &&
               clickPoint[1] >= minY && clickPoint[1] <= maxY
      })()

      // Check 4: Click on a selected drawn object
      const clickedOnDrawnObj = moveGroupState.drawnObjectIds.some(id => {
        const obj = drawnObjects.objects.find(o => o.id === id)
        if (!obj) return false
        const minX = Math.min(...obj.points.map(p => p[0]))
        const maxX = Math.max(...obj.points.map(p => p[0]))
        const minY = Math.min(...obj.points.map(p => p[1]))
        const maxY = Math.max(...obj.points.map(p => p[1]))
        return clickPoint[0] >= minX && clickPoint[0] <= maxX &&
               clickPoint[1] >= minY && clickPoint[1] <= maxY
      })

      if (clickedOnJoint || clickedOnGroupLink || clickedInBoundingBox || clickedOnDrawnObj) {
        setMoveGroupState(prev => ({
          ...prev,
          isDragging: true,
          dragStartPoint: clickPoint
        }))
        const itemCount = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
        showStatus(`Moving ${itemCount} items — drag to reposition`, 'action')
        return
      } else {
        // Clicked outside selection - exit move mode
        exitMoveGroupMode()
        return  // Important: return to prevent other handlers from running
      }
    }

    // Handle select mode - drag joints
    if (toolMode === 'select') {
    const jointsWithPositions = getJointsWithPositions()
    const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)

    if (nearestJoint) {
      // Start dragging
      setDragState({
        isDragging: true,
        draggedJoint: nearestJoint.name,
        dragStartPosition: nearestJoint.position,
        currentPosition: nearestJoint.position,
        mergeTarget: null,
        mergeProximity: Infinity
      })
      setSelectedJoints([nearestJoint.name])
      setSelectedLinks([])
      showStatus(`Dragging ${nearestJoint.name}`, 'action')
    }
    }
  }, [toolMode, selectedJoints, getJointsWithPositions, getJointPosition, showStatus])

  // Handle mouse move on canvas
  const handleCanvasMouseMove = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const currentPoint: [number, number] = [x, y]

    // Handle group selection box preview
    if (groupSelectionState.isSelecting && groupSelectionState.startPoint) {
      setGroupSelectionState(prev => ({
        ...prev,
        currentPoint
      }))

      // Preview what would be selected
      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const box = {
        x1: groupSelectionState.startPoint[0],
        y1: groupSelectionState.startPoint[1],
        x2: currentPoint[0],
        y2: currentPoint[1]
      }
      const preview = findElementsInBox(box, jointsWithPositions, linksWithPositions)
      showStatus(`Selecting ${preview.joints.length} joints, ${preview.links.length} links`, 'action')
      return
    }

    // Handle link creation preview
    if (linkCreationState.isDrawing) {
      const jointsWithPositions = getJointsWithPositions()
      const nearestJoint = findNearestJoint(currentPoint, jointsWithPositions)
      const endPoint: [number, number] = nearestJoint?.position || currentPoint

      setPreviewLine({
        start: linkCreationState.startPoint!,
        end: endPoint
      })
      return
    }

    // Handle move group dragging (multiple joints and/or drawn objects)
    // Uses rigid body translation to preserve mechanism structure
    if (moveGroupState.isDragging && moveGroupState.dragStartPoint) {
      const dx = currentPoint[0] - moveGroupState.dragStartPoint[0]
      const dy = currentPoint[1] - moveGroupState.dragStartPoint[1]

      // Use rigid body translation - moves from ORIGINAL positions by delta
      // This preserves all distances and angles in the mechanism
      translateGroupRigid(
        moveGroupState.joints,
        moveGroupState.startPositions,  // Original positions before drag started
        dx,
        dy
      )

      // Move all drawn objects by the delta (from their original positions)
      if (moveGroupState.drawnObjectIds.length > 0) {
        setDrawnObjects(prev => ({
          ...prev,
          objects: prev.objects.map(obj => {
            if (moveGroupState.drawnObjectIds.includes(obj.id)) {
              const originalPoints = moveGroupState.drawnObjectStartPositions[obj.id]
              if (originalPoints) {
                return {
                  ...obj,
                  points: originalPoints.map(p => [p[0] + dx, p[1] + dy] as [number, number])
                }
              }
            }
            return obj
          })
        }))
      }

      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      showStatus(`Moving ${totalItems} items (Δ${dx.toFixed(1)}, ${dy.toFixed(1)})`, 'action')
      return
    }

    // Handle dragging single joint
    if (dragState.isDragging && dragState.draggedJoint) {
      const jointsWithPositions = getJointsWithPositions()
      const mergeTarget = findMergeTarget(currentPoint, jointsWithPositions, dragState.draggedJoint, jointMergeRadius)

      if (mergeTarget) {
        // Near a merge target
        setDragState(prev => ({
          ...prev,
          currentPosition: currentPoint,
          mergeTarget: mergeTarget.name,
          mergeProximity: mergeTarget.distance
        }))
        showStatus(`Release to merge into ${mergeTarget.name}`, 'action')
      } else {
        // Just moving
        setDragState(prev => ({
          ...prev,
          currentPosition: currentPoint,
          mergeTarget: null,
          mergeProximity: Infinity
        }))
        showStatus(`Moving ${dragState.draggedJoint} to (${x.toFixed(1)}, ${y.toFixed(1)})`, 'action')
      }

      // Update the joint position in real-time
      moveJoint(dragState.draggedJoint, currentPoint)
    }
  }, [linkCreationState.isDrawing, linkCreationState.startPoint, dragState.isDragging, dragState.draggedJoint, groupSelectionState.isSelecting, groupSelectionState.startPoint, moveGroupState, getJointsWithPositions, getLinksWithPositions, moveJoint, showStatus])

  // Handle mouse up on canvas (for drag end)
  const handleCanvasMouseUp = useCallback(() => {
    // Handle group selection completion
    if (groupSelectionState.isSelecting && groupSelectionState.startPoint && groupSelectionState.currentPoint) {
      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const box = {
        x1: groupSelectionState.startPoint[0],
        y1: groupSelectionState.startPoint[1],
        x2: groupSelectionState.currentPoint[0],
        y2: groupSelectionState.currentPoint[1]
      }
      const selected = findElementsInBox(box, jointsWithPositions, linksWithPositions)

      // Also check for drawn objects in the box
      const drawnObjectsInBox = drawnObjects.objects.filter(obj => {
        // Check if any point of the object is in the box
        const minX = Math.min(box.x1, box.x2)
        const maxX = Math.max(box.x1, box.x2)
        const minY = Math.min(box.y1, box.y2)
        const maxY = Math.max(box.y1, box.y2)
        return obj.points.some(p => p[0] >= minX && p[0] <= maxX && p[1] >= minY && p[1] <= maxY)
      }).map(obj => obj.id)

      // Also include DrawnObjects merged with selected links
      const mergedDrawnObjects = drawnObjects.objects
        .filter(obj => obj.mergedLinkName && selected.links.includes(obj.mergedLinkName))
        .map(obj => obj.id)

      // Combine both - those in box and those merged with selected links
      const allSelectedDrawnObjects = [...new Set([...drawnObjectsInBox, ...mergedDrawnObjects])]

      setSelectedJoints(selected.joints)
      setSelectedLinks(selected.links)
      setDrawnObjects(prev => ({ ...prev, selectedIds: allSelectedDrawnObjects }))
      setGroupSelectionState(initialGroupSelectionState)

      if (selected.joints.length > 0 || selected.links.length > 0 || allSelectedDrawnObjects.length > 0) {
        // Enter move mode with all selected items
        enterMoveGroupMode(selected.joints, allSelectedDrawnObjects)
      } else {
        showStatus('No elements selected', 'info', 1500)
      }
      return
    }

    // Handle move group drag completion
    if (moveGroupState.isDragging) {
      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      showStatus(`Moved ${totalItems} items`, 'success', 2000)
      // Reset drag state but keep move mode active
      setMoveGroupState(prev => ({
        ...prev,
        isDragging: false,
        dragStartPoint: null,
        // Update start positions to current positions for next drag
        startPositions: Object.fromEntries(
          prev.joints.map(jointName => {
            const pos = getJointPosition(jointName)
            return [jointName, pos || [0, 0]]
          })
        ),
        drawnObjectStartPositions: Object.fromEntries(
          prev.drawnObjectIds.map(id => {
            const obj = drawnObjects.objects.find(o => o.id === id)
            return [id, obj ? [...obj.points] : []]
          })
        )
      }))
      triggerMechanismChange()
      return
    }

    // Handle single joint drag completion
    if (!dragState.isDragging || !dragState.draggedJoint) return

    if (dragState.mergeTarget) {
      // Perform merge
      mergeJoints(dragState.draggedJoint, dragState.mergeTarget)
    } else if (dragState.currentPosition) {
      // Just moved - show final position
      showStatus(`Moved ${dragState.draggedJoint} to (${dragState.currentPosition[0].toFixed(1)}, ${dragState.currentPosition[1].toFixed(1)})`, 'success', 2000)
    }

    setDragState(initialDragState)
  }, [dragState, groupSelectionState, moveGroupState, drawnObjects.objects, getJointPosition, mergeJoints, getJointsWithPositions, getLinksWithPositions, enterMoveGroupMode, showStatus, triggerMechanismChange])

  // Batch delete multiple items at once using hypergraph operations
  const batchDelete = useCallback((jointsToDelete: string[], linksToDelete: string[], drawnObjectsToDelete: string[] = []) => {
    let doc = linkageDoc
    let totalDeletedEdges = linksToDelete.length
    let totalDeletedNodes = jointsToDelete.length

    // First delete edges (links) - this handles orphan detection
    if (linksToDelete.length > 0) {
      const edgeResult = deleteEdgesOp(doc, linksToDelete)
      doc = edgeResult.doc
      totalDeletedEdges = edgeResult.deletedEdges.length
      // Orphaned nodes are counted separately
      totalDeletedNodes += edgeResult.orphanedNodes.length
    }

    // Then delete nodes (joints) - this also handles connected edges and orphans
    if (jointsToDelete.length > 0) {
      const nodeResult = deleteNodesOp(doc, jointsToDelete)
      doc = nodeResult.doc
      totalDeletedNodes = nodeResult.deletedNodes.length
      totalDeletedEdges += nodeResult.deletedEdges.length
    }

    // Also delete DrawnObjects that are merged with any deleted link
    const allDrawnObjectsToDelete = new Set(drawnObjectsToDelete)
    const allDeletedEdges = new Set(linksToDelete)
    drawnObjects.objects.forEach(obj => {
      if (obj.mergedLinkName && allDeletedEdges.has(obj.mergedLinkName)) {
        allDrawnObjectsToDelete.add(obj.id)
      }
    })

    // Apply state update
    setLinkageDoc(doc)

    if (allDrawnObjectsToDelete.size > 0) {
      const newDrawnObjects = drawnObjects.objects.filter(obj => !allDrawnObjectsToDelete.has(obj.id))
      setDrawnObjects(prev => ({
        ...prev,
        objects: newDrawnObjects,
        selectedIds: prev.selectedIds.filter(id => !allDrawnObjectsToDelete.has(id))
      }))
    }

    // Clear selections and trigger update
    setSelectedJoints([])
    setSelectedLinks([])
    clearTrajectory()
    triggerMechanismChange()

    // Exit move mode if active
    if (moveGroupState.isActive) {
      setMoveGroupState(initialMoveGroupState)
    }

    return {
      deletedJoints: totalDeletedNodes,
      deletedLinks: totalDeletedEdges,
      deletedDrawnObjects: allDrawnObjectsToDelete.size
    }
  }, [linkageDoc, drawnObjects.objects, moveGroupState.isActive, triggerMechanismChange])

  // Handle delete with confirmation for multiple items
  const handleDeleteSelected = useCallback(() => {
    const totalItems = selectedJoints.length + selectedLinks.length + drawnObjects.selectedIds.length

    if (totalItems === 0) {
      showStatus('Nothing selected to delete', 'info', 1500)
      return
    }

    if (totalItems > 1) {
      // Show confirmation dialog for multiple items
      setDeleteConfirmDialog({
        open: true,
        joints: selectedJoints,
        links: selectedLinks
      })
    } else {
      // Single item - delete directly
      const result = batchDelete(selectedJoints, selectedLinks, drawnObjects.selectedIds)
      showStatus(`Deleted ${result.deletedJoints} joints, ${result.deletedLinks} links`, 'success', 2500)
    }
  }, [selectedJoints, selectedLinks, drawnObjects.selectedIds, batchDelete, showStatus])

  // Confirm delete multiple items
  const confirmDelete = useCallback(() => {
    const { joints, links } = deleteConfirmDialog
    const drawnObjectIds = drawnObjects.selectedIds

    const result = batchDelete(joints, links, drawnObjectIds)

    setDeleteConfirmDialog({ open: false, joints: [], links: [] })
    showStatus(`Deleted ${result.deletedJoints} joints, ${result.deletedLinks} links${result.deletedDrawnObjects > 0 ? `, ${result.deletedDrawnObjects} objects` : ''}`, 'success', 2500)
  }, [deleteConfirmDialog, drawnObjects.selectedIds, batchDelete, showStatus])

  // Complete path drawing (called by Enter key or double-click)
  const completePathDrawing = useCallback(() => {
    if (pathDrawState.isDrawing && pathDrawState.points.length >= 2) {
      const newPath = createTargetPath(pathDrawState.points, targetPaths)
      setTargetPaths(prev => [...prev, newPath])
      setSelectedPathId(newPath.id)
      setPathDrawState(initialPathDrawState)
      showStatus(`Created target path with ${pathDrawState.points.length} points`, 'success', 2500)
    } else if (pathDrawState.isDrawing) {
      showStatus('Path needs at least 2 points', 'warning', 2000)
    }
  }, [pathDrawState, targetPaths, showStatus])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return
      }

      // Escape to cancel
      if (event.key === 'Escape') {
        cancelAction()
        return
      }

      // Enter to complete path drawing
      if (event.key === 'Enter' && pathDrawState.isDrawing) {
        completePathDrawing()
        event.preventDefault()
        return
      }

      // Spacebar to play/pause animation
      if (event.key === ' ' || event.code === 'Space') {
        event.preventDefault()  // Prevent page scroll
        if (animationState.isAnimating) {
          pauseAnimation()
          showStatus('Animation paused', 'info', 1500)
        } else if (trajectoryData && trajectoryData.nSteps > 0) {
          playAnimation()
          showStatus('Animation playing', 'info', 1500)
        } else if (canSimulate(pylinkDoc.pylinkage.joints)) {
          // No trajectory yet, run simulation first then play
          runSimulation().then(() => {
            setTimeout(() => playAnimation(), 100)
          })
          showStatus('Running simulation...', 'action')
        }
        return
      }

      // Delete/Backspace/X to delete selected items (with confirmation for multiple)
      // Now also handles DrawnObjects (polygons)
      const hasSelectedItems = selectedJoints.length > 0 || selectedLinks.length > 0 || drawnObjects.selectedIds.length > 0
      if ((event.key === 'Delete' || event.key === 'Backspace' || event.key === 'x' || event.key === 'X') && hasSelectedItems) {
        // If X is pressed and it's not the shortcut intent (no modifier), delete selected
        if (event.key === 'x' || event.key === 'X') {
          // X is also the delete tool shortcut, so only delete if items are selected
          // and switch to delete mode if nothing selected
          if (!hasSelectedItems) {
            // Let it fall through to tool selection
          } else {
            handleDeleteSelected()
            event.preventDefault()
            return
          }
        } else {
          handleDeleteSelected()
          event.preventDefault()
          return
        }
      }

      const key = event.key.toUpperCase()
      const tool = TOOLS.find(t => t.shortcut === key)
      if (tool) {
        // If switching away from draw_link while drawing, cancel
        if (linkCreationState.isDrawing && tool.id !== 'draw_link') {
          setLinkCreationState(initialLinkCreationState)
          setPreviewLine(null)
        }
        // Cancel group selection if switching tools
        if (groupSelectionState.isSelecting && tool.id !== 'group_select') {
          setGroupSelectionState(initialGroupSelectionState)
        }
        // Cancel polygon drawing if switching tools
        if (polygonDrawState.isDrawing && tool.id !== 'draw_polygon') {
          setPolygonDrawState(initialPolygonDrawState)
        }
        // Cancel measurement if switching tools
        if (measureState.isMeasuring && tool.id !== 'measure') {
          setMeasureState(initialMeasureState)
        }
        // Cancel merge if switching tools
        if (mergePolygonState.step !== 'idle' && tool.id !== 'merge') {
          setMergePolygonState(initialMergePolygonState)
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
          setSelectedLinks([])
        }
        // Cancel path drawing if switching tools
        if (pathDrawState.isDrawing && tool.id !== 'draw_path') {
          setPathDrawState(initialPathDrawState)
        }
        setToolMode(tool.id)

        // Show appropriate message for merge mode
        if (tool.id === 'merge') {
          setMergePolygonState({ step: 'awaiting_selection', selectedPolygonId: null, selectedLinkName: null })
          showStatus('Select a link or a polygon to begin merge', 'action')
        } else if (tool.id === 'draw_path') {
          showStatus('Click to start drawing target path', 'action')
        } else {
          showStatus(`${tool.label} mode`, 'info', 1500)
        }
        event.preventDefault()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [linkCreationState.isDrawing, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, pathDrawState.isDrawing, cancelAction, showStatus, selectedJoints, selectedLinks, handleDeleteSelected, animationState.isAnimating, playAnimation, pauseAnimation, trajectoryData, runSimulation, pylinkDoc.pylinkage.joints, completePathDrawing])

  // Handle canvas click
  const handleCanvasClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    // Don't process click if we just finished dragging or group selecting
    if (dragState.isDragging || groupSelectionState.isSelecting) return

    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const clickPoint: [number, number] = [x, y]

    const jointsWithPositions = getJointsWithPositions()
    const linksWithPositions = getLinksWithPositions()
    const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)
    const nearestLink = findNearestLink(clickPoint, linksWithPositions)

    // Handle mechanism select mode - select and enter move mode
    if (toolMode === 'mechanism_select') {
      // Helper to find DrawnObjects merged with mechanism links
      const findMergedDrawnObjects = (linkNames: string[]): string[] => {
        return drawnObjects.objects
          .filter(obj => obj.mergedLinkName && linkNames.includes(obj.mergedLinkName))
          .map(obj => obj.id)
      }

      if (nearestJoint) {
        const mechanism = findConnectedMechanism(nearestJoint.name, pylinkDoc.meta.links)
        setSelectedJoints(mechanism.joints)
        setSelectedLinks(mechanism.links)
        // Find DrawnObjects merged with this mechanism's links
        const mergedDrawnObjects = findMergedDrawnObjects(mechanism.links)
        setDrawnObjects(prev => ({ ...prev, selectedIds: mergedDrawnObjects }))
        // Enter move mode with the selected mechanism and merged DrawnObjects
        enterMoveGroupMode(mechanism.joints, mergedDrawnObjects)
      } else if (nearestLink) {
        // Get joints from the link and find their connected mechanism
        const linkMeta = pylinkDoc.meta.links[nearestLink.name]
        if (linkMeta && linkMeta.connects.length > 0) {
          const mechanism = findConnectedMechanism(linkMeta.connects[0], pylinkDoc.meta.links)
          setSelectedJoints(mechanism.joints)
          setSelectedLinks(mechanism.links)
          // Find DrawnObjects merged with this mechanism's links
          const mergedDrawnObjects = findMergedDrawnObjects(mechanism.links)
          setDrawnObjects(prev => ({ ...prev, selectedIds: mergedDrawnObjects }))
          // Enter move mode with the selected mechanism and merged DrawnObjects
          enterMoveGroupMode(mechanism.joints, mergedDrawnObjects)
        }
      } else {
        setSelectedJoints([])
        setSelectedLinks([])
        setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
        exitMoveGroupMode()
        showStatus('Click on a joint or link to select its mechanism', 'info', 1500)
      }
      return
    }

    // Handle measure mode
    if (toolMode === 'measure') {
      // Snap to existing joint if within MERGE_THRESHOLD
      const snappedPoint: [number, number] = nearestJoint?.position || clickPoint
      const snappedX = snappedPoint[0]
      const snappedY = snappedPoint[1]
      const snappedToJoint = nearestJoint ? ` (${nearestJoint.name})` : ''

      if (!measureState.isMeasuring) {
        // First click - set start point
        setMeasureState({
          isMeasuring: true,
          startPoint: snappedPoint,
          endPoint: null,
          measurementId: Date.now()
        })
        // Add start marker
        setMeasurementMarkers(prev => [...prev, {
          id: Date.now(),
          point: snappedPoint,
          timestamp: Date.now()
        }])
        showStatus(`Start: (${snappedX.toFixed(1)}, ${snappedY.toFixed(1)})${snappedToJoint} — click second point`, 'action')
      } else {
        // Second click - complete measurement
        const startPoint = measureState.startPoint!
        const distance = calculateDistance(startPoint, snappedPoint)
        const dx = snappedPoint[0] - startPoint[0]
        const dy = snappedPoint[1] - startPoint[1]

        // Add end marker
        setMeasurementMarkers(prev => [...prev, {
          id: Date.now(),
          point: snappedPoint,
          timestamp: Date.now()
        }])

        showStatus(`Distance: ${distance.toFixed(2)} units (Δx: ${dx.toFixed(1)}, Δy: ${dy.toFixed(1)})`, 'success', 5000)

        // Reset measurement state
        setMeasureState(initialMeasureState)

        // Start fade timer for markers (3 seconds)
        setTimeout(() => {
          setMeasurementMarkers(prev => prev.filter(m => Date.now() - m.timestamp < 3000))
        }, 3000)
      }
      return
    }

    // Handle merge mode - merge polygon with enclosed link (either order works)
    if (toolMode === 'merge') {
      // Helper to check if click is on a polygon (inside or on the stroke/outline)
      const findClickedPolygon = () => {
        return drawnObjects.objects.find(obj => {
          if (obj.type !== 'polygon' || obj.points.length < 3) return false
          // Skip already-merged polygons
          if (obj.mergedLinkName) return false
          // Check if inside polygon
          if (isPointInPolygon(clickPoint, obj.points)) return true
          // Check if on the polygon's stroke (near any edge)
          for (let i = 0; i < obj.points.length; i++) {
            const p1 = obj.points[i]
            const p2 = obj.points[(i + 1) % obj.points.length]
            // Point-to-line-segment distance check
            const lineLen = calculateDistance(p1, p2)
            if (lineLen === 0) continue
            const t = Math.max(0, Math.min(1, ((clickPoint[0] - p1[0]) * (p2[0] - p1[0]) + (clickPoint[1] - p1[1]) * (p2[1] - p1[1])) / (lineLen * lineLen)))
            const projX = p1[0] + t * (p2[0] - p1[0])
            const projY = p1[1] + t * (p2[1] - p1[1])
            const dist = calculateDistance(clickPoint, [projX, projY])
            if (dist < 0.5) return true  // Within 0.5 units of edge
          }
          return false
        })
      }

      // Helper to complete the merge
      const completeMerge = (polygonId: string, linkName: string) => {
        const polygon = drawnObjects.objects.find(obj => obj.id === polygonId)
        const linkMeta = pylinkDoc.meta.links[linkName]

        if (!polygon || !linkMeta) {
          showStatus('Error: Could not find polygon or link', 'error', 2000)
          setMergePolygonState(initialMergePolygonState)
          return false
        }

        const startPos = getJointPosition(linkMeta.connects[0])
        const endPos = getJointPosition(linkMeta.connects[1])

        if (!startPos || !endPos) {
          showStatus('Error: Could not get link endpoint positions', 'error', 2000)
          return false
        }

        // Check if both endpoints are inside the polygon
        if (!areLinkEndpointsInPolygon(startPos, endPos, polygon.points)) {
          showStatus(`Link "${linkName}" endpoints are not inside the polygon. Both ends must be enclosed.`, 'warning', 3500)
          return false
        }

        // Success! Merge the polygon with the link
        const linkColor = linkMeta.color || getDefaultColor(0)

        setDrawnObjects(prev => ({
          ...prev,
          objects: prev.objects.map(obj => {
            if (obj.id === polygonId) {
              return {
                ...obj,
                mergedLinkName: linkName,
                fillColor: linkColor,
                fillOpacity: 0.25,  // Subtle shaded color
                strokeColor: linkColor
              }
            }
            return obj
          }),
          selectedIds: []
        }))

        showStatus(`Merged polygon "${polygon.name}" with link "${linkName}"`, 'success', 3000)
        setMergePolygonState(initialMergePolygonState)
        return true
      }

      // Initial state: waiting for user to select either a polygon or a link
      // PRIORITY: If clicking near a link, select the link (even if inside a polygon)
      // This allows selecting links that are inside polygons
      if (mergePolygonState.step === 'awaiting_selection') {
        // Check for link FIRST - prioritize link selection over polygon
        if (nearestLink) {
          // User clicked on a link first
          setMergePolygonState({
            step: 'link_selected',
            selectedPolygonId: null,
            selectedLinkName: nearestLink.name
          })
          setSelectedLinks([nearestLink.name])
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))  // Clear polygon selection
          showStatus(`Selected link "${nearestLink.name}" — now click a polygon to merge`, 'action')
          return
        }

        // No link clicked - check for polygon
        const clickedPolygon = findClickedPolygon()
        if (clickedPolygon) {
          // User clicked on an unmerged polygon
          setMergePolygonState({
            step: 'polygon_selected',
            selectedPolygonId: clickedPolygon.id,
            selectedLinkName: null
          })
          setDrawnObjects(prev => ({ ...prev, selectedIds: [clickedPolygon.id] }))
          setSelectedLinks([])  // Clear link selection
          showStatus(`Selected polygon "${clickedPolygon.name}" — now click a link to merge`, 'action')
          return
        }

        // Clicked on empty space - check if they clicked on an already-merged polygon
        const mergedPolygon = drawnObjects.objects.find(obj => {
          if (obj.type !== 'polygon' || obj.points.length < 3) return false
          if (!obj.mergedLinkName) return false
          return isPointInPolygon(clickPoint, obj.points)
        })
        if (mergedPolygon) {
          showStatus(`Polygon "${mergedPolygon.name}" is already merged with link "${mergedPolygon.mergedLinkName}"`, 'info', 2500)
        } else {
          showStatus('Select a link or a polygon to begin merge', 'info', 2000)
        }
        return
      }

      // Polygon selected: waiting for link
      if (mergePolygonState.step === 'polygon_selected') {
        // Link takes priority - complete the merge
        if (nearestLink) {
          const success = completeMerge(mergePolygonState.selectedPolygonId!, nearestLink.name)
          if (!success) {
            // Merge failed (endpoints not in polygon) - keep selection, let user try another link
            showStatus('Link endpoints must be inside the polygon. Try another link.', 'warning', 2500)
          }
          return
        }

        // No link - check if user clicked a different polygon to switch
        const clickedPolygon = findClickedPolygon()
        if (clickedPolygon && clickedPolygon.id !== mergePolygonState.selectedPolygonId) {
          // User clicked a different polygon - switch selection
          setMergePolygonState({
            step: 'polygon_selected',
            selectedPolygonId: clickedPolygon.id,
            selectedLinkName: null
          })
          setDrawnObjects(prev => ({ ...prev, selectedIds: [clickedPolygon.id] }))
          showStatus(`Switched to polygon "${clickedPolygon.name}" — now click a link to merge`, 'action')
        } else {
          showStatus('Click a link to merge with the selected polygon', 'info', 2000)
        }
        return
      }

      // Link selected: waiting for polygon
      if (mergePolygonState.step === 'link_selected') {
        // First check if user clicked a different link to switch
        if (nearestLink && nearestLink.name !== mergePolygonState.selectedLinkName) {
          // User clicked a different link - switch selection
          setMergePolygonState({
            step: 'link_selected',
            selectedPolygonId: null,
            selectedLinkName: nearestLink.name
          })
          setSelectedLinks([nearestLink.name])
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
          showStatus(`Switched to link "${nearestLink.name}" — now click a polygon to merge`, 'action')
          return
        }

        // Check for polygon (clicking inside or on edge)
        const clickedPolygon = findClickedPolygon()
        if (clickedPolygon) {
          const success = completeMerge(clickedPolygon.id, mergePolygonState.selectedLinkName!)
          if (!success) {
            // Merge failed - keep selection, let user try another polygon
            showStatus('Link endpoints must be inside the polygon. Try another polygon.', 'warning', 2500)
          }
          return
        }

        showStatus('Click inside a polygon to merge with the selected link', 'info', 2000)
        return
      }
    }

    // Handle draw_polygon mode - creates DrawnObjects only (no joints/links)
    if (toolMode === 'draw_polygon') {
      // Just use click position directly, no snapping to joints for polygons
      const point = clickPoint

      if (!polygonDrawState.isDrawing) {
        // First click - start polygon
        setPolygonDrawState({
          isDrawing: true,
          points: [point]
        })
        showStatus('Click to add polygon sides, click near start to close', 'action')
      } else {
        // Check if near starting point to close polygon
        const startPoint = polygonDrawState.points[0]
        const distanceToStart = calculateDistance(point, startPoint)

        if (distanceToStart <= MERGE_THRESHOLD && polygonDrawState.points.length >= 3) {
          // Close the polygon - create DrawnObject only (no joints/links)
          const polygonPoints = polygonDrawState.points
          const newDrawnObject = createDrawnObject(
            'polygon',
            polygonPoints,
            drawnObjects.objects.map(o => o.id)
          )
          setDrawnObjects(prev => ({
            ...prev,
            objects: [...prev.objects, newDrawnObject],
            selectedIds: [newDrawnObject.id]  // Select the new polygon
          }))

          setPolygonDrawState(initialPolygonDrawState)
          showStatus(`Completed ${polygonDrawState.points.length}-sided polygon (${newDrawnObject.id})`, 'success', 2500)
        } else {
          // Add new point to polygon
          setPolygonDrawState(prev => ({
            ...prev,
            points: [...prev.points, point]
          }))

          const sides = polygonDrawState.points.length
          showStatus(`${sides + 1} points — click near start to close polygon`, 'action')
        }
      }
      return
    }

    // Handle draw_path mode - for trajectory optimization target paths
    if (toolMode === 'draw_path') {
      const point = clickPoint

      if (!pathDrawState.isDrawing) {
        // First click - start path
        setPathDrawState({
          isDrawing: true,
          points: [point]
        })
        showStatus('Click to add points. Click near start or double-click to close path.', 'action')
      } else {
        // Check if clicking near start point to close the path (need at least 3 points)
        const startPoint = pathDrawState.points[0]
        const distanceToStart = calculateDistance(point, startPoint)

        if (distanceToStart <= jointMergeRadius && pathDrawState.points.length >= 3) {
          // Close the path - don't add the clicked point, just complete with existing points
          // The path is implicitly closed (last point connects back to first)
          const newPath = createTargetPath(pathDrawState.points, targetPaths)
          setTargetPaths(prev => [...prev, newPath])
          setSelectedPathId(newPath.id)
          setPathDrawState(initialPathDrawState)
          showStatus(`Created closed path with ${pathDrawState.points.length} points`, 'success', 2500)
        } else {
          // Add new point to path
          setPathDrawState(prev => ({
            ...prev,
            points: [...prev.points, point]
          }))

          const pointCount = pathDrawState.points.length + 1
          if (pointCount >= 3) {
            showStatus(`${pointCount} points — click near start to close, or double-click to finish`, 'action')
          } else {
            showStatus(`${pointCount} points — need at least 3 for a path`, 'action')
          }
        }
      }
      return
    }

    // Handle delete mode
    if (toolMode === 'delete') {
      // If items are selected, delete them
      if (selectedJoints.length > 0 || selectedLinks.length > 0) {
        handleDeleteSelected()
        return
      }

      // Otherwise, prioritize joints over links
      if (nearestJoint) {
        deleteJoint(nearestJoint.name)
      } else if (nearestLink) {
        deleteLink(nearestLink.name)
      } else {
        showStatus('Click on a joint or link to delete it', 'info', 2000)
      }
      return
    }

    // Handle draw_link mode
    if (toolMode === 'draw_link') {
      if (!linkCreationState.isDrawing) {
        // First click - start the link
        const startJointName = nearestJoint?.name || null
        const startPoint = nearestJoint?.position || clickPoint

        setLinkCreationState({
          isDrawing: true,
          startPoint,
          startJointName,
          endPoint: null
        })

        if (startJointName) {
          showStatus(`Drawing from ${startJointName} — click to complete`, 'action')
        } else {
          showStatus(`Drawing from (${startPoint[0].toFixed(1)}, ${startPoint[1].toFixed(1)}) — click to complete`, 'action')
        }
      } else {
        // Second click - complete the link
        const endJointName = nearestJoint?.name || null
        const endPoint = nearestJoint?.position || clickPoint

        // Don't allow connecting a joint to itself
        if (endJointName && endJointName === linkCreationState.startJointName) {
          showStatus('Cannot connect a joint to itself', 'warning', 2000)
          return
        }

        createLinkWithRevoluteDefault(
          linkCreationState.startPoint!,
          endPoint,
          linkCreationState.startJointName,
          endJointName
        )

        // Reset link creation state
        setLinkCreationState(initialLinkCreationState)
        setPreviewLine(null)
      }
      return
    }

    // Handle select mode (only for clicks, not drags)
    if (toolMode === 'select') {
      if (nearestJoint) {
        setSelectedJoints([nearestJoint.name])
        setSelectedLinks([])
        showStatus(`Selected ${nearestJoint.name}`, 'info', 1500)
      } else if (nearestLink) {
        setSelectedLinks([nearestLink.name])
        setSelectedJoints([])
        showStatus(`Selected ${nearestLink.name}`, 'info', 1500)
      } else {
        setSelectedJoints([])
        setSelectedLinks([])
        clearStatus()
      }
    }
  }, [toolMode, linkCreationState, dragState.isDragging, groupSelectionState.isSelecting, measureState, polygonDrawState, pathDrawState, selectedJoints, selectedLinks, getJointsWithPositions, getLinksWithPositions, pylinkDoc.meta.links, pylinkDoc.pylinkage.joints, deleteJoint, deleteLink, handleDeleteSelected, showStatus, clearStatus, triggerMechanismChange, createLinkWithRevoluteDefault])

  // Handle canvas double-click (for completing path drawing)
  const handleCanvasDoubleClick = useCallback((_event: React.MouseEvent<SVGSVGElement>) => {
    // Complete path drawing on double-click
    if (toolMode === 'draw_path' && pathDrawState.isDrawing && pathDrawState.points.length >= 2) {
      const newPath = createTargetPath(pathDrawState.points, targetPaths)
      setTargetPaths(prev => [...prev, newPath])
      setSelectedPathId(newPath.id)
      setPathDrawState(initialPathDrawState)
      showStatus(`Created target path with ${pathDrawState.points.length} points`, 'success', 2500)
      return
    }
  }, [toolMode, pathDrawState, targetPaths, showStatus])

  // Get cursor style based on tool mode
  const getCursorStyle = () => {
    if (moveGroupState.isDragging) return 'grabbing'
    if (moveGroupState.isActive) return 'move'
    if (dragState.isDragging) return 'grabbing'
    if (groupSelectionState.isSelecting) return 'crosshair'
    switch (toolMode) {
      case 'draw_link': return 'crosshair'
      case 'draw_polygon': return 'crosshair'
      case 'draw_path': return 'crosshair'
      case 'measure': return 'crosshair'
      case 'delete': return 'pointer'
      case 'select': return 'default'
      case 'group_select': return 'crosshair'
      case 'mechanism_select': return 'pointer'
      default: return 'default'
    }
  }

  // Render joints
  const renderJoints = () => {
    return pylinkDoc.pylinkage.joints.map((joint, index) => {
      const pos = getJointPosition(joint.name)
      if (!pos) return null

      const isSelected = selectedJoints.includes(joint.name)
      const isInMoveGroup = moveGroupState.isActive && moveGroupState.joints.includes(joint.name)
      const isHovered = hoveredJoint === joint.name
      const isDragging = dragState.draggedJoint === joint.name
      const isMergeTarget = dragState.mergeTarget === joint.name
      const meta = pylinkDoc.meta.joints[joint.name]
      const color = meta?.color || getDefaultColor(index)

      const isStatic = joint.type === 'Static'
      const isCrank = joint.type === 'Crank'

      // Highlight state flags (must be declared before using them)
      const showMergeHighlight = isMergeTarget || (isDragging && dragState.mergeTarget)
      const showMoveGroupHighlight = isInMoveGroup && !showMergeHighlight

      // Use theme colors for joint fill - keep original colors, selection indicated by glow
      const staticColor = jointColors.static      // #E74C3C - red for fixed joints
      const crankColor = jointColors.crank        // #F39C12 - amber for crank joints

      // Joint fill color based on type (NOT changed by selection - selection uses glow)
      const jointFillColor = isStatic
        ? staticColor
        : isCrank
          ? crankColor
          : color

      // Determine highlight type for glow effect
      const highlightType: HighlightType = showMergeHighlight
        ? 'merge'
        : showMoveGroupHighlight
          ? 'move_group'
          : (isSelected || isDragging)
            ? 'selected'
            : isHovered
              ? 'hovered'
              : 'none'

      // Get highlight styling - joints glow in their original color
      const defaultStroke = darkMode ? '#333333' : '#FFFFFF'
      const highlightStyle = getHighlightStyle('joint', highlightType, jointFillColor, 2)

      // Use the highlight filter, or no filter if not highlighted
      // Stroke color: use object color for glow, but keep default stroke for the actual outline
      const strokeColor = highlightType === 'none'
        ? defaultStroke
        : highlightType === 'move_group' || highlightType === 'merge'
          ? highlightStyle.stroke
          : defaultStroke  // Keep white/dark outline, the glow provides the color

      return (
        <g key={joint.name}>
          <circle
            cx={unitsToPixels(pos[0])}
            cy={unitsToPixels(pos[1])}
            r={isDragging ? jointSize * 1.5 : (isHovered || isSelected || isMergeTarget || isInMoveGroup ? jointSize * 1.25 : jointSize)}
            fill={jointFillColor}
            stroke={strokeColor}
            strokeWidth={highlightStyle.strokeWidth}
            filter={highlightStyle.filter}
            style={{ cursor: moveGroupState.isActive ? 'move' : (toolMode === 'select' ? 'grab' : 'pointer') }}
            onMouseEnter={() => !dragState.isDragging && !moveGroupState.isDragging && setHoveredJoint(joint.name)}
            onMouseLeave={() => setHoveredJoint(null)}
            onDoubleClick={(e) => {
              if (toolMode === 'select') {
                e.stopPropagation()
                openJointEditModal(joint.name)
              }
            }}
          />
          {/* Joint label - always show on hover/drag, or when showJointLabels is enabled */}
          {(showJointLabels || isHovered || isDragging || isSelected) && (
            <text
              x={unitsToPixels(pos[0])}
              y={unitsToPixels(pos[1]) - 14}
              textAnchor="middle"
              fontSize="11"
              fontWeight={isHovered || isDragging ? 'bold' : 'normal'}
              fill={showMergeHighlight ? jointColors.mergeHighlight : (darkMode ? '#e0e0e0' : '#333')}
              style={{ pointerEvents: 'none' }}
            >
              {joint.name}
            </text>
          )}
          {isStatic && !isDragging && (
            <rect
              x={unitsToPixels(pos[0]) - 4}
              y={unitsToPixels(pos[1]) + 10}
              width={8}
              height={4}
              fill="#e74c3c"
            />
          )}
          {isCrank && !isDragging && (
            <path
              d={`M ${unitsToPixels(pos[0])} ${unitsToPixels(pos[1]) + 10} L ${unitsToPixels(pos[0]) - 6} ${unitsToPixels(pos[1]) + 18} L ${unitsToPixels(pos[0]) + 6} ${unitsToPixels(pos[1]) + 18} Z`}
              fill="#f39c12"
            />
          )}
        </g>
      )
    })
  }

  // Render trajectory dots for simulation results
  const renderTrajectories = () => {
    if (!trajectoryData || !showTrajectory) return null

    // ─────────────────────────────────────────────────────────────────────────
    // TRAJECTORY COLOR CYCLE
    // ─────────────────────────────────────────────────────────────────────────
    // This function generates colors for trajectory dots that cycle back to
    // the starting color at the end of the animation. This creates a smooth
    // visual loop where the last dot's color nearly matches the first dot's.
    //
    // Available cycle types (configured in theme.ts):
    //   - 'rainbow': Full hue rotation through color wheel
    //   - 'fire':    Orange → dark → orange (warm, dramatic)
    //   - 'glow':    Orange → light/white → orange (bright, ethereal)
    //
    // The color at step 0 will be very close to the color at step (totalSteps-1),
    // ensuring trajectories that complete a full cycle look visually continuous.
    // ─────────────────────────────────────────────────────────────────────────
    // Use the user-configurable trajectory color cycle from Settings
    const getTrajectoryColor = (stepIndex: number, totalSteps: number) => {
      return getCyclicColor(stepIndex, totalSteps, trajectoryColorCycle)
    }

    return Object.entries(trajectoryData.trajectories).map(([jointName, positions]) => {
      // Skip if no trajectory data for this joint
      if (!positions || !Array.isArray(positions) || positions.length === 0) {
        return null
      }

      // Show trajectories for Revolute and Crank joints (they move during simulation)
      const jointType = trajectoryData.jointTypes?.[jointName]
      if (jointType !== 'Revolute' && jointType !== 'Crank') return null

      // Check if this joint has show_path enabled in meta (per-joint visibility control)
      const jointMeta = pylinkDoc.meta.joints[jointName]
      if (jointMeta && jointMeta.show_path === false) return null

      // Validate first position exists and has coordinates
      const firstPos = positions[0]
      if (!firstPos || typeof firstPos[0] !== 'number' || typeof firstPos[1] !== 'number') {
        return null
      }

      // Check if joint actually moves (has varying positions)
      const hasMovement = positions.length > 1 && positions.some((pos, i) =>
        i > 0 && pos && pos[0] !== undefined && pos[1] !== undefined &&
        (Math.abs(pos[0] - firstPos[0]) > 0.001 || Math.abs(pos[1] - firstPos[1]) > 0.001)
      )

      // Get color based on joint type for non-moving indicators
      const jointColor = jointType === 'Crank' ? jointColors.crank : jointColors.pivot

      // Filter out any invalid positions for rendering
      const validPositions = positions.filter(pos =>
        pos && typeof pos[0] === 'number' && typeof pos[1] === 'number' &&
        isFinite(pos[0]) && isFinite(pos[1])
      )

      if (validPositions.length === 0) return null

      return (
        <g key={`trajectory-${jointName}`}>
          {hasMovement && validPositions.length > 1 ? (
            <>
              {/* Draw trajectory path as a line (if trajectoryStyle includes line) */}
              {(trajectoryStyle === 'line' || trajectoryStyle === 'both') && (
                <path
                  d={validPositions
                    .filter(pos => pos && pos[0] !== undefined && pos[1] !== undefined)
                    .map((pos, i) =>
                      `${i === 0 ? 'M' : 'L'} ${unitsToPixels(pos[0])} ${unitsToPixels(pos[1])}`
                    ).join(' ')}
                  fill="none"
                  stroke="rgba(33, 150, 243, 0.3)"
                  strokeWidth={2}
                  strokeDasharray="4,2"
                />
              )}
              {/* Draw dots at each timestep (if trajectoryStyle includes dots) */}
              {(trajectoryStyle === 'dots' || trajectoryStyle === 'both') && validPositions.map((pos, stepIndex) => {
                // Extra safety check for each position
                if (!pos || pos[0] === undefined || pos[1] === undefined) return null
                return (
                  <circle
                    key={`${jointName}-step-${stepIndex}`}
                    cx={unitsToPixels(pos[0])}
                    cy={unitsToPixels(pos[1])}
                    r={trajectoryDotSize}
                    fill={getTrajectoryColor(stepIndex, validPositions.length)}
                    stroke="#fff"
                    strokeWidth={1}
                    opacity={0.8}
                  >
                    <title>{`${jointName} step ${stepIndex + 1}/${validPositions.length}: (${pos[0]?.toFixed(1) ?? '?'}, ${pos[1]?.toFixed(1) ?? '?'})`}</title>
                  </circle>
                )
              })}
            </>
          ) : (
            /* Non-moving joint: show a stationary indicator ring */
            <circle
              cx={unitsToPixels(validPositions[0][0])}
              cy={unitsToPixels(validPositions[0][1])}
              r={10}
              fill="none"
              stroke={jointColor}
              strokeWidth={2}
              strokeDasharray="3,3"
              opacity={0.6}
            >
              <title>{`${jointName}: stationary (not constrained to moving joints)`}</title>
            </circle>
          )}
        </g>
      )
    })
  }

  // Render links between joints
  const renderLinks = () => {
    const links: JSX.Element[] = []

    Object.entries(pylinkDoc.meta.links).forEach(([linkName, linkMeta], index) => {
      if (linkMeta.connects.length < 2) return

      const pos0 = getJointPosition(linkMeta.connects[0])
      const pos1 = getJointPosition(linkMeta.connects[1])

      if (!pos0 || !pos1) return

      const isHovered = hoveredLink === linkName
      const isSelected = selectedLinks.includes(linkName)
      // Check if link's joints are in move group
      const isInMoveGroup = moveGroupState.isActive &&
        linkMeta.connects.every(jointName => moveGroupState.joints.includes(jointName))

      // Check if this is a ground link
      const isGroundLink = linkMeta.isGround || false

      // Check if this link would stretch (invalid kinematic constraint)
      const isStretchingLink = stretchingLinks.includes(linkName)

      // Base link color - stretching links are red, ground links use gray if no custom color
      const baseLinkColor = isStretchingLink
        ? '#ff0000'  // Red for invalid/stretching links
        : isGroundLink && !linkMeta.color
          ? '#7f7f7f'  // Default gray for ground links
          : (linkMeta.color || getDefaultColor(index))

      // Determine highlight type for glow effect
      const linkHighlightType: HighlightType = isInMoveGroup
        ? 'move_group'
        : isSelected
          ? 'selected'
          : isHovered
            ? 'hovered'
            : 'none'

      // Get highlight styling - links glow in their original color
      const linkHighlightStyle = getHighlightStyle('link', linkHighlightType, baseLinkColor, linkThickness)

      // Ground links are rendered slightly thinner
      const effectiveStrokeWidth = isGroundLink
        ? Math.max(linkHighlightStyle.strokeWidth * 0.8, 2)
        : linkHighlightStyle.strokeWidth

      // Calculate midpoint for label
      const midX = (pos0[0] + pos1[0]) / 2
      const midY = (pos0[1] + pos1[1]) / 2

      links.push(
        <g key={linkName}>
          <line
            x1={unitsToPixels(pos0[0])}
            y1={unitsToPixels(pos0[1])}
            x2={unitsToPixels(pos1[0])}
            y2={unitsToPixels(pos1[1])}
            stroke={baseLinkColor}
            strokeWidth={effectiveStrokeWidth}
            strokeLinecap="round"
            strokeDasharray={isGroundLink ? '8,4' : undefined}  // Dashed for ground links
            filter={linkHighlightStyle.filter}
            style={{ cursor: moveGroupState.isActive ? 'move' : 'pointer' }}
            onMouseEnter={() => !moveGroupState.isDragging && setHoveredLink(linkName)}
            onMouseLeave={() => setHoveredLink(null)}
            onDoubleClick={(e) => {
              if (toolMode === 'select') {
                e.stopPropagation()
                openLinkEditModal(linkName)
              }
            }}
          />
          {/* Link label - show on hover/selected, or when showLinkLabels is enabled */}
          {(showLinkLabels || isHovered || isSelected) && (
            <g>
              {/* Background for readability */}
              <rect
                x={unitsToPixels(midX) - linkName.length * 3.5 - 4}
                y={unitsToPixels(midY) - 8}
                width={linkName.length * 7 + 8}
                height={14}
                fill={darkMode ? 'rgba(30, 30, 30, 0.85)' : 'rgba(255, 255, 255, 0.85)'}
                rx={3}
                style={{ pointerEvents: 'none' }}
              />
              <text
                x={unitsToPixels(midX)}
                y={unitsToPixels(midY) + 3}
                textAnchor="middle"
                fontSize="10"
                fontWeight={isHovered ? 'bold' : 'normal'}
                fill={isHovered || isSelected ? baseLinkColor : (darkMode ? '#b0b0b0' : '#555')}
                style={{ pointerEvents: 'none' }}
              >
                {linkName}
              </text>
            </g>
          )}
        </g>
      )
    })

    return links
  }

  // Render grid - now uses full canvas dimensions
  // Uses CSS variables for dark mode support
  const renderGrid = () => {
    const lines: JSX.Element[] = []
    const maxUnitsX = pixelsToUnits(canvasDimensions.width)
    const maxUnitsY = pixelsToUnits(canvasDimensions.height)

    // Get CSS variable values for grid colors (dark mode aware)
    const gridMajorColor = darkMode ? '#444444' : '#dddddd'
    const gridMinorColor = darkMode ? '#333333' : '#eeeeee'
    const gridTextColor = darkMode ? '#666666' : '#999999'

    // Major grid every 20 units
    for (let i = 0; i <= Math.ceil(maxUnitsX); i += 20) {
      lines.push(
        <line
          key={`v-major-${i}`}
          x1={unitsToPixels(i)}
          y1={0}
          x2={unitsToPixels(i)}
          y2={canvasDimensions.height}
          stroke={gridMajorColor}
          strokeWidth={1}
        />
      )
      if (i > 0) {
        lines.push(
          <text key={`vl-${i}`} x={unitsToPixels(i) + 2} y={12} fontSize="10" fill={gridTextColor}>
            {i}
          </text>
        )
      }
    }

    for (let i = 0; i <= Math.ceil(maxUnitsY); i += 20) {
      lines.push(
        <line
          key={`h-major-${i}`}
          x1={0}
          y1={unitsToPixels(i)}
          x2={canvasDimensions.width}
          y2={unitsToPixels(i)}
          stroke={gridMajorColor}
          strokeWidth={1}
        />
      )
      if (i > 0) {
        lines.push(
          <text key={`hl-${i}`} x={2} y={unitsToPixels(i) - 2} fontSize="10" fill={gridTextColor}>
            {i}
          </text>
        )
      }
    }

    // Minor grid every 10 units
    for (let i = 10; i <= Math.ceil(maxUnitsX); i += 20) {
      lines.push(
        <line
          key={`v-minor-${i}`}
          x1={unitsToPixels(i)}
          y1={0}
          x2={unitsToPixels(i)}
          y2={canvasDimensions.height}
          stroke={gridMinorColor}
          strokeWidth={0.5}
          strokeDasharray="2,4"
        />
      )
    }

    for (let i = 10; i <= Math.ceil(maxUnitsY); i += 20) {
      lines.push(
        <line
          key={`h-minor-${i}`}
          x1={0}
          y1={unitsToPixels(i)}
          x2={canvasDimensions.width}
          y2={unitsToPixels(i)}
          stroke={gridMinorColor}
          strokeWidth={0.5}
          strokeDasharray="2,4"
        />
      )
    }

    return lines
  }

  // Render preview line during link creation
  const renderPreviewLine = () => {
    if (!previewLine) return null

    return (
      <line
        x1={unitsToPixels(previewLine.start[0])}
        y1={unitsToPixels(previewLine.start[1])}
        x2={unitsToPixels(previewLine.end[0])}
        y2={unitsToPixels(previewLine.end[1])}
        stroke="#ff8c00"
        strokeWidth={3}
        strokeDasharray="8,4"
        strokeLinecap="round"
        opacity={0.7}
      />
    )
  }

  // Render group selection box (dashed rectangle)
  const renderSelectionBox = () => {
    if (!groupSelectionState.isSelecting || !groupSelectionState.startPoint || !groupSelectionState.currentPoint) {
      return null
    }

    const x1 = unitsToPixels(groupSelectionState.startPoint[0])
    const y1 = unitsToPixels(groupSelectionState.startPoint[1])
    const x2 = unitsToPixels(groupSelectionState.currentPoint[0])
    const y2 = unitsToPixels(groupSelectionState.currentPoint[1])

    const minX = Math.min(x1, x2)
    const minY = Math.min(y1, y2)
    const width = Math.abs(x2 - x1)
    const height = Math.abs(y2 - y1)

    return (
      <rect
        x={minX}
        y={minY}
        width={width}
        height={height}
        fill="rgba(25, 118, 210, 0.1)"
        stroke="#1976d2"
        strokeWidth={2}
        strokeDasharray="8,4"
        pointerEvents="none"
      />
    )
  }

  // Render completed drawn objects (polygons, shapes)
  const renderDrawnObjects = () => {
    if (drawnObjects.objects.length === 0) return null

    return drawnObjects.objects.map(obj => {
      if (obj.type === 'polygon' && obj.points.length >= 3) {
        const pathData = obj.points.map((p, i) =>
          `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
        ).join(' ') + ' Z'

        const isSelected = drawnObjects.selectedIds.includes(obj.id)
        const isInMoveGroup = moveGroupState.isActive && moveGroupState.drawnObjectIds.includes(obj.id)

        // Determine highlight type for glow effect
        const polygonHighlightType: HighlightType = isInMoveGroup
          ? 'move_group'
          : isSelected
            ? 'selected'
            : 'none'

        // Get highlight styling - keeps original fill, adds glow outline in object's color
        const polygonHighlightStyle = getHighlightStyle('polygon', polygonHighlightType, obj.strokeColor, obj.strokeWidth)

        return (
          <g key={obj.id}>
            <path
              d={pathData}
              fill={obj.fillColor}  // Keep original fill color
              stroke={obj.strokeColor}  // Keep original stroke color
              strokeWidth={polygonHighlightStyle.strokeWidth}
              fillOpacity={obj.fillOpacity}
              filter={polygonHighlightStyle.filter}
              style={{ cursor: moveGroupState.isActive ? 'move' : 'pointer' }}
              onClick={(e) => {
                // In merge mode, let the canvas handler process this click
                // (don't stopPropagation, don't handle selection here)
                if (toolMode === 'merge') {
                  // Don't stop propagation - let the canvas click handler handle merge logic
                  return
                }

                e.stopPropagation()
                // Toggle selection (non-merge mode)
                setDrawnObjects(prev => ({
                  ...prev,
                  selectedIds: prev.selectedIds.includes(obj.id)
                    ? prev.selectedIds.filter(id => id !== obj.id)
                    : [...prev.selectedIds, obj.id]
                }))
              }}
            />
            {/* Show object name on hover/selection */}
            {isSelected && (
              <text
                x={unitsToPixels(obj.points[0][0])}
                y={unitsToPixels(obj.points[0][1]) - 8}
                fontSize="10"
                fill={obj.strokeColor}
                fontWeight="500"
              >
                {obj.name}
              </text>
            )}
          </g>
        )
      }
      return null
    })
  }

  // Render polygon preview during drawing
  const renderPolygonPreview = () => {
    if (!polygonDrawState.isDrawing || polygonDrawState.points.length === 0) return null

    const points = polygonDrawState.points

    // Draw lines between points
    const lines = []
    for (let i = 0; i < points.length - 1; i++) {
      lines.push(
        <line
          key={`poly-line-${i}`}
          x1={unitsToPixels(points[i][0])}
          y1={unitsToPixels(points[i][1])}
          x2={unitsToPixels(points[i + 1][0])}
          y2={unitsToPixels(points[i + 1][1])}
          stroke="#9c27b0"
          strokeWidth={3}
          strokeDasharray="6,3"
          opacity={0.8}
        />
      )
    }

    // Draw points
    const pointMarkers = points.map((point, i) => (
      <circle
        key={`poly-point-${i}`}
        cx={unitsToPixels(point[0])}
        cy={unitsToPixels(point[1])}
        r={i === 0 ? 10 : 6}
        fill={i === 0 ? '#9c27b0' : '#ce93d8'}
        stroke="#fff"
        strokeWidth={2}
        opacity={0.9}
      />
    ))

    // If we have 3+ points, show a preview fill
    let polygonFill = null
    if (points.length >= 3) {
      const pathData = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
      ).join(' ') + ' Z'

      polygonFill = (
        <path
          d={pathData}
          fill="rgba(156, 39, 176, 0.15)"
          stroke="none"
          pointerEvents="none"
        />
      )
    }

    return (
      <g>
        {polygonFill}
        {lines}
        {pointMarkers}
        {/* Show hint circle around starting point */}
        {points.length >= 3 && (
          <circle
            cx={unitsToPixels(points[0][0])}
            cy={unitsToPixels(points[0][1])}
            r={unitsToPixels(MERGE_THRESHOLD)}
            fill="none"
            stroke="#9c27b0"
            strokeWidth={1}
            strokeDasharray="4,4"
            opacity={0.5}
          />
        )}
      </g>
    )
  }

  // Render target path preview during drawing
  const renderPathPreview = () => {
    if (!pathDrawState.isDrawing || pathDrawState.points.length === 0) return null

    const points = pathDrawState.points
    const canClose = points.length >= 3

    // Draw lines between points
    const lines = []
    for (let i = 0; i < points.length - 1; i++) {
      lines.push(
        <line
          key={`path-line-${i}`}
          x1={unitsToPixels(points[i][0])}
          y1={unitsToPixels(points[i][1])}
          x2={unitsToPixels(points[i + 1][0])}
          y2={unitsToPixels(points[i + 1][1])}
          stroke="#e91e63"
          strokeWidth={3}
          strokeDasharray="8,4"
          opacity={0.9}
        />
      )
    }

    // Draw closing line preview (dashed, faded) - shows the path will be closed
    const closingLine = points.length >= 2 ? (
      <line
        key="path-closing-line"
        x1={unitsToPixels(points[points.length - 1][0])}
        y1={unitsToPixels(points[points.length - 1][1])}
        x2={unitsToPixels(points[0][0])}
        y2={unitsToPixels(points[0][1])}
        stroke="#e91e63"
        strokeWidth={2}
        strokeDasharray="4,8"
        opacity={0.4}
      />
    ) : null

    // Draw snap circle around start point when path can be closed
    const snapCircle = canClose ? (
      <circle
        cx={unitsToPixels(points[0][0])}
        cy={unitsToPixels(points[0][1])}
        r={unitsToPixels(jointMergeRadius)}
        fill="none"
        stroke="#e91e63"
        strokeWidth={2}
        strokeDasharray="4,4"
        opacity={0.6}
      />
    ) : null

    // Draw points
    const pointMarkers = points.map((point, i) => (
      <circle
        key={`path-point-${i}`}
        cx={unitsToPixels(point[0])}
        cy={unitsToPixels(point[1])}
        r={i === 0 ? 8 : 5}
        fill={i === 0 ? '#e91e63' : '#f48fb1'}
        stroke="#fff"
        strokeWidth={2}
        opacity={0.9}
      />
    ))

    return (
      <g>
        {closingLine}
        {lines}
        {snapCircle}
        {pointMarkers}
      </g>
    )
  }

  // Render completed target paths (for trajectory optimization)
  // All target paths are rendered as closed curves (cyclic trajectories)
  const renderTargetPaths = () => {
    if (targetPaths.length === 0) return null

    return targetPaths.map(path => {
      const isSelected = selectedPathId === path.id
      const points = path.points

      if (points.length < 2) return null

      // Create path data for closed curve (Z closes the path back to start)
      const pathData = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
      ).join(' ') + ' Z'  // Close the path

      return (
        <g key={path.id}>
          {/* Path line - closed shape */}
          <path
            d={pathData}
            fill="none"
            stroke={path.color}
            strokeWidth={isSelected ? 4 : 3}
            strokeDasharray="10,5"
            opacity={isSelected ? 1 : 0.7}
            style={{ cursor: 'pointer' }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedPathId(isSelected ? null : path.id)
            }}
          />
          {/* Path points */}
          {points.map((point, i) => (
            <circle
              key={`${path.id}-point-${i}`}
              cx={unitsToPixels(point[0])}
              cy={unitsToPixels(point[1])}
              r={isSelected ? 5 : 4}
              fill={path.color}
              stroke="#fff"
              strokeWidth={1.5}
              opacity={isSelected ? 1 : 0.7}
              style={{ pointerEvents: 'none' }}
            />
          ))}
          {/* Start point indicator (larger) */}
          <circle
            cx={unitsToPixels(points[0][0])}
            cy={unitsToPixels(points[0][1])}
            r={isSelected ? 7 : 6}
            fill={path.color}
            stroke="#fff"
            strokeWidth={2}
            opacity={isSelected ? 1 : 0.8}
            style={{ pointerEvents: 'none' }}
          />
          {/* Path label */}
          {isSelected && points.length > 0 && (
            <text
              x={unitsToPixels(points[0][0])}
              y={unitsToPixels(points[0][1]) - 14}
              textAnchor="middle"
              fontSize="11"
              fontWeight="bold"
              fill={path.color}
            >
              {path.name} (closed)
            </text>
          )}
        </g>
      )
    })
  }

  // Render measurement markers (X marks that fade)
  const renderMeasurementMarkers = () => {
    const now = Date.now()

    return measurementMarkers.map((marker) => {
      const age = now - marker.timestamp
      const opacity = Math.max(0, 1 - age / 3000) // Fade over 3 seconds
      const size = 8
      const x = unitsToPixels(marker.point[0])
      const y = unitsToPixels(marker.point[1])

      return (
        <g key={marker.id} opacity={opacity}>
          {/* X mark */}
          <line
            x1={x - size}
            y1={y - size}
            x2={x + size}
            y2={y + size}
            stroke="#f44336"
            strokeWidth={3}
            strokeLinecap="round"
          />
          <line
            x1={x + size}
            y1={y - size}
            x2={x - size}
            y2={y + size}
            stroke="#f44336"
            strokeWidth={3}
            strokeLinecap="round"
          />
          {/* Coordinate label */}
          <text
            x={x}
            y={y - 14}
            textAnchor="middle"
            fontSize="10"
            fill="#f44336"
            fontWeight="500"
          >
            ({marker.point[0].toFixed(1)}, {marker.point[1].toFixed(1)})
          </text>
        </g>
      )
    })
  }

  // Render measurement line (if measuring)
  const renderMeasurementLine = () => {
    if (!measureState.isMeasuring || !measureState.startPoint) return null

    const startX = unitsToPixels(measureState.startPoint[0])
    const startY = unitsToPixels(measureState.startPoint[1])

    return (
      <g>
        {/* Start point indicator */}
        <circle
          cx={startX}
          cy={startY}
          r={6}
          fill="#f44336"
          stroke="#fff"
          strokeWidth={2}
        />
      </g>
    )
  }

  // Toolbar toggle handler
  const handleToggleToolbar = useCallback((id: string) => {
    setOpenToolbars(prev => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }, [])

  // Toolbar position change handler
  const handleToolbarPositionChange = useCallback((id: string, position: ToolbarPosition) => {
    setToolbarPositions(prev => ({ ...prev, [id]: position }))
  }, [])

  // Get toolbar position (use saved or default)
  // Negative x values mean "offset from right edge of canvas"
  const getToolbarPosition = (id: string): ToolbarPosition => {
    if (toolbarPositions[id]) return toolbarPositions[id]
    const config = TOOLBAR_CONFIGS.find(c => c.id === id)
    const defaultPos = config?.defaultPosition || { x: 100, y: 100 }

    // Convert negative x to position from right edge
    if (defaultPos.x < 0) {
      return {
        x: canvasDimensions.width + defaultPos.x,
        y: defaultPos.y
      }
    }
    return defaultPos
  }

  // Get toolbar dimensions based on type
  const getToolbarDimensions = (id: string): { minWidth: number; maxHeight: number } => {
    switch (id) {
      case 'tools':
        // Tools should NEVER scroll - large maxHeight to fit all content
        return { minWidth: 220, maxHeight: 600 }
      case 'more':
        return { minWidth: 180, maxHeight: 350 }  // Simplified, shorter now (optimization moved out)
      case 'optimize':
        return { minWidth: 320, maxHeight: 700 }  // Wide for hyperparameters, tall for all controls
      case 'links':
        return { minWidth: 200, maxHeight: 480 }  // 1.5x taller for links list
      case 'nodes':
        return { minWidth: 200, maxHeight: 320 }  // Taller for nodes list
      case 'settings':
        return { minWidth: 280, maxHeight: 900 }  // Wide enough for form fields, tall for all settings sections
      default:
        return { minWidth: 200, maxHeight: 400 }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // TOOLBAR CONTENT COMPONENTS (using extracted components from builder/toolbars/)
  // ═══════════════════════════════════════════════════════════════════════════════

  const ToolsContent = () => (
    <ToolsToolbar
      toolMode={toolMode}
      setToolMode={setToolMode}
      hoveredTool={hoveredTool}
      setHoveredTool={setHoveredTool}
      linkCreationState={linkCreationState}
      setLinkCreationState={setLinkCreationState}
      setPreviewLine={setPreviewLine}
    />
  )

  // Update link (edge) property - using hypergraph operations
  const updateLinkProperty = useCallback((linkName: string, property: string, value: string | string[] | boolean) => {
    // Map legacy property names to edge meta
    const metaUpdate: Record<string, unknown> = { [property]: value }
    setLinkageDoc(prev => updateEdgeMeta(prev, linkName, metaUpdate))
  }, [])

  // Rename a link (edge) - using hypergraph operations
  const renameLink = useCallback((oldName: string, newName: string) => {
    if (oldName === newName || !newName.trim()) return

    const result = renameEdgeOperation(linkageDoc, oldName, newName)
    if (!result.success) {
      showStatus(result.error || `Link "${newName}" already exists`, 'error', 2000)
      return
    }

    setLinkageDoc(result.doc)

    // Update the modal data with new name if modal is open
    if (editingLinkData && editingLinkData.name === oldName) {
      setEditingLinkData(prev => prev ? { ...prev, name: newName } : null)
    }
    showStatus(`Renamed to ${newName}`, 'success', 1500)
  }, [linkageDoc, showStatus, editingLinkData])

  // Links toolbar content - double-click opens edit modal
  const LinksContent = () => (
    <LinksToolbar
      links={pylinkDoc.meta.links}
      selectedLinks={selectedLinks}
      setSelectedLinks={setSelectedLinks}
      setSelectedJoints={setSelectedJoints}
      hoveredLink={hoveredLink}
      setHoveredLink={setHoveredLink}
      selectionColor={selectionColor}
      getJointPosition={getJointPosition}
      openLinkEditModal={openLinkEditModal}
    />
  )

  // Valid pylinkage joint types
  const JOINT_TYPES = ['Static', 'Crank', 'Revolute'] as const

  // Update joint property - using hypergraph operations
  // IMPORTANT: For non-fixed roles, position is stored in the node
  const updateJointProperty = useCallback((jointName: string, property: string, value: string) => {
    if (property !== 'type') {
      // Only handle type changes for now
      return
    }

    // Map legacy type names to hypergraph roles
    const typeToRole: Record<string, 'fixed' | 'crank' | 'follower'> = {
      'Static': 'fixed',
      'Crank': 'crank',
      'Revolute': 'follower'
    }

    const newRole = typeToRole[value]
    if (!newRole) {
      showStatus(`Unknown joint type: ${value}`, 'error', 2000)
      return
    }

    // Use the hypergraph role change function
    const result = changeNodeRole(
      linkageDoc,
      jointName,
      newRole,
      getJointPosition
    )

    if (!result.success) {
      showStatus(result.error || `Failed to change ${jointName} to ${value}`, 'error', 2000)
      return
    }

    showStatus(`Changed ${jointName} to ${value}`, 'success', 1500)

    // Clear trajectory (will trigger auto-simulation via effect)
    clearTrajectory()

    // Update state
    setLinkageDoc(result.doc)

    // Trigger auto-simulation after state update
    triggerMechanismChange()
  }, [linkageDoc, getJointPosition, showStatus, triggerMechanismChange])

  // Rename a joint (node) - using hypergraph operations
  const renameJoint = useCallback((oldName: string, newName: string) => {
    if (oldName === newName || !newName.trim()) return

    const result = renameNodeOperation(linkageDoc, oldName, newName)
    if (!result.success) {
      showStatus(result.error || `Joint "${newName}" already exists`, 'error', 2000)
      return
    }

    setLinkageDoc(result.doc)

    // Update the modal data with new name if modal is open
    if (editingJointData && editingJointData.name === oldName) {
      setEditingJointData(prev => prev ? { ...prev, name: newName } : null)
    }
    showStatus(`Renamed to ${newName}`, 'success', 1500)
  }, [linkageDoc, showStatus, editingJointData])

  // Nodes toolbar content - double-click opens edit modal
  const NodesContent = () => (
    <NodesToolbar
      joints={pylinkDoc.pylinkage.joints}
      selectedJoints={selectedJoints}
      setSelectedJoints={setSelectedJoints}
      setSelectedLinks={setSelectedLinks}
      hoveredJoint={hoveredJoint}
      setHoveredJoint={setHoveredJoint}
      selectionColor={selectionColor}
      getJointPosition={getJointPosition}
      openJointEditModal={openJointEditModal}
    />
  )

  // More toolbar content
  const MoreContent = () => (
    <MoreToolbar
      joints={pylinkDoc.pylinkage.joints}
      animationState={animationState}
      playAnimation={playAnimation}
      pauseAnimation={pauseAnimation}
      stopAnimation={stopAnimation}
      setPlaybackSpeed={setPlaybackSpeed}
      setAnimatedPositions={setAnimatedPositions}
      isSimulating={isSimulating}
      trajectoryData={trajectoryData}
      autoSimulateEnabled={autoSimulateEnabled}
      setAutoSimulateEnabled={setAutoSimulateEnabled}
      autoSimulateDelayMs={autoSimulateDelayMs}
      runSimulation={runSimulation}
      triggerMechanismChange={triggerMechanismChange}
      showTrajectory={showTrajectory}
      setShowTrajectory={setShowTrajectory}
      stretchingLinks={stretchingLinks}
      loadDemo4Bar={loadDemo4Bar}
      loadPylinkGraph={loadPylinkGraph}
      savePylinkGraph={savePylinkGraph}
      validateMechanism={validateMechanism}
      showStatus={showStatus}
    />
  )

  // Run trajectory optimization
  // Extract current dimensions from pylink document (for showing before/after)
  const extractCurrentDimensions = (doc: PylinkDocument): Record<string, number> => {
    const dims: Record<string, number> = {}
    for (const joint of doc.pylinkage.joints) {
      if (joint.type === 'Crank' && joint.distance !== undefined) {
        dims[`${joint.name}_distance`] = joint.distance
      } else if (joint.type === 'Revolute') {
        if (joint.distance0 !== undefined) dims[`${joint.name}_distance0`] = joint.distance0
        if (joint.distance1 !== undefined) dims[`${joint.name}_distance1`] = joint.distance1
      }
    }
    return dims
  }

  // Revert to pre-optimization state
  const revertOptimization = () => {
    if (preOptimizationDoc) {
      setLinkageDoc(preOptimizationDoc)
      setPreOptimizationDoc(null)
      setOptimizationResult(null)
      triggerMechanismChange()
      showStatus('Reverted to pre-optimization state', 'info', 2000)
    }
  }

  // Preprocess trajectory (smooth and/or resample)
  const preprocessTrajectory = async () => {
    const selectedPath = targetPaths.find(p => p.id === selectedPathId)
    if (!selectedPath || selectedPath.points.length < 3) {
      showStatus('Select a path with at least 3 points', 'warning', 2000)
      return
    }

    try {
      setIsPreprocessing(true)
      setPreprocessResult(null)
      showStatus('Preprocessing trajectory...', 'action')

      const response = await fetch('/api/prepare-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trajectory: selectedPath.points,
          target_n_steps: prepTargetNSteps,
          smooth: prepEnableSmooth,
          smooth_window: prepSmoothWindow,
          smooth_polyorder: prepSmoothPolyorder,
          smooth_method: prepSmoothMethod,
          resample: prepEnableResample,
          resample_method: prepResampleMethod,
          closed: true  // All target paths are treated as closed/cyclic
        })
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        // Update the path with preprocessed points
        const newPoints: [number, number][] = result.trajectory.map((p: number[]) => [p[0], p[1]] as [number, number])

        setTargetPaths(prev => prev.map(p =>
          p.id === selectedPathId
            ? { ...p, points: newPoints, name: `${p.name} (processed)` }
            : p
        ))

        setPreprocessResult({
          originalPoints: result.original_points,
          outputPoints: result.output_points,
          analysis: result.analysis
        })

        showStatus(
          `Preprocessed: ${result.original_points} → ${result.output_points} points`,
          'success',
          3000
        )
      } else {
        showStatus(result.message || 'Preprocessing failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Preprocessing error: ${error}`, 'error', 3000)
    } finally {
      setIsPreprocessing(false)
    }
  }

  const runOptimization = async () => {
    const selectedPath = targetPaths.find(p => p.id === selectedPathId)
    if (!selectedPath || !selectedPath.targetJoint) {
      showStatus('Select a path and target joint first', 'warning', 2000)
      return
    }

    // Warn if target path points don't match simulation steps
    if (selectedPath.points.length !== simulationSteps) {
      console.warn(`⚠️ Path has ${selectedPath.points.length} points but simulation uses ${simulationSteps} steps. Consider preprocessing the path.`)
    }

    try {
      setIsOptimizing(true)
      setOptimizationResult(null)
      showStatus(`Running ${optMethod.toUpperCase()} optimization (N=${simulationSteps})...`, 'action')

      // Save current state before optimization (deep copy)
      const savedDoc = JSON.parse(JSON.stringify(linkageDoc)) as LinkageDocument
      setPreOptimizationDoc(savedDoc)

      // Extract original dimensions for comparison (uses legacy pylinkDoc view)
      const originalDims = extractCurrentDimensions(pylinkDoc)

      // Build optimization options based on method
      const optimizationOptions: Record<string, unknown> = {
        method: optMethod,
        verbose: optVerbose,
        bounds_factor: optBoundsFactor,
        min_length: optMinLength
      }

      // PSO-specific options
      if (optMethod === 'pso' || optMethod === 'pylinkage') {
        optimizationOptions.n_particles = optNParticles
        optimizationOptions.iterations = optIterations
      }

      // SciPy-specific options
      if (optMethod === 'scipy' || optMethod === 'powell' || optMethod === 'nelder-mead') {
        optimizationOptions.max_iterations = optMaxIterations
        optimizationOptions.tolerance = optTolerance
      }

      const response = await fetch('/api/optimize-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pylink_data: {
            ...linkageDoc,
            n_steps: simulationSteps  // Use simulation steps for consistency
          },
          target_path: {
            joint_name: selectedPath.targetJoint,
            positions: selectedPath.points
          },
          optimization_options: optimizationOptions
        })
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.result) {
        const optResult = result.result

        // Apply the optimized data (backend returns hypergraph format)
        if (optResult.optimized_pylink_data) {
          const resultData = optResult.optimized_pylink_data
          if (isHypergraphFormat(resultData)) {
            setLinkageDoc(resultData)
            triggerMechanismChange()
          } else {
            console.error('Backend returned unexpected legacy format')
            showStatus('Optimization failed - unexpected result format', 'error', 4000)
          }
        }

        const improvement = optResult.initial_error > 0
          ? ((1 - optResult.final_error / optResult.initial_error) * 100)
          : 0

        setOptimizationResult({
          success: optResult.success,
          initialError: optResult.initial_error,
          finalError: optResult.final_error,
          message: result.message,
          iterations: optResult.iterations,
          executionTimeMs: result.execution_time_ms,
          optimizedDimensions: optResult.optimized_dimensions,
          originalDimensions: originalDims
        })

        if (optResult.success) {
          showStatus(`Optimization complete: ${improvement.toFixed(1)}% improvement`, 'success', 4000)
        } else {
          showStatus(`Optimization finished with limited improvement: ${improvement.toFixed(1)}%`, 'warning', 4000)
        }
      } else {
        // Optimization failed - clear the saved state
        setPreOptimizationDoc(null)
        setOptimizationResult({
          success: false,
          initialError: 0,
          finalError: 0,
          message: result.message || 'Optimization failed'
        })
        showStatus(result.message || 'Optimization failed', 'error', 3000)
      }
    } catch (error) {
      // Error - clear the saved state
      setPreOptimizationDoc(null)
      setOptimizationResult({
        success: false,
        initialError: 0,
        finalError: 0,
        message: `Error: ${error}`
      })
      showStatus(`Optimization error: ${error}`, 'error', 3000)
    } finally {
      setIsOptimizing(false)
    }
  }

  // Settings toolbar content
  const SettingsContent = () => (
    <SettingsToolbar
      darkMode={darkMode}
      setDarkMode={setDarkMode}
      showGrid={showGrid}
      setShowGrid={setShowGrid}
      showJointLabels={showJointLabels}
      setShowJointLabels={setShowJointLabels}
      showLinkLabels={showLinkLabels}
      setShowLinkLabels={setShowLinkLabels}
      simulationStepsInput={simulationStepsInput}
      setSimulationStepsInput={setSimulationStepsInput}
      autoSimulateDelayMs={autoSimulateDelayMs}
      setAutoSimulateDelayMs={setAutoSimulateDelayMs}
      trajectoryColorCycle={trajectoryColorCycle}
      setTrajectoryColorCycle={setTrajectoryColorCycle}
      trajectoryData={trajectoryData}
      autoSimulateEnabled={autoSimulateEnabled}
      triggerMechanismChange={triggerMechanismChange}
      jointMergeRadius={jointMergeRadius}
      setJointMergeRadius={setJointMergeRadius}
      canvasBgColor={canvasBgColor as CanvasBgColor}
      setCanvasBgColor={setCanvasBgColor}
      jointSize={jointSize}
      setJointSize={setJointSize}
      linkThickness={linkThickness}
      setLinkThickness={setLinkThickness}
      trajectoryDotSize={trajectoryDotSize}
      setTrajectoryDotSize={setTrajectoryDotSize}
      selectionHighlightColor={selectionHighlightColor as SelectionHighlightColor}
      setSelectionHighlightColor={setSelectionHighlightColor}
      showMeasurementUnits={showMeasurementUnits}
      setShowMeasurementUnits={setShowMeasurementUnits}
      trajectoryStyle={trajectoryStyle as TrajectoryStyle}
      setTrajectoryStyle={setTrajectoryStyle}
    />
  )

  // Optimization toolbar content
  const OptimizationContent = () => (
    <OptimizationToolbar
      joints={pylinkDoc.pylinkage.joints}
      targetPaths={targetPaths}
      setTargetPaths={setTargetPaths}
      selectedPathId={selectedPathId}
      setSelectedPathId={setSelectedPathId}
      preprocessResult={preprocessResult}
      isPreprocessing={isPreprocessing}
      prepEnableSmooth={prepEnableSmooth}
      setPrepEnableSmooth={setPrepEnableSmooth}
      prepSmoothMethod={prepSmoothMethod as SmoothMethod}
      setPrepSmoothMethod={setPrepSmoothMethod}
      prepSmoothWindow={prepSmoothWindow}
      setPrepSmoothWindow={setPrepSmoothWindow}
      prepSmoothPolyorder={prepSmoothPolyorder}
      setPrepSmoothPolyorder={setPrepSmoothPolyorder}
      prepEnableResample={prepEnableResample}
      setPrepEnableResample={setPrepEnableResample}
      prepTargetNSteps={prepTargetNSteps}
      setPrepTargetNSteps={setPrepTargetNSteps}
      prepResampleMethod={prepResampleMethod as ResampleMethod}
      setPrepResampleMethod={setPrepResampleMethod}
      preprocessTrajectory={preprocessTrajectory}
      simulationSteps={simulationSteps}
      simulationStepsInput={simulationStepsInput}
      setSimulationStepsInput={setSimulationStepsInput}
      optMethod={optMethod as OptMethod}
      setOptMethod={setOptMethod}
      optNParticles={optNParticles}
      setOptNParticles={setOptNParticles}
      optIterations={optIterations}
      setOptIterations={setOptIterations}
      optMaxIterations={optMaxIterations}
      setOptMaxIterations={setOptMaxIterations}
      optTolerance={optTolerance}
      setOptTolerance={setOptTolerance}
      optBoundsFactor={optBoundsFactor}
      setOptBoundsFactor={setOptBoundsFactor}
      optMinLength={optMinLength}
      setOptMinLength={setOptMinLength}
      optVerbose={optVerbose}
      setOptVerbose={setOptVerbose}
      isOptimizing={isOptimizing}
      runOptimization={runOptimization}
      optimizationResult={optimizationResult}
      preOptimizationDoc={preOptimizationDoc}
      revertOptimization={revertOptimization}
    />
  )

  const renderToolbarContent = (id: string) => {
    switch (id) {
      case 'tools': return <ToolsContent />
      case 'links': return <LinksContent />
      case 'nodes': return <NodesContent />
      case 'more': return <MoreContent />
      case 'optimize': return <OptimizationContent />
      case 'settings': return <SettingsContent />
      default: return null
    }
  }

  // Demo: load a 4-bar linkage (hypergraph format)
  const loadDemo4Bar = () => {
    // 4-bar demo positions
    const crankAnchorPos: [number, number] = [90, 90]
    const rockerAnchorPos: [number, number] = [150, 90]
    const crankAngle = 0.2618
    const crankDistance = 20
    const crankPos: [number, number] = [
      crankAnchorPos[0] + crankDistance * Math.cos(crankAngle),
      crankAnchorPos[1] + crankDistance * Math.sin(crankAngle)
    ]

    // Calculate coupler position using circle-circle intersection
    const d0 = 50  // distance from crank
    const d1 = 40  // distance from rocker_anchor
    const dx = rockerAnchorPos[0] - crankPos[0]
    const dy = rockerAnchorPos[1] - crankPos[1]
    const d = Math.sqrt(dx * dx + dy * dy)
    const a = (d0 * d0 - d1 * d1 + d * d) / (2 * d)
    const h = Math.sqrt(Math.max(0, d0 * d0 - a * a))
    const px = crankPos[0] + (a * dx) / d
    const py = crankPos[1] + (a * dy) / d
    const couplerJointPos: [number, number] = [
      px - (h * dy) / d,
      py + (h * dx) / d
    ]

    // Create demo in hypergraph format directly
    const demo: LinkageDocument = {
      name: '4bar',
      version: '2.0.0',
      linkage: {
        name: '4bar',
        nodes: {
          crank_anchor: {
            id: 'crank_anchor',
            position: crankAnchorPos,
            role: 'fixed',
            jointType: 'revolute',
            name: 'crank_anchor'
          },
          rocker_anchor: {
            id: 'rocker_anchor',
            position: rockerAnchorPos,
            role: 'fixed',
            jointType: 'revolute',
            name: 'rocker_anchor'
          },
          crank: {
            id: 'crank',
            position: crankPos,
            role: 'crank',
            jointType: 'revolute',
            angle: crankAngle,
            name: 'crank'
          },
          coupler_rocker_joint: {
            id: 'coupler_rocker_joint',
            position: couplerJointPos,
            role: 'follower',
            jointType: 'revolute',
            name: 'coupler_rocker_joint'
          }
        },
        edges: {
          ground: {
            id: 'ground',
            source: 'crank_anchor',
            target: 'rocker_anchor',
            distance: 60
          },
          crank_link: {
            id: 'crank_link',
            source: 'crank_anchor',
            target: 'crank',
            distance: crankDistance
          },
          coupler: {
            id: 'coupler',
            source: 'crank',
            target: 'coupler_rocker_joint',
            distance: d0
          },
          rocker: {
            id: 'rocker',
            source: 'coupler_rocker_joint',
            target: 'rocker_anchor',
            distance: d1
          }
        },
        hyperedges: {}
      },
      meta: {
        nodes: {
          crank: { color: '#ff7f0e', zlevel: 0, showPath: true },
          coupler_rocker_joint: { color: '#2ca02c', zlevel: 0, showPath: true }
        },
        edges: {
          ground: { color: '#7f7f7f', isGround: true },
          crank_link: { color: '#ff7f0e' },
          coupler: { color: '#2ca02c' },
          rocker: { color: '#1f77b4' }
        }
      }
    }

    setLinkageDoc(demo)
    clearTrajectory()
    showStatus('Loaded demo 4-bar linkage', 'success', 2000)
  }

  // Save pylink graph to server
  const savePylinkGraph = async () => {
    try {
      showStatus('Saving...', 'action')
      const response = await fetch('/api/save-pylink-graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(linkageDoc)
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        showStatus(`Saved as ${result.filename}`, 'success', 3000)
      } else {
        showStatus(result.message || 'Save failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Save error: ${error}`, 'error', 3000)
    }
  }

  // Validate mechanism - calls backend to check if mechanism can be simulated
  const validateMechanism = async () => {
    try {
      showStatus('Validating mechanism...', 'action')
      const response = await fetch('/api/validate-mechanism', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(linkageDoc)
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'error') {
        showStatus(`Validation error: ${result.message}`, 'error', 5000)
        return
      }

      // Handle validation result
      if (result.valid) {
        const groupCount = result.valid_groups?.length || 0
        const jointCount = result.valid_groups?.reduce((acc: number, g: { joints: string[] }) => acc + g.joints.length, 0) || 0
        showStatus(`✓ Valid mechanism: ${groupCount} group(s), ${jointCount} joints`, 'success', 4000)
      } else {
        // Collect error messages
        const errors = result.errors || []
        const errorMsg = errors.length > 0
          ? errors.slice(0, 2).join('; ') + (errors.length > 2 ? ` (+${errors.length - 2} more)` : '')
          : 'No valid mechanism groups found'
        showStatus(`⚠ Invalid: ${errorMsg}`, 'warning', 5000)
      }
    } catch (error) {
      showStatus(`Validation error: ${error}`, 'error', 3000)
    }
  }

  // Detect if data is in new hypergraph format (version 2.0.0 with linkage property)
  const isHypergraphFormat = (data: unknown): data is LinkageDocument => {
    return (
      typeof data === 'object' &&
      data !== null &&
      'version' in data &&
      (data as { version: string }).version === '2.0.0' &&
      'linkage' in data
    )
  }

  // Load pylink graph from server (most recent)
  const loadPylinkGraph = async () => {
    try {
      showStatus('Loading...', 'action')
      const response = await fetch('/api/load-pylink-graph')

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.data) {
        // Check if it's the new hypergraph format
        if (isHypergraphFormat(result.data)) {
          // New format - load directly
          setLinkageDoc(result.data)
          setSelectedJoints([])
          setSelectedLinks([])
          clearTrajectory()
          triggerMechanismChange()
          showStatus(`Loaded ${result.filename}`, 'success', 3000)
        } else {
          // Legacy format - not supported anymore
          // TODO: Add migration tool or backend conversion endpoint
          console.warn(`File ${result.filename} is in legacy format - cannot load directly`)
          showStatus(`Cannot load ${result.filename} - legacy format not supported. Re-save the file to update.`, 'error', 5000)
        }
      } else {
        showStatus(result.message || 'No graphs to load', 'warning', 3000)
      }
    } catch (error) {
      showStatus(`Load error: ${error}`, 'error', 3000)
    }
  }

  return (
    <Box
      ref={containerRef}
      sx={{
        position: 'relative',
        width: '100%',
        maxWidth: '100vw',
        height: 'calc(100vh - 140px)',
        minWidth: CANVAS_MIN_WIDTH_PX,
        minHeight: CANVAS_MIN_HEIGHT_PX,
        pb: '38px', // Space for fixed footer (reduced from 44px)
        overflow: 'hidden'
      }}
    >
      {/* Canvas area */}
      <Paper
        ref={canvasRef}
        sx={{
          width: '100%',
          height: '100%',
          overflow: 'hidden',
          backgroundColor: canvasBgColor === 'default'
            ? 'var(--color-canvas-bg)'
            : canvasBgColor === 'white' ? '#ffffff'
            : canvasBgColor === 'cream' ? '#FAF3E1'
            : '#1a1a1a',
          border: '1px solid var(--color-border)',
          borderRadius: 2,
          position: 'relative',
          cursor: getCursorStyle()
        }}
      >
        <svg
          width="100%"
          height="100%"
          style={{ display: 'block' }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
          onClick={handleCanvasClick}
          onDoubleClick={handleCanvasDoubleClick}
        >
          {/* SVG Filter Definitions for Glow Effects */}
          <SVGFilters />

          {/* Grid (toggleable via Settings) */}
          {showGrid && renderGrid()}

          {/* Completed drawn objects (polygons, shapes) - render BEFORE links so links are on top */}
          {renderDrawnObjects()}

          {/* Links - rendered on top of polygons so they can be clicked even inside polygons */}
          {renderLinks()}

          {/* Preview line during link creation */}
          {renderPreviewLine()}

          {/* Polygon preview during drawing */}
          {renderPolygonPreview()}

          {/* Target paths for trajectory optimization */}
          {renderTargetPaths()}

          {/* Target path preview during drawing */}
          {renderPathPreview()}

          {/* Trajectory dots for simulation results */}
          {renderTrajectories()}

          {/* Joints */}
          {renderJoints()}

          {/* Group selection box */}
          {renderSelectionBox()}

          {/* Measurement markers */}
          {renderMeasurementMarkers()}
          {renderMeasurementLine()}
        </svg>

        {/* ToolbarToggleButtonsContainer - Horizontal button bar for toggling toolbars */}
        {/* Contains: Tools, Links, Nodes, More buttons in a horizontal row */}
        <ToolbarToggleButtons
          openToolbars={openToolbars}
          onToggleToolbar={handleToggleToolbar}
        />

        {/* Draggable floating toolbars */}
        {Array.from(openToolbars).map(toolbarId => {
          const config = TOOLBAR_CONFIGS.find(c => c.id === toolbarId)
          if (!config) return null
          const dimensions = getToolbarDimensions(toolbarId)
          return (
            <DraggableToolbar
              key={toolbarId}
              id={toolbarId}
              title={config.title}
              icon={config.icon}
              initialPosition={getToolbarPosition(toolbarId)}
              onClose={() => handleToggleToolbar(toolbarId)}
              onPositionChange={handleToolbarPositionChange}
              minWidth={dimensions.minWidth}
              maxHeight={dimensions.maxHeight}
            >
              {renderToolbarContent(toolbarId)}
            </DraggableToolbar>
          )
        })}
      </Paper>

      {/* Fixed Footer Toolbar */}
      <FooterToolbar
        toolMode={toolMode}
        jointCount={pylinkDoc.pylinkage.joints.length}
        linkCount={Object.keys(pylinkDoc.meta.links).length}
        selectedJoints={selectedJoints}
        selectedLinks={selectedLinks}
        statusMessage={statusMessage}
        linkCreationState={linkCreationState}
        polygonDrawState={polygonDrawState}
        measureState={measureState}
        groupSelectionState={groupSelectionState}
        mergePolygonState={mergePolygonState}
        pathDrawState={pathDrawState}
        canvasWidth={canvasDimensions.width}
        onCancelAction={cancelAction}
        darkMode={darkMode}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, joints: [], links: [] })}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            confirmDelete()
          }
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>
          Confirm Delete
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete {deleteConfirmDialog.joints.length + deleteConfirmDialog.links.length} items?
          </DialogContentText>
          <Box sx={{ mt: 2 }}>
            {deleteConfirmDialog.joints.length > 0 && (
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                <strong>Joints:</strong> {deleteConfirmDialog.joints.join(', ')}
              </Typography>
            )}
            {deleteConfirmDialog.links.length > 0 && (
              <Typography variant="body2">
                <strong>Links:</strong> {deleteConfirmDialog.links.join(', ')}
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDeleteConfirmDialog({ open: false, joints: [], links: [] })}
            color="inherit"
          >
            Cancel
          </Button>
          <Button
            onClick={confirmDelete}
            color="error"
            variant="contained"
            autoFocus
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Joint Edit Modal */}
      <JointEditModal
        open={editingJointData !== null}
        onClose={() => setEditingJointData(null)}
        jointData={editingJointData}
        jointTypes={JOINT_TYPES}
        onRename={renameJoint}
        onTypeChange={(jointName, newType) => updateJointProperty(jointName, 'type', newType)}
        onShowPathChange={(jointName, showPath) => {
          // Use hypergraph mutation
          setLinkageDoc(prev => updateNodeMeta(prev, jointName, { showPath }))
          // Update modal data
          setEditingJointData(prev => prev ? { ...prev, showPath } : null)
          // Trigger re-render of trajectories (paths are already computed, just need to re-render)
          // No need to re-simulate - just triggering state update refreshes the render
        }}
        darkMode={darkMode}
      />

      {/* Link Edit Modal */}
      <LinkEditModal
        open={editingLinkData !== null}
        onClose={() => setEditingLinkData(null)}
        linkData={editingLinkData}
        onRename={renameLink}
        onColorChange={(linkName, color) => {
          updateLinkProperty(linkName, 'color', color)
          // Update modal data
          setEditingLinkData(prev => prev ? { ...prev, color } : null)
        }}
        onGroundChange={(linkName, isGround) => {
          updateLinkProperty(linkName, 'isGround', isGround)
          // Update modal data
          setEditingLinkData(prev => prev ? { ...prev, isGround } : null)
        }}
        darkMode={darkMode}
      />
    </Box>
  )
}

export default BuilderTab
