import React, { useRef, useCallback, useEffect, useMemo } from 'react'
import { Box } from '@mui/material'
import {
  TOOLS,
  initialLinkCreationState,
  initialDragState,
  initialGroupSelectionState,
  initialPolygonDrawState,
  initialMeasureState,
  DrawnObjectsState,
  createDrawnObject,
  initialMoveGroupState,
  initialMergePolygonState,
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
  transformPolygonPoints,
  MERGE_THRESHOLD,
  JOINT_SNAP_THRESHOLD,
  TOOLBAR_CONFIGS,
  ToolbarPosition,
  JointData,
  LinkData
} from './BuilderTools'
import {
  jointColors,
  getCyclicColor
} from '../theme'
import {
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
  AnimateToolbar,
  // Toolbar types
  type CanvasBgColor,
  type TrajectoryStyle,
  type OptMethod,
  type SmoothMethod,
  type ResampleMethod,
  // Rendering (layer renders moved to useCanvasLayerRenders)
  getHighlightStyle as getHighlightStyleFromBuilder,
  BuilderModals,
  BuilderCanvasArea,
  BuilderToolbars,
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
  changeNodeRole,
  // Optimization Controller
  useOptimizationController,
  // Animation Controller
  useAnimationController,
  // Tool handlers
  getToolHandler,
  handleMergeLinkClick,
  handleMergePolygonClick,
  useMoveGroup,
  useModalState,
  useToolSelectionState,
  useDrawnPathState,
  useToolFlowsState,
  useStatusState,
  useCanvasSettingsState,
  useDocumentState,
  type ToolContext,
  applyLoadedDocument,
  computeOptimizerSyncStatus,
  useCanvasLayerRenders,
  type UseCanvasLayerRendersParams
} from './builder'
import { getStoredGraphFilename, setStoredGraphFilename } from '../prefs'

// ═══════════════════════════════════════════════════════════════════════════════
// BUILDER TAB — Main builder view orchestrator
// ═══════════════════════════════════════════════════════════════════════════════
//
// Responsibilities:
// - Own top-level state (linkage doc, selection, tool mode, draw/measure/merge/path state,
//   modals, move group, animation, optimization, settings).
// - Compose OptimizationController, AnimationController, useMoveGroup, buildToolContext/toolContext.
// - Compose presentational components: BuilderModals, BuilderCanvasArea, BuilderToolbars, AnimateToolbar.
// - Provide canvas event handlers and layer render functions to BuilderCanvasArea.
// - Provide toolbar content via renderToolbarContent to BuilderToolbars.
//
// Layer render data prep is in useCanvasLayerRenders; load/sync helpers in builder/helpers.
// Line count still above ~1,500; further extractions (e.g. toolbar content factory) optional.
// ═══════════════════════════════════════════════════════════════════════════════

const BuilderTab: React.FC = () => {
  // Canvas scaling helpers
  const pixelsToUnits = (pixels: number) => pixels / PIXELS_PER_UNIT
  const unitsToPixels = (units: number) => units * PIXELS_PER_UNIT

  // Tool + selection state (consolidated: tool mode, toolbars, selection, hover)
  const {
    toolMode,
    setToolMode,
    openToolbars,
    setOpenToolbars,
    toolbarPositions,
    setToolbarPositions,
    hoveredTool,
    setHoveredTool,
    selectedJoints,
    setSelectedJoints,
    selectedLinks,
    setSelectedLinks,
    hoveredJoint,
    setHoveredJoint,
    hoveredLink,
    setHoveredLink,
    hoveredPolygonId,
    setHoveredPolygonId
  } = useToolSelectionState()

  // Modal state (consolidated: joint edit, link edit, delete confirm)
  const {
    editingJointData,
    setEditingJointData,
    editingLinkData,
    setEditingLinkData,
    deleteConfirmDialog,
    setDeleteConfirmDialog
  } = useModalState()

  // Document state (consolidated: linkage hypergraph)
  const { linkageDoc, setLinkageDoc } = useDocumentState()

  // ═══════════════════════════════════════════════════════════════════════════════
  // LEGACY FORMAT VIEW - Computed from hypergraph state for backward-compatible APIs
  // ═══════════════════════════════════════════════════════════════════════════════

  // Convert new LinkageDocument to legacy PylinkDocument format for backward compatibility
  // Used by: toolbars, rendering, validation, etc. that still use legacy format
  const pylinkDoc: PylinkDocument = useMemo(() => {
    return convertLinkageDocumentToLegacy(linkageDoc)
  }, [linkageDoc])

  // Status state (consolidated: statusMessage, showStatus, clearStatus)
  const { statusMessage, showStatus, clearStatus } = useStatusState()

  // Tool flows state (consolidated: link creation, drag, group select, polygon draw, measure, move group)
  const {
    linkCreationState,
    setLinkCreationState,
    previewLine,
    setPreviewLine,
    dragState,
    setDragState,
    groupSelectionState,
    setGroupSelectionState,
    polygonDrawState,
    setPolygonDrawState,
    measureState,
    setMeasureState,
    measurementMarkers,
    setMeasurementMarkers,
    moveGroupState,
    setMoveGroupState
  } = useToolFlowsState()

  // Drawn / path / merge state (consolidated: polygons, target paths, path drawing, merge tool)
  const {
    drawnObjects,
    setDrawnObjects,
    targetPaths,
    setTargetPaths,
    selectedPathId,
    setSelectedPathId,
    pathDrawState,
    setPathDrawState,
    mergePolygonState,
    setMergePolygonState
  } = useDrawnPathState()

  // Canvas/settings state (consolidated: display, simulation, dimensions)
  const settings = useCanvasSettingsState()
  const {
    simulationSteps,
    setSimulationSteps,
    simulationStepsInput,
    setSimulationStepsInput,
    mechanismVersion,
    setMechanismVersion,
    showTrajectory,
    setShowTrajectory,
    autoSimulateDelayMs,
    setAutoSimulateDelayMs,
    jointMergeRadius,
    setJointMergeRadius,
    trajectoryColorCycle,
    setTrajectoryColorCycle,
    darkMode,
    setDarkMode,
    showGrid,
    setShowGrid,
    showJointLabels,
    setShowJointLabels,
    showLinkLabels,
    setShowLinkLabels,
    canvasBgColor,
    setCanvasBgColor,
    jointSize,
    setJointSize,
    linkThickness,
    setLinkThickness,
    trajectoryDotSize,
    setTrajectoryDotSize,
    trajectoryDotOutline,
    setTrajectoryDotOutline,
    trajectoryDotOpacity,
    setTrajectoryDotOpacity,
    selectionHighlightColor,
    trajectoryStyle,
    setTrajectoryStyle,
    canvasDimensions,
    setCanvasDimensions
  } = settings

  // Derived selection color
  const selectionColorMap = { blue: '#1976d2', orange: '#FA8112', green: '#2e7d32', purple: '#9c27b0' }
  const selectionColor = selectionColorMap[selectionHighlightColor]

  // Use builder's getHighlightStyle with theme jointColors (4-arg for renderers)
  const getHighlightStyle = useCallback(
    (objectType: 'joint' | 'link' | 'polygon', highlightType: 'none' | 'selected' | 'hovered' | 'move_group' | 'merge', baseColor: string, baseStrokeWidth: number) =>
      getHighlightStyleFromBuilder(objectType, highlightType, baseColor, baseStrokeWidth, jointColors),
    [jointColors]
  )

  const canvasRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Animation: positions override during playback - now managed by AnimationController

  // Update canvas dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setCanvasDimensions({
          width: rect.width,
          height: Math.max(rect.height - 48 + 12, CANVAS_MIN_HEIGHT_PX) // Account for footer + extra 12px
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

  // Debounced simulation steps: parse input after user stops typing and update steps used for sim. Never overwrite the input.
  const simulationStepsInputRef = useRef(simulationStepsInput)
  simulationStepsInputRef.current = simulationStepsInput
  const prevSimulationStepsRef = useRef(simulationSteps)

  useEffect(() => {
    const timer = setTimeout(() => {
      const val = parseInt(simulationStepsInputRef.current, 10)
      if (!isNaN(val)) {
        const clamped = Math.max(MIN_SIMULATION_STEPS, Math.min(MAX_SIMULATION_STEPS, val))
        setSimulationSteps(prev => (clamped !== prev ? clamped : prev))
      }
    }, 400)
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

  // Trigger mechanism change (for auto-simulation)
  const triggerMechanismChange = useCallback(() => {
    setMechanismVersion(v => v + 1)
  }, [setMechanismVersion])

  // CRITICAL: Logging helper for mechanism state tracking
  const logMechanismState = useCallback((label: string, doc: LinkageDocument) => {
    const nodeCount = Object.keys(doc.linkage?.nodes || {}).length
    const edgeCount = Object.keys(doc.linkage?.edges || {}).length
    const edgeDistances = Object.entries(doc.linkage?.edges || {}).slice(0, 5).map(([id, edge]) => ({
      id,
      distance: (edge as { distance?: number }).distance
    }))
    console.log(`[Mechanism State] ${label}:`, {
      nodeCount,
      edgeCount,
      edgeDistances,
      timestamp: Date.now()
    })
  }, [])

  // Extract dimensions from LinkageDocument using the same naming scheme as backend
  // Backend uses edge IDs: "edge_id_distance"
  // This matches the format returned in optimized_dimensions
  const extractDimensionsFromLinkageDoc = useCallback((doc: LinkageDocument): Record<string, number> => {
    const dims: Record<string, number> = {}
    const linkage = doc.linkage
    const nodes = linkage.nodes
    const edges = linkage.edges

    // Get static node IDs (for filtering ground links) - only 'fixed' is a node role; ground is an edge property
    const staticNodes = new Set(
      Object.entries(nodes)
        .filter(([_, node]) => node.role === 'fixed')
        .map(([nodeId, _]) => nodeId)
    )

    // Extract dimensions from edges (matches backend's extract_dimensions logic)
    for (const [edgeId, edge] of Object.entries(edges)) {
      const source = edge.source
      const target = edge.target

      // Skip edges between two static nodes (ground links - not optimizable)
      if (staticNodes.has(source) && staticNodes.has(target)) {
        continue
      }

      const distance = edge.distance
      if (distance !== undefined && distance !== null && distance > 0) {
        // Use same naming scheme as backend: "edge_id_distance"
        dims[`${edgeId}_distance`] = distance
      }
    }

    return dims
  }, [])

  // Detect if data is in new hypergraph format (version 2.0.0 with linkage property)
  const isHypergraphFormat = useCallback((data: unknown): data is LinkageDocument => {
    return (
      typeof data === 'object' &&
      data !== null &&
      'version' in data &&
      (data as { version: string }).version === '2.0.0' &&
      'linkage' in data
    )
  }, [])

  // ═══════════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION CONTROLLER - Step 1.2: State and functions
  // ═══════════════════════════════════════════════════════════════════════════════
  const optimization = useOptimizationController({
    linkageDoc,
    setLinkageDoc,
    targetPaths,
    setTargetPaths,
    selectedPathId,
    simulationSteps,
    showStatus,
    triggerMechanismChange,
    autoSimulateDelayMs,
    pylinkDoc,
    isHypergraphFormat,
    logMechanismState,
    extractDimensionsFromLinkageDoc
  })

  // ═══════════════════════════════════════════════════════════════════════════════
  // ANIMATION CONTROLLER - Step 2.2: State and functions
  // ═══════════════════════════════════════════════════════════════════════════════

  // Handle animation frame changes - update joint positions visually
  const handleAnimationFrameChange = useCallback((_frame: number) => {
    // This will be called by the animation hook when the frame changes
    // AnimationController handles updating animatedPositions internally
  }, [])

  // Dimension fetching is now handled by OptimizationController

  const animation = useAnimationController({
    // Simulation dependencies
    linkageDoc,
    simulationSteps,
    autoSimulateDelayMs,
    mechanismVersion,
    onSimulationComplete: optimization.handleSimulationComplete,
    showStatus,
    // Animation dependencies
    onFrameChange: handleAnimationFrameChange,
    frameIntervalMs: 50  // 20fps default
  })

  // Restore graph from cookie on mount (persisted current graph filename)
  const hasRestoredGraphRef = useRef(false)
  useEffect(() => {
    if (hasRestoredGraphRef.current) return
    const filename = getStoredGraphFilename()
    if (!filename) return
    hasRestoredGraphRef.current = true
    fetch(`/api/load-pylink-graph?filename=${encodeURIComponent(filename)}`)
      .then(res => res.ok ? res.json() : Promise.reject(new Error(res.statusText)))
      .then(result => {
        if (result.status === 'success' && result.data && isHypergraphFormat(result.data)) {
          applyLoadedDocument({
            doc: result.data,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setSelectedJoints,
            setSelectedLinks,
            clearTrajectory: animation.clearTrajectory,
            triggerMechanismChange
          })
          showStatus(`Restored ${result.filename}`, 'success', 2000)
        } else {
          setStoredGraphFilename(null)
        }
      })
      .catch(() => setStoredGraphFilename(null))
  }, [animation.clearTrajectory, triggerMechanismChange, isHypergraphFormat])

  // Validate links after simulation completes - detect links that would stretch
  useEffect(() => {
    if (animation.trajectoryData && animation.trajectoryData.trajectories) {
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
        animation.trajectoryData.jointTypes,
        animation.trajectoryData.trajectories
      )

      // Use the stretchingLinks directly from validation result
      if (validation.stretchingLinks.length > 0) {
        animation.setStretchingLinks(validation.stretchingLinks)

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
        if (animation.animationState.isAnimating) {
          animation.pauseAnimation()
        }
      } else {
        animation.setStretchingLinks([])
      }
    } else {
      animation.setStretchingLinks([])
    }
  }, [animation.trajectoryData, pylinkDoc.meta.links, showStatus, animation.animationState.isAnimating, animation.pauseAnimation])

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
  const getJointPosition = useCallback((jointName: string, visited: Set<string> = new Set()): [number, number] | null => {
    // Cycle detection: prevent infinite recursion
    if (visited.has(jointName)) {
      console.warn(`Cycle detected in joint relationships for ${jointName}`)
      return null
    }
    visited.add(jointName)

    // When animating, use animated positions for moving joints
    if (animation.animatedPositions && animation.animatedPositions[jointName]) {
      return animation.animatedPositions[jointName]
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
        const pos0 = getJointPosition(parent0.name, new Set(visited))
        const pos1 = getJointPosition(parent1.name, new Set(visited))
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
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints, animation.animatedPositions])

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
   * CRITICAL: Must read distance from LinkageDocument edge, not calculate from positions
   * This ensures we show the actual optimized edge distance, not a calculated value
   */
  const buildLinkData = useCallback((linkName: string): LinkData | null => {
    const linkMeta = pylinkDoc.meta.links[linkName]
    if (!linkMeta || linkMeta.connects.length < 2) return null

    // CRITICAL FIX: Read distance from LinkageDocument edge, not from visual positions
    // This ensures we show the actual optimized edge distance, not a calculated value
    const edge = linkageDoc.linkage?.edges?.[linkName]
    const edgeDistance = edge?.distance

    const pos0 = getJointPosition(linkMeta.connects[0])
    const pos1 = getJointPosition(linkMeta.connects[1])

    // Only calculate from positions if edge distance is not available (fallback)
    const calculatedLength = pos0 && pos1 ? calculateDistance(pos0, pos1) : null
    const length = edgeDistance !== undefined && edgeDistance !== null ? edgeDistance : calculatedLength

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
  }, [pylinkDoc.meta.links, linkageDoc, getJointPosition])

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

  // Clear optimized mechanism flag when mechanism is manually modified significantly
  // (e.g., deleting nodes/edges changes structure, not just dimensions)
  const clearOptimizedMechanismFlag = useCallback(() => {
    optimization.setIsOptimizedMechanism(false)
    optimization.setIsSyncedToOptimizer(false)
    optimization.lastOptimizerResultRef.current = null
  }, [])

  // Delete a link (edge) and any orphan nodes
  const deleteLink = useCallback((linkName: string) => {
    // Use new hypergraph operation
    const result = deleteEdgeOp(linkageDoc, linkName)

    // Clear trajectory and trigger auto-simulation
    animation.clearTrajectory()
    triggerMechanismChange()

    // Update state directly with hypergraph format
    setLinkageDoc(result.doc)

    // Clear optimized mechanism flag - structure changed (deleted link)
    clearOptimizedMechanismFlag()

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, clearOptimizedMechanismFlag])

  // Delete a joint (node) and all connected edges, plus any resulting orphans
  const deleteJoint = useCallback((jointName: string) => {
    // Use new hypergraph operation
    const result = deleteNodeOp(linkageDoc, jointName)

    // Clear trajectory and trigger auto-simulation
    animation.clearTrajectory()
    triggerMechanismChange()

    // Update state directly with hypergraph format
    setLinkageDoc(result.doc)

    // Clear optimized mechanism flag - structure changed (deleted node)
    clearOptimizedMechanismFlag()

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, clearOptimizedMechanismFlag])

  // Move a joint (node) to a new position
  // In hypergraph format, we just update the node position and sync edge distances
  const moveJoint = useCallback((jointName: string, newPosition: [number, number]) => {
    // CRITICAL: Check if mechanism is locked (optimization just completed)
    if (optimization.optimizationJustCompletedRef.current) {
      const timeSinceOpt = Date.now() - (optimization.optimizationCompleteTimeRef.current || 0)
      if (timeSinceOpt < 2000) {
        console.warn('[CRITICAL] Attempted to move node within 2s of optimization completion. Blocking to preserve optimizer result.', {
          timeSinceOpt,
          stack: new Error().stack
        })
        showStatus('CRITICAL: Blocked mechanism modification immediately after optimization', 'error', 5000)
        return // Block the modification
      }
    }

    const node = getNode(linkageDoc, jointName)
    if (!node) return

    // Move the node to new position
    let newDoc = moveNodeMutation(linkageDoc, jointName, newPosition)

    // CRITICAL: Only sync edge distances if NOT an optimized mechanism
    // Optimized mechanisms have exact distances that must be preserved
    if (!optimization.isOptimizedMechanism) {
      // Sync all edge distances from the new positions
      // This ensures the kinematic constraints match the visual positions
      newDoc = syncAllEdgeDistances(newDoc)
    } else {
      console.warn('[CRITICAL] Skipping syncAllEdgeDistances on optimized mechanism to preserve optimizer distances')
    }

    // Clear trajectory and trigger auto-simulation if enabled
    animation.clearTrajectory()
    triggerMechanismChange()

    setLinkageDoc(newDoc)

    // Mark as unsynced if manually modified (but keep isOptimizedMechanism true)
    // setIsSyncedToOptimizer will be updated by the useEffect
  }, [linkageDoc, showStatus, optimization.isOptimizedMechanism])

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

    // Mark as unsynced if manually modified (but keep isOptimizedMechanism true)
    // setIsSyncedToOptimizer will be updated by the useEffect
  }, [linkageDoc, showStatus])

  const getLinkConnects = useCallback((linkName: string): [string, string] | null => {
    const m = pylinkDoc.meta.links[linkName]
    return m?.connects?.length >= 2 ? (m.connects as [string, string]) : null
  }, [pylinkDoc.meta.links])

  const moveGroup = useMoveGroup({
    moveGroupState,
    setMoveGroupState,
    getJointsWithPositions,
    getLinksWithPositions,
    getJointPosition,
    findNearestJoint,
    findNearestLink,
    getLinkConnects,
    drawnObjects,
    setDrawnObjects: setDrawnObjects as React.Dispatch<React.SetStateAction<{ objects: unknown[]; selectedIds: string[] }>>,
    translateGroupRigid,
    exitMoveGroupMode,
    showStatus: showStatus as (message: string, type?: string, duration?: number) => void,
    triggerMechanismChange
  })

  // Merge two joints (nodes) together (source is absorbed into target)
  const mergeJoints = useCallback((sourceJoint: string, targetJoint: string) => {
    // Use new hypergraph operation
    const result = mergeNodesOperation(linkageDoc, sourceJoint, targetJoint)

    // Clear trajectory and trigger auto-simulation
    animation.clearTrajectory()
    triggerMechanismChange()

    setLinkageDoc(result.doc)
    setSelectedJoints([targetJoint])

    // Clear optimized mechanism flag - structure changed (merged nodes)
    clearOptimizedMechanismFlag()

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, triggerMechanismChange, clearOptimizedMechanismFlag])
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
    animation.clearTrajectory()
    triggerMechanismChange()

    // Update state
    setLinkageDoc(result.doc)

    showStatus(result.message, 'success', 2500)

    return result.edgeId
  }, [linkageDoc, showStatus, triggerMechanismChange])

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
    animation.clearTrajectory()
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

  function buildToolContext(params: Omit<
    ToolContext,
    'getPointFromEvent' | 'snapThreshold' | 'mergeThreshold' | 'getLinkMeta' | 'mergeLinkThreshold' | 'setDrawnObjects' | 'setTargetPaths' | 'createDrawnObject' | 'createTargetPath'
  > & {
    linkMeta: PylinkDocument['meta']['links']
    setDrawnObjects: React.Dispatch<React.SetStateAction<DrawnObjectsState>>
    setTargetPaths: React.Dispatch<React.SetStateAction<TargetPath[]>>
    createDrawnObject: (type: 'polygon', points: [number, number][], existingIds: string[]) => { id: string; type: string; points: [number, number][]; name: string }
    createTargetPath: (points: [number, number][], existingPaths: TargetPath[]) => TargetPath
  }): ToolContext {
    const {
      canvasRef: cr,
      pixelsToUnits: pu,
      linkMeta,
      setDrawnObjects: setDrawn,
      setTargetPaths: setPaths,
      createTargetPath: createPath,
      createDrawnObject: createDrawn,
      ...rest
    } = params
    return {
      ...rest,
      canvasRef: cr as React.RefObject<SVGSVGElement | HTMLDivElement | null>,
      pixelsToUnits: pu,
      getPointFromEvent: (event: React.MouseEvent<SVGSVGElement>) => {
        if (!cr.current) return null
        const rect = cr.current.getBoundingClientRect()
        const x = pu(event.clientX - rect.left)
        const y = pu(event.clientY - rect.top)
        return [x, y] as [number, number]
      },
      snapThreshold: JOINT_SNAP_THRESHOLD,
      mergeThreshold: MERGE_THRESHOLD,
      getLinkMeta: (linkName: string) => {
        const m = linkMeta[linkName]
        return m ? { connects: m.connects as [string, string], color: m.color } : null
      },
      mergeLinkThreshold: 8.0,
      setDrawnObjects: setDrawn as unknown as ToolContext['setDrawnObjects'],
      createDrawnObject: createDrawn as unknown as ToolContext['createDrawnObject'],
      setTargetPaths: setPaths as unknown as ToolContext['setTargetPaths'],
      createTargetPath: createPath as unknown as ToolContext['createTargetPath']
    } as ToolContext
  }

  // Tool context for per-tool handlers (select, etc.)
  const toolContext: ToolContext = useMemo(
    () =>
      buildToolContext({
        canvasRef,
        pixelsToUnits,
        toolMode,
        setToolMode,
        getJointsWithPositions,
        getLinksWithPositions,
        getJointPosition,
        findNearestJoint,
        findNearestLink,
        findMergeTarget,
        jointMergeRadius,
        dragState,
        setDragState,
        selectedJoints,
        setSelectedJoints,
        selectedLinks,
        setSelectedLinks,
        moveJoint,
        mergeJoints,
        showStatus,
        clearStatus,
        linkCreationState,
        setLinkCreationState,
        setPreviewLine,
        createLinkWithRevoluteDefault,
        deleteJoint,
        deleteLink,
        handleDeleteSelected,
        measureState,
        setMeasureState,
        setMeasurementMarkers,
        calculateDistance,
        groupSelectionState,
        setGroupSelectionState,
        findElementsInBox,
        drawnObjects,
        setDrawnObjects,
        enterMoveGroupMode,
        polygonDrawState,
        setPolygonDrawState,
        createDrawnObject,
        mergePolygonState,
        setMergePolygonState,
        linkMeta: pylinkDoc.meta.links,
        isPointInPolygon,
        areLinkEndpointsInPolygon,
        getDefaultColor,
        transformPolygonPoints,
        pathDrawState,
        setPathDrawState,
        targetPaths,
        setTargetPaths,
        createTargetPath,
        setSelectedPathId
      }),
    [
      toolMode,
      dragState,
      selectedJoints,
      selectedLinks,
      jointMergeRadius,
      linkCreationState,
      measureState,
      groupSelectionState,
      polygonDrawState,
      mergePolygonState,
      pathDrawState,
      drawnObjects,
      targetPaths,
      pylinkDoc.meta.links,
      getJointsWithPositions,
      getLinksWithPositions,
      getJointPosition,
      moveJoint,
      mergeJoints,
      showStatus,
      clearStatus,
      createLinkWithRevoluteDefault,
      handleDeleteSelected,
      transformPolygonPoints
    ]
  )

  // Layer render functions (data prep + render wiring) — built by hook
  const layerRenders = useCanvasLayerRenders({
    pylinkDoc,
    getJointPosition,
    getDefaultColor,
    getHighlightStyle,
    unitsToPixels,
    pixelsToUnits,
    toolMode,
    selectedJoints,
    selectedLinks,
    hoveredJoint,
    hoveredLink,
    hoveredPolygonId,
    moveGroupState,
    dragState,
    groupSelectionState,
    polygonDrawState,
    pathDrawState,
    measureState,
    measurementMarkers,
    previewLine,
    trajectoryData: animation.trajectoryData ?? null,
    stretchingLinks: animation.stretchingLinks,
    showTrajectory,
    canvasDimensions,
    darkMode,
    jointSize,
    linkThickness,
    trajectoryDotSize,
    trajectoryDotOutline,
    trajectoryDotOpacity,
    trajectoryStyle,
    trajectoryColorCycle,
    showJointLabels,
    showLinkLabels,
    jointMergeRadius,
    mergeThreshold: MERGE_THRESHOLD,
    targetPaths,
    selectedPathId,
    drawnObjects,
    jointColors,
    getCyclicColor: getCyclicColor as UseCanvasLayerRendersParams['getCyclicColor'],
    setHoveredJoint,
    setHoveredLink,
    setHoveredPolygonId,
    setDrawnObjects: setDrawnObjects as UseCanvasLayerRendersParams['setDrawnObjects'],
    setSelectedPathId,
    openJointEditModal,
    openLinkEditModal,
    handleMergeLinkClick,
    handleMergePolygonClick,
    toolContext,
    transformPolygonPoints
  })

  // Handle mouse down on canvas (for drag start)
  const handleCanvasMouseDown = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!canvasRef.current) return

    // Auto-pause animation on any canvas interaction
    if (animation.animationState.isAnimating) {
      animation.pauseAnimation()
      showStatus('Animation paused for editing', 'info', 1500)
    }

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const clickPoint: [number, number] = [x, y]

    if (moveGroup.handleMouseDown(event, clickPoint)) return

    const point = toolContext.getPointFromEvent(event)
    if (point) {
      const handler = getToolHandler(toolMode)
      if (handler.onMouseDown?.(event, point, toolContext)) return
    }
  }, [toolMode, toolContext, moveGroup, animation.animationState.isAnimating, animation.pauseAnimation])

  // Handle mouse move on canvas
  const handleCanvasMouseMove = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const currentPoint: [number, number] = [x, y]

    if (moveGroup.handleMouseMove(event, currentPoint)) return

    const handler = getToolHandler(toolMode)
    if (handler.onMouseMove?.(event, currentPoint, toolContext)) return
  }, [toolMode, toolContext, moveGroup, linkCreationState.isDrawing, linkCreationState.startPoint, dragState.isDragging, dragState.draggedJoint, groupSelectionState.isSelecting, groupSelectionState.startPoint])

  // Handle mouse up on canvas (for drag end)
  const handleCanvasMouseUp = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (moveGroup.handleMouseUp(event)) return

    const handler = getToolHandler(toolMode)
    if (handler.onMouseUp?.(event, toolContext)) return
  }, [toolMode, toolContext, moveGroup])

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

      // Escape to cancel (move mode first, then other actions)
      if (event.key === 'Escape') {
        if (moveGroupState.isActive) {
          exitMoveGroupMode()
          showStatus('Move mode cancelled', 'info', 2000)
          event.preventDefault()
          return
        }
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
        if (animation.animationState.isAnimating) {
          animation.pauseAnimation()
          showStatus('Animation paused', 'info', 1500)
        } else if (animation.trajectoryData && animation.trajectoryData.nSteps > 0) {
          animation.playAnimation()
          showStatus('Animation playing', 'info', 1500)
        } else if (canSimulate(pylinkDoc.pylinkage.joints)) {
          // No trajectory yet, run simulation first then play
          animation.runSimulation().then(() => {
            setTimeout(() => animation.playAnimation(), 100)
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
  }, [moveGroupState.isActive, exitMoveGroupMode, linkCreationState.isDrawing, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, pathDrawState.isDrawing, cancelAction, showStatus, selectedJoints, selectedLinks, handleDeleteSelected, animation.animationState.isAnimating, animation.playAnimation, animation.pauseAnimation, animation.trajectoryData, animation.runSimulation, pylinkDoc.pylinkage.joints, completePathDrawing])

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
    // Use larger threshold (8 units) in merge mode for easier link clicking
    const linkThreshold = toolMode === 'merge' ? 8.0 : JOINT_SNAP_THRESHOLD
    const nearestLink = findNearestLink(clickPoint, linksWithPositions, linkThreshold)

    // Delegate to tool handler (e.g. select: click to select joint/link)
    const handler = getToolHandler(toolMode)
    if (handler.onClick?.(event, clickPoint, toolContext)) return

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

  }, [toolMode, toolContext, linkCreationState, dragState.isDragging, groupSelectionState.isSelecting, measureState, polygonDrawState, pathDrawState, selectedJoints, selectedLinks, getJointsWithPositions, getLinksWithPositions, pylinkDoc.meta.links, pylinkDoc.pylinkage.joints, deleteJoint, deleteLink, handleDeleteSelected, showStatus, clearStatus, triggerMechanismChange, createLinkWithRevoluteDefault])

  // Handle canvas double-click (for completing path drawing)
  const handleCanvasDoubleClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    const point = toolContext.getPointFromEvent(event)
    if (point) {
      const handler = getToolHandler(toolMode)
      if (handler.onDoubleClick?.(event, point, toolContext)) return
    }
  }, [toolMode, toolContext])

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
  // Negative x/y values mean "offset from right/bottom edge of canvas"
  const getToolbarPosition = (id: string): ToolbarPosition => {
    if (toolbarPositions[id]) return toolbarPositions[id]
    const config = TOOLBAR_CONFIGS.find(c => c.id === id)
    const defaultPos = config?.defaultPosition || { x: 100, y: 100 }

    let x = defaultPos.x
    let y = defaultPos.y

    // Convert negative x to position from right edge
    if (defaultPos.x < 0) {
      x = canvasDimensions.width + defaultPos.x
    }
    // Convert negative y to position from bottom edge
    if (defaultPos.y < 0) {
      y = canvasDimensions.height + defaultPos.y
    }

    return { x, y }
  }

  // Get toolbar dimensions based on type
  const getToolbarDimensions = (id: string): { minWidth: number; maxHeight: number } => {
    switch (id) {
      case 'tools':
        // Tools should NEVER scroll - large maxHeight to fit all content
        return { minWidth: 220, maxHeight: 600 }
      case 'more':
        return { minWidth: 180, maxHeight: 500 }  // Tall enough to fit all tools without scrolling
      case 'optimize':
        return { minWidth: 960, maxHeight: 650 }  // 3x wider (320*3), shorter height for horizontal layout
      case 'links':
        return { minWidth: 200, maxHeight: 480 }  // 1.5x taller for links list
      case 'nodes':
        return { minWidth: 200, maxHeight: 320 }  // Taller for nodes list
      case 'settings':
        return { minWidth: 320, maxHeight: 900 }  // Wide for label+control rows
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
      onPauseAnimation={animation.animationState.isAnimating ? animation.pauseAnimation : undefined}
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
      linkageDoc={linkageDoc}
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
    animation.clearTrajectory()

    // Update state
    setLinkageDoc(result.doc)

    // Update modal data if it's open for this joint
    setEditingJointData(prev => {
      if (prev && prev.name === jointName) {
        return { ...prev, type: value as 'Static' | 'Crank' | 'Revolute' }
      }
      return prev
    })

    // Trigger auto-simulation after state update
    triggerMechanismChange()
  }, [linkageDoc, getJointPosition, showStatus, triggerMechanismChange])

  // Keyboard shortcuts for changing node type (Q=Revolute, W=Static, A=Crank)
  useEffect(() => {
    const handleNodeTypeShortcut = (event: KeyboardEvent) => {
      // Skip if typing in input field
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return
      }

      // Only handle if exactly one joint is selected
      if (selectedJoints.length !== 1) return

      const jointName = selectedJoints[0]
      let newType: string | null = null

      if (event.key === 'q' || event.key === 'Q') {
        newType = 'Revolute'
      } else if (event.key === 'w' || event.key === 'W') {
        newType = 'Static'
      } else if (event.key === 'a' || event.key === 'A') {
        newType = 'Crank'
      }

      if (newType) {
        updateJointProperty(jointName, 'type', newType)
        event.preventDefault()
      }
    }

    document.addEventListener('keydown', handleNodeTypeShortcut)
    return () => document.removeEventListener('keydown', handleNodeTypeShortcut)
  }, [selectedJoints, updateJointProperty])

  // Rename a joint (node) - using hypergraph operations
  const renameJoint = useCallback((oldName: string, newName: string) => {
    if (oldName === newName || !newName.trim()) return

    const result = renameNodeOperation(linkageDoc, oldName, newName)
    if (!result.success) {
      showStatus(result.error || `Joint "${newName}" already exists`, 'error', 2000)
      return
    }

    setLinkageDoc(result.doc)

    // Update animated positions to use new name (prevents visual jump during animation)
    animation.setAnimatedPositions(prev => {
      if (!prev || !prev[oldName]) return prev
      const updated = { ...prev }
      updated[newName] = updated[oldName]
      delete updated[oldName]
      return updated
    })

    // Update selection state if the renamed joint was selected
    setSelectedJoints(prev =>
      prev.includes(oldName)
        ? prev.map(name => name === oldName ? newName : name)
        : prev
    )

    // Update hovered state if the renamed joint was hovered
    if (hoveredJoint === oldName) {
      setHoveredJoint(newName)
    }

    // Update the modal data with new name if modal is open
    if (editingJointData && editingJointData.name === oldName) {
      setEditingJointData(prev => prev ? { ...prev, name: newName } : null)
    }
    showStatus(`Renamed to ${newName}`, 'success', 1500)
  }, [linkageDoc, showStatus, editingJointData, hoveredJoint])

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

  // More toolbar content (Demos, File Operations, Validation)
  // Note: Animation controls are now in AnimateToolbar
  const MoreContent = () => (
    <MoreToolbar
      loadDemo4Bar={loadDemo4Bar}
      loadDemoLeg={loadDemoLeg}
      loadDemoWalker={loadDemoWalker}
      loadDemoComplex={loadDemoComplex}
      loadPylinkGraphLast={loadPylinkGraphLast}
      loadFromFile={loadFromFile}
      savePylinkGraph={savePylinkGraph}
      savePylinkGraphAs={savePylinkGraphAs}
    />
  )

  // ═══════════════════════════════════════════════════════════════════════════════
  // CRITICAL: Check if mechanism is synced to optimizer result - FAIL HARD ON MISMATCH
  // ═══════════════════════════════════════════════════════════════════════════════
  useEffect(() => {
    const result = computeOptimizerSyncStatus(
      optimization.lastOptimizerResultRef.current,
      linkageDoc,
      optimization.isOptimizedMechanism
    )

    if (result.kind === 'no_check') {
      optimization.setIsSyncedToOptimizer(false)
      return
    }
    if (result.kind === 'structural_mismatch') {
      const errorMsg = `CRITICAL ERROR: Mechanism structure changed! Missing: ${result.structuralMismatches.join(', ')}`
      console.error(errorMsg, { structuralMismatches: result.structuralMismatches })
      showStatus(errorMsg, 'error', 15000)
      console.error('[CRITICAL] Forcing re-application of optimizer result due to structural mismatch')
      setLinkageDoc(result.optimizerDoc)
      triggerMechanismChange()
      optimization.setIsSyncedToOptimizer(true)
      return
    }
    if (result.kind === 'value_mismatch') {
      optimization.setIsSyncedToOptimizer(false)
      console.log('[Sync] Mechanism modified by user - marked as unsynced', { valueMismatches: result.valueMismatches.length })
      return
    }
    optimization.setIsSyncedToOptimizer(true)
  }, [linkageDoc, optimization.isOptimizedMechanism, showStatus, triggerMechanismChange])

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
      trajectoryData={animation.trajectoryData}
      autoSimulateEnabled={animation.autoSimulateEnabled}
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
      trajectoryDotOutline={trajectoryDotOutline}
      setTrajectoryDotOutline={setTrajectoryDotOutline}
      trajectoryDotOpacity={trajectoryDotOpacity}
      setTrajectoryDotOpacity={setTrajectoryDotOpacity}
      trajectoryStyle={trajectoryStyle as TrajectoryStyle}
      setTrajectoryStyle={setTrajectoryStyle}
    />
  )

  // Optimization toolbar content
  const OptimizationContent = () => (
      <OptimizationToolbar
      joints={pylinkDoc.pylinkage.joints}
      linkageDoc={linkageDoc}
      trajectoryData={animation.trajectoryData}
      stretchingLinks={animation.stretchingLinks}
      targetPaths={targetPaths}
      setTargetPaths={setTargetPaths}
      selectedPathId={selectedPathId}
      setSelectedPathId={setSelectedPathId}
      preprocessResult={optimization.preprocessResult}
      isPreprocessing={optimization.isPreprocessing}
      prepEnableSmooth={optimization.prepEnableSmooth}
      setPrepEnableSmooth={optimization.setPrepEnableSmooth}
      prepSmoothMethod={optimization.prepSmoothMethod as SmoothMethod}
      setPrepSmoothMethod={optimization.setPrepSmoothMethod}
      prepSmoothWindow={optimization.prepSmoothWindow}
      setPrepSmoothWindow={optimization.setPrepSmoothWindow}
      prepSmoothPolyorder={optimization.prepSmoothPolyorder}
      setPrepSmoothPolyorder={optimization.setPrepSmoothPolyorder}
      prepEnableResample={optimization.prepEnableResample}
      setPrepEnableResample={optimization.setPrepEnableResample}
      prepTargetNSteps={optimization.prepTargetNSteps}
      setPrepTargetNSteps={optimization.setPrepTargetNSteps}
      prepResampleMethod={optimization.prepResampleMethod as ResampleMethod}
      setPrepResampleMethod={optimization.setPrepResampleMethod}
      preprocessTrajectory={optimization.preprocessTrajectory}
      simulationSteps={simulationSteps}
      simulationStepsInput={simulationStepsInput}
      setSimulationStepsInput={setSimulationStepsInput}
      optMethod={optimization.optMethod as OptMethod}
      setOptMethod={optimization.setOptMethod}
      optNParticles={optimization.optNParticles}
      setOptNParticles={optimization.setOptNParticles}
      optIterations={optimization.optIterations}
      setOptIterations={optimization.setOptIterations}
      optMaxIterations={optimization.optMaxIterations}
      setOptMaxIterations={optimization.setOptMaxIterations}
      optTolerance={optimization.optTolerance}
      setOptTolerance={optimization.setOptTolerance}
      optBoundsFactor={optimization.optBoundsFactor}
      setOptBoundsFactor={optimization.setOptBoundsFactor}
      optMinLength={optimization.optMinLength}
      setOptMinLength={optimization.setOptMinLength}
      optVerbose={optimization.optVerbose}
      setOptVerbose={optimization.setOptVerbose}
      isOptimizing={optimization.isOptimizing}
      runOptimization={(config?: Record<string, unknown>) => { optimization.runOptimization(config ?? {}).catch(err => console.error('Optimization error:', err)) }}
      optimizationResult={optimization.optimizationResult}
      preOptimizationDoc={optimization.preOptimizationDoc}
      revertOptimization={optimization.revertOptimization}
      syncToOptimizerResult={optimization.syncToOptimizerResult}
      isSyncedToOptimizer={optimization.isSyncedToOptimizer}
      dimensionInfo={optimization.dimensionInfo}
      isLoadingDimensions={optimization.isLoadingDimensions}
      dimensionInfoError={optimization.dimensionInfoError}
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

  // Load demo from backend
  const loadDemoFromBackend = async (demoName: string) => {
    try {
      showStatus(`Loading ${demoName} demo...`, 'action')
      const response = await fetch(`/api/load-demo?name=${demoName}`)

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.data) {
        // Check if it's the new hypergraph format
        if (isHypergraphFormat(result.data)) {
          // Store target_joint in metadata if provided by demo
          const docWithTargetJoint = result.data
          if (result.target_joint) {
            if (!docWithTargetJoint.meta) {
              docWithTargetJoint.meta = {}
            }
            docWithTargetJoint.meta.target_joint = result.target_joint
          }
          applyLoadedDocument({
            doc: docWithTargetJoint,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setSelectedJoints,
            setSelectedLinks,
            clearTrajectory: animation.clearTrajectory,
            triggerMechanismChange
          })
          showStatus(`Loaded ${demoName} demo`, 'success', 2000)
        } else {
          showStatus(`Demo ${demoName} is in legacy format - cannot load`, 'error', 3000)
        }
      } else {
        showStatus(result.message || `Failed to load ${demoName} demo`, 'error', 3000)
      }
    } catch (error) {
      showStatus(`Load error: ${error}`, 'error', 3000)
    }
  }

  // Demo loaders
  const loadDemo4Bar = () => loadDemoFromBackend('4bar')
  const loadDemoLeg = () => loadDemoFromBackend('leg')
  const loadDemoWalker = () => loadDemoFromBackend('walker')
  const loadDemoComplex = () => loadDemoFromBackend('complex')

  // Save pylink graph to server
  const savePylinkGraph = async () => {
    try {
      showStatus('Saving...', 'action')
      // Include drawnObjects in the document for persistence
      const docToSave = {
        ...linkageDoc,
        drawnObjects: drawnObjects.objects.length > 0 ? drawnObjects.objects : undefined
      }
      const response = await fetch('/api/save-pylink-graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(docToSave)
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        setStoredGraphFilename(result.filename)
        showStatus(`Saved as ${result.filename}`, 'success', 3000)
      } else {
        showStatus(result.message || 'Save failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Save error: ${error}`, 'error', 3000)
    }
  }

  // Load pylink graph from server (most recent) - used by "Load Last" button
  const loadPylinkGraphLast = async () => {
    try {
      showStatus('Loading last saved...', 'action')
      const response = await fetch('/api/load-pylink-graph')

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.data) {
        // Check if it's the new hypergraph format
        if (isHypergraphFormat(result.data)) {
          applyLoadedDocument({
            doc: result.data,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setSelectedJoints,
            setSelectedLinks,
            clearTrajectory: animation.clearTrajectory,
            triggerMechanismChange
          })
          setStoredGraphFilename(result.filename)
          showStatus(`Loaded ${result.filename}`, 'success', 3000)
        } else {
          // Legacy format - not supported anymore
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

  // Load pylink graph from server - show file picker dialog
  const loadFromFile = async () => {
    try {
      showStatus('Fetching file list...', 'action')
      const listResponse = await fetch('/api/list-pylink-graphs')

      if (!listResponse.ok) throw new Error(`HTTP error! status: ${listResponse.status}`)
      const listResult = await listResponse.json()

      if (listResult.status !== 'success' || !listResult.files || listResult.files.length === 0) {
        showStatus('No saved graphs found', 'warning', 3000)
        return
      }

      // Create a simple file selection dialog using browser prompt
      const files = listResult.files as Array<{ filename: string; name: string; saved_at: string }>
      const fileOptions = files.slice(0, 20).map((f, i) => `${i + 1}. ${f.name} (${f.saved_at})`).join('\n')
      const selection = prompt(`Select a file to load:\n\n${fileOptions}\n\nEnter number (1-${Math.min(files.length, 20)}):`)

      if (!selection) {
        showStatus('Load cancelled', 'info', 2000)
        return
      }

      const idx = parseInt(selection, 10) - 1
      if (isNaN(idx) || idx < 0 || idx >= files.length) {
        showStatus('Invalid selection', 'error', 3000)
        return
      }

      const selectedFile = files[idx]
      showStatus(`Loading ${selectedFile.name}...`, 'action')

      const response = await fetch(`/api/load-pylink-graph?filename=${encodeURIComponent(selectedFile.filename)}`)
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.data) {
        if (isHypergraphFormat(result.data)) {
          applyLoadedDocument({
            doc: result.data,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setSelectedJoints,
            setSelectedLinks,
            clearTrajectory: animation.clearTrajectory,
            triggerMechanismChange
          })
          setStoredGraphFilename(result.filename)
          showStatus(`Loaded ${result.filename}`, 'success', 3000)
        } else {
          showStatus(`Cannot load ${result.filename} - legacy format not supported`, 'error', 5000)
        }
      } else {
        showStatus(result.message || 'Load failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Load error: ${error}`, 'error', 3000)
    }
  }

  // Save pylink graph with custom filename
  const savePylinkGraphAs = async () => {
    const suggestedName = linkageDoc.name || 'untitled'
    const filename = prompt('Enter filename to save as:', suggestedName)

    if (!filename) {
      showStatus('Save cancelled', 'info', 2000)
      return
    }

    try {
      showStatus('Saving...', 'action')
      // Include drawnObjects in the document for persistence
      const docToSave = {
        ...linkageDoc,
        drawnObjects: drawnObjects.objects.length > 0 ? drawnObjects.objects : undefined
      }
      const response = await fetch('/api/save-pylink-graph-as', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: docToSave,
          filename: filename
        })
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        setStoredGraphFilename(result.filename)
        showStatus(`Saved as ${result.filename}`, 'success', 3000)
      } else {
        showStatus(result.message || 'Save failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Save error: ${error}`, 'error', 3000)
    }
  }

  // Compose: container → canvas area (with toolbars) → animate toolbar → modals
  return (
    <Box
      ref={containerRef}
      sx={{
        position: 'relative',
        width: '100%',
        maxWidth: '100vw',
        height: 'calc(100vh - 90px)',
        minWidth: CANVAS_MIN_WIDTH_PX,
        minHeight: CANVAS_MIN_HEIGHT_PX,
        overflow: 'hidden'
      }}
    >
      <BuilderCanvasArea
        canvasRef={canvasRef}
        canvasBgColor={canvasBgColor}
        cursor={getCursorStyle()}
        optimization={{
          isOptimizedMechanism: optimization.isOptimizedMechanism,
          isSyncedToOptimizer: optimization.isSyncedToOptimizer,
          syncToOptimizerResult: optimization.syncToOptimizerResult
        }}
        showGrid={showGrid}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseUp}
        onClick={handleCanvasClick}
        onDoubleClick={handleCanvasDoubleClick}
        renderGrid={layerRenders.renderGrid}
        renderDrawnObjects={layerRenders.renderDrawnObjects}
        renderLinks={layerRenders.renderLinks}
        renderPreviewLine={layerRenders.renderPreviewLine}
        renderPolygonPreview={layerRenders.renderPolygonPreview}
        renderTargetPaths={layerRenders.renderTargetPaths}
        renderPathPreview={layerRenders.renderPathPreview}
        renderTrajectories={layerRenders.renderTrajectories}
        renderJoints={layerRenders.renderJoints}
        renderSelectionBox={layerRenders.renderSelectionBox}
        renderMeasurementMarkers={layerRenders.renderMeasurementMarkers}
        renderMeasurementLine={layerRenders.renderMeasurementLine}
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
        openToolbars={openToolbars}
        onToggleToolbar={handleToggleToolbar}
        onToolbarInteract={animation.animationState.isAnimating ? animation.pauseAnimation : undefined}
        darkMode={darkMode}
      >
        <BuilderToolbars
          openToolbars={openToolbars}
          toolbarConfigs={TOOLBAR_CONFIGS}
          getToolbarPosition={getToolbarPosition}
          getToolbarDimensions={getToolbarDimensions}
          onToggleToolbar={handleToggleToolbar}
          onPositionChange={handleToolbarPositionChange}
          renderToolbarContent={renderToolbarContent}
          onInteract={animation.animationState.isAnimating ? animation.pauseAnimation : undefined}
        />
      </BuilderCanvasArea>

      {/* Animation Toolbar - Centered at bottom */}
      <AnimateToolbar
        joints={pylinkDoc.pylinkage.joints}
        animationState={animation.animationState}
        playAnimation={animation.playAnimation}
        pauseAnimation={animation.pauseAnimation}
        stopAnimation={animation.stopAnimation}
        setPlaybackSpeed={animation.setPlaybackSpeed}
        setAnimatedPositions={animation.setAnimatedPositions}
        setFrame={animation.setAnimationFrame}
        isSimulating={animation.isSimulating}
        trajectoryData={animation.trajectoryData}
        autoSimulateEnabled={animation.autoSimulateEnabled}
        setAutoSimulateEnabled={animation.setAutoSimulateEnabled}
        runSimulation={animation.runSimulation}
        triggerMechanismChange={triggerMechanismChange}
        showTrajectory={showTrajectory}
        setShowTrajectory={setShowTrajectory}
        stretchingLinks={animation.stretchingLinks}
        showStatus={showStatus}
        darkMode={darkMode}
      />

      <BuilderModals
        deleteConfirmDialog={deleteConfirmDialog}
        onCloseDeleteConfirm={() => setDeleteConfirmDialog({ open: false, joints: [], links: [] })}
        onConfirmDelete={confirmDelete}
        editingJointData={editingJointData}
        onCloseJointEdit={() => setEditingJointData(null)}
        editingLinkData={editingLinkData}
        onCloseLinkEdit={() => setEditingLinkData(null)}
        renameJoint={renameJoint}
        renameLink={renameLink}
        updateJointProperty={updateJointProperty}
        updateLinkProperty={updateLinkProperty}
        onJointShowPathChange={(jointName, showPath) =>
          setLinkageDoc(prev => updateNodeMeta(prev, jointName, { showPath }))
        }
        setEditingJointData={setEditingJointData}
        setEditingLinkData={setEditingLinkData}
        jointTypes={JOINT_TYPES}
        darkMode={darkMode}
      />
    </Box>
  )
}

export default BuilderTab
