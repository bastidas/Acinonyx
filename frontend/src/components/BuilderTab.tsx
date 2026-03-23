import React, { useRef, useCallback, useEffect, useLayoutEffect, useMemo } from 'react'
import { Box } from '@mui/material'
import {
  TOOLS,
  initialLinkCreationState,
  initialDragState,
  initialGroupSelectionState,
  initialPolygonDrawState,
  initialMeasureState,
  DrawnObject,
  DrawnObjectsState,
  createDrawnObject,
  initialMoveGroupState,
  initialMergePolygonState,
  initialPathDrawState,
  createTargetPath,
  findNearestJoint,
  findNearestLink,
  findMergeTarget,
  calculateDistance,
  getDefaultColor,
  findElementsInBox,
  isPointInPolygon,
  areLinkEndpointsInPolygon,
  transformPolygonPoints,
  MERGE_THRESHOLD,
  TOOLBAR_CONFIGS,
  ToolbarPosition,
  JointData,
  LinkData,
  FormData,
  type ExploreTrajectoriesState,
  type ExploreTrajectorySample
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
  DRAG_MOVE_THRESHOLD,
  CANVAS_MIN_WIDTH_PX,
  CANVAS_MIN_HEIGHT_PX,
  // Toolbar (AnimateToolbar; other toolbars rendered via useToolbarContent)
  type ZLevelRow,
  type ZLevelHeuristicConfig,
  DEFAULT_Z_LEVEL_CONFIG,
  AnimateToolbar,
  // Toolbar types
  type CanvasBgColor,
  type TrajectoryStyle,
  // Rendering (layer renders moved to useCanvasLayerRenders)
  getHighlightStyle as getHighlightStyleFromBuilder,
  BuilderModals,
  BuilderCanvasArea,
  BuilderToolbars,
  // Hypergraph helpers (new API)
  getNode,
  getNodeMeta,
  getConnectedNodes,
  // Hypergraph mutations (new API)
  moveNode as moveNodeMutation,
  syncAllEdgeDistances,
  exploreRegion,
  getExploreRegionOptionsForMaxPoints,
  getCombinatorialSecondOptions,
  remapEdgeReferencesInDrawnObjects,
  removeDrawnObjectsReferencingDeletedEdges,
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
  applyLoadedDocument,
  computeOptimizerSyncStatus,
  useCanvasLayerRenders,
  type UseCanvasLayerRendersParams,
  useDisplayFrame,
  useMechanismPositions,
  type PendingDropPosition,
  useToolContext,
  useCanvasEventHandlers,
  useToolbarContent,
  createEmptyLinkageDocument,
  useViewportState,
  buildSyncedDocAfterDrop
} from './builder'
import { getStoredGraphFilename, setStoredGraphFilename } from '../prefs'

// ═══════════════════════════════════════════════════════════════════════════════
// BUILDER TAB — Main builder view orchestrator
// ═══════════════════════════════════════════════════════════════════════════════
//
// Responsibilities:
// - Own top-level state (linkage doc, selection, tool mode, draw/measure/merge/path state,
//   modals, move group, animation, optimization, settings).
// - Compose OptimizationController, AnimationController, useMoveGroup, useToolContext.
// - Compose presentational components: BuilderModals, BuilderCanvasArea, BuilderToolbars, AnimateToolbar.
// - Provide canvas event handlers and layer render functions to BuilderCanvasArea.
// - Provide toolbar content via renderToolbarContent to BuilderToolbars.
//
// Layer render data prep is in useCanvasLayerRenders; load/sync helpers in builder/helpers.
// Line count still above ~1,500; further extractions (e.g. toolbar content factory) optional.
// ═══════════════════════════════════════════════════════════════════════════════

const BuilderTab: React.FC = () => {
  // Canvas scaling helpers (world space: used by rendering and distances)
  const pixelsToUnits = (pixels: number) => pixels / PIXELS_PER_UNIT
  const unitsToPixels = (units: number) => units * PIXELS_PER_UNIT

  const canvasRef = useRef<HTMLDivElement>(null)
  const viewport = useViewportState({ canvasRef })

  /** Convert screen pixel position (relative to canvas) to world units; used for mouse events when zoom/pan is active */
  const screenToUnit = useCallback((screenPx: number, screenPy: number): [number, number] => {
    const { zoom, panX, panY } = viewport.viewport
    const worldPx = (screenPx - panX) / zoom
    const worldPy = (screenPy - panY) / zoom
    return [worldPx / PIXELS_PER_UNIT, worldPy / PIXELS_PER_UNIT]
  }, [viewport.viewport])

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

  /**
   * Apply default metaValue heuristic without overwriting user-defined values.
   * - large: fixed/crank nodes
   * - medium: direct neighbors of fixed/crank nodes
   * - small: all others
   */
  const applyMetaValueHeuristicDefaults = useCallback((doc: LinkageDocument): LinkageDocument => {
    const nodes = Object.values(doc.linkage.nodes)
    if (nodes.length === 0) return doc

    const primaryNodeIds = new Set(
      nodes
        .filter(node => node.role === 'fixed' || node.role === 'crank')
        .map(node => node.id)
    )

    const mediumNodeIds = new Set<string>()
    for (const edge of Object.values(doc.linkage.edges)) {
      if (primaryNodeIds.has(edge.source) && !primaryNodeIds.has(edge.target)) {
        mediumNodeIds.add(edge.target)
      }
      if (primaryNodeIds.has(edge.target) && !primaryNodeIds.has(edge.source)) {
        mediumNodeIds.add(edge.source)
      }
    }

    let changed = false
    const nextNodeMeta = { ...doc.meta.nodes }

    for (const node of nodes) {
      const existingMeta = doc.meta.nodes[node.id]
      if (existingMeta?.metaValue !== undefined) continue

      const heuristicValue = primaryNodeIds.has(node.id)
        ? 'large'
        : mediumNodeIds.has(node.id)
          ? 'medium'
          : 'small'

      nextNodeMeta[node.id] = {
        ...(existingMeta ?? { color: '', zlevel: 0 }),
        metaValue: heuristicValue
      }
      changed = true
    }

    if (!changed) return doc

    return {
      ...doc,
      meta: {
        ...doc.meta,
        nodes: nextNodeMeta
      }
    }
  }, [])

  // Keep heuristic defaults populated as structure changes; preserve user-set values.
  useEffect(() => {
    const nextDoc = applyMetaValueHeuristicDefaults(linkageDoc)
    if (nextDoc !== linkageDoc) {
      setLinkageDoc(nextDoc)
    }
  }, [linkageDoc, applyMetaValueHeuristicDefaults, setLinkageDoc])

  // Status state (consolidated: statusMessage, showStatus, clearStatus, statusHistory)
  const { statusMessage, showStatus, clearStatus, statusHistory, clearStatusHistory } = useStatusState()

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

  // Drawn / path / merge state (consolidated: polygons, target paths, path drawing, merge tool, canvas images)
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
    setMergePolygonState,
    canvases,
    setCanvases
  } = useDrawnPathState()

  const [editingCanvasId, setEditingCanvasId] = React.useState<string | null>(null)

  // Forms toolbar and form edit modal
  const [editingFormData, setEditingFormData] = React.useState<FormData | null>(null)
  const runComputeAfterFormSaveRef = useRef(false)
  const [formPaddingUnits, setFormPaddingUnits] = React.useState(5)
  const [createRigidForms, setCreateRigidForms] = React.useState(true)
  const [createLinkForms, setCreateLinkForms] = React.useState(true)
  const [computeZLevelsAfterCreate, setComputeZLevelsAfterCreate] = React.useState(true)
  const [zLevelConfig, setZLevelConfig] = React.useState<ZLevelHeuristicConfig>(() => ({ ...DEFAULT_Z_LEVEL_CONFIG }))

  // Explore node trajectories mode: state cleared when switching tools
  const [exploreTrajectoriesState, setExploreTrajectoriesState] = React.useState<ExploreTrajectoriesState>({
    exploreNodeId: null,
    exploreCenter: null,
    exploreSamples: [],
    exploreHoveredIndex: null,
    exploreHoveredFromTrajectoryPath: false,
    exploreLoading: false,
    exploreMode: 'single',
    exploreSecondNodeId: null,
    exploreSecondCenter: null,
    explorePinnedFirstPosition: null
  })
  const resetExploreTrajectories = useCallback(() => {
    setExploreTrajectoriesState({
      exploreNodeId: null,
      exploreCenter: null,
      exploreSamples: [],
      exploreHoveredIndex: null,
      exploreHoveredFromTrajectoryPath: false,
      exploreLoading: false,
      exploreMode: 'single',
      exploreSecondNodeId: null,
      exploreSecondCenter: null,
      explorePinnedFirstPosition: null
    })
  }, [])

  /** Position of just-released joint to show until new trajectory arrives (avoids jump on release). */
  const [pendingDropPosition, setPendingDropPosition] = React.useState<PendingDropPosition>(null)

  const getNodeShowPath = useCallback((nodeId: string): boolean => {
    const meta = getNodeMeta(linkageDoc, nodeId)
    return meta?.showPath === true
  }, [linkageDoc])

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
    jointOutline,
    setJointOutline,
    linkThickness,
    setLinkThickness,
    linkTransparency,
    setLinkTransparency,
    linkColorMode,
    setLinkColorMode,
    linkColorSingle,
    setLinkColorSingle,
    trajectoryDotSize,
    setTrajectoryDotSize,
    trajectoryDotOutline,
    setTrajectoryDotOutline,
    trajectoryDotOpacity,
    setTrajectoryDotOpacity,
    showTrajectoryStepNumbers,
    setShowTrajectoryStepNumbers,
    selectionHighlightColor,
    trajectoryStyle,
    setTrajectoryStyle,
    canvasDimensions,
    setCanvasDimensions,
    exploreRadius,
    exploreRadialSamples,
    exploreAzimuthalSamples,
    exploreNMaxCombinatorial,
    exploreColormapEnabled,
    exploreColormapType
  } = settings

  const runExploreTrajectories = useCallback(async (nodeId: string, center: [number, number]) => {
    setExploreTrajectoriesState(prev => ({
      ...prev,
      exploreLoading: true,
      exploreNodeId: nodeId,
      exploreCenter: center,
      exploreSamples: [],
      exploreMode: 'single',
      exploreSecondNodeId: null,
      exploreSecondCenter: null,
      explorePinnedFirstPosition: null
    }))
    showStatus(`Exploring trajectories for ${nodeId}...`, 'action')
    try {
      const points = exploreRegion(center, { deltaDegrees: 360 / exploreAzimuthalSamples, R: exploreRadius, nRadialSamples: exploreRadialSamples })
      const requests = points.map(({ position }) => {
        let doc = moveNodeMutation(linkageDoc, nodeId, position)
        doc = syncAllEdgeDistances(doc)
        return { ...doc, n_steps: simulationSteps }
      })
      const response = await fetch('/api/compute-pylink-trajectories-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ requests })
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const data = await response.json()
      const results = data.results as Array<{ status: string; trajectories?: Record<string, [number, number][]>; n_steps?: number; joint_types?: Record<string, string>; message?: string }>
      if (!Array.isArray(results) || results.length !== points.length) {
        throw new Error('Invalid batch response')
      }
      const exploreSamples: ExploreTrajectorySample[] = points.map((p, i) => {
        const r = results[i]
        const valid = r?.status === 'success' && r?.trajectories != null
        return {
          position: p.position,
          valid,
          trajectory: valid && r.trajectories && r.n_steps != null
            ? { trajectories: r.trajectories, nSteps: r.n_steps, jointTypes: r.joint_types ?? {} }
            : null
        }
      })
      const validCount = exploreSamples.filter(s => s.valid).length
      setExploreTrajectoriesState(prev => ({
        ...prev,
        exploreSamples,
        exploreLoading: false
      }))
      showStatus(`Exploration complete: ${validCount} valid, ${exploreSamples.length - validCount} invalid`, 'success', 2500)
      showStatus('Click a trajectory to apply, or click another node to explore combinations.', 'action', 4000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Exploration failed'
      showStatus(msg, 'error', 3000)
      setExploreTrajectoriesState(prev => ({ ...prev, exploreSamples: [], exploreLoading: false }))
    }
  }, [linkageDoc, simulationSteps, showStatus, exploreRadius, exploreRadialSamples, exploreAzimuthalSamples])

  const runExploreTrajectoriesCombinatorial = useCallback(
    async (
      nodeId1: string,
      _center1: [number, number],
      samples1: Array<{ position: [number, number]; valid: boolean }>,
      nodeId2: string,
      center2: [number, number]
    ) => {
      const N1 = Math.max(1, samples1.length)
      const N2_max = Math.floor(exploreNMaxCombinatorial / N1)
      if (N2_max < 1) {
        showStatus(`Too many first samples for combinatorial exploration (max ${exploreNMaxCombinatorial} total).`, 'warning', 3000)
        return
      }
      // First run uses exploreAzimuthalSamples angles and exploreRadialSamples radial; use same desired shape and reduce until under cap
      const options = getCombinatorialSecondOptions(N2_max, exploreRadialSamples, exploreAzimuthalSamples)
      const points2 = exploreRegion(center2, {
        R: exploreRadius,
        deltaDegrees: options.deltaDegrees,
        nRadialSamples: options.nRadialSamples
      })
      const N2 = points2.length
      const total = N1 * N2
      setExploreTrajectoriesState(prev => ({
        ...prev,
        exploreLoading: true,
        exploreMode: 'combinatorial',
        exploreSecondNodeId: nodeId2,
        exploreSecondCenter: center2,
        exploreSamples: []
      }))
      showStatus(`Exploring combinations: ${N1}×${N2} = ${total}...`, 'action')
      try {
        const requests: unknown[] = []
        for (let i = 0; i < N1; i++) {
          for (let j = 0; j < N2; j++) {
            let doc = moveNodeMutation(linkageDoc, nodeId1, samples1[i].position)
            doc = moveNodeMutation(doc, nodeId2, points2[j].position)
            doc = syncAllEdgeDistances(doc)
            requests.push({ ...doc, n_steps: simulationSteps })
          }
        }
        const response = await fetch('/api/compute-pylink-trajectories-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ requests })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        const results = data.results as Array<{
          status: string
          trajectories?: Record<string, [number, number][]>
          n_steps?: number
          joint_types?: Record<string, string>
        }>
        if (!Array.isArray(results) || results.length !== total) {
          throw new Error('Invalid batch response')
        }
        const flatSamples: ExploreTrajectorySample[] = []
        for (let idx = 0; idx < results.length; idx++) {
          const i = Math.floor(idx / N2)
          const j = idx % N2
          const r = results[idx]
          const valid = r?.status === 'success' && r?.trajectories != null
          flatSamples.push({
            position: points2[j].position,
            positionFirst: samples1[i].position,
            valid,
            trajectory:
              valid && r.trajectories && r.n_steps != null
                ? { trajectories: r.trajectories, nSteps: r.n_steps, jointTypes: r.joint_types ?? {} }
                : null
          })
        }
        const validCount = flatSamples.filter((s) => s.valid).length
        setExploreTrajectoriesState((prev) => ({
          ...prev,
          exploreSamples: flatSamples,
          exploreCenter: null,
          exploreLoading: false
        }))
        showStatus(`Combinatorial complete: ${validCount} valid of ${total}`, 'success', 2500)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Combinatorial exploration failed'
        showStatus(msg, 'error', 3000)
        setExploreTrajectoriesState(prev => ({ ...prev, exploreSamples: [], exploreLoading: false }))
      }
    },
    [linkageDoc, simulationSteps, showStatus, exploreRadius, exploreRadialSamples, exploreAzimuthalSamples, exploreNMaxCombinatorial]
  )

  const runExploreTrajectoriesSecond = useCallback(
    async (
      fixedNodeId: string,
      pinnedPosition: [number, number],
      nodeId2: string,
      center2: [number, number]
    ) => {
      setExploreTrajectoriesState(prev => ({
        ...prev,
        exploreLoading: true,
        explorePinnedFirstPosition: pinnedPosition,
        exploreSecondNodeId: nodeId2,
        exploreSecondCenter: center2,
        exploreMode: 'path',
        exploreSamples: []
      }))
      showStatus(`Exploring path: ${nodeId2} with ${fixedNodeId} fixed...`, 'action')
      try {
        const points = exploreRegion(center2, {
          deltaDegrees: 360 / exploreAzimuthalSamples,
          R: exploreRadius,
          nRadialSamples: exploreRadialSamples
        })
        const requests = points.map(({ position }) => {
          let doc = moveNodeMutation(linkageDoc, fixedNodeId, pinnedPosition)
          doc = moveNodeMutation(doc, nodeId2, position)
          doc = syncAllEdgeDistances(doc)
          return { ...doc, n_steps: simulationSteps }
        })
        const response = await fetch('/api/compute-pylink-trajectories-batch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ requests })
        })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        const results = data.results as Array<{
          status: string
          trajectories?: Record<string, [number, number][]>
          n_steps?: number
          joint_types?: Record<string, string>
        }>
        if (!Array.isArray(results) || results.length !== points.length) {
          throw new Error('Invalid batch response')
        }
        const exploreSamples: ExploreTrajectorySample[] = points.map((p, i) => {
          const r = results[i]
          const valid = r?.status === 'success' && r?.trajectories != null
          return {
            position: p.position,
            valid,
            trajectory:
              valid && r.trajectories && r.n_steps != null
                ? { trajectories: r.trajectories, nSteps: r.n_steps, jointTypes: r.joint_types ?? {} }
                : null
          }
        })
        const validCount = exploreSamples.filter(s => s.valid).length
        setExploreTrajectoriesState(prev => ({
          ...prev,
          exploreSamples,
          exploreCenter: center2,
          exploreLoading: false
        }))
        showStatus(`Path exploration complete: ${validCount} valid`, 'success', 2500)
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Path exploration failed'
        showStatus(msg, 'error', 3000)
        setExploreTrajectoriesState(prev => ({ ...prev, exploreSamples: [], exploreLoading: false }))
      }
    },
    [linkageDoc, simulationSteps, showStatus, exploreRadius, exploreRadialSamples]
  )

  // Derived selection color
  const selectionColorMap = { blue: '#1976d2', orange: '#FA8112', green: '#2e7d32', purple: '#9c27b0' }
  const selectionColor = selectionColorMap[selectionHighlightColor]

  // Use builder's getHighlightStyle with theme jointColors (4-arg for renderers)
  const getHighlightStyle = useCallback(
    (objectType: 'joint' | 'link' | 'polygon', highlightType: 'none' | 'selected' | 'hovered' | 'move_group' | 'merge', baseColor: string, baseStrokeWidth: number) =>
      getHighlightStyleFromBuilder(objectType, highlightType, baseColor, baseStrokeWidth, jointColors),
    [jointColors]
  )

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

  // Logging helper for mechanism state tracking (no-op in production)
  const logMechanismState = useCallback((_label: string, _doc: LinkageDocument) => {}, [])

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
    isOptimizerPanelOpen: openToolbars.has('optimize'),
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

  /** Ref kept in sync with user-set frame (slider) for setAnimationFrameWithDisplay. */
  const displayFrameRef = useRef<number | null>(null)

  // Dimension fetching is now handled by OptimizationController

  const animation = useAnimationController({
    // Simulation dependencies
    linkageDoc,
    simulationSteps,
    autoSimulateDelayMs,
    mechanismVersion,
    isDragging: dragState.isDragging || moveGroupState.isDragging,
    onSimulationComplete: optimization.handleSimulationComplete,
    showStatus,
    // Animation dependencies
    onFrameChange: handleAnimationFrameChange
  })

  /** Set frame and update displayFrameRef so drag start captures the frame the user is viewing (not N-1 from loop). */
  const setAnimationFrameWithDisplay = useCallback(
    (frame: number) => {
      displayFrameRef.current = frame
      animation.setAnimationFrame(frame)
    },
    [animation.setAnimationFrame]
  )

  // Sync display ref when trajectory first appears so we have an initial frame (e.g. 0) before user scrubs.
  useEffect(() => {
    if (animation.trajectoryData && displayFrameRef.current === null) {
      displayFrameRef.current = animation.animationState?.currentFrame ?? 0
    }
  }, [animation.trajectoryData, animation.animationState?.currentFrame])

  // Single owner for display frame and edit mode (drag start → frame 0, drag end → follow animation).
  const { displayFrame, displayFrameOverrideRef, enterEditMode, exitEditMode } = useDisplayFrame({
    animationCurrentFrame: animation.animationState?.currentFrame ?? 0,
    setAnimationFrame: animation.setAnimationFrame,
    pauseAnimation: animation.pauseAnimation,
    isAnimating: animation.animationState?.isAnimating ?? false
  })
  /** Effective display frame: override ref (0 on drag start) so first paint shows 1/N, not N/N.
   * Requires enterEditMode to be called synchronously (e.g. flushSync) on drag start—see useDisplayFrame. */
  const effectiveDisplayFrame =
    displayFrameOverrideRef.current !== null ? displayFrameOverrideRef.current : displayFrame
  /** Called on drag start (single-node or group) so mechanism and label show 1/N. */
  const resetAnimationToFirstFrame = enterEditMode

  // Single display frame for mechanism and toolbar (effectiveDisplayFrame so drag start paints 0 synchronously)
  const { getJointPosition } = useMechanismPositions({
    linkageDoc,
    pylinkDoc,
    trajectoryData: animation.trajectoryData,
    displayFrame: effectiveDisplayFrame,
    animatedPositions: animation.animatedPositions,
    dragState,
    pendingDropPosition
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
          optimization.clearOptimizerState()
          applyLoadedDocument({
            doc: result.data,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setCanvases: (c) => setCanvases(Array.isArray(c) ? c : []),
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
  }, [optimization.clearOptimizerState, animation.clearTrajectory, triggerMechanismChange, isHypergraphFormat])

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
    if (toolMode === 'explore_node_trajectories') {
      resetExploreTrajectories()
    }
    setToolMode('select')
  }, [linkCreationState.isDrawing, dragState.isDragging, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, mergePolygonState.step, pathDrawState.isDrawing, showStatus, toolMode, resetExploreTrajectories])

  // On drag end: clear display frame override only when neither single-node nor group drag is active.
  // TRAJECTORY HOP FIX: We only call triggerMechanismChange when the user actually moved (actualMove);
  // otherwise a click-without-drag would bump mechanismVersion and trigger auto-sim, causing a hop to N/N.
  const prevSingleNodeDraggingRef = useRef(false)
  /** Bake dragged joint position once per drag so document position = trajectory[X] (stops dragged node jumping to N/N). */
  const dragBakedRef = useRef(false)
  /** Capture last drag positions so we can tell at drag-end if user actually moved (H1). */
  const lastDragStartRef = useRef<[number, number] | null>(null)
  const lastDragCurrentRef = useRef<[number, number] | null>(null)
  /** Capture dragged joint name for drag-end sync (state is reset on mouseup). */
  const lastDraggedJointRef = useRef<string | null>(null)
  const anyDragging = dragState.isDragging || moveGroupState.isDragging
  useEffect(() => {
    if (dragState.isDragging && dragState.dragStartPosition && dragState.currentPosition) {
      lastDragStartRef.current = dragState.dragStartPosition
      lastDragCurrentRef.current = dragState.currentPosition
      if (dragState.draggedJoint) lastDraggedJointRef.current = dragState.draggedJoint
    }
    const wasSingleNodeDragging = prevSingleNodeDraggingRef.current
    prevSingleNodeDraggingRef.current = dragState.isDragging
    if (!anyDragging) {
      dragBakedRef.current = false
      exitEditMode()
      const start = lastDragStartRef.current
      const current = lastDragCurrentRef.current
      const actualMove = start != null && current != null && Math.hypot(current[0] - start[0], current[1] - start[1]) >= DRAG_MOVE_THRESHOLD
      if (wasSingleNodeDragging && actualMove) {
        const trajectories = animation.trajectoryData?.trajectories
        const draggedJoint = lastDraggedJointRef.current
        const dropPosition = current
        const frame = 0 // We always edit at frame 0 (enterEditMode sets it).
        // Show dropped position until new trajectory arrives (avoids jump to old trajectory for one frame).
        if (draggedJoint && dropPosition) {
          setPendingDropPosition({ jointName: draggedJoint, position: [dropPosition[0], dropPosition[1]] })
        }
        // TRAJECTORY HOP FIX: Sync doc to edited frame (1/N) with dropped joint at user position, then run sim
        // with that doc. If we don't, the backend would get an inconsistent doc (only dragged node updated)
        // and return a wrong frame 0, so the mechanism would "hop" to N/N or show cyclic shift (0->1, 1->2).
        if (trajectories && draggedJoint && dropPosition && linkageDoc?.linkage?.nodes) {
          const syncedDoc = buildSyncedDocAfterDrop(
            linkageDoc,
            trajectories,
            draggedJoint,
            [dropPosition[0], dropPosition[1]],
            frame
          )
          setLinkageDoc(syncedDoc)
          // Run sim with synced doc so backend gets consistent 1/N state (avoids hop); do not rely on timer.
          ;(animation.runSimulation as (doc?: unknown) => Promise<void>)(syncedDoc)
        } else {
          triggerMechanismChange()
        }
      }
      lastDragStartRef.current = null
      lastDragCurrentRef.current = null
      lastDraggedJointRef.current = null
    }
  }, [anyDragging, dragState.isDragging, dragState.dragStartPosition, dragState.currentPosition, dragState.draggedJoint, triggerMechanismChange, exitEditMode, animation.trajectoryData, animation.runSimulation, linkageDoc, setLinkageDoc, setPendingDropPosition])

  // Clear pending drop position when trajectory updates (e.g. after runSimulation(syncedDoc) completes).
  useEffect(() => {
    if (pendingDropPosition) setPendingDropPosition(null)
  }, [animation.trajectoryData])

  // TRAJECTORY HOP FIX: Only bake when the user has actually moved (not on click-only). If we baked on every
  // mousedown, linkageDoc would change and the auto-sim effect would run after the delay, causing a hop.
  // When we do bake, use currentPosition so we don't overwrite moveJoint. See also: triggerMechanismChange
  // only when actualMove, and drag-end sync + runSimulation(syncedDoc).
  useLayoutEffect(() => {
    if (
      !dragState.isDragging ||
      !dragState.draggedJoint ||
      !dragState.dragStartPosition ||
      !dragState.currentPosition ||
      dragBakedRef.current
    ) {
      return
    }
    const dist = Math.hypot(
      dragState.currentPosition[0] - dragState.dragStartPosition[0],
      dragState.currentPosition[1] - dragState.dragStartPosition[1]
    )
    if (dist < DRAG_MOVE_THRESHOLD) {
      return
    }
    dragBakedRef.current = true
    const jointName = dragState.draggedJoint
    const pos = dragState.currentPosition
    setLinkageDoc((prev) => {
      const linkage = prev?.linkage
      if (!linkage?.nodes?.[jointName]) return prev
      const nodes = { ...linkage.nodes }
      nodes[jointName] = { ...nodes[jointName], position: [pos[0], pos[1]] }
      return { ...prev, linkage: { ...linkage, nodes } }
    })
  }, [dragState.isDragging, dragState.draggedJoint, dragState.dragStartPosition, dragState.currentPosition, setLinkageDoc])

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
      showPath: meta?.show_path ?? false,
      metaValue: linkageDoc.meta.nodes[jointName]?.metaValue
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
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints, pylinkDoc.meta.links, linkageDoc.meta.nodes, getJointPosition])

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
      zlevel: linkMeta.zlevel,
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

    setDrawnObjects(prev => {
      const { objects, removedIds } = removeDrawnObjectsReferencingDeletedEdges(
        prev.objects,
        new Set(result.deletedEdges)
      )
      const removed = new Set(removedIds)
      return {
        ...prev,
        objects: objects as DrawnObject[],
        selectedIds: prev.selectedIds.filter(id => !removed.has(id))
      }
    })

    // Clear optimized mechanism flag - structure changed (deleted link)
    clearOptimizedMechanismFlag()

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, clearOptimizedMechanismFlag, setDrawnObjects])

  // Delete a joint (node) and all connected edges, plus any resulting orphans
  const deleteJoint = useCallback((jointName: string) => {
    // Use new hypergraph operation
    const result = deleteNodeOp(linkageDoc, jointName)

    // Clear trajectory and trigger auto-simulation
    animation.clearTrajectory()
    triggerMechanismChange()

    // Update state directly with hypergraph format
    setLinkageDoc(result.doc)

    setDrawnObjects(prev => {
      const { objects, removedIds } = removeDrawnObjectsReferencingDeletedEdges(
        prev.objects,
        new Set(result.deletedEdges)
      )
      const removed = new Set(removedIds)
      return {
        ...prev,
        objects: objects as DrawnObject[],
        selectedIds: prev.selectedIds.filter(id => !removed.has(id))
      }
    })

    // Clear optimized mechanism flag - structure changed (deleted node)
    clearOptimizedMechanismFlag()

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, clearOptimizedMechanismFlag, setDrawnObjects])

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

    // Do not clear trajectory or trigger simulation during drag: keep trajectory and frame (X/N).
    // Clear and trigger only when not dragging; on drag end we trigger via effect below.
    if (!dragState.isDragging) {
      animation.clearTrajectory()
      triggerMechanismChange()
    }

    setLinkageDoc(newDoc)

    // Mark as unsynced if manually modified (but keep isOptimizedMechanism true)
    // setIsSyncedToOptimizer will be updated by the useEffect
  }, [linkageDoc, showStatus, optimization.isOptimizedMechanism, dragState.isDragging, animation.clearTrajectory, triggerMechanismChange])

  /** Move two joints in one document update (used when applying combinatorial explore result). */
  const moveTwoJoints = useCallback(
    (
      jointName1: string,
      position1: [number, number],
      jointName2: string,
      position2: [number, number]
    ) => {
      if (optimization.optimizationJustCompletedRef.current) {
        const timeSinceOpt = Date.now() - (optimization.optimizationCompleteTimeRef.current || 0)
        if (timeSinceOpt < 2000) {
          showStatus('CRITICAL: Blocked mechanism modification immediately after optimization', 'error', 5000)
          return
        }
      }
      const node1 = getNode(linkageDoc, jointName1)
      const node2 = getNode(linkageDoc, jointName2)
      if (!node1 || !node2) return
      let newDoc = moveNodeMutation(linkageDoc, jointName1, position1)
      newDoc = moveNodeMutation(newDoc, jointName2, position2)
      if (!optimization.isOptimizedMechanism) {
        newDoc = syncAllEdgeDistances(newDoc)
      }
      animation.clearTrajectory()
      triggerMechanismChange()
      setLinkageDoc(newDoc)
    },
    [linkageDoc, showStatus, optimization.isOptimizedMechanism]
  )

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

  const drawnObjectsRef = useRef(drawnObjects.objects)
  drawnObjectsRef.current = drawnObjects.objects

  const apiValidatePolygonRigidity = useCallback(async () => {
    const trajectoryData = animation.trajectoryData
    if (!trajectoryData?.trajectories) return { status: 'success' as const, polygons: {} }
    const objects = drawnObjectsRef.current
    const polygons = objects.filter(
      (o: { type?: string; contained_links?: string[] }) =>
        o.type === 'polygon' && (o.contained_links?.length ?? 0) > 0
    )
    if (polygons.length === 0) return { status: 'success' as const, polygons: {} }
    const drawn_objects = polygons.map((obj: { id: string; type?: string; contained_links?: string[] }) => ({
      id: obj.id,
      type: 'polygon',
      contained_links: obj.contained_links ?? []
    }))
    const body = {
      trajectories: trajectoryData.trajectories,
      linkage: linkageDoc.linkage,
      drawn_objects
    }
    const res = await fetch('/api/validate-polygon-rigidity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    const data = await res.json()
    if (data.status === 'success' && data.polygons) {
      setDrawnObjects(prev => ({
        ...prev,
        objects: prev.objects.map(obj => {
          const rigid = data.polygons[obj.id]
          if (rigid == null) return obj
          const prevValid = (obj as { contained_links_valid?: boolean }).contained_links_valid
          const rigid_valid = rigid.rigid_valid !== false
          const contained_links_valid = prevValid !== false && rigid_valid
          return { ...obj, contained_links_valid }
        })
      }))
    }
    return data
  }, [animation.trajectoryData, linkageDoc.linkage, setDrawnObjects])

  /** Build a copy of linkage with each node position replaced by current position (for merge/find-associated to match current view). */
  const buildLinkageWithCurrentPositions = useCallback(
    (linkage: { nodes?: Record<string, { position?: number[] }>; edges?: unknown }, getPos: (id: string) => [number, number] | null) => {
      const nodes = linkage?.nodes ?? {}
      const nextNodes: Record<string, { position?: number[] }> = {}
      for (const [id, node] of Object.entries(nodes)) {
        const pos = getPos(id)
        nextNodes[id] = { ...node, position: pos ?? node?.position ?? [0, 0] }
      }
      return { ...linkage, nodes: nextNodes }
    },
    []
  )

  const apiFindAssociatedPolygons = useCallback(async () => {
    const polygons = drawnObjects.objects.filter(
      (o: { type?: string; points?: unknown[] }) => o.type === 'polygon' && o.points && o.points.length >= 3
    )
    const edges = linkageDoc.meta?.edges ?? {}
    const drawn_objects = polygons.map(obj => {
      let points: [number, number][] = obj.points as [number, number][]
      const mergedLinkName = (obj as { mergedLinkName?: string }).mergedLinkName
      const mergedLinkOriginalStart = (obj as { mergedLinkOriginalStart?: [number, number] }).mergedLinkOriginalStart
      const mergedLinkOriginalEnd = (obj as { mergedLinkOriginalEnd?: [number, number] }).mergedLinkOriginalEnd
      if (mergedLinkName && mergedLinkOriginalStart && mergedLinkOriginalEnd) {
        const linkMeta = edges[mergedLinkName] as { connects?: [string, string] } | undefined
        if (linkMeta?.connects) {
          const currentStart = getJointPosition(linkMeta.connects[0])
          const currentEnd = getJointPosition(linkMeta.connects[1])
          if (currentStart && currentEnd) {
            points = transformPolygonPoints(
              obj.points as [number, number][],
              mergedLinkOriginalStart,
              mergedLinkOriginalEnd,
              currentStart,
              currentEnd
            )
          }
        }
      }
      return { ...obj, points }
    })
    const linkageForRequest = buildLinkageWithCurrentPositions(linkageDoc.linkage, getJointPosition)
    const body = {
      pylink_data: { linkage: linkageForRequest, meta: linkageDoc.meta },
      drawn_objects
    }
    const res = await fetch('/api/find-associated-polygons', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    })
    const data = await res.json()
    if (data.status === 'success' && data.polygons) {
      const hasContainedLinks = Object.values(data.polygons as Record<string, { contained_links?: string[] }>).some(
        r => (r.contained_links?.length ?? 0) > 0
      )
      let rigidResult: Record<string, { rigid_valid?: boolean }> = {}
      if (animation.trajectoryData?.trajectories && hasContainedLinks) {
        const validateRes = await fetch('/api/validate-polygon-rigidity', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            trajectories: animation.trajectoryData.trajectories,
            linkage: linkageDoc.linkage,
            drawn_objects: polygons.map((obj: { id: string; type?: string }) => ({
              id: obj.id,
              type: 'polygon',
              contained_links: data.polygons[obj.id]?.contained_links ?? (obj as { contained_links?: string[] }).contained_links ?? []
            }))
          })
        })
        const validateData = await validateRes.json()
        if (validateData.status === 'success' && validateData.polygons) rigidResult = validateData.polygons
      }
      setDrawnObjects(prev => ({
        ...prev,
        objects: prev.objects.map(obj => {
          const r = data.polygons[obj.id]
          if (r == null) return obj
          const all_inside = r.all_inside
          const rigid = rigidResult[obj.id]
          const rigid_valid = rigid == null ? true : rigid.rigid_valid !== false
          return { ...obj, contained_links_valid: all_inside && rigid_valid }
        })
      }))
    }
    return data
  }, [linkageDoc.linkage, linkageDoc.meta, linkageDoc.meta?.edges, drawnObjects.objects, setDrawnObjects, getJointPosition, transformPolygonPoints, animation.trajectoryData, buildLinkageWithCurrentPositions])

  // After simulation, validate polygon rigidity once (only if polygons with contained_links exist).
  // Ref and stable callback avoid depending on drawnObjects.objects so updating it doesn't retrigger and loop.
  useEffect(() => {
    if (!animation.trajectoryData?.trajectories) return
    const objects = drawnObjectsRef.current
    const hasPolygonsWithLinks = objects.some(
      (o: { type?: string; contained_links?: string[] }) =>
        o.type === 'polygon' && (o.contained_links?.length ?? 0) > 0
    )
    if (!hasPolygonsWithLinks) return
    apiValidatePolygonRigidity()
  }, [animation.trajectoryData, apiValidatePolygonRigidity])

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
    triggerMechanismChange,
    resetAnimationToFirstFrame,
    apiFindAssociatedPolygons
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

    if (result.deletedEdges.length > 0) {
      setDrawnObjects(prev => {
        const { objects, removedIds } = removeDrawnObjectsReferencingDeletedEdges(
          prev.objects,
          new Set(result.deletedEdges)
        )
        const removed = new Set(removedIds)
        return {
          ...prev,
          objects: objects as DrawnObject[],
          selectedIds: prev.selectedIds.filter(id => !removed.has(id))
        }
      })
    }

    // Clear optimized mechanism flag - structure changed (merged nodes)
    clearOptimizedMechanismFlag()

    showStatus(result.message, 'success', 2500)
  }, [linkageDoc, showStatus, triggerMechanismChange, clearOptimizedMechanismFlag, setDrawnObjects])
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
    const allDeletedEdgeIds = new Set<string>()

    // First delete edges (links) - this handles orphan detection
    if (linksToDelete.length > 0) {
      const edgeResult = deleteEdgesOp(doc, linksToDelete)
      doc = edgeResult.doc
      totalDeletedEdges = edgeResult.deletedEdges.length
      for (const id of edgeResult.deletedEdges) {
        allDeletedEdgeIds.add(id)
      }
      // Orphaned nodes are counted separately
      totalDeletedNodes += edgeResult.orphanedNodes.length
    }

    // Then delete nodes (joints) - this also handles connected edges and orphans
    if (jointsToDelete.length > 0) {
      const nodeResult = deleteNodesOp(doc, jointsToDelete)
      doc = nodeResult.doc
      totalDeletedNodes = nodeResult.deletedNodes.length
      totalDeletedEdges += nodeResult.deletedEdges.length
      for (const id of nodeResult.deletedEdges) {
        allDeletedEdgeIds.add(id)
      }
    }

    // Apply state update
    setLinkageDoc(doc)

    let deletedDrawnObjects = 0
    setDrawnObjects(prev => {
      const { objects: afterPrune, removedIds } = removeDrawnObjectsReferencingDeletedEdges(
        prev.objects,
        allDeletedEdgeIds
      )
      const explicit = new Set(drawnObjectsToDelete)
      const nextObjects = afterPrune.filter(
        o => !explicit.has((o as DrawnObject).id)
      ) as DrawnObject[]
      deletedDrawnObjects = prev.objects.length - nextObjects.length
      const removedForSelection = new Set([...removedIds, ...drawnObjectsToDelete])
      return {
        ...prev,
        objects: nextObjects,
        selectedIds: prev.selectedIds.filter(id => !removedForSelection.has(id))
      }
    })

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
      deletedDrawnObjects
    }
  }, [linkageDoc, moveGroupState.isActive, triggerMechanismChange, setDrawnObjects])

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

  // Tool context for per-tool handlers (select, etc.) — built by useToolContext
  const toolContext = useToolContext({
    buildLinkageWithCurrentPositions,
    canvasRef,
    pixelsToUnits,
    viewport: viewport.viewport,
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
    moveTwoJoints,
    mergeJoints,
    resetAnimationToFirstFrame,
    pauseAnimation: animation.pauseAnimation,
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
    setSelectedPathId,
    linkageDoc,
    setLinkageDoc,
    apiFindAssociatedPolygons,
    exploreTrajectoriesState,
    setExploreTrajectoriesState,
    runExploreTrajectories,
    runExploreTrajectoriesSecond,
    runExploreTrajectoriesCombinatorial,
    getNodeShowPath,
    getJointType: (name: string) => pylinkDoc.pylinkage.joints.find(j => j.name === name)?.type ?? null
  })

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
    jointOutline,
    linkThickness,
    linkTransparency,
    linkColorMode,
    linkColorSingle,
    trajectoryDotSize,
    trajectoryDotOutline,
    trajectoryDotOpacity,
    showTrajectoryStepNumbers,
    trajectoryStyle,
    trajectoryColorCycle,
    showJointLabels,
    showLinkLabels,
    jointMergeRadius,
    mergeThreshold: MERGE_THRESHOLD,
    targetPaths,
    selectedPathId,
    drawnObjects,
    canvases,
    setCanvases,
    openCanvasEdit: (id: string) => setEditingCanvasId(id),
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
    transformPolygonPoints,
    exploreTrajectoriesState,
    exploreColormapEnabled,
    exploreColormapType,
    exploreRadius
  })

  // Canvas event handlers (mouse, click, double-click) — built by useCanvasEventHandlers
  const {
    handleCanvasMouseDown,
    handleCanvasMouseMove,
    handleCanvasMouseUp,
    handleCanvasMouseLeave,
    handleCanvasClick,
    handleCanvasDoubleClick
  } = useCanvasEventHandlers({
    canvasRef,
    viewport,
    screenToUnit,
    toolMode,
    toolContext,
    moveGroup,
    enterEditMode,
    animation,
    showStatus,
    dragState,
    groupSelectionState,
    getJointsWithPositions,
    getLinksWithPositions,
    findNearestJoint,
    findNearestLink,
    drawnObjects,
    setDrawnObjects,
    setSelectedJoints,
    setSelectedLinks,
    enterMoveGroupMode,
    exitMoveGroupMode,
    pylinkDoc
  })

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
      // Let toolbar content (Optimize, etc.) handle selection and copy; don't trigger tool shortcuts
      if (event.target instanceof Node && (event.target as HTMLElement).closest?.('[data-draggable-toolbar-content]')) {
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

      // Skip tool shortcuts when modifier held so Cmd+C / Ctrl+V etc. work in logs and about
      if (event.metaKey || event.ctrlKey) return

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
        // Clear explore trajectories when switching away
        if (toolMode === 'explore_node_trajectories' && tool.id !== 'explore_node_trajectories') {
          resetExploreTrajectories()
        }
        setToolMode(tool.id)

        // Show appropriate message for merge mode
        if (tool.id === 'merge') {
          setMergePolygonState({ step: 'awaiting_selection', selectedPolygonId: null, selectedLinkName: null })
          showStatus('Select a link or a polygon to begin merge', 'action')
        } else if (tool.id === 'draw_path') {
          showStatus('Click to start drawing target path', 'action')
        } else if (tool.id === 'explore_node_trajectories') {
          showStatus('Click a node to explore trajectories', 'action')
        } else {
          showStatus(`${tool.label} mode`, 'info', 1500)
        }
        event.preventDefault()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [toolMode, moveGroupState.isActive, exitMoveGroupMode, linkCreationState.isDrawing, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, pathDrawState.isDrawing, cancelAction, showStatus, selectedJoints, selectedLinks, handleDeleteSelected, animation.animationState.isAnimating, animation.playAnimation, animation.pauseAnimation, animation.trajectoryData, animation.runSimulation, pylinkDoc.pylinkage.joints, completePathDrawing, resetExploreTrajectories])

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
        // Tools should NEVER scroll - 20px narrower so it doesn't crowd the canvas
        return { minWidth: 200, maxHeight: 600 }
      case 'more':
        return { minWidth: 180, maxHeight: 500 }  // Tall enough to fit all tools without scrolling
      case 'optimize':
        return { minWidth: 960, maxHeight: 650 }  // 3x wider (320*3), shorter height for horizontal layout
      case 'links':
        return { minWidth: 200, maxHeight: 480 }  // 1.5x taller for links list
      case 'nodes':
        return { minWidth: 200, maxHeight: 320 }  // Taller for nodes list
      case 'forms':
        return { minWidth: 255, maxHeight: 580 }
      case 'settings':
        return { minWidth: 280, maxHeight: 900 }  // 40px narrower; x=-280 keeps it on-screen
      default:
        return { minWidth: 200, maxHeight: 400 }
    }
  }

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

    setDrawnObjects(prev => ({
      ...prev,
      objects: remapEdgeReferencesInDrawnObjects(prev.objects, oldName, newName) as DrawnObject[]
    }))

    // Update the modal data with new name if modal is open
    if (editingLinkData && editingLinkData.name === oldName) {
      setEditingLinkData(prev => prev ? { ...prev, name: newName } : null)
    }
    showStatus(`Renamed to ${newName}`, 'success', 1500)
  }, [linkageDoc, showStatus, editingLinkData, setDrawnObjects])

  // Z-level palette: distinct colors for each layer (same z => same color)
  const Z_LEVEL_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78'
  ]

  const zLevelRows = React.useMemo((): ZLevelRow[] => {
    const edges = linkageDoc.meta?.edges ?? {}
    const objects = drawnObjects.objects as DrawnObject[]
    const zToColor = new Map<number, string>()
    const isNumericZ = (z: unknown): z is number =>
      typeof z === 'number' && Number.isFinite(z)
    for (const meta of Object.values(edges)) {
      const z = (meta as { zlevel?: unknown }).zlevel
      if (!isNumericZ(z) || zToColor.has(z)) continue
      const color = (meta as { color?: string }).color
      zToColor.set(z, typeof color === 'string' ? color : Z_LEVEL_PALETTE[zToColor.size % Z_LEVEL_PALETTE.length])
    }
    for (const o of objects) {
      if (o.type !== 'polygon') continue
      const z = o.z_level
      if (!isNumericZ(z) || zToColor.has(z)) continue
      const color = o.fillColor
      zToColor.set(z, typeof color === 'string' ? color : Z_LEVEL_PALETTE[zToColor.size % Z_LEVEL_PALETTE.length])
    }
    const uniqueZ = [...zToColor.keys()].sort((a, b) => a - b)
    return uniqueZ.map(z => ({ z, color: zToColor.get(z) ?? '#888888' }))
  }, [linkageDoc.meta?.edges, drawnObjects.objects])

  const getColorForZLevel = useCallback((z: number): string => {
    const row = zLevelRows.find(r => r.z === z)
    if (row) return row.color
    const idx = zLevelRows.length ? zLevelRows.length : Math.abs(z) % Z_LEVEL_PALETTE.length
    return Z_LEVEL_PALETTE[idx % Z_LEVEL_PALETTE.length]
  }, [zLevelRows])

  const onZLevelColorChange = useCallback((z: number, color: string) => {
    setLinkageDoc(prev => {
      const nextEdges = { ...prev.meta.edges }
      for (const [linkId, meta] of Object.entries(nextEdges)) {
        if ((meta as { zlevel?: number }).zlevel === z) {
          nextEdges[linkId] = { ...meta, color }
        }
      }
      return { ...prev, meta: { ...prev.meta, edges: nextEdges } }
    })
    setDrawnObjects(prev => ({
      ...prev,
      objects: prev.objects.map(obj => {
        const o = obj as DrawnObject
        if (o.type === 'polygon' && o.z_level === z) {
          return { ...o, fillColor: color, strokeColor: color }
        }
        return obj
      })
    }))
    setLinkColorMode('z-level')
  }, [setLinkageDoc, setDrawnObjects, setLinkColorMode])

  const onReorderForms = useCallback(async (formId: string, newZLevel: number) => {
    const fixed_entity_z_levels: Record<string, number> = {}
    for (const obj of drawnObjects.objects as DrawnObject[]) {
      if (obj.type !== 'polygon' || obj.z_level == null) continue
      const isDragged = obj.id === formId
      if (isDragged || obj.z_level_fixed) {
        fixed_entity_z_levels['polygon:' + obj.id] = isDragged ? newZLevel : obj.z_level
      }
    }
    setDrawnObjects(prev => ({
      ...prev,
      objects: prev.objects.map(obj => {
        if (obj.id !== formId) return obj
        const o = obj as DrawnObject
        return { ...o, z_level: newZLevel, z_level_fixed: true }
      })
    }))
    try {
      const drawn_objects = drawnObjects.objects
        .filter((o: { type?: string; id?: string; contained_links?: string[] }) =>
          o.type === 'polygon' && o.id && (o.contained_links?.length ?? 0) > 0
        )
        .map((o: DrawnObject) => ({
          id: o.id,
          type: 'polygon' as const,
          contained_links: o.contained_links ?? [],
          ...(Array.isArray(o.points) && o.points.length >= 3 && { points: o.points })
        }))
      const formSoftPinsReorder: Record<string, [number, number]> = {}
      for (const o of drawnObjects.objects as DrawnObject[]) {
        if (o.type === 'polygon' && o.target_z_level != null && Number.isFinite(o.target_z_level)) {
          formSoftPinsReorder['polygon:' + o.id] = [o.target_z_level, 1]
        }
      }
      const body: Record<string, unknown> = {
        linkage: linkageDoc.linkage,
        meta: linkageDoc.meta,
        n_steps: simulationSteps || 32,
        ...(drawn_objects.length > 0 && { drawn_objects }),
        ...(drawn_objects.length > 0 && { margin_units: formPaddingUnits }),
        fixed_entity_z_levels,
        z_level_config: {
          ...zLevelConfig,
          crank_z: zLevelConfig.crank_z ?? null,
          soft_pins: { ...(zLevelConfig.soft_pins ?? {}), ...formSoftPinsReorder }
        }
      }
      if (animation.trajectoryData?.trajectories && animation.trajectoryData.nSteps > 0) {
        body.trajectories = animation.trajectoryData.trajectories
        body.n_steps = animation.trajectoryData.nSteps
      }
      const res = await fetch('/api/compute-link-z-levels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      const data = (await res.json()) as {
        status: string
        assignments?: Array<Record<string, number>>
        polygon_z_levels?: Record<string, number>
        message?: string
      }
      if (data.status === 'error' || !res.ok) {
        showStatus(data.message ?? 'Z-level recompute failed', 'error', 5000)
        return
      }
      if (!Array.isArray(data.assignments) || data.assignments.length === 0) {
        showStatus(data.message ?? 'No z-level assignment returned', 'error', 3000)
        return
      }
      const assignment = data.assignments[0]
      const uniqueZ = [...new Set(Object.values(assignment))].sort((a, b) => a - b)
      const zToColor = new Map<number, string>()
      uniqueZ.forEach((z, i) => {
        zToColor.set(z, Z_LEVEL_PALETTE[i % Z_LEVEL_PALETTE.length])
      })
      setLinkageDoc(prev => {
        const nextEdges = { ...prev.meta.edges }
        for (const [linkId, z] of Object.entries(assignment)) {
          const existing = nextEdges[linkId] ?? {}
          nextEdges[linkId] = {
            ...existing,
            zlevel: z,
            color: zToColor.get(z) ?? (existing as { color?: string }).color ?? '#888888'
          }
        }
        return { ...prev, meta: { ...prev.meta, edges: nextEdges } }
      })
      setDrawnObjects(prev => ({
        ...prev,
        objects: prev.objects.map((obj): DrawnObject => {
          const o = obj as DrawnObject
          if (o.type !== 'polygon' || !data.polygon_z_levels || !(o.id in data.polygon_z_levels)) return o
          const z = data.polygon_z_levels[o.id]
          // Preserve custom fill when z-level unchanged so form color edits are not reverted
          const color = z === o.z_level ? o.fillColor : (zToColor.get(z) ?? o.fillColor)
          return {
            ...o,
            z_level: z,
            fillColor: color,
            strokeColor: color,
            fillOpacity: 0.25
          }
        })
      }))
      setLinkColorMode('z-level')
      animation.setAnimationFrame(0)
      showStatus('Z-levels updated', 'success', 3000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Z-level recompute failed'
      showStatus(msg, 'error', 5000)
    }
  }, [drawnObjects.objects, linkageDoc, simulationSteps, formPaddingUnits, zLevelConfig, animation.trajectoryData, animation.setAnimationFrame, setLinkageDoc, setDrawnObjects, setLinkColorMode, showStatus])

  const onFormZLevelFixedChange = useCallback((formId: string, fixed: boolean) => {
    setDrawnObjects(prev => ({
      ...prev,
      objects: prev.objects.map(obj =>
        obj.id !== formId ? obj : { ...obj, z_level_fixed: fixed }
      )
    }))
  }, [setDrawnObjects])

  const handleComputeLinkZLevels = useCallback(async () => {
    try {
      const drawn_objects = drawnObjects.objects
        .filter((o: { type?: string; id?: string; contained_links?: string[] }) =>
          o.type === 'polygon' && o.id && (o.contained_links?.length ?? 0) > 0
        )
        .map((o: DrawnObject) => ({
          id: o.id,
          type: 'polygon' as const,
          contained_links: o.contained_links ?? [],
          ...(Array.isArray(o.points) && o.points.length >= 3 && { points: o.points })
        }))
      const fixedPolygons = (drawnObjects.objects as DrawnObject[]).filter(
        (o): o is DrawnObject => o.type === 'polygon' && !!o.z_level_fixed && o.z_level != null
      )
      const fixed_entity_z_levels: Record<string, number> = {}
      for (const p of fixedPolygons) {
        fixed_entity_z_levels['polygon:' + p.id] = p.z_level!
      }
      const formSoftPins: Record<string, [number, number]> = {}
      for (const o of drawnObjects.objects as DrawnObject[]) {
        if (o.type === 'polygon' && o.target_z_level != null && Number.isFinite(o.target_z_level)) {
          formSoftPins['polygon:' + o.id] = [o.target_z_level, 1]
        }
      }
      const body: Record<string, unknown> = {
        linkage: linkageDoc.linkage,
        meta: linkageDoc.meta,
        n_steps: simulationSteps || 32,
        ...(drawn_objects.length > 0 && { drawn_objects }),
        ...(drawn_objects.length > 0 && { margin_units: formPaddingUnits }),
        ...(Object.keys(fixed_entity_z_levels).length > 0 && { fixed_entity_z_levels }),
        z_level_config: {
          ...zLevelConfig,
          crank_z: zLevelConfig.crank_z ?? null,
          soft_pins: { ...(zLevelConfig.soft_pins ?? {}), ...formSoftPins }
        }
      }
      if (animation.trajectoryData?.trajectories && animation.trajectoryData.nSteps > 0) {
        body.trajectories = animation.trajectoryData.trajectories
        body.n_steps = animation.trajectoryData.nSteps
      }
      const res = await fetch('/api/compute-link-z-levels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = (await res.json()) as {
        status: string
        assignments?: Array<Record<string, number>>
        polygon_z_levels?: Record<string, number>
        message?: string
      }
      if (data.status === 'error') {
        showStatus(data.message ?? 'Z-level assignment failed', 'error', 5000)
        return
      }
      if (data.status !== 'success' || !Array.isArray(data.assignments) || data.assignments.length === 0) {
        showStatus(data.message ?? 'No z-level assignment returned', 'error', 3000)
        return
      }
      const assignment = data.assignments[0]
      const uniqueZ = [...new Set(Object.values(assignment))].sort((a, b) => a - b)
      const zToColor = new Map<number, string>()
      uniqueZ.forEach((z, i) => {
        zToColor.set(z, Z_LEVEL_PALETTE[i % Z_LEVEL_PALETTE.length])
      })
      setLinkageDoc(prev => {
        const nextEdges = { ...prev.meta.edges }
        for (const [linkId, z] of Object.entries(assignment)) {
          const existing = nextEdges[linkId] ?? {}
          nextEdges[linkId] = {
            ...existing,
            zlevel: z,
            color: zToColor.get(z) ?? (existing as { color?: string }).color ?? '#888888'
          }
        }
        return { ...prev, meta: { ...prev.meta, edges: nextEdges } }
      })
      if (data.polygon_z_levels && Object.keys(data.polygon_z_levels).length > 0) {
        const fillOpacityZ = 0.25
        setDrawnObjects(prev => ({
          ...prev,
          objects: prev.objects.map((obj): DrawnObject => {
            const o = obj as DrawnObject
            if (data.polygon_z_levels && o.id in data.polygon_z_levels) {
              const z = data.polygon_z_levels[o.id]
              // Preserve custom fill when z-level unchanged so form color edits are not reverted
              const color = z === o.z_level ? o.fillColor : (zToColor.get(z) ?? o.fillColor)
              return {
                ...o,
                z_level: z,
                fillColor: color,
                strokeColor: color,
                fillOpacity: fillOpacityZ
              }
            }
            return o
          })
        }))
      }
      setLinkColorMode('z-level')
      animation.setAnimationFrame(0)
      showStatus('Z-levels computed; links colored by layer', 'success', 3000)
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Z-level computation failed'
      showStatus(msg, 'error', 3000)
    }
  }, [linkageDoc, simulationSteps, drawnObjects.objects, setLinkageDoc, setDrawnObjects, setLinkColorMode, showStatus, animation, formPaddingUnits])

  useEffect(() => {
    if (!runComputeAfterFormSaveRef.current) return
    runComputeAfterFormSaveRef.current = false
    void handleComputeLinkZLevels()
  }, [drawnObjects.objects, handleComputeLinkZLevels])

  type SuggestedPolygon = { polygon_id: string; points: [number, number][]; contained_links: string[]; z_level: number }

  const handleCreateForms = useCallback(async () => {
    if (!createRigidForms && !createLinkForms) {
      showStatus('Turn on "Create rigid forms" and/or "Create link forms"', 'warning', 3000)
      return
    }
    try {
      // Flow: 1) Create and merge rigid forms  2) Create and merge link forms (each with only its link)
      //       3) Z-level calculation on full set (when computeZLevelsAfterCreate) so layers are valid
      const nodes = linkageDoc.linkage?.nodes ?? {}
      const getStep0Position = (nodeId: string): [number, number] | undefined => {
        const pos = nodes[nodeId]?.position
        if (Array.isArray(pos) && pos.length >= 2) return [Number(pos[0]), Number(pos[1])]
        return undefined
      }
      animation.setAnimationFrame(0)

      let suggested: SuggestedPolygon[] = []
      let assignments: Record<string, number> = {}
      let polygon_z_levels: Record<string, number> = {}

      const buildBody = (): Record<string, unknown> => {
        const body: Record<string, unknown> = {
          linkage: linkageDoc.linkage,
          meta: linkageDoc.meta,
          n_steps: simulationSteps || 32,
          margin_units: formPaddingUnits,
          z_level_config: { ...zLevelConfig, crank_z: zLevelConfig.crank_z ?? null },
          skip_existing_forms: true,
          existing_drawn_objects: drawnObjects.objects
            .filter((o: { type?: string; id?: string; contained_links?: string[] }) =>
              o.type === 'polygon' && o.id && (o.contained_links?.length ?? 0) > 0
            )
            .map((o: { id: string; type?: string; contained_links?: string[] }) => ({
              id: o.id,
              type: 'polygon',
              contained_links: o.contained_links ?? []
            }))
        }
        if (animation.trajectoryData?.trajectories && animation.trajectoryData.nSteps > 0) {
          body.trajectories = animation.trajectoryData.trajectories
          body.n_steps = animation.trajectoryData.nSteps
        }
        return body
      }

      const fetchAndParse = async (mode: 'rigid_groups' | 'per_link') => {
        const res = await fetch('/api/create-polygons-from-rigid-groups', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...buildBody(), mode })
        })
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return (await res.json()) as {
          status: string
          assignments?: Record<string, number>
          polygon_z_levels?: Record<string, number>
          suggested_polygons?: SuggestedPolygon[]
          message?: string
        }
      }

      if (createRigidForms) {
        const data = await fetchAndParse('rigid_groups')
        if (data.status !== 'success') {
          showStatus(data.message ?? 'Failed to create rigid forms', 'error', 3000)
          return
        }
        const rigidSuggested = data.suggested_polygons ?? []
        suggested = rigidSuggested
        assignments = data.assignments ?? {}
        polygon_z_levels = data.polygon_z_levels ?? {}
      }

      if (createLinkForms) {
        const data = await fetchAndParse('per_link')
        if (data.status !== 'success') {
          showStatus(data.message ?? 'Failed to create link forms', 'error', 3000)
          return
        }
        const linkSuggested = data.suggested_polygons ?? []
        if (createRigidForms && suggested.length > 0) {
          const linksInRigid = new Set(suggested.flatMap(p => p.contained_links))
          const extra = linkSuggested.filter(p => p.contained_links[0] && !linksInRigid.has(p.contained_links[0]))
          suggested = [...suggested, ...extra]
          polygon_z_levels = { ...polygon_z_levels, ...(data.polygon_z_levels ?? {}) }
          assignments = { ...assignments, ...(data.assignments ?? {}) }
        } else {
          suggested = linkSuggested
          assignments = data.assignments ?? {}
          polygon_z_levels = data.polygon_z_levels ?? {}
        }
      }

      if (suggested.length === 0) {
        showStatus(createRigidForms ? 'No rigid groups detected' : 'No links', 'info', 3000)
        return
      }

      animation.setAnimationFrame(0)
      const uniqueZ = [...new Set(Object.values(assignments))].sort((a, b) => a - b)
      const zToColor = new Map<number, string>()
      uniqueZ.forEach((z, i) => {
        zToColor.set(z, Z_LEVEL_PALETTE[i % Z_LEVEL_PALETTE.length])
      })
      setLinkageDoc(prev => {
        const nextEdges = { ...prev.meta.edges }
        for (const [linkId, z] of Object.entries(assignments)) {
          const existing = nextEdges[linkId] ?? {}
          nextEdges[linkId] = {
            ...existing,
            zlevel: z,
            color: zToColor.get(z) ?? (existing as { color?: string }).color ?? '#888888'
          }
        }
        return { ...prev, meta: { ...prev.meta, edges: nextEdges } }
      })
      const fillOpacityZ = 0.25
      const edges = linkageDoc.meta?.edges ?? {}
      const traj = animation.trajectoryData?.trajectories
      const newObjects: DrawnObject[] = suggested.map(sp => {
        const color = zToColor.get(sp.z_level) ?? '#888888'
        const primaryLink = sp.contained_links[0]
        const isLinkForm = sp.contained_links.length === 1
        const linkFormSuffix = primaryLink?.replace(/^link_/, '') ?? ''
        const formId = isLinkForm ? `link_form_${linkFormSuffix}` : sp.polygon_id
        const formName = isLinkForm ? formId : sp.polygon_id.replace('rigid_group_', 'rigid_form_')
        const linkMeta = primaryLink ? (edges[primaryLink] as { connects?: [string, string] } | undefined) : undefined
        const connects = linkMeta?.connects
        const mergedLinkOriginalStart =
          connects && traj?.[connects[0]]?.[0]
            ? (traj[connects[0]][0] as [number, number])
            : connects ? getStep0Position(connects[0]) : undefined
        const mergedLinkOriginalEnd =
          connects && traj?.[connects[1]]?.[0]
            ? (traj[connects[1]][0] as [number, number])
            : connects ? getStep0Position(connects[1]) : undefined
        return {
          id: formId,
          type: 'polygon',
          name: formName,
          points: sp.points,
          fillColor: color,
          strokeColor: color,
          strokeWidth: 1.5,
          fillOpacity: fillOpacityZ,
          closed: true,
          contained_links: sp.contained_links,
          z_level: sp.z_level,
          mergedLinkName: primaryLink ?? undefined,
          mergedLinkOriginalStart: mergedLinkOriginalStart ?? undefined,
          mergedLinkOriginalEnd: mergedLinkOriginalEnd ?? undefined,
          contained_links_valid: true
        }
      })

      let linkageForMerge: typeof linkageDoc.linkage = linkageDoc.linkage
      if (traj && linkageDoc.linkage?.nodes) {
        const nodesCopy = { ...linkageDoc.linkage.nodes }
        for (const nodeId of Object.keys(nodesCopy)) {
          const pos = traj[nodeId]?.[0]
          if (pos) nodesCopy[nodeId] = { ...nodesCopy[nodeId], position: pos }
        }
        linkageForMerge = { ...linkageDoc.linkage, nodes: nodesCopy }
      }
      const mergePayload = {
        pylink_data: { linkage: linkageForMerge, meta: linkageDoc.meta },
        polygon_id: '' as string,
        polygon_points: [] as [number, number][]
      }
      const mergeResponses = await Promise.all(
        suggested.map(async sp => {
          const isLinkForm = sp.contained_links.length === 1
          const formId = isLinkForm
            ? `link_form_${(sp.contained_links[0] ?? '').replace(/^link_/, '')}`
            : sp.polygon_id
          const restrictToLinks = isLinkForm ? [sp.contained_links[0]] : sp.contained_links
          const res = await fetch('/api/merge-polygon', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              ...mergePayload,
              polygon_id: formId,
              polygon_points: sp.points,
              restrict_to_links: restrictToLinks
            })
          })
          const data = await res.json() as { status?: string; polygon?: { contained_links?: string[]; mergedLinkName?: string; mergedLinkOriginalStart?: [number, number]; mergedLinkOriginalEnd?: [number, number]; fill_color?: string; stroke_color?: string } }
          return { polygonId: formId, data }
        })
      )

      const linkageAfterMerge = (() => {
        const nextEdges = { ...linkageDoc.meta.edges }
        for (const r of mergeResponses) {
          const sp = suggested.find(s => {
            const sid = s.contained_links.length === 1 ? `link_form_${(s.contained_links[0] ?? '').replace(/^link_/, '')}` : s.polygon_id
            return sid === r.polygonId
          })
          const p = r.data.polygon
          const fillColor = p?.fill_color
          if (r.data.status !== 'success' || !sp?.contained_links?.length || !fillColor) continue
          const links = sp.contained_links
          const polygonObj = newObjects.find(o => o.id === r.polygonId) as { z_level?: number } | undefined
          const polygonZ = polygonObj?.z_level ?? (nextEdges[links[0]] as { zlevel?: number } | undefined)?.zlevel
          for (const lid of links) {
            const existing = nextEdges[lid] ?? {}
            nextEdges[lid] = { ...existing, color: fillColor, ...(polygonZ !== undefined && { zlevel: polygonZ }) }
          }
        }
        return { ...linkageDoc, meta: { ...linkageDoc.meta, edges: nextEdges } }
      })()

      const mergedObjects: DrawnObject[] = drawnObjects.objects.concat(newObjects).map(obj => {
        const r = mergeResponses.find(m => m.polygonId === obj.id)
        const sp = r ? suggested.find(s => {
          const sid = s.contained_links.length === 1 ? `link_form_${(s.contained_links[0] ?? '').replace(/^link_/, '')}` : s.polygon_id
          return sid === r.polygonId
        }) : null
        if (!r || !sp || r.data.status !== 'success' || !r.data.polygon) return obj
        const p = r.data.polygon
        const rigidGroup = new Set(sp.contained_links)
        const containedOnly = (p.contained_links ?? []).filter((lid: string) => rigidGroup.has(lid))
        const primary = containedOnly.length > 0 ? containedOnly[0] : sp.contained_links[0]
        return {
          ...obj,
          contained_links: sp.contained_links,
          mergedLinkName: primary ?? undefined,
          mergedLinkOriginalStart: obj.mergedLinkOriginalStart ?? p.mergedLinkOriginalStart ?? undefined,
          mergedLinkOriginalEnd: obj.mergedLinkOriginalEnd ?? p.mergedLinkOriginalEnd ?? undefined,
          fillColor: p.fill_color ?? obj.fillColor,
          strokeColor: p.stroke_color ?? obj.strokeColor,
          fillOpacity: 0.25,
          contained_links_valid: true
        }
      })

      setLinkageDoc(linkageAfterMerge)
      setDrawnObjects(prev => ({ ...prev, objects: mergedObjects, selectedIds: [] }))

      showStatus(`Created ${newObjects.length} form(s)`, 'success', 3000)

      if (computeZLevelsAfterCreate) {
        try {
          const drawn_objects = mergedObjects
            .filter((o: { type?: string; id?: string; contained_links?: string[] }) =>
              o.type === 'polygon' && o.id && (o.contained_links?.length ?? 0) > 0
            )
            .map((o: DrawnObject) => ({
              id: o.id,
              type: 'polygon' as const,
              contained_links: o.contained_links ?? [],
              ...(Array.isArray(o.points) && o.points.length >= 3 && { points: o.points })
            }))
          const formSoftPinsCreate: Record<string, [number, number]> = {}
          for (const o of mergedObjects as DrawnObject[]) {
            if (o.type === 'polygon' && o.target_z_level != null && Number.isFinite(o.target_z_level)) {
              formSoftPinsCreate['polygon:' + o.id] = [o.target_z_level, 1]
            }
          }
          const zBody: Record<string, unknown> = {
            linkage: linkageAfterMerge.linkage,
            meta: linkageAfterMerge.meta,
            n_steps: simulationSteps || 32,
            ...(drawn_objects.length > 0 && { drawn_objects }),
            ...(drawn_objects.length > 0 && { margin_units: formPaddingUnits }),
            z_level_config: {
              ...zLevelConfig,
              crank_z: zLevelConfig.crank_z ?? null,
              soft_pins: { ...(zLevelConfig.soft_pins ?? {}), ...formSoftPinsCreate }
            }
          }
          if (animation.trajectoryData?.trajectories && animation.trajectoryData.nSteps > 0) {
            zBody.trajectories = animation.trajectoryData.trajectories
            zBody.n_steps = animation.trajectoryData.nSteps
          }
          const zRes = await fetch('/api/compute-link-z-levels', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(zBody)
          })
          if (!zRes.ok) throw new Error(`HTTP ${zRes.status}`)
          const zData = (await zRes.json()) as {
            status: string
            assignments?: Array<Record<string, number>>
            polygon_z_levels?: Record<string, number>
            message?: string
          }
          if (zData.status === 'success' && Array.isArray(zData.assignments) && zData.assignments.length > 0) {
            const assignment = zData.assignments[0]
            const uniqueZ = [...new Set(Object.values(assignment))].sort((a, b) => a - b)
            const zToColor = new Map<number, string>()
            uniqueZ.forEach((z, i) => {
              zToColor.set(z, Z_LEVEL_PALETTE[i % Z_LEVEL_PALETTE.length])
            })
            setLinkageDoc(prev => {
              const nextEdges = { ...prev.meta.edges }
              for (const [linkId, z] of Object.entries(assignment)) {
                const existing = nextEdges[linkId] ?? {}
                nextEdges[linkId] = {
                  ...existing,
                  zlevel: z,
                  color: zToColor.get(z) ?? (existing as { color?: string }).color ?? '#888888'
                }
              }
              return { ...prev, meta: { ...prev.meta, edges: nextEdges } }
            })
            const pz = zData.polygon_z_levels
            if (pz && Object.keys(pz).length > 0) {
              const fillOpacityZ = 0.25
              const zLeveledObjects = mergedObjects.map((obj): DrawnObject => {
                const o = obj as DrawnObject
                if (o.id in pz) {
                  const z = pz[o.id]
                  const color = zToColor.get(z) ?? o.fillColor
                  return {
                    ...o,
                    z_level: z,
                    fillColor: color,
                    strokeColor: color,
                    fillOpacity: fillOpacityZ
                  }
                }
                return o
              })
              setDrawnObjects(prev => ({ ...prev, objects: zLeveledObjects }))
            }
            setLinkColorMode('z-level')
            showStatus('Z-levels computed; links and forms colored by layer', 'success', 3000)
          }
        } catch (zErr) {
          const zMsg = zErr instanceof Error ? zErr.message : 'Compute z-levels failed'
          showStatus(zMsg, 'error', 3000)
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Create forms failed'
      showStatus(msg, 'error', 3000)
    }
  }, [
    linkageDoc,
    drawnObjects.objects,
    simulationSteps,
    setLinkageDoc,
    setDrawnObjects,
    setLinkColorMode,
    showStatus,
    formPaddingUnits,
    zLevelConfig,
    createRigidForms,
    createLinkForms,
    computeZLevelsAfterCreate,
    animation
  ])

  const openFormEdit = useCallback((formId: string) => {
    const obj = drawnObjects.objects.find((o: { id: string }) => o.id === formId) as DrawnObject | undefined
    if (!obj || obj.type !== 'polygon') return
    setEditingFormData({
      id: obj.id,
      name: obj.name,
      fillColor: obj.fillColor,
      z_level: obj.z_level,
      z_level_fixed: obj.z_level_fixed,
      target_z_level: obj.target_z_level
    })
  }, [drawnObjects.objects])

  const onSaveForm = useCallback((id: string, updates: { name: string; fillColor: string; strokeColor: string; z_level?: number; z_level_fixed?: boolean; target_z_level?: number }) => {
    setDrawnObjects(prev => ({
      ...prev,
      objects: prev.objects.map(obj =>
        obj.id !== id
          ? obj
          : {
              ...obj,
              name: updates.name,
              fillColor: updates.fillColor,
              strokeColor: updates.strokeColor,
              ...(updates.z_level !== undefined && { z_level: updates.z_level }),
              ...(updates.z_level_fixed !== undefined && { z_level_fixed: updates.z_level_fixed }),
              ...(updates.target_z_level !== undefined && { target_z_level: updates.target_z_level })
            }
      )
    }))
    setEditingFormData(null)
    showStatus('Form updated', 'success', 1500)
    if (updates.z_level !== undefined || updates.z_level_fixed !== undefined) {
      runComputeAfterFormSaveRef.current = true
    }
  }, [setDrawnObjects, showStatus])

  const handleDeleteAllForms = useCallback(() => {
    if (!window.confirm('Remove all forms? This cannot be undone.')) return
    setDrawnObjects(prev => ({
      ...prev,
      objects: prev.objects.filter((o: { type?: string }) => o.type !== 'polygon'),
      selectedIds: []
    }))
    showStatus('All forms removed', 'success', 2000)
  }, [setDrawnObjects, showStatus])

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
      return
    }
    optimization.setIsSyncedToOptimizer(true)
  }, [linkageDoc, optimization.isOptimizedMechanism, showStatus, triggerMechanismChange])

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
          optimization.clearOptimizerState()
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
            setCanvases: (c) => setCanvases(Array.isArray(c) ? c : []),
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
      const msg = error instanceof Error ? error.message : String(error)
      showStatus(`Load failed: ${msg}`, 'error', 5000)
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
      // Include drawnObjects and canvases in the document for persistence
      const docToSave = {
        ...linkageDoc,
        drawnObjects: drawnObjects.objects.length > 0 ? drawnObjects.objects : undefined,
        canvases: canvases.length > 0 ? canvases : undefined
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
          optimization.clearOptimizerState()
          applyLoadedDocument({
            doc: result.data,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setCanvases: (c) => setCanvases(Array.isArray(c) ? c : []),
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
      const msg = error instanceof Error ? error.message : String(error)
      showStatus(`Load failed: ${msg}`, 'error', 5000)
    }
  }

  // Load pylink graph from a .json file (native file picker; read in browser)
  const handleLoadFileSelected = useCallback(
    (file: File) => {
      const reader = new FileReader()
      reader.onload = () => {
        try {
          const text = reader.result as string
          const data = JSON.parse(text) as unknown
          if (!isHypergraphFormat(data)) {
            showStatus('Load failed: file is not a valid mechanism document (legacy format not supported)', 'error', 5000)
            return
          }
          optimization.clearOptimizerState()
          applyLoadedDocument({
            doc: data,
            setLinkageDoc,
            setDrawnObjects: (s) => setDrawnObjects(s as DrawnObjectsState),
            setCanvases: (c) => setCanvases(Array.isArray(c) ? c : []),
            setSelectedJoints,
            setSelectedLinks,
            clearTrajectory: animation.clearTrajectory,
            triggerMechanismChange
          })
          setStoredGraphFilename(file.name)
          showStatus(`Loaded ${file.name}`, 'success', 3000)
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err)
          showStatus(`Load failed: ${msg}`, 'error', 5000)
        }
      }
      reader.onerror = () => showStatus('Load failed: could not read file', 'error', 5000)
      reader.readAsText(file)
    },
    [
      isHypergraphFormat,
      optimization.clearOptimizerState,
      applyLoadedDocument,
      setLinkageDoc,
      setDrawnObjects,
      setCanvases,
      setSelectedJoints,
      setSelectedLinks,
      animation.clearTrajectory,
      triggerMechanismChange,
      showStatus
    ]
  )

  // Suggested default for Save As: doc name or acinonyx-YYYYMMDD
  const suggestedSaveAsName =
    linkageDoc.name && linkageDoc.name !== 'untitled'
      ? linkageDoc.name
      : `acinonyx-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`

  const handleSaveAs = useCallback(
    async (filename: string) => {
      if (!filename.trim()) return
      try {
        showStatus('Saving...', 'action')
        const docToSave = {
          ...linkageDoc,
          drawnObjects: drawnObjects.objects.length > 0 ? drawnObjects.objects : undefined,
          canvases: canvases.length > 0 ? canvases : undefined
        }
        const response = await fetch('/api/save-pylink-graph-as', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            data: docToSave,
            filename: filename.trim()
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
    },
    [linkageDoc, drawnObjects.objects, canvases, showStatus]
  )

  const handleClearAll = useCallback(() => {
    optimization.clearOptimizerState()
    setLinkageDoc(createEmptyLinkageDocument('untitled'))
    setDrawnObjects({ objects: [], selectedIds: [] })
    setCanvases([])
    setSelectedJoints([])
    setSelectedLinks([])
    animation.clearTrajectory()
    setStoredGraphFilename(null)
    showStatus('Cleared. Starting with an empty mechanism.', 'info', 3000)
  }, [
    optimization.clearOptimizerState,
    setLinkageDoc,
    setDrawnObjects,
    setCanvases,
    setSelectedJoints,
    setSelectedLinks,
    animation.clearTrajectory,
    showStatus
  ])

  // Toolbar content by id — built by useToolbarContent (after all load/save callbacks)
  const toolbarContentParams = useMemo(
    () => ({
      toolsProps: {
        toolMode,
        setToolMode,
        hoveredTool,
        setHoveredTool,
        linkCreationState,
        setLinkCreationState,
        setPreviewLine,
        onPauseAnimation: animation.animationState.isAnimating ? animation.pauseAnimation : undefined
      },
      linksProps: {
        links: pylinkDoc.meta.links,
        linkageDoc,
        selectedLinks,
        setSelectedLinks,
        setSelectedJoints,
        hoveredLink,
        setHoveredLink,
        selectionColor,
        getJointPosition,
        openLinkEditModal
      },
      nodesProps: {
        joints: pylinkDoc.pylinkage.joints,
        selectedJoints,
        setSelectedJoints,
        setSelectedLinks,
        hoveredJoint,
        setHoveredJoint,
        selectionColor,
        getJointPosition,
        openJointEditModal
      },
      moreProps: {
        loadDemo4Bar,
        loadDemoLeg,
        loadDemoWalker,
        loadDemoComplex,
        loadPylinkGraphLast,
        onLoadFileSelected: handleLoadFileSelected,
        savePylinkGraph,
        suggestedSaveAsName,
        onSaveAs: handleSaveAs,
        onClearAll: handleClearAll,
        canvases,
        setCanvases,
        editingCanvasId,
        setEditingCanvasId,
        showStatus
      },
      formsProps: {
        formPaddingUnits,
        onFormPaddingChange: setFormPaddingUnits,
        createRigidForms,
        setCreateRigidForms,
        createLinkForms,
        setCreateLinkForms,
        computeZLevelsAfterCreate,
        setComputeZLevelsAfterCreate,
        onCreateForms: handleCreateForms,
        onComputeLinkZLevels: handleComputeLinkZLevels,
        objects: drawnObjects.objects as import('./builder/rendering/types').DrawnObject[],
        selectedIds: drawnObjects.selectedIds,
        onSelectForms: (ids: string[]) => setDrawnObjects(prev => ({ ...prev, selectedIds: ids })),
        openFormEdit,
        showStatus,
        darkMode,
        zLevelRows,
        onZLevelColorChange,
        onReorderForms,
        onFormZLevelFixedChange,
        zLevelConfig,
        onZLevelConfigChange: setZLevelConfig,
        onDeleteAllForms: handleDeleteAllForms
      },
      settingsProps: {
        darkMode,
        setDarkMode,
        showGrid,
        setShowGrid,
        showJointLabels,
        setShowJointLabels,
        showLinkLabels,
        setShowLinkLabels,
        simulationStepsInput,
        setSimulationStepsInput,
        autoSimulateDelayMs,
        setAutoSimulateDelayMs,
        trajectoryColorCycle,
        setTrajectoryColorCycle,
        trajectoryData: animation.trajectoryData,
        autoSimulateEnabled: animation.autoSimulateEnabled,
        setAutoSimulateEnabled: animation.setAutoSimulateEnabled,
        triggerMechanismChange,
        jointMergeRadius,
        setJointMergeRadius,
        canvasBgColor: canvasBgColor as CanvasBgColor,
        setCanvasBgColor,
        jointSize,
        setJointSize,
        jointOutline,
        setJointOutline,
        linkThickness,
        setLinkThickness,
        linkTransparency,
        setLinkTransparency,
        linkColorMode,
        setLinkColorMode,
        linkColorSingle,
        setLinkColorSingle,
        trajectoryDotSize,
        setTrajectoryDotSize,
        trajectoryDotOutline,
        setTrajectoryDotOutline,
        trajectoryDotOpacity,
        setTrajectoryDotOpacity,
        showTrajectoryStepNumbers,
        setShowTrajectoryStepNumbers,
        trajectoryStyle: trajectoryStyle as TrajectoryStyle,
        setTrajectoryStyle,
        exploreRadius,
        setExploreRadius: settings.setExploreRadius,
        exploreRadialSamples,
        setExploreRadialSamples: settings.setExploreRadialSamples,
        exploreAzimuthalSamples,
        setExploreAzimuthalSamples: settings.setExploreAzimuthalSamples,
        exploreNMaxCombinatorial,
        setExploreNMaxCombinatorial: settings.setExploreNMaxCombinatorial,
        exploreColormapEnabled,
        setExploreColormapEnabled: settings.setExploreColormapEnabled,
        exploreColormapType,
        setExploreColormapType: settings.setExploreColormapType
      },
      optimizeProps: {
        joints: pylinkDoc.pylinkage.joints,
        linkageDoc,
        trajectoryData: animation.trajectoryData,
        stretchingLinks: animation.stretchingLinks,
        targetPaths,
        setTargetPaths,
        selectedPathId,
        setSelectedPathId,
        preprocessResult: optimization.preprocessResult,
        isPreprocessing: optimization.isPreprocessing,
        prepEnableSmooth: optimization.prepEnableSmooth,
        setPrepEnableSmooth: optimization.setPrepEnableSmooth,
        prepSmoothMethod: optimization.prepSmoothMethod as import('./builder/toolbars/OptimizationToolbar').SmoothMethod,
        setPrepSmoothMethod: optimization.setPrepSmoothMethod,
        prepSmoothWindow: optimization.prepSmoothWindow,
        setPrepSmoothWindow: optimization.setPrepSmoothWindow,
        prepSmoothPolyorder: optimization.prepSmoothPolyorder,
        setPrepSmoothPolyorder: optimization.setPrepSmoothPolyorder,
        prepEnableResample: optimization.prepEnableResample,
        setPrepEnableResample: optimization.setPrepEnableResample,
        prepTargetNSteps: optimization.prepTargetNSteps,
        setPrepTargetNSteps: optimization.setPrepTargetNSteps,
        prepResampleMethod: optimization.prepResampleMethod as import('./builder/toolbars/OptimizationToolbar').ResampleMethod,
        setPrepResampleMethod: optimization.setPrepResampleMethod,
        preprocessTrajectory: optimization.preprocessTrajectory,
        simulationSteps,
        simulationStepsInput,
        setSimulationStepsInput,
        optMethod: optimization.optMethod as import('./builder/toolbars/OptimizationToolbar').OptMethod,
        setOptMethod: optimization.setOptMethod,
        optNParticles: optimization.optNParticles,
        setOptNParticles: optimization.setOptNParticles,
        optIterations: optimization.optIterations,
        setOptIterations: optimization.setOptIterations,
        optMaxIterations: optimization.optMaxIterations,
        setOptMaxIterations: optimization.setOptMaxIterations,
        optTolerance: optimization.optTolerance,
        setOptTolerance: optimization.setOptTolerance,
        optInertia: optimization.optInertia,
        setOptInertia: optimization.setOptInertia,
        optC1: optimization.optC1,
        setOptC1: optimization.setOptC1,
        optC2: optimization.optC2,
        setOptC2: optimization.setOptC2,
        optDiscretizationSteps: optimization.optDiscretizationSteps,
        setOptDiscretizationSteps: optimization.setOptDiscretizationSteps,
        optTimeLimit: optimization.optTimeLimit,
        setOptTimeLimit: optimization.setOptTimeLimit,
        optGapLimit: optimization.optGapLimit,
        setOptGapLimit: optimization.setOptGapLimit,
        optBoundsFactor: optimization.optBoundsFactor,
        setOptBoundsFactor: optimization.setOptBoundsFactor,
        optMinLength: optimization.optMinLength,
        setOptMinLength: optimization.setOptMinLength,
        isOptimizing: optimization.isOptimizing,
        runOptimization: (config?: Record<string, unknown>) => {
          optimization.runOptimization(config ?? {}).catch(err => console.error('Optimization error:', err))
        },
        optimizationResult: optimization.optimizationResult,
        preOptimizationDoc: optimization.preOptimizationDoc,
        revertOptimization: optimization.revertOptimization,
        syncToOptimizerResult: optimization.syncToOptimizerResult,
        isSyncedToOptimizer: optimization.isSyncedToOptimizer,
        dimensionInfo: optimization.dimensionInfo,
        isLoadingDimensions: optimization.isLoadingDimensions,
        dimensionInfoError: optimization.dimensionInfoError
      }
    }),
    [
      toolMode,
      setToolMode,
      hoveredTool,
      setHoveredTool,
      linkCreationState,
      setLinkCreationState,
      setPreviewLine,
      animation.animationState.isAnimating,
      animation.pauseAnimation,
      pylinkDoc.meta.links,
      linkageDoc,
      selectedLinks,
      setSelectedLinks,
      setSelectedJoints,
      hoveredLink,
      setHoveredLink,
      selectionColor,
      getJointPosition,
      openLinkEditModal,
      pylinkDoc.pylinkage.joints,
      selectedJoints,
      setSelectedJoints,
      hoveredJoint,
      setHoveredJoint,
      openJointEditModal,
      loadDemo4Bar,
      loadDemoLeg,
      loadDemoWalker,
      loadDemoComplex,
      loadPylinkGraphLast,
      handleLoadFileSelected,
      savePylinkGraph,
      suggestedSaveAsName,
      handleSaveAs,
      handleClearAll,
      canvases,
      setCanvases,
      editingCanvasId,
      setEditingCanvasId,
      showStatus,
      formPaddingUnits,
      setFormPaddingUnits,
      createRigidForms,
      setCreateRigidForms,
      createLinkForms,
      setCreateLinkForms,
      computeZLevelsAfterCreate,
      setComputeZLevelsAfterCreate,
      handleCreateForms,
      handleComputeLinkZLevels,
      drawnObjects.objects,
      drawnObjects.selectedIds,
      setDrawnObjects,
      openFormEdit,
      darkMode,
      zLevelRows,
      onZLevelColorChange,
      onReorderForms,
      onFormZLevelFixedChange,
      setDarkMode,
      showGrid,
      setShowGrid,
      showJointLabels,
      setShowJointLabels,
      showLinkLabels,
      setShowLinkLabels,
      simulationStepsInput,
      setSimulationStepsInput,
      autoSimulateDelayMs,
      setAutoSimulateDelayMs,
      trajectoryColorCycle,
      setTrajectoryColorCycle,
      animation.trajectoryData,
      animation.autoSimulateEnabled,
      animation.setAutoSimulateEnabled,
      triggerMechanismChange,
      jointMergeRadius,
      setJointMergeRadius,
      canvasBgColor,
      setCanvasBgColor,
      jointSize,
      setJointSize,
      jointOutline,
      setJointOutline,
      linkThickness,
      setLinkThickness,
      linkTransparency,
      setLinkTransparency,
      linkColorMode,
      setLinkColorMode,
      linkColorSingle,
      setLinkColorSingle,
      trajectoryDotSize,
      setTrajectoryDotSize,
      trajectoryDotOutline,
      setTrajectoryDotOutline,
      trajectoryDotOpacity,
      setTrajectoryDotOpacity,
      showTrajectoryStepNumbers,
      setShowTrajectoryStepNumbers,
      trajectoryStyle,
      setTrajectoryStyle,
      exploreRadius,
      settings.setExploreRadius,
      exploreRadialSamples,
      settings.setExploreRadialSamples,
      exploreAzimuthalSamples,
      settings.setExploreAzimuthalSamples,
      exploreNMaxCombinatorial,
      settings.setExploreNMaxCombinatorial,
      exploreColormapEnabled,
      settings.setExploreColormapEnabled,
      exploreColormapType,
      settings.setExploreColormapType,
      targetPaths,
      setTargetPaths,
      selectedPathId,
      setSelectedPathId,
      optimization.preprocessResult,
      optimization.isPreprocessing,
      optimization.prepEnableSmooth,
      optimization.setPrepEnableSmooth,
      optimization.prepSmoothMethod,
      optimization.setPrepSmoothMethod,
      optimization.prepSmoothWindow,
      optimization.setPrepSmoothWindow,
      optimization.prepSmoothPolyorder,
      optimization.setPrepSmoothPolyorder,
      optimization.prepEnableResample,
      optimization.setPrepEnableResample,
      optimization.prepTargetNSteps,
      optimization.setPrepTargetNSteps,
      optimization.prepResampleMethod,
      optimization.setPrepResampleMethod,
      optimization.preprocessTrajectory,
      simulationSteps,
      simulationStepsInput,
      setSimulationStepsInput,
      optimization.optMethod,
      optimization.setOptMethod,
      optimization.optNParticles,
      optimization.setOptNParticles,
      optimization.optIterations,
      optimization.setOptIterations,
      optimization.optMaxIterations,
      optimization.setOptMaxIterations,
      optimization.optTolerance,
      optimization.setOptTolerance,
      optimization.optInertia,
      optimization.setOptInertia,
      optimization.optC1,
      optimization.setOptC1,
      optimization.optC2,
      optimization.setOptC2,
      optimization.optDiscretizationSteps,
      optimization.setOptDiscretizationSteps,
      optimization.optTimeLimit,
      optimization.setOptTimeLimit,
      optimization.optGapLimit,
      optimization.setOptGapLimit,
      optimization.optBoundsFactor,
      optimization.setOptBoundsFactor,
      optimization.optMinLength,
      optimization.setOptMinLength,
      optimization.isOptimizing,
      optimization.runOptimization,
      optimization.optimizationResult,
      optimization.preOptimizationDoc,
      optimization.revertOptimization,
      optimization.syncToOptimizerResult,
      optimization.isSyncedToOptimizer,
      optimization.dimensionInfo,
      optimization.isLoadingDimensions,
      optimization.dimensionInfoError
    ]
  )
  const renderToolbarContent = useToolbarContent(toolbarContentParams)

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
        viewport={viewport.viewport}
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={handleCanvasMouseMove}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseLeave}
        onClick={handleCanvasClick}
        onDoubleClick={handleCanvasDoubleClick}
        renderGrid={layerRenders.renderGrid}
        renderCanvases={layerRenders.renderCanvases}
        renderDrawnObjects={layerRenders.renderDrawnObjects}
        renderLinks={layerRenders.renderLinks}
        renderPreviewLine={layerRenders.renderPreviewLine}
        renderPolygonPreview={layerRenders.renderPolygonPreview}
        renderTargetPaths={layerRenders.renderTargetPaths}
        renderPathPreview={layerRenders.renderPathPreview}
        renderExplorationTrajectories={layerRenders.renderExplorationTrajectories}
        renderTrajectories={layerRenders.renderTrajectories}
        renderExplorationDots={layerRenders.renderExplorationDots}
        renderJoints={layerRenders.renderJoints}
        renderSelectionBox={layerRenders.renderSelectionBox}
        renderMeasurementMarkers={layerRenders.renderMeasurementMarkers}
        renderMeasurementLine={layerRenders.renderMeasurementLine}
        exploreModeActive={toolMode === 'explore_node_trajectories' && exploreTrajectoriesState.exploreSamples.length > 0}
        toolMode={toolMode}
        jointCount={pylinkDoc.pylinkage.joints.length}
        linkCount={Object.keys(pylinkDoc.meta.links).length}
        selectedJoints={selectedJoints}
        selectedLinks={selectedLinks}
        statusMessage={statusMessage}
        statusHistory={statusHistory}
        clearStatusHistory={clearStatusHistory}
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

      {/* Animation Toolbar - effectiveDisplayFrame so first paint after drag start shows 1/N */}
      <AnimateToolbar
        joints={pylinkDoc.pylinkage.joints}
        animationState={{ ...animation.animationState, currentFrame: effectiveDisplayFrame }}
        playAnimation={animation.playAnimation}
        pauseAnimation={animation.pauseAnimation}
        stopAnimation={animation.stopAnimation}
        setPlaybackFps={animation.setPlaybackFps}
        setPlaybackDirection={animation.setPlaybackDirection}
        setAnimatedPositions={animation.setAnimatedPositions}
        setFrame={setAnimationFrameWithDisplay}
        isSimulating={animation.isSimulating}
        trajectoryData={animation.trajectoryData}
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
        editingFormData={editingFormData}
        onCloseFormEdit={() => setEditingFormData(null)}
        onSaveForm={onSaveForm}
        renameJoint={renameJoint}
        renameLink={renameLink}
        updateJointProperty={updateJointProperty}
        updateLinkProperty={updateLinkProperty}
        onJointMetaValueChange={(jointName, metaValue) => {
          setLinkageDoc(prev => updateNodeMeta(prev, jointName, { metaValue }))
          setEditingJointData(prev => (prev && prev.name === jointName ? { ...prev, metaValue } : prev))
        }}
        onJointShowPathChange={(jointName, showPath) =>
          setLinkageDoc(prev => updateNodeMeta(prev, jointName, { showPath }))
        }
        setEditingJointData={setEditingJointData}
        setEditingLinkData={setEditingLinkData}
        jointTypes={JOINT_TYPES}
        getColorForZLevel={getColorForZLevel}
        darkMode={darkMode}
      />
    </Box>
  )
}

export default BuilderTab
