import { useState, useRef, useCallback, useEffect } from 'react'
import type { LinkageDocument, PylinkDocument } from './types'
import type { OptMethod, SmoothMethod, ResampleMethod } from './toolbars'
import { DEFAULT_SIMULATION_STEPS } from './constants'
import type { TargetPath } from '../BuilderTools'
import type { TrajectoryData } from '../AnimateSimulate'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface OptimizationResult {
  success: boolean
  initialError: number | null
  finalError: number | null
  message: string
  iterations?: number
  executionTimeMs?: number
  optimizedDimensions?: Record<string, number>
  originalDimensions?: Record<string, number>
}

export interface DimensionInfo {
  names: string[]
  initial_values: number[]
  bounds: [number, number][]
  n_dimensions: number
}

export interface PreprocessResult {
  originalPoints: number
  outputPoints: number
  analysis: Record<string, unknown>
}

export interface OptimizationControllerContext {
  linkageDoc: LinkageDocument
  setLinkageDoc: (doc: LinkageDocument) => void
  targetPaths: TargetPath[]
  setTargetPaths: React.Dispatch<React.SetStateAction<TargetPath[]>>
  selectedPathId: string | null
  simulationSteps: number
  showStatus: (text: string, type?: 'info' | 'success' | 'warning' | 'error' | 'action', duration?: number) => void
  triggerMechanismChange: () => void
  autoSimulateDelayMs: number
  pylinkDoc: PylinkDocument
  // Helper functions
  isHypergraphFormat: (data: unknown) => data is LinkageDocument
  logMechanismState: (label: string, doc: LinkageDocument) => void
  extractDimensionsFromLinkageDoc: (doc: LinkageDocument) => Record<string, number>
}

export interface UseOptimizationControllerReturn {
  // State
  isOptimizing: boolean
  preOptimizationDoc: LinkageDocument | null
  optimizationResult: OptimizationResult | null
  isOptimizedMechanism: boolean
  isSyncedToOptimizer: boolean
  optMethod: OptMethod
  optNParticles: number
  optIterations: number
  optMaxIterations: number
  optTolerance: number
  optBoundsFactor: number
  optMinLength: number
  optVerbose: boolean
  prepEnableSmooth: boolean
  prepSmoothWindow: number
  prepSmoothPolyorder: number
  prepSmoothMethod: SmoothMethod
  prepEnableResample: boolean
  prepTargetNSteps: number
  prepResampleMethod: ResampleMethod
  isPreprocessing: boolean
  preprocessResult: PreprocessResult | null
  dimensionInfo: DimensionInfo | null
  isLoadingDimensions: boolean
  dimensionInfoError: string | null

  // Setters
  setIsOptimizing: React.Dispatch<React.SetStateAction<boolean>>
  setPreOptimizationDoc: React.Dispatch<React.SetStateAction<LinkageDocument | null>>
  setOptimizationResult: React.Dispatch<React.SetStateAction<OptimizationResult | null>>
  setIsOptimizedMechanism: React.Dispatch<React.SetStateAction<boolean>>
  setIsSyncedToOptimizer: React.Dispatch<React.SetStateAction<boolean>>
  setOptMethod: React.Dispatch<React.SetStateAction<OptMethod>>
  setOptNParticles: React.Dispatch<React.SetStateAction<number>>
  setOptIterations: React.Dispatch<React.SetStateAction<number>>
  setOptMaxIterations: React.Dispatch<React.SetStateAction<number>>
  setOptTolerance: React.Dispatch<React.SetStateAction<number>>
  setOptBoundsFactor: React.Dispatch<React.SetStateAction<number>>
  setOptMinLength: React.Dispatch<React.SetStateAction<number>>
  setOptVerbose: React.Dispatch<React.SetStateAction<boolean>>
  setPrepEnableSmooth: React.Dispatch<React.SetStateAction<boolean>>
  setPrepSmoothWindow: React.Dispatch<React.SetStateAction<number>>
  setPrepSmoothPolyorder: React.Dispatch<React.SetStateAction<number>>
  setPrepSmoothMethod: React.Dispatch<React.SetStateAction<SmoothMethod>>
  setPrepEnableResample: React.Dispatch<React.SetStateAction<boolean>>
  setPrepTargetNSteps: React.Dispatch<React.SetStateAction<number>>
  setPrepResampleMethod: React.Dispatch<React.SetStateAction<ResampleMethod>>
  setIsPreprocessing: React.Dispatch<React.SetStateAction<boolean>>
  setPreprocessResult: React.Dispatch<React.SetStateAction<PreprocessResult | null>>
  setDimensionInfo: React.Dispatch<React.SetStateAction<DimensionInfo | null>>
  setIsLoadingDimensions: React.Dispatch<React.SetStateAction<boolean>>
  setDimensionInfoError: React.Dispatch<React.SetStateAction<string | null>>

  // Functions
  revertOptimization: () => void
  preprocessTrajectory: () => Promise<void>
  runOptimization: (config: Record<string, unknown>) => Promise<void>
  syncToOptimizerResult: () => void
  handleSimulationComplete: (data: TrajectoryData) => void
  // Refs (for external access if needed)
  lastOptimizerResultRef: React.MutableRefObject<LinkageDocument | null>
  optimizationJustCompletedRef: React.MutableRefObject<boolean>
  optimizationCompleteTimeRef: React.MutableRefObject<number | null>
}

// ═══════════════════════════════════════════════════════════════════════════════
// HOOK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Hook for managing optimization state and logic
 * Step 1.1: State only (no functions yet)
 */
export function useOptimizationController(
  context: OptimizationControllerContext
): UseOptimizationControllerReturn {
  // Optimization state
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [preOptimizationDoc, setPreOptimizationDoc] = useState<LinkageDocument | null>(null)
  const [optimizationResult, setOptimizationResult] = useState<OptimizationResult | null>(null)

  // CRITICAL: Track optimizer result and sync state
  const lastOptimizerResultRef = useRef<LinkageDocument | null>(null)
  const [isOptimizedMechanism, setIsOptimizedMechanism] = useState(false)
  const [isSyncedToOptimizer, setIsSyncedToOptimizer] = useState(false)

  // CRITICAL: Mechanism lock to prevent modifications after optimization
  const optimizationJustCompletedRef = useRef(false)
  const optimizationCompleteTimeRef = useRef<number | null>(null)

  // Optimization hyperparameters
  const [optMethod, setOptMethod] = useState<OptMethod>('pylinkage')
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
  const [prepSmoothMethod, setPrepSmoothMethod] = useState<SmoothMethod>('savgol')
  const [prepEnableResample, setPrepEnableResample] = useState(true)
  const [prepTargetNSteps, setPrepTargetNSteps] = useState(DEFAULT_SIMULATION_STEPS)
  const [prepResampleMethod, setPrepResampleMethod] = useState<ResampleMethod>('parametric')
  const [isPreprocessing, setIsPreprocessing] = useState(false)
  const [preprocessResult, setPreprocessResult] = useState<PreprocessResult | null>(null)

  // Dimension info state - fetched after trajectory computation
  const [dimensionInfo, setDimensionInfo] = useState<DimensionInfo | null>(null)
  const [isLoadingDimensions, setIsLoadingDimensions] = useState(false)
  const [dimensionInfoError, setDimensionInfoError] = useState<string | null>(null)
  const dimensionFetchTimerRef = useRef<number | null>(null)
  const prevTrajectoryKeyRef = useRef<string | null>(null)

  // Use refs to access latest context values without causing callback re-creation
  const contextRef = useRef(context)
  useEffect(() => {
    contextRef.current = context
  }, [context])

  // ═══════════════════════════════════════════════════════════════════════════════
  // HELPER FUNCTIONS
  // ═══════════════════════════════════════════════════════════════════════════════

  // Serialize config to ensure plain object
  const serializeMechVariationConfig = useCallback((config: Record<string, unknown>): Record<string, unknown> => {
    const serialized = JSON.parse(JSON.stringify(config))
    if (serialized.dimension_variation) {
      serialized.dimension_variation = {
        default_variation_range: serialized.dimension_variation.default_variation_range ?? 0.5,
        default_enabled: serialized.dimension_variation.default_enabled ?? true,
        dimension_overrides: serialized.dimension_variation.dimension_overrides || {},
        exclude_dimensions: serialized.dimension_variation.exclude_dimensions || [],
      }
    }
    if (serialized.static_joint_movement) {
      serialized.static_joint_movement = {
        enabled: serialized.static_joint_movement.enabled ?? false,
        max_x_movement: serialized.static_joint_movement.max_x_movement ?? 10.0,
        max_y_movement: serialized.static_joint_movement.max_y_movement ?? 10.0,
        joint_overrides: serialized.static_joint_movement.joint_overrides || {},
        linked_joints: serialized.static_joint_movement.linked_joints || [],
      }
    }
    if (serialized.topology_changes) {
      serialized.topology_changes = {
        enabled: serialized.topology_changes.enabled ?? false,
        add_node_probability: serialized.topology_changes.add_node_probability ?? 0.0,
        remove_node_probability: serialized.topology_changes.remove_node_probability ?? 0.0,
        add_link_probability: serialized.topology_changes.add_link_probability ?? 0.0,
        remove_link_probability: serialized.topology_changes.remove_link_probability ?? 0.0,
        min_nodes: serialized.topology_changes.min_nodes ?? 3,
        max_nodes: serialized.topology_changes.max_nodes ?? 32,
        preserve_crank: serialized.topology_changes.preserve_crank ?? true,
      }
    }
    return serialized
  }, [])

  // ═══════════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION FUNCTIONS
  // ═══════════════════════════════════════════════════════════════════════════════

  // Revert to pre-optimization state
  const revertOptimization = useCallback(() => {
    if (preOptimizationDoc) {
      context.setLinkageDoc(preOptimizationDoc)
      setPreOptimizationDoc(null)
      setOptimizationResult(null)
      context.triggerMechanismChange()
      context.showStatus('Reverted to pre-optimization state', 'info', 2000)
    }
  }, [preOptimizationDoc, context])

  // Preprocess trajectory (smooth and/or resample)
  const preprocessTrajectory = useCallback(async () => {
    const selectedPath = context.targetPaths.find(p => p.id === context.selectedPathId)
    if (!selectedPath || selectedPath.points.length < 3) {
      context.showStatus('Select a path with at least 3 points', 'warning', 2000)
      return
    }

    try {
      setIsPreprocessing(true)
      setPreprocessResult(null)
      context.showStatus('Preprocessing trajectory...', 'action')

      // Send in TargetTrajectory format (preferred) or raw trajectory (backward compatible)
      const requestBody: Record<string, unknown> = {
        target_trajectory: {
          joint_name: selectedPath.targetJoint || 'unknown',
          positions: selectedPath.points
        },
        target_n_steps: prepTargetNSteps,
        smooth: prepEnableSmooth,
        smooth_window: prepSmoothWindow,
        smooth_polyorder: prepSmoothPolyorder,
        smooth_method: prepSmoothMethod,
        resample: prepEnableResample,
        resample_method: prepResampleMethod,
        closed: true  // All target paths are treated as closed/cyclic
      }

      const response = await fetch('/api/prepare-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        // Backend now returns TargetTrajectory format
        const targetTrajectory = result.target_trajectory
        if (!targetTrajectory || !targetTrajectory.positions) {
          throw new Error('Invalid response: missing target_trajectory.positions')
        }

        // Update the path with preprocessed points from TargetTrajectory
        const newPoints: [number, number][] = targetTrajectory.positions.map(
          (p: number[]) => [p[0], p[1]] as [number, number]
        )

        // Store the full TargetTrajectory for use in optimization
        context.setTargetPaths(prev => prev.map(p =>
          p.id === context.selectedPathId
            ? {
                ...p,
                points: newPoints,
                // Store TargetTrajectory for optimizer (keep original name)
                targetTrajectory: {
                  joint_name: targetTrajectory.joint_name || selectedPath.targetJoint || 'unknown',
                  positions: targetTrajectory.positions,
                  n_steps: targetTrajectory.n_steps || targetTrajectory.positions.length
                }
              }
            : p
        ))

        setPreprocessResult({
          originalPoints: result.original_points,
          outputPoints: result.output_points,
          analysis: result.analysis
        })

        context.showStatus(
          `Preprocessed: ${result.original_points} → ${result.output_points} points`,
          'success',
          3000
        )
      } else {
        context.showStatus(result.message || 'Preprocessing failed', 'error', 3000)
      }
    } catch (error) {
      context.showStatus(`Preprocessing error: ${error}`, 'error', 3000)
    } finally {
      setIsPreprocessing(false)
    }
  }, [context, prepTargetNSteps, prepEnableSmooth, prepSmoothWindow, prepSmoothPolyorder, prepSmoothMethod, prepEnableResample, prepResampleMethod])

  // Run optimization
  const runOptimization = useCallback(async (config: Record<string, unknown>) => {
    const selectedPath = context.targetPaths.find(p => p.id === context.selectedPathId)
    if (!selectedPath || !selectedPath.targetJoint) {
      context.showStatus('Select a path and target joint first', 'warning', 2000)
      return
    }

    // Warn if target path points don't match simulation steps
    if (selectedPath.points.length !== context.simulationSteps) {
      console.warn(`⚠️ Path has ${selectedPath.points.length} points but simulation uses ${context.simulationSteps} steps. Consider preprocessing the path.`)
    }

    try {
      setIsOptimizing(true)
      setOptimizationResult(null)
      context.showStatus(`Running ${optMethod.toUpperCase()} optimization (N=${context.simulationSteps})...`, 'action')

      // Save current state before optimization (deep copy)
      const savedDoc = JSON.parse(JSON.stringify(context.linkageDoc)) as LinkageDocument
      setPreOptimizationDoc(savedDoc)

      // Extract original dimensions for comparison from PRE-OPTIMIZATION document
      const originalDims = context.extractDimensionsFromLinkageDoc(savedDoc)

      // Serialize and log optimization config before sending
      const serializedConfig = serializeMechVariationConfig(config as Record<string, unknown>)

      // Build optimization options based on method
      const optimizationOptions: Record<string, unknown> = {
        method: optMethod,
        verbose: optVerbose,
        mech_variation_config: serializedConfig
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

      // Use stored TargetTrajectory if available (from preprocessing), otherwise construct from points
      const targetTrajectory = selectedPath.targetTrajectory || {
        joint_name: selectedPath.targetJoint,
        positions: selectedPath.points,
        n_steps: selectedPath.points.length
      }

      const response = await fetch('/api/optimize-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pylink_data: {
            ...context.linkageDoc,
            n_steps: context.simulationSteps
          },
          target_path: {
            joint_name: targetTrajectory.joint_name,
            positions: targetTrajectory.positions
          },
          optimization_options: optimizationOptions
        })
      })

      if (!response.ok) {
        const errorResult = await response.json().catch(() => ({ status: 'error', message: `HTTP error! status: ${response.status}` }))
        context.showStatus(`Optimization failed: ${errorResult.message || `HTTP ${response.status}`}`, 'error', 6000)
        setOptimizationResult({
          success: false,
          initialError: null,
          finalError: null,
          message: errorResult.message || `HTTP ${response.status}`,
          iterations: 0,
          executionTimeMs: 0,
          optimizedDimensions: {},
          originalDimensions: originalDims
        })
        return
      }

      const result = await response.json()

      // Handle API-level errors
      if (result.status === 'error') {
        const errorMsg = result.message || result.error_type || 'Optimization failed'
        context.showStatus(`Optimization failed: ${errorMsg}`, 'error', 6000)
        setOptimizationResult({
          success: false,
          initialError: 0,
          finalError: 0,
          message: errorMsg,
          iterations: 0,
          executionTimeMs: result.execution_time_ms || 0,
          optimizedDimensions: {},
          originalDimensions: originalDims
        })
        return
      }

      if (result.status === 'success' && result.result) {
        const optResult = result.result

        // CRITICAL VALIDATION: Fail hard if optimized_pylink_data is missing
        if (!optResult.optimized_pylink_data) {
          const errorMsg = 'CRITICAL ERROR: Optimizer returned success but no optimized_pylink_data'
          console.error(errorMsg, optResult)
          context.showStatus(errorMsg, 'error', 10000)
          setOptimizationResult({
            success: false,
            initialError: 0,
            finalError: 0,
            message: errorMsg,
            iterations: 0,
            executionTimeMs: result.execution_time_ms || 0,
            optimizedDimensions: {},
            originalDimensions: originalDims
          })
          setIsOptimizing(false)
          throw new Error(errorMsg)
        }

        const resultData = optResult.optimized_pylink_data

        // CRITICAL VALIDATION: Ensure it's hypergraph format
        if (!context.isHypergraphFormat(resultData)) {
          const errorMsg = 'CRITICAL ERROR: Backend returned unexpected format (not hypergraph)'
          console.error(errorMsg, resultData)
          context.showStatus(errorMsg, 'error', 10000)
          setOptimizationResult({
            success: false,
            initialError: 0,
            finalError: 0,
            message: errorMsg,
            iterations: 0,
            executionTimeMs: result.execution_time_ms || 0,
            optimizedDimensions: {},
            originalDimensions: originalDims
          })
          setIsOptimizing(false)
          throw new Error(errorMsg)
        }

        // CRITICAL VALIDATION: Ensure basic structure exists
        if (!resultData.linkage || !resultData.linkage.nodes || !resultData.linkage.edges) {
          const errorMsg = 'CRITICAL ERROR: Optimized mechanism missing required structure (linkage.nodes or linkage.edges)'
          console.error(errorMsg, resultData)
          context.showStatus(errorMsg, 'error', 10000)
          setOptimizationResult({
            success: false,
            initialError: 0,
            finalError: 0,
            message: errorMsg,
            iterations: 0,
            executionTimeMs: result.execution_time_ms || 0,
            optimizedDimensions: {},
            originalDimensions: originalDims
          })
          setIsOptimizing(false)
          throw new Error(errorMsg)
        }

        // CRITICAL VALIDATION: Ensure optimized_dimensions match edge distances
        if (optResult.optimized_dimensions && Object.keys(optResult.optimized_dimensions).length > 0) {
          const optimizedDims = optResult.optimized_dimensions
          const edges = resultData.linkage?.edges || {}
          const mismatches: string[] = []
          const tolerance = 1e-6

          for (const [dimName, dimValue] of Object.entries(optimizedDims)) {
            if (!dimName.endsWith('_distance')) continue

            const edgeId = dimName.replace(/_distance$/, '')
            const edge = edges[edgeId]

            if (!edge) {
              mismatches.push(`Dimension '${dimName}' maps to edge '${edgeId}' which is missing from returned edges`)
              continue
            }

            const edgeDistance = edge.distance
            if (edgeDistance === undefined || edgeDistance === null) {
              mismatches.push(`Dimension '${dimName}' maps to edge '${edgeId}' which has no distance`)
              continue
            }

            if (Math.abs(Number(dimValue) - Number(edgeDistance)) > tolerance) {
              mismatches.push(
                `Dimension '${dimName}': reported=${dimValue}, edge '${edgeId}' distance=${edgeDistance}, diff=${Math.abs(Number(dimValue) - Number(edgeDistance))}`
              )
            }
          }

          if (mismatches.length > 0) {
            const errorMsg = `CRITICAL ERROR: optimized_dimensions do not match edge distances!\nMismatches:\n${mismatches.map(m => `  - ${m}`).join('\n')}`
            console.error(errorMsg, {
              optimized_dimensions: optimizedDims,
              edge_distances: Object.fromEntries(Object.entries(edges).map(([id, e]) => [id, e.distance]))
            })
            context.showStatus(errorMsg, 'error', 15000)
            setOptimizationResult({
              success: false,
              initialError: 0,
              finalError: 0,
              message: errorMsg,
              iterations: 0,
              executionTimeMs: result.execution_time_ms || 0,
              optimizedDimensions: {},
              originalDimensions: originalDims
            })
            setIsOptimizing(false)
            throw new Error(errorMsg)
          }

          console.log(`[Optimization] ✓ Validation passed: All ${Object.keys(optimizedDims).length} optimized dimensions match edge distances`)
        }

        // CRITICAL VALIDATION: Ensure node positions match edge distances
        const nodes = resultData.linkage?.nodes || {}
        const edges = resultData.linkage?.edges || {}
        const positionMismatches: string[] = []
        const tolerance = 1e-3

        for (const [edgeId, edge] of Object.entries(edges)) {
          const sourceNode = nodes[edge.source]
          const targetNode = nodes[edge.target]

          if (!sourceNode || !targetNode) continue

          const storedDistance = edge.distance
          if (storedDistance === undefined || storedDistance === null) continue

          const sourcePos = sourceNode.position
          const targetPos = targetNode.position
          if (!sourcePos || !targetPos || sourcePos.length < 2 || targetPos.length < 2) continue

          const calculatedDistance = Math.sqrt(
            Math.pow(targetPos[0] - sourcePos[0], 2) +
            Math.pow(targetPos[1] - sourcePos[1], 2)
          )

          const diff = Math.abs(calculatedDistance - storedDistance)
          if (diff > tolerance) {
            positionMismatches.push(
              `Edge ${edgeId}: stored distance=${storedDistance.toFixed(3)}, calculated from positions=${calculatedDistance.toFixed(3)}, diff=${diff.toFixed(3)}`
            )
          }
        }

        if (positionMismatches.length > 0) {
          const errorMsg = `CRITICAL ERROR: Node positions do not match edge distances!\nMismatches:\n${positionMismatches.map(m => `  - ${m}`).join('\n')}\nThis indicates inconsistent data - positions must match edge distances!`
          console.error(errorMsg, { resultData })
          context.showStatus(errorMsg, 'error', 15000)
          setOptimizationResult({
            success: false,
            initialError: 0,
            finalError: 0,
            message: errorMsg,
            iterations: 0,
            executionTimeMs: result.execution_time_ms || 0,
            optimizedDimensions: {},
            originalDimensions: originalDims
          })
          setIsOptimizing(false)
          throw new Error(errorMsg)
        }

        console.log(`[Optimization] ✓ Position validation passed: All ${Object.keys(edges).length} edge distances match node positions`)

        // CRITICAL: Apply optimized mechanism - this is the ONLY source of truth
        context.logMechanismState('Before optimization', context.linkageDoc)
        context.logMechanismState('After optimization (optimizer result)', resultData)

        console.log('[Optimization] Applying optimized mechanism:', {
          nodeCount: Object.keys(resultData.linkage.nodes || {}).length,
          edgeCount: Object.keys(resultData.linkage.edges || {}).length,
          success: optResult.success
        })

        // Track optimizer result for verification and sync
        lastOptimizerResultRef.current = resultData

        // Apply the optimized mechanism
        context.setLinkageDoc(resultData)
        context.triggerMechanismChange()

        // CRITICAL: Keep preOptimizationDoc so user can revert
        // Don't clear it - the revert button needs it
        if (optResult.success) {
          // Keep preOptimizationDoc for revert functionality
          setIsOptimizedMechanism(true)
          setIsSyncedToOptimizer(true)

          // Lock mechanism for 2 seconds to prevent accidental modifications
          optimizationJustCompletedRef.current = true
          optimizationCompleteTimeRef.current = Date.now()
          setTimeout(() => {
            optimizationJustCompletedRef.current = false
            optimizationCompleteTimeRef.current = null
            console.log('[Optimization] Mechanism lock released after 2 seconds')
          }, 2000)

          console.log('[Optimization] Applied optimized mechanism - mechanism is synced to optimizer result and locked for 2 seconds')
        }

        // Safely calculate improvement
        const initialError = optResult.initial_error
        const finalError = optResult.final_error
        const hasValidErrors = (
          initialError != null &&
          typeof initialError === 'number' &&
          !isNaN(initialError) &&
          isFinite(initialError) &&
          finalError != null &&
          typeof finalError === 'number' &&
          !isNaN(finalError) &&
          isFinite(finalError) &&
          initialError > 0
        )

        const improvement = hasValidErrors
          ? ((1 - finalError / initialError) * 100)
          : 0

        setOptimizationResult({
          success: optResult.success,
          initialError: initialError ?? null,
          finalError: finalError ?? null,
          message: result.message || optResult.error || 'Optimization completed',
          iterations: optResult.iterations,
          executionTimeMs: result.execution_time_ms,
          optimizedDimensions: optResult.optimized_dimensions || {},
          originalDimensions: originalDims
        })

        if (optResult.success && hasValidErrors) {
          context.showStatus(`Optimization complete: ${improvement.toFixed(1)}% improvement`, 'success', 4000)
        } else if (optResult.success) {
          context.showStatus('Optimization complete', 'success', 4000)
        } else {
          const errorMsg = optResult.error || optResult.error_type || 'Optimization failed'
          context.showStatus(`Optimization failed: ${errorMsg}`, 'error', 6000)
        }
      } else {
        // Optimization failed - keep preOptimizationDoc for potential revert
        setOptimizationResult({
          success: false,
          initialError: 0,
          finalError: 0,
          message: result.message || 'Optimization failed',
          iterations: 0,
          executionTimeMs: 0,
          optimizedDimensions: {},
          originalDimensions: originalDims
        })
        context.showStatus(result.message || 'Optimization failed', 'error', 3000)
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
      context.showStatus(`Optimization error: ${error}`, 'error', 3000)
    } finally {
      setIsOptimizing(false)
    }
  }, [context, optMethod, optNParticles, optIterations, optMaxIterations, optTolerance, optVerbose, serializeMechVariationConfig])

  // Sync mechanism to optimizer result
  const syncToOptimizerResult = useCallback(() => {
    if (!lastOptimizerResultRef.current) {
      context.showStatus('No optimizer result available to sync to', 'warning', 3000)
      return
    }

    const optimizerDoc = lastOptimizerResultRef.current
    console.log('[Optimization] Syncing mechanism to optimizer result')
    context.setLinkageDoc(optimizerDoc)
    context.triggerMechanismChange()
    setIsSyncedToOptimizer(true)
    context.showStatus('Mechanism synced to optimizer result', 'success', 2000)
  }, [context])

  // ═══════════════════════════════════════════════════════════════════════════════
  // DIMENSION FETCHING
  // ═══════════════════════════════════════════════════════════════════════════════

  // Fetch dimensions after simulation completes
  // CRITICAL: Use refs to access context values to prevent infinite loops
  // The callback should be stable and not recreate on every render
  const handleSimulationComplete = useCallback((data: TrajectoryData) => {
    // Access latest context values via ref to avoid dependency on changing object
    const currentContext = contextRef.current
    if (!currentContext.linkageDoc) return

    // Generate stable key for this trajectory
    const generateKey = () => {
      try {
        // LinkageDocument has name, linkage.nodes, linkage.edges structure
        // Use pylinkDoc for joint count (converted legacy format)
        const keyParts = {
          name: currentContext.linkageDoc.name,
          jointCount: currentContext.pylinkDoc.pylinkage.joints.length,
          edgeCount: Object.keys(currentContext.linkageDoc.linkage?.edges || {}).length,
          trajectorySteps: data.nSteps,
          trajectoryJoints: Object.keys(data.trajectories || {}).length
        }
        return JSON.stringify(keyParts)
      } catch {
        return null
      }
    }

    const trajectoryKey = generateKey()
    if (!trajectoryKey) return

    // Skip if already fetched for this trajectory
    if (trajectoryKey === prevTrajectoryKeyRef.current) {
      console.log('[OptimizationController] Already fetched dimensions for this trajectory, skipping')
      return
    }

    // Clear any existing timer
    if (dimensionFetchTimerRef.current) {
      clearTimeout(dimensionFetchTimerRef.current)
    }

    // Fetch with delay (auto-simulate delay + a few ms)
    const fetchDelay = currentContext.autoSimulateDelayMs + 50
    console.log('[OptimizationController] Scheduling dimension fetch after', fetchDelay, 'ms')
    dimensionFetchTimerRef.current = setTimeout(async () => {
      setIsLoadingDimensions(true)
      setDimensionInfoError(null)

      try {
        // Access latest context values inside timeout (they may have changed)
        const latestContext = contextRef.current
        // Validate linkageDoc has required structure before calling API
        if (!latestContext.linkageDoc || typeof latestContext.linkageDoc !== 'object' || !('linkage' in latestContext.linkageDoc)) {
          setDimensionInfoError('Invalid linkage structure - missing linkage field')
          setIsLoadingDimensions(false)
          return
        }

        // Send LinkageDocument directly (same format as compute-pylink-trajectory)
        // Backend expects hypergraph format with linkage.nodes and linkage.edges
        const response = await fetch('/api/get-optimizable-dimensions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ pylink_data: latestContext.linkageDoc })
        })

        if (response.ok) {
          const result = await response.json()
          if (result.status === 'success' && result.dimension_bounds_spec) {
            const spec = result.dimension_bounds_spec
            setDimensionInfo({
              names: spec.names || [],
              initial_values: spec.initial_values || [],
              bounds: spec.bounds || [],
              n_dimensions: spec.names?.length || 0
            })
            prevTrajectoryKeyRef.current = trajectoryKey
            console.log('[OptimizationController] Successfully fetched dimensions:', spec.names?.length || 0)
          } else {
            setDimensionInfoError(result.message || 'Failed to load dimensions')
          }
        } else {
          setDimensionInfoError('Failed to fetch dimensions')
        }
      } catch (error) {
        console.error('[OptimizationController] Error fetching dimensions:', error)
        setDimensionInfoError('Error fetching dimensions')
      } finally {
        setIsLoadingDimensions(false)
      }
    }, fetchDelay)
  }, []) // Empty deps - callback is stable, uses refs to access latest values

  return {
    // State
    isOptimizing,
    preOptimizationDoc,
    optimizationResult,
    isOptimizedMechanism,
    isSyncedToOptimizer,
    optMethod,
    optNParticles,
    optIterations,
    optMaxIterations,
    optTolerance,
    optBoundsFactor,
    optMinLength,
    optVerbose,
    prepEnableSmooth,
    prepSmoothWindow,
    prepSmoothPolyorder,
    prepSmoothMethod,
    prepEnableResample,
    prepTargetNSteps,
    prepResampleMethod,
    isPreprocessing,
    preprocessResult,
    dimensionInfo,
    isLoadingDimensions,
    dimensionInfoError,

    // Setters
    setIsOptimizing,
    setPreOptimizationDoc,
    setOptimizationResult,
    setIsOptimizedMechanism,
    setIsSyncedToOptimizer,
    setOptMethod,
    setOptNParticles,
    setOptIterations,
    setOptMaxIterations,
    setOptTolerance,
    setOptBoundsFactor,
    setOptMinLength,
    setOptVerbose,
    setPrepEnableSmooth,
    setPrepSmoothWindow,
    setPrepSmoothPolyorder,
    setPrepSmoothMethod,
    setPrepEnableResample,
    setPrepTargetNSteps,
    setPrepResampleMethod,
    setIsPreprocessing,
    setPreprocessResult,
    setDimensionInfo,
    setIsLoadingDimensions,
    setDimensionInfoError,

    // Functions
    revertOptimization,
    preprocessTrajectory,
    runOptimization,
    syncToOptimizerResult,
    handleSimulationComplete,
    // Refs
    lastOptimizerResultRef,
    optimizationJustCompletedRef,
    optimizationCompleteTimeRef,
  }
}
