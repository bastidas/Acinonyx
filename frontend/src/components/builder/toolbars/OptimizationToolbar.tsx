/**
 * Optimization Toolbar - Refactored
 *
 * Thin wrapper that composes TargetTrajectoryPanel and OptimizationPanel.
 * All complex logic has been moved to the individual panel components.
 *
 * Fetches dimension info ONCE after trajectory computation.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Box, Divider } from '@mui/material'
import type { TrajectoryData } from '../../AnimateSimulate'
import type { PylinkJoint } from '../types'
import type { TargetPath } from '../../BuilderTools'
import { TargetTrajectoryPanel } from './TargetTrajectoryPanel'
import { OptimizationPanel } from './OptimizationPanel'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES (re-export TargetPath for consumers that import from this module)
// ═══════════════════════════════════════════════════════════════════════════════
export type { TargetPath } from '../../BuilderTools'

export interface PreprocessResult {
  originalPoints: number
  outputPoints: number
  analysis?: {
    total_path_length?: number
    is_closed?: boolean
  }
}

/** Compatible with OptimizationController result (initialError/finalError may be null) */
export interface OptimizationResult {
  success: boolean
  initialError: number | null
  finalError: number | null
  iterations?: number
  executionTimeMs?: number
  optimizedDimensions?: Record<string, number>
  originalDimensions?: Record<string, number>
  message?: string
}

export type OptMethod = 'pylinkage' | 'scipy' | 'powell' | 'nelder-mead' | 'scip'
export type SmoothMethod = 'savgol' | 'moving_avg' | 'gaussian'
export type ResampleMethod = 'parametric' | 'cubic' | 'linear'

export interface OptimizationToolbarProps {
  // Joint data for simulation checks and selection
  joints: PylinkJoint[]

  // Mechanism data for dimension info
  linkageDoc?: unknown  // LinkageDocument or PylinkDocument

  // Trajectory data - when this exists and mechanism is valid, fetch dimensions
  trajectoryData?: TrajectoryData | null
  stretchingLinks?: string[]  // Empty array means mechanism is valid for trajectory drawing

  // Dimension info (fetched by parent after simulation completes)
  dimensionInfo?: {
    names: string[]
    initial_values: number[]
    bounds: [number, number][]
    n_dimensions: number
  } | null
  isLoadingDimensions?: boolean
  dimensionInfoError?: string | null

  // Target paths
  targetPaths: TargetPath[]
  setTargetPaths: React.Dispatch<React.SetStateAction<TargetPath[]>>
  selectedPathId: string | null
  setSelectedPathId: (id: string | null) => void

  // Preprocessing state
  preprocessResult: PreprocessResult | null
  isPreprocessing: boolean
  prepEnableSmooth: boolean
  setPrepEnableSmooth: (enable: boolean) => void
  prepSmoothMethod: SmoothMethod
  setPrepSmoothMethod: (method: SmoothMethod) => void
  prepSmoothWindow: number
  setPrepSmoothWindow: (window: number) => void
  prepSmoothPolyorder: number
  setPrepSmoothPolyorder: (order: number) => void
  prepEnableResample: boolean
  setPrepEnableResample: (enable: boolean) => void
  prepTargetNSteps: number
  setPrepTargetNSteps: (steps: number) => void
  prepResampleMethod: ResampleMethod
  setPrepResampleMethod: (method: ResampleMethod) => void
  preprocessTrajectory: () => void

  // Simulation steps
  simulationSteps: number
  simulationStepsInput: string
  setSimulationStepsInput: (value: string) => void

  // Optimization method
  optMethod: OptMethod
  setOptMethod: (method: OptMethod) => void

  // PSO parameters
  optNParticles: number
  setOptNParticles: (n: number) => void
  optIterations: number
  setOptIterations: (n: number) => void

  // SciPy parameters
  optMaxIterations: number
  setOptMaxIterations: (n: number) => void
  optTolerance: number
  setOptTolerance: (tol: number) => void

  // PSO (pylinkage) extra parameters
  optInertia: number
  setOptInertia: (n: number) => void
  optC1: number
  setOptC1: (n: number) => void
  optC2: number
  setOptC2: (n: number) => void

  // SCIP parameters
  optDiscretizationSteps: number
  setOptDiscretizationSteps: (n: number) => void
  optTimeLimit: number
  setOptTimeLimit: (n: number) => void
  optGapLimit: number
  setOptGapLimit: (n: number) => void

  // Bounds (deprecated - use optimizationConfig instead)
  optBoundsFactor: number
  setOptBoundsFactor: (factor: number) => void
  optMinLength: number
  setOptMinLength: (len: number) => void

  // Optimization config (new way)
  optimizationConfig?: Record<string, unknown> | null
  setOptimizationConfig?: (config: Record<string, unknown>) => void

  // Verbose & run
  isOptimizing: boolean
  /** Called with optional config when Run is clicked (e.g. MechVariationConfig from OptimizationPanel) */
  runOptimization: (config?: Record<string, unknown>) => void

  // Results
  optimizationResult: OptimizationResult | null
  preOptimizationDoc: unknown
  revertOptimization: () => void
  syncToOptimizerResult?: () => void
  isSyncedToOptimizer?: boolean
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const OptimizationToolbar: React.FC<OptimizationToolbarProps> = (props) => {
  // Use dimension info from props (fetched by parent after simulation completes)
  const dimensionInfo = props.dimensionInfo
  const isLoadingDimensions = props.isLoadingDimensions || false
  const dimensionInfoError = props.dimensionInfoError || null

  const [targetJoint, setTargetJointState] = useState<string | null>(null)
  const [isUserTargetJointLocked, setIsUserTargetJointLocked] = useState(false)
  const isUserTargetJointLockedRef = useRef(false)

  const selectedPathId = props.selectedPathId
  const selectedPath = useMemo(
    () => props.targetPaths.find(p => p.id === selectedPathId) || null,
    [props.targetPaths, selectedPathId]
  )

  const mechanismKey = useMemo(() => {
    if (!props.linkageDoc) return null
    try {
      const doc = props.linkageDoc as { name?: string; linkage?: { nodes?: Record<string, unknown>; edges?: Record<string, unknown> } }
      const keyParts = {
        name: doc.name,
        jointCount: Object.keys(doc.linkage?.nodes || {}).length,
        edgeCount: Object.keys(doc.linkage?.edges || {}).length
      }
      return JSON.stringify(keyParts)
    } catch {
      return null
    }
  }, [props.linkageDoc])

  const prevMechanismKeyRef = useRef<string | null>(null)
  useEffect(() => {
    // Only clear target joint when mechanism actually changed (user loaded different mechanism), not on first mount
    const mechanismChanged = prevMechanismKeyRef.current !== null && prevMechanismKeyRef.current !== mechanismKey
    prevMechanismKeyRef.current = mechanismKey
    if (mechanismChanged) {
      setIsUserTargetJointLocked(false)
      isUserTargetJointLockedRef.current = false
      setTargetJointState(null)
    }
  }, [mechanismKey])

  // Clear targetJoint when it's not in the current mechanism's joints (e.g. after load/switch) to avoid MUI Select out-of-range
  const jointNames = useMemo(() => new Set(props.joints.map(j => j.name)), [props.joints])
  useEffect(() => {
    if (targetJoint != null && !jointNames.has(targetJoint)) {
      setTargetJointState(null)
      setIsUserTargetJointLocked(false)
      isUserTargetJointLockedRef.current = false
    }
  }, [targetJoint, jointNames])

  const updateSelectedPathTargetJoint = useCallback((joint: string | null) => {
    if (!selectedPath) return
    const normalized = joint || undefined
    if (selectedPath.targetJoint === normalized) {
      return
    }
    props.setTargetPaths(prev => prev.map(path =>
      path.id === selectedPath.id ? { ...path, targetJoint: normalized } : path
    ))
  }, [selectedPath, props.setTargetPaths])

  const selectedPathTargetJoint = selectedPath?.targetJoint ?? null

  useEffect(() => {
    if (!selectedPathId) {
      // When no path is selected, leave targetJoint as-is (user selection or auto-select both stick)
      return
    }

    if (!selectedPath) {
      return
    }

    // Path has no joint but toolbar shows one: always apply toolbar to path so the visible
    // selection applies to the current path (e.g. after drawing a new trajectory). Do this
    // regardless of "user locked" so preprocessing/optimization use the visible joint.
    const jointNames = new Set(props.joints.map(j => j.name))
    if (selectedPathTargetJoint == null && targetJoint != null && jointNames.has(targetJoint)) {
      updateSelectedPathTargetJoint(targetJoint)
      return
    }

    if (selectedPathTargetJoint !== targetJoint) {
      if (isUserTargetJointLockedRef.current) {
        return
      }
      setTargetJointState(selectedPathTargetJoint)
    }
  }, [selectedPathId, selectedPath, selectedPathTargetJoint, targetJoint, props.joints, updateSelectedPathTargetJoint])

  const setTargetJointFromSystem = useCallback((joint: string | null) => {
    setIsUserTargetJointLocked(false)
    isUserTargetJointLockedRef.current = false
    setTargetJointState(joint)
    updateSelectedPathTargetJoint(joint)
  }, [updateSelectedPathTargetJoint])

  const handleTargetJointChange = useCallback((joint: string | null) => {
    const locked = Boolean(joint)
    setIsUserTargetJointLocked(locked)
    isUserTargetJointLockedRef.current = locked
    setTargetJointState(joint)
    updateSelectedPathTargetJoint(joint)
  }, [updateSelectedPathTargetJoint])

  return (
    <Box sx={{ p: 1.5, display: 'flex', gap: 2 }}>
      {/* Left Column: Target Trajectory Panel */}
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <TargetTrajectoryPanel
          joints={props.joints}
          linkageDoc={props.linkageDoc}
          trajectoryData={props.trajectoryData}
          stretchingLinks={props.stretchingLinks}
          targetPaths={props.targetPaths}
          setTargetPaths={props.setTargetPaths}
          selectedPathId={props.selectedPathId}
          setSelectedPathId={props.setSelectedPathId}
          preprocessResult={props.preprocessResult}
          isPreprocessing={props.isPreprocessing}
          prepEnableSmooth={props.prepEnableSmooth}
          setPrepEnableSmooth={props.setPrepEnableSmooth}
          prepSmoothMethod={props.prepSmoothMethod}
          setPrepSmoothMethod={props.setPrepSmoothMethod}
          prepSmoothWindow={props.prepSmoothWindow}
          setPrepSmoothWindow={props.setPrepSmoothWindow}
          prepSmoothPolyorder={props.prepSmoothPolyorder}
          setPrepSmoothPolyorder={props.setPrepSmoothPolyorder}
          prepEnableResample={props.prepEnableResample}
          setPrepEnableResample={props.setPrepEnableResample}
          prepTargetNSteps={props.prepTargetNSteps}
          setPrepTargetNSteps={props.setPrepTargetNSteps}
          prepResampleMethod={props.prepResampleMethod}
          setPrepResampleMethod={props.setPrepResampleMethod}
          preprocessTrajectory={props.preprocessTrajectory}
          simulationSteps={props.simulationSteps}
          dimensionInfo={dimensionInfo}
          isLoadingDimensions={isLoadingDimensions}
          dimensionInfoError={dimensionInfoError}
          targetJoint={targetJoint}
          setTargetJoint={setTargetJointFromSystem}
          onTargetJointChange={handleTargetJointChange}
          allowAutoTargetJointSelection={!isUserTargetJointLocked}
        />
      </Box>

      <Divider orientation="vertical" flexItem />

      {/* Right Column: Optimization Panel */}
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <OptimizationPanel
          joints={props.joints}
          linkageDoc={props.linkageDoc}
          trajectoryData={props.trajectoryData}
          stretchingLinks={props.stretchingLinks}
          targetPaths={props.targetPaths}
          selectedPathId={props.selectedPathId}
          optimizationConfig={props.optimizationConfig || undefined}
          setOptimizationConfig={props.setOptimizationConfig ? (config) => props.setOptimizationConfig?.(config as Record<string, unknown>) : undefined}
          simulationSteps={props.simulationSteps}
          simulationStepsInput={props.simulationStepsInput}
          setSimulationStepsInput={props.setSimulationStepsInput}
          optMethod={props.optMethod}
          setOptMethod={props.setOptMethod}
          optNParticles={props.optNParticles}
          setOptNParticles={props.setOptNParticles}
          optIterations={props.optIterations}
          setOptIterations={props.setOptIterations}
          optMaxIterations={props.optMaxIterations}
          setOptMaxIterations={props.setOptMaxIterations}
          optTolerance={props.optTolerance}
          setOptTolerance={props.setOptTolerance}
          optInertia={props.optInertia}
          setOptInertia={props.setOptInertia}
          optC1={props.optC1}
          setOptC1={props.setOptC1}
          optC2={props.optC2}
          setOptC2={props.setOptC2}
          optDiscretizationSteps={props.optDiscretizationSteps}
          setOptDiscretizationSteps={props.setOptDiscretizationSteps}
          optTimeLimit={props.optTimeLimit}
          setOptTimeLimit={props.setOptTimeLimit}
          optGapLimit={props.optGapLimit}
          setOptGapLimit={props.setOptGapLimit}
          isOptimizing={props.isOptimizing}
          runOptimization={props.runOptimization}
          optimizationResult={props.optimizationResult}
          preOptimizationDoc={props.preOptimizationDoc}
          revertOptimization={props.revertOptimization}
          syncToOptimizerResult={props.syncToOptimizerResult}
          isSyncedToOptimizer={props.isSyncedToOptimizer}
          dimensionInfo={dimensionInfo}
          isLoadingDimensions={isLoadingDimensions}
          dimensionInfoError={dimensionInfoError}
          targetJoint={targetJoint}
          onTargetJointChange={handleTargetJointChange}
        />
      </Box>
    </Box>
  )
}

export default OptimizationToolbar
