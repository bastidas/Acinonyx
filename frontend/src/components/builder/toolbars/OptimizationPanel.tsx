/**
 * Optimization Panel
 *
 * Configures optimization bounds, method, parameters, and runs optimization.
 * Separated from target trajectory management for better maintainability.
 */

import React, { useMemo, useState } from 'react'
import {
  Box, Typography, Button, Tooltip, Divider, FormControl, Select, MenuItem,
  TextField, FormControlLabel, Switch, Dialog, DialogTitle, DialogContent,
  DialogActions, Accordion, AccordionSummary, AccordionDetails, Stack,
  IconButton, CircularProgress
} from '@mui/material'
import { Settings, Close, ExpandMore } from '@mui/icons-material'
import { colors } from '../../../theme'
import { canSimulate, type TrajectoryData } from '../../AnimateSimulate'
import type { PylinkJoint } from '../types'
import { MIN_SIMULATION_STEPS, MAX_SIMULATION_STEPS } from '../constants'
// OptimizationBoundsConfig is no longer used inline - only in dialog
import { DimensionVariationConfig } from './DimensionVariationConfig'
import { StaticJointMovementConfig } from './StaticJointMovementConfig'

// Removed module-level state - using React.memo instead to prevent remounting

// Types (avoid circular dependency by defining locally)
export type OptMethod = 'pylinkage' | 'scipy' | 'powell' | 'nelder-mead' | 'scip'

export interface TargetPath {
  id: string
  name: string
  points: [number, number][]
  color: string
  targetJoint?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface OptimizationResult {
  success: boolean
  initialError: number
  finalError: number
  iterations?: number
  executionTimeMs?: number
  optimizedDimensions?: Record<string, number>
  originalDimensions?: Record<string, number>
  message?: string
}

export interface MechVariationConfig {
  dimension_variation: {
    default_variation_range: number
    default_enabled: boolean
    dimension_overrides: Record<string, [boolean, number, number]>
    exclude_dimensions: string[]
  }
  static_joint_movement?: {
    enabled: boolean
    max_x_movement: number
    max_y_movement: number
    joint_overrides: Record<string, [boolean, number, number]>
    linked_joints: [string, string][]
  }
  topology_changes?: {
    enabled: boolean
    add_node_probability: number
    remove_node_probability: number
    add_link_probability: number
    remove_link_probability: number
    min_nodes: number
    max_nodes: number
    preserve_crank: boolean
  }
  max_attempts?: number
  fallback_ranges?: number[]
  random_seed?: number | null
}

export interface DimensionInfo {
  names: string[]
  initial_values: number[]
  bounds: [number, number][]
  n_dimensions: number
}

export interface OptimizationPanelProps {
  // Mechanism data
  joints: PylinkJoint[]
  linkageDoc?: unknown
  trajectoryData?: TrajectoryData | null
  stretchingLinks?: string[]

  // Target paths (for optimization)
  targetPaths: TargetPath[]
  selectedPathId: string | null
  targetJoint: string | null
  onTargetJointChange: (joint: string | null) => void

  // Dimension info (fetched by parent after trajectory computation)
  dimensionInfo?: DimensionInfo | null
  isLoadingDimensions?: boolean
  dimensionInfoError?: string | null

  // Simulation steps
  simulationSteps: number
  simulationStepsInput: string
  setSimulationStepsInput: (value: string) => void

  // Optimization method
  optMethod: OptMethod
  setOptMethod: (method: OptMethod) => void

  // Method-specific parameters
  optNParticles: number
  setOptNParticles: (n: number) => void
  optIterations: number
  setOptIterations: (n: number) => void
  optMaxIterations: number
  setOptMaxIterations: (n: number) => void
  optTolerance: number
  setOptTolerance: (tol: number) => void
  optInertia: number
  setOptInertia: (n: number) => void
  optC1: number
  setOptC1: (n: number) => void
  optC2: number
  setOptC2: (n: number) => void
  optDiscretizationSteps: number
  setOptDiscretizationSteps: (n: number) => void
  optTimeLimit: number
  setOptTimeLimit: (n: number) => void
  optGapLimit: number
  setOptGapLimit: (n: number) => void

  // Execution
  isOptimizing: boolean
  runOptimization: (config: MechVariationConfig) => void
  optimizationResult: OptimizationResult | null
  preOptimizationDoc: unknown
  revertOptimization: () => void
  syncToOptimizerResult?: () => void
  isSyncedToOptimizer?: boolean
}

// Method descriptions: compact tooltip with class, scope, and what it does
const methodDescriptions: Record<string, { name: string; class: string; scope: string; summary: string }> = {
  'pylinkage': {
    name: 'Particle Swarm',
    class: 'Swarm / population-based',
    scope: 'Global',
    summary: 'Particles explore the space and share best positions. Good for non-convex and linkage-specific tuning.'
  },
  'scipy': {
    name: 'L-BFGS-B (SciPy)',
    class: 'Gradient-based (quasi-Newton)',
    scope: 'Local',
    summary: 'Bounded quasi-Newton. Fast convergence when the landscape is smooth and a good initial guess is available.'
  },
  'powell': {
    name: "Powell's Method",
    class: 'Gradient-free (direction set)',
    scope: 'Local',
    summary: 'Minimizes along coordinate directions in sequence. No derivatives; good when gradients are noisy or unavailable.'
  },
  'nelder-mead': {
    name: 'Nelder-Mead Simplex',
    class: 'Gradient-free (simplex)',
    scope: 'Local',
    summary: 'Direct search with a simplex of points. Robust, no gradients; can handle discontinuities.'
  },
  'scip': {
    name: 'SCIP',
    class: 'Grid / discrete',
    scope: 'Global (over grid)',
    summary: 'Discretized grid search over bounds. Suited to discrete or heavily constrained dimensions; grid size limits apply.'
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const OptimizationPanel: React.FC<OptimizationPanelProps> = ({
  joints,
  linkageDoc,
  trajectoryData,
  stretchingLinks = [],
  targetPaths,
  selectedPathId,
  targetJoint,
  onTargetJointChange,
  simulationSteps,
  simulationStepsInput,
  setSimulationStepsInput,
  optMethod,
  setOptMethod,
  optNParticles,
  setOptNParticles,
  optIterations,
  setOptIterations,
  optMaxIterations,
  setOptMaxIterations,
  optTolerance,
  setOptTolerance,
  optInertia,
  setOptInertia,
  optC1,
  setOptC1,
  optC2,
  setOptC2,
  optDiscretizationSteps,
  setOptDiscretizationSteps,
  optTimeLimit,
  setOptTimeLimit,
  optGapLimit,
  setOptGapLimit,
  isOptimizing,
  runOptimization,
  optimizationResult,
  preOptimizationDoc,
  revertOptimization,
  syncToOptimizerResult,
  isSyncedToOptimizer,
  dimensionInfo,
  isLoadingDimensions = false,
  dimensionInfoError = null
}) => {
  const hasCrank = canSimulate(joints)
  const selectedPath = targetPaths.find(p => p.id === selectedPathId)
  const canOptimize = Boolean(selectedPath && targetJoint && hasCrank && !isOptimizing)
  const currentMethodInfo = methodDescriptions[optMethod]

  // Dialog state
  const [showOptimizationBoundsDialog, setShowOptimizationBoundsDialog] = useState(false)
  const [dimensionSyncStates, setDimensionSyncStates] = useState<Record<string, boolean>>({})

  // Optimization config state (local state like TargetTrajectoryPanel)
  const defaultOptimizationConfig: MechVariationConfig = {
    dimension_variation: {
      default_variation_range: 0.5,
      default_enabled: true,
      dimension_overrides: {},
      exclude_dimensions: []
    },
    static_joint_movement: {
      enabled: false,
      max_x_movement: 10.0,
      max_y_movement: 10.0,
      joint_overrides: {},
      linked_joints: []
    },
    topology_changes: {
      enabled: false,
      add_node_probability: 0.0,
      remove_node_probability: 0.0,
      add_link_probability: 0.0,
      remove_link_probability: 0.0,
      min_nodes: 3,
      max_nodes: 32,
      preserve_crank: true
    },
    max_attempts: 128,
    fallback_ranges: [0.15, 0.15, 0.15],
    random_seed: null
  }

  const [optimizationConfig, setOptimizationConfig] = useState<MechVariationConfig>(defaultOptimizationConfig)

  const labelWidth = 140

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
      {/* Right column: N_STEPS first (label left, control right) */}
      <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
        <Tooltip title="Number of trajectory points for simulation and optimization. Should match preprocessed trajectory points." placement="top">
          <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>
            Trajectory steps
          </Typography>
        </Tooltip>
        <TextField
          type="number"
          size="small"
          value={simulationStepsInput}
          onChange={(e) => setSimulationStepsInput(e.target.value)}
          inputProps={{ min: MIN_SIMULATION_STEPS, max: MAX_SIMULATION_STEPS, step: 4 }}
          sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
        />
        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.75rem' }}>
          {MIN_SIMULATION_STEPS}–{MAX_SIMULATION_STEPS}
        </Typography>
      </Stack>

      {/* Configure Optimization Bounds — button only, no section label */}
      <Button
        size="small"
        variant="outlined"
        startIcon={<Settings />}
        onClick={() => setShowOptimizationBoundsDialog(true)}
        sx={{ fontSize: '0.75rem', textTransform: 'none', alignSelf: 'flex-start' }}
      >
        Configure Optimization Bounds
      </Button>

      <Divider />

      {/* Method: label left, selector right */}
      <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
        <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, fontSize: '0.8rem' }}>
          Method
        </Typography>
        <Tooltip
          title={
            <Box component="span" sx={{ display: 'block', maxWidth: 300 }}>
              <strong>{currentMethodInfo.name}</strong>
              <br />
              <span style={{ opacity: 0.9 }}>{currentMethodInfo.class} · {currentMethodInfo.scope}</span>
              <br />
              {currentMethodInfo.summary}
            </Box>
          }
          placement="right"
        >
          <FormControl size="small" sx={{ minWidth: 180 }}>
            <Select
              value={optMethod}
              onChange={(e) => setOptMethod(e.target.value as OptMethod)}
              sx={{ fontSize: '0.85rem' }}
            >
              <MenuItem value="pylinkage">Particle Swarm</MenuItem>
              <MenuItem value="scipy">L-BFGS-B (SciPy)</MenuItem>
              <MenuItem value="powell">Powell's Method</MenuItem>
              <MenuItem value="nelder-mead">Nelder-Mead Simplex</MenuItem>
              <MenuItem value="scip">SCIP</MenuItem>
            </Select>
          </FormControl>
        </Tooltip>
      </Stack>

      <Divider />

      {/* Hyperparameters — compact: label + control on same row; iterations first */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#7b1fa2', mb: 0.5, fontSize: '0.8rem' }}>
          Hyperparameters
        </Typography>

        {/* Particle Swarm (pylinkage): Iterations first, then rest */}
        {optMethod === 'pylinkage' && (
          <>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Iterations for the swarm. Typical: 256–1024" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Iterations</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optIterations}
                onChange={(e) => setOptIterations(Math.max(10, Math.min(10000, parseInt(e.target.value) || 512)))}
                inputProps={{ min: 10, max: 10000, step: 64 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Swarm size. Typical: 20–50" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Particles</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optNParticles}
                onChange={(e) => setOptNParticles(Math.max(5, Math.min(1024, parseInt(e.target.value) || 32)))}
                inputProps={{ min: 5, max: 1024, step: 16 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Inertia weight (w). Typical: 0.5–0.9" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Inertia (w)</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optInertia}
                onChange={(e) => setOptInertia(Math.max(0.1, Math.min(2, parseFloat(e.target.value) || 0.7)))}
                inputProps={{ min: 0.1, max: 2, step: 0.1 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Cognitive coefficient (c1)" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>C1 (leader)</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optC1}
                onChange={(e) => setOptC1(Math.max(0, Math.min(5, parseFloat(e.target.value) || 1.5)))}
                inputProps={{ min: 0, max: 5, step: 0.1 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Social coefficient (c2)" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>C2 (follower)</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optC2}
                onChange={(e) => setOptC2(Math.max(0, Math.min(5, parseFloat(e.target.value) || 1.5)))}
                inputProps={{ min: 0, max: 5, step: 0.1 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
          </>
        )}

        {/* SciPy (L-BFGS-B, Powell, Nelder-Mead): Max iterations first, then tolerance */}
        {(optMethod === 'scipy' || optMethod === 'powell' || optMethod === 'nelder-mead') && (
          <>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Max function evaluations" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Max iterations</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optMaxIterations}
                onChange={(e) => setOptMaxIterations(Math.max(10, Math.min(10000, parseInt(e.target.value) || (optMethod === 'powell' || optMethod === 'nelder-mead' ? 512 : 100))))}
                inputProps={{ min: 10, max: 10000, step: 50 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Convergence tolerance" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Tolerance</Typography>
              </Tooltip>
              <Select
                size="small"
                value={optTolerance}
                onChange={(e) => setOptTolerance(e.target.value as number)}
                sx={{ minWidth: 140, fontSize: '0.85rem' }}
              >
                <MenuItem value={1e-4}>1e-4</MenuItem>
                <MenuItem value={1e-5}>1e-5</MenuItem>
                <MenuItem value={1e-6}>1e-6</MenuItem>
                <MenuItem value={1e-7}>1e-7</MenuItem>
                <MenuItem value={1e-8}>1e-8</MenuItem>
              </Select>
            </Stack>
          </>
        )}

        {/* SCIP: Discretization steps first (budget-like), then time and gap */}
        {optMethod === 'scip' && (
          <>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Grid points per dimension" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Discretization</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optDiscretizationSteps}
                onChange={(e) => setOptDiscretizationSteps(Math.max(2, Math.min(50, parseInt(e.target.value) || 20)))}
                inputProps={{ min: 2, max: 50, step: 2 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Max solve time (seconds)" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Time limit (s)</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optTimeLimit}
                onChange={(e) => setOptTimeLimit(Math.max(0, Math.min(3600, parseInt(e.target.value) || 300)))}
                inputProps={{ min: 0, max: 3600, step: 60 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
            <Stack direction="row" alignItems="center" gap={1.5} sx={{ minHeight: 40 }}>
              <Tooltip title="Relative optimality gap (e.g. 0.01 = 1%)" placement="left">
                <Typography variant="body2" sx={{ color: 'text.secondary', minWidth: labelWidth, cursor: 'help', fontSize: '0.8rem' }}>Gap limit</Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                value={optGapLimit}
                onChange={(e) => setOptGapLimit(Math.max(0, Math.min(1, parseFloat(e.target.value) || 0.01)))}
                inputProps={{ min: 0, max: 1, step: 0.01 }}
                sx={{ width: 96, '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Stack>
          </>
        )}
      </Box>

      <Divider />

      {/* Run Optimization — MUI Button with CircularProgress when loading */}
      <Box>
        <Button
          variant="contained"
          fullWidth
          size="medium"
          onClick={() => runOptimization(optimizationConfig)}
          disabled={!canOptimize}
          startIcon={isOptimizing ? <CircularProgress size={18} color="inherit" /> : null}
          sx={{
            textTransform: 'none',
            fontSize: '0.95rem',
            fontWeight: 600,
            py: 1.1,
            backgroundColor: colors.primary,
            '&:hover': { backgroundColor: colors.primaryDark },
            '&.Mui-disabled': { backgroundColor: '#e0e0e0' }
          }}
        >
          {isOptimizing ? 'Optimizing…' : 'Run Optimization'}
        </Button>

        {/* Disabled reason */}
        {!canOptimize && !isOptimizing && (
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1, textAlign: 'center' }}>
            {!hasCrank ? 'Need a valid mechanism with Crank' :
             !selectedPath ? 'Select a target path above' :
             !targetJoint ? 'Select a joint to optimize' :
             'Ready to optimize'}
          </Typography>
        )}

        {/* Warning if path points don't match simulation steps */}
        {canOptimize && selectedPath && selectedPath.points.length !== simulationSteps && (
          <Box sx={{
            mt: 1,
            p: 1,
            borderRadius: 1,
            bgcolor: 'rgba(237, 108, 2, 0.1)',
            border: '1px solid #ed6c02'
          }}>
            <Typography variant="caption" sx={{ color: '#ed6c02', display: 'flex', alignItems: 'center', gap: 0.5 }}>
              Warning: path has {selectedPath.points.length} points but simulation uses {simulationSteps}. Preprocess or adjust N_STEPS for best results.
            </Typography>
          </Box>
        )}
      </Box>

      <Divider />

      {/* Optimization Results */}
      {optimizationResult && (
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: optimizationResult.success ? '#2e7d32' : '#d32f2f' }}>
              {optimizationResult.success ? 'Results' : 'Optimization Failed'}
            </Typography>
            {(preOptimizationDoc != null) && optimizationResult.success && (
              <Button
                size="small"
                variant="outlined"
                color="warning"
                onClick={revertOptimization}
                sx={{
                  textTransform: 'none',
                  fontSize: '0.7rem',
                  py: 0.25,
                  px: 1,
                  minWidth: 'auto'
                }}
              >
                Revert
              </Button>
            )}
          </Box>

          <Box sx={{
            p: 1.5,
            borderRadius: 1,
            bgcolor: optimizationResult.success ? 'rgba(46, 125, 50, 0.08)' : 'rgba(211, 47, 47, 0.08)',
            border: '1px solid',
            borderColor: optimizationResult.success ? '#4caf50' : '#f44336'
          }}>
            {optimizationResult.success ? (
              <>
                {/* Sync button - show if out of sync */}
                {syncToOptimizerResult && isSyncedToOptimizer === false && (
                  <Box sx={{ mb: 1.5 }}>
                    <Button
                      variant="contained"
                      color="primary"
                      size="small"
                      fullWidth
                      onClick={syncToOptimizerResult}
                      sx={{ mb: 1 }}
                    >
                      Sync to Optimizer Result
                    </Button>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', textAlign: 'center' }}>
                      Mechanism has been modified. Click to restore optimizer result.
                    </Typography>
                  </Box>
                )}
                {/* Sync status indicator */}
                {isSyncedToOptimizer !== undefined && (
                  <Box sx={{
                    mb: 1.5,
                    p: 1,
                    borderRadius: 0.5,
                    bgcolor: isSyncedToOptimizer ? 'rgba(46, 125, 50, 0.1)' : 'rgba(255, 152, 0, 0.1)',
                    border: '1px solid',
                    borderColor: isSyncedToOptimizer ? '#4caf50' : '#ff9800'
                  }}>
                    <Typography variant="caption" sx={{
                      color: isSyncedToOptimizer ? '#2e7d32' : '#f57c00',
                      fontWeight: 500,
                      display: 'flex',
                      alignItems: 'center'
                    }}>
                      {isSyncedToOptimizer ? 'Synced to optimizer' : 'Out of sync'}
                    </Typography>
                  </Box>
                )}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>Improvement</Typography>
                  <Typography variant="caption" sx={{
                    fontWeight: 700,
                    color: (optimizationResult.initialError != null &&
                            typeof optimizationResult.initialError === 'number' &&
                            !isNaN(optimizationResult.initialError) &&
                            isFinite(optimizationResult.initialError) &&
                            optimizationResult.initialError > 0 &&
                            optimizationResult.finalError != null &&
                            typeof optimizationResult.finalError === 'number' &&
                            !isNaN(optimizationResult.finalError) &&
                            isFinite(optimizationResult.finalError))
                      ? ((1 - optimizationResult.finalError / optimizationResult.initialError) * 100 > 50 ? '#2e7d32' : '#ed6c02')
                      : '#666'
                  }}>
                    {optimizationResult.initialError != null &&
                     typeof optimizationResult.initialError === 'number' &&
                     !isNaN(optimizationResult.initialError) &&
                     isFinite(optimizationResult.initialError) &&
                     optimizationResult.initialError > 0 &&
                     optimizationResult.finalError != null &&
                     typeof optimizationResult.finalError === 'number' &&
                     !isNaN(optimizationResult.finalError) &&
                     isFinite(optimizationResult.finalError)
                      ? `${((1 - optimizationResult.finalError / optimizationResult.initialError) * 100).toFixed(1)}%`
                      : 'N/A'}
                  </Typography>
                </Box>
                {optimizationResult.iterations && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>Iterations</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {optimizationResult.iterations}
                    </Typography>
                  </Box>
                )}
                {optimizationResult.executionTimeMs && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>Time</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                      {(optimizationResult.executionTimeMs / 1000).toFixed(2)}s
                    </Typography>
                  </Box>
                )}

                {/* Dimension changes */}
                {optimizationResult.optimizedDimensions && optimizationResult.originalDimensions &&
                 Object.keys(optimizationResult.optimizedDimensions).length > 0 && (
                  <Box sx={{ mt: 1.5, pt: 1.5, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                    <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 1 }}>
                      Dimension Changes
                    </Typography>
                    <Box sx={{
                      display: 'grid',
                      gridTemplateColumns: '1fr auto auto auto',
                      gap: 0.5,
                      fontSize: '0.65rem',
                      '& > *': { py: 0.25 }
                    }}>
                      {/* Header */}
                      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: '0.6rem' }}>
                        Parameter
                      </Typography>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#ff7043', fontSize: '0.6rem', textAlign: 'right' }}>
                        Before
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem', textAlign: 'center' }}>
                        →
                      </Typography>
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#2e7d32', fontSize: '0.6rem', textAlign: 'right' }}>
                        After
                      </Typography>

                      {/* Data rows */}
                      {Object.entries(optimizationResult.optimizedDimensions).map(([name, newValue]) => {
                        const oldValue = optimizationResult.originalDimensions?.[name] ?? newValue
                        const changed = Math.abs((newValue as number) - (oldValue as number)) > 0.001
                        return (
                          <React.Fragment key={name}>
                            <Typography variant="caption" sx={{
                              color: 'text.secondary',
                              fontSize: '0.65rem',
                              fontWeight: changed ? 500 : 400
                            }}>
                              {name.replace(/_/g, ' ')}
                            </Typography>
                            <Typography variant="caption" sx={{
                              fontFamily: 'monospace',
                              fontSize: '0.65rem',
                              textAlign: 'right',
                              color: '#ff7043'
                            }}>
                              {(oldValue as number).toFixed(3)}
                            </Typography>
                            <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem', textAlign: 'center' }}>
                              →
                            </Typography>
                            <Typography variant="caption" sx={{
                              fontFamily: 'monospace',
                              fontSize: '0.65rem',
                              textAlign: 'right',
                              color: changed ? '#2e7d32' : 'text.secondary',
                              fontWeight: changed ? 600 : 400
                            }}>
                              {(newValue as number).toFixed(3)}
                            </Typography>
                          </React.Fragment>
                        )
                      })}
                    </Box>
                  </Box>
                )}
              </>
            ) : (
              <Typography variant="caption" sx={{ color: '#d32f2f' }}>
                {optimizationResult.message || 'Optimization failed'}
              </Typography>
            )}
          </Box>
        </Box>
      )}

      {/* Optimization Bounds Config Dialog */}
      <Dialog
        open={showOptimizationBoundsDialog}
        onClose={(event, reason) => {
          // Blur any focused elements before closing to prevent aria-hidden warning
          if (document.activeElement instanceof HTMLElement) {
            document.activeElement.blur()
          }
          setShowOptimizationBoundsDialog(false)
        }}
        maxWidth="md"
        fullWidth
        disableRestoreFocus
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Optimization Bounds Config</Typography>
            <IconButton size="small" onClick={() => setShowOptimizationBoundsDialog(false)}>
              <Close />
            </IconButton>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 1 }}>
            {/* Dimension Variation */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Dimension Variation
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                {dimensionInfo ? (
                  <DimensionVariationConfig
                    dimensionInfo={dimensionInfo}
                    dimensionOverrides={optimizationConfig.dimension_variation.dimension_overrides}
                    dimensionSyncStates={dimensionSyncStates}
                    defaultEnabled={optimizationConfig.dimension_variation.default_enabled}
                    defaultVariationRange={optimizationConfig.dimension_variation.default_variation_range}
                    onOverrideChange={(name, override) => {
                      setOptimizationConfig(prev => ({
                        ...prev,
                        dimension_variation: {
                          ...prev.dimension_variation,
                          dimension_overrides: {
                            ...prev.dimension_variation.dimension_overrides,
                            [name]: override
                          }
                        }
                      }))
                    }}
                    onSyncStateChange={(name, synced) => {
                      setDimensionSyncStates(prev => ({
                        ...prev,
                        [name]: synced
                      }))
                    }}
                    onDefaultEnabledChange={(enabled) => {
                      setOptimizationConfig(prev => ({
                        ...prev,
                        dimension_variation: { ...prev.dimension_variation, default_enabled: enabled }
                      }))
                    }}
                    onDefaultRangeChange={(range) => {
                      setOptimizationConfig(prev => ({
                        ...prev,
                        dimension_variation: { ...prev.dimension_variation, default_variation_range: range }
                      }))
                    }}
                    compact={true}
                    showTopControls={true}
                  />
                ) : (
                  <Box sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      {isLoadingDimensions ? 'Loading dimensions...' : dimensionInfoError || 'Loading dimensions...'}
                    </Typography>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>

            {/* Static Joint Movement */}
            <Accordion defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Static Joint Movement
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <StaticJointMovementConfig
                  joints={joints}
                  enabled={optimizationConfig.static_joint_movement?.enabled ?? false}
                  defaultMaxX={optimizationConfig.static_joint_movement?.max_x_movement ?? 10.0}
                  defaultMaxY={optimizationConfig.static_joint_movement?.max_y_movement ?? 10.0}
                  jointOverrides={optimizationConfig.static_joint_movement?.joint_overrides ?? {}}
                  onEnabledChange={(enabled) => {
                    setOptimizationConfig(prev => ({
                      ...prev,
                      static_joint_movement: {
                        ...prev.static_joint_movement,
                        enabled,
                        max_x_movement: prev.static_joint_movement?.max_x_movement ?? 10.0,
                        max_y_movement: prev.static_joint_movement?.max_y_movement ?? 10.0,
                        joint_overrides: prev.static_joint_movement?.joint_overrides ?? {},
                        linked_joints: prev.static_joint_movement?.linked_joints ?? []
                      }
                    }))
                  }}
                  onDefaultMaxXChange={() => {}}
                  onDefaultMaxYChange={() => {}}
                  onJointOverrideChange={(jointName, override) => {
                    setOptimizationConfig(prev => ({
                      ...prev,
                      static_joint_movement: {
                        enabled: prev.static_joint_movement?.enabled ?? false,
                        max_x_movement: prev.static_joint_movement?.max_x_movement ?? 10.0,
                        max_y_movement: prev.static_joint_movement?.max_y_movement ?? 10.0,
                        joint_overrides: {
                          ...(prev.static_joint_movement?.joint_overrides ?? {}),
                          [jointName]: override
                        },
                        linked_joints: prev.static_joint_movement?.linked_joints ?? []
                      }
                    }))
                  }}
                  compact={true}
                />
              </AccordionDetails>
            </Accordion>

            {/* Topology Changes (Not Implemented) */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, color: 'text.secondary' }}>
                  Topology Changes (Not Implemented)
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box sx={{ p: 2, bgcolor: 'rgba(0,0,0,0.02)', borderRadius: 1, border: '1px dashed rgba(0,0,0,0.2)' }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic', display: 'block', mb: 2 }}>
                    Future functionality: Add/remove links and nodes during optimization.
                    This feature is not yet implemented.
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={optimizationConfig.topology_changes?.enabled ?? false}
                        disabled
                        onChange={() => {}}
                      />
                    }
                    label="Enable Topology Changes (Disabled - Not Implemented)"
                  />
                </Box>
              </AccordionDetails>
            </Accordion>

            {/* Advanced Settings */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  Advanced Settings
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Stack spacing={2}>
                  <TextField
                    type="number"
                    size="small"
                    fullWidth
                    label="Max Attempts"
                    value={optimizationConfig.max_attempts}
                    onChange={(e) => {
                      setOptimizationConfig(prev => ({
                        ...prev,
                        max_attempts: parseInt(e.target.value) || 128
                      }))
                    }}
                    inputProps={{ min: 1, max: 1000 }}
                    helperText="Maximum attempts to find a valid configuration per range"
                  />
                  <Box>
                    <Typography variant="caption" sx={{ display: 'block', mb: 1 }}>
                      Fallback Ranges (comma-separated)
                    </Typography>
                    <TextField
                      size="small"
                      fullWidth
                      value={(optimizationConfig.fallback_ranges ?? [0.15, 0.15, 0.15]).join(', ')}
                      onChange={(e) => {
                        const ranges = e.target.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
                        if (ranges.length > 0) {
                          setOptimizationConfig(prev => ({ ...prev, fallback_ranges: ranges }))
                        }
                      }}
                      placeholder="0.15, 0.15, 0.15"
                      helperText="Progressively smaller ranges to try if primary fails"
                    />
                  </Box>
                  <TextField
                    type="number"
                    size="small"
                    fullWidth
                    label="Random Seed (optional)"
                    value={optimizationConfig.random_seed || ''}
                    onChange={(e) => {
                      setOptimizationConfig(prev => ({
                        ...prev,
                        random_seed: e.target.value ? parseInt(e.target.value) : null
                      }))
                    }}
                    inputProps={{ min: 0 }}
                    helperText="Leave empty for random seed"
                  />
                </Stack>
              </AccordionDetails>
            </Accordion>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowOptimizationBoundsDialog(false)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            onClick={() => setShowOptimizationBoundsDialog(false)}
          >
            Done
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default OptimizationPanel
