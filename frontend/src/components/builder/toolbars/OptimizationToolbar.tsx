/**
 * Optimization Toolbar - Path optimization settings and controls
 */
import React from 'react'
import {
  Box, Typography, Button, Tooltip, Divider, FormControl, Select, MenuItem,
  TextField, FormControlLabel, Switch, IconButton, Chip
} from '@mui/material'
import { canSimulate } from '../../AnimateSimulate'
import type { PylinkJoint } from '../types'
import { MIN_SIMULATION_STEPS, MAX_SIMULATION_STEPS } from '../constants'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface TargetPath {
  id: string
  name: string
  color: string
  points: [number, number][]
  targetJoint: string | null
  isComplete: boolean
}

export interface PreprocessResult {
  originalPoints: number
  outputPoints: number
  analysis?: {
    total_path_length?: number
    is_closed?: boolean
    [key: string]: unknown
  }
}

export interface OptimizationResult {
  success: boolean
  message?: string
  initialError: number
  finalError: number
  iterations?: number
  executionTimeMs?: number
  optimizedDimensions?: Record<string, number>
  originalDimensions?: Record<string, number>
}

export type OptMethod = 'pso' | 'pylinkage' | 'scipy' | 'powell' | 'nelder-mead'
export type SmoothMethod = 'savgol' | 'moving_avg' | 'gaussian'
export type ResampleMethod = 'parametric' | 'cubic' | 'linear'

export interface OptimizationToolbarProps {
  // Joint data for simulation checks and selection
  joints: PylinkJoint[]

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

  // Bounds
  optBoundsFactor: number
  setOptBoundsFactor: (factor: number) => void
  optMinLength: number
  setOptMinLength: (len: number) => void

  // Verbose & run
  optVerbose: boolean
  setOptVerbose: (verbose: boolean) => void
  isOptimizing: boolean
  runOptimization: () => void

  // Results
  optimizationResult: OptimizationResult | null
  preOptimizationDoc: unknown
  revertOptimization: () => void
}

// Method descriptions for tooltips
const methodDescriptions: Record<string, { name: string; description: string; pros: string; cons: string }> = {
  'pso': {
    name: 'Particle Swarm Optimization',
    description: 'Bio-inspired algorithm where particles explore the solution space, sharing information about good solutions.',
    pros: 'Robust, handles non-convex problems well, good at avoiding local minima',
    cons: 'Slower than gradient methods, requires tuning particles/iterations'
  },
  'pylinkage': {
    name: 'Pylinkage PSO',
    description: 'Native PSO implementation from the pylinkage library, optimized for linkage mechanisms.',
    pros: 'Designed specifically for linkages, well-tested',
    cons: 'Similar tradeoffs to standard PSO'
  },
  'scipy': {
    name: 'L-BFGS-B (SciPy)',
    description: 'Quasi-Newton method with bounded constraints. Uses gradient approximation for fast convergence.',
    pros: 'Very fast convergence for smooth problems',
    cons: 'Can get stuck in local minima, requires good initial guess'
  },
  'powell': {
    name: "Powell's Method",
    description: 'Direction-set method that minimizes along each coordinate direction sequentially.',
    pros: 'Gradient-free, good for noisy functions',
    cons: 'Slower than gradient methods, may not find global optimum'
  },
  'nelder-mead': {
    name: 'Nelder-Mead Simplex',
    description: 'Direct search method using a simplex of N+1 points that adapts to the function landscape.',
    pros: 'Very robust, no gradients needed, handles discontinuities',
    cons: 'Slow for high dimensions, local optimizer only'
  }
}

export const OptimizationToolbar: React.FC<OptimizationToolbarProps> = ({
  joints,
  targetPaths,
  setTargetPaths,
  selectedPathId,
  setSelectedPathId,
  preprocessResult,
  isPreprocessing,
  prepEnableSmooth,
  setPrepEnableSmooth,
  prepSmoothMethod,
  setPrepSmoothMethod,
  prepSmoothWindow,
  setPrepSmoothWindow,
  prepSmoothPolyorder,
  setPrepSmoothPolyorder,
  prepEnableResample,
  setPrepEnableResample,
  prepTargetNSteps,
  setPrepTargetNSteps,
  prepResampleMethod,
  setPrepResampleMethod,
  preprocessTrajectory,
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
  optBoundsFactor,
  setOptBoundsFactor,
  optMinLength,
  setOptMinLength,
  optVerbose,
  setOptVerbose,
  isOptimizing,
  runOptimization,
  optimizationResult,
  preOptimizationDoc,
  revertOptimization
}) => {
  const hasCrank = canSimulate(joints)
  const selectedPath = targetPaths.find(p => p.id === selectedPathId)
  const canOptimize = selectedPath && selectedPath.targetJoint && hasCrank && !isOptimizing
  const currentMethodInfo = methodDescriptions[optMethod]

  return (
    <Box sx={{ p: 2 }}>
      {/* ═══════════════════════════════════════════════════════════════════════
          TARGET PATH SELECTION
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#e91e63', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
        <span>📍</span> Target Path
      </Typography>

      {targetPaths.length > 0 ? (
        <Box sx={{ mb: 2 }}>
          {targetPaths.map(path => (
            <Box
              key={path.id}
              onClick={() => setSelectedPathId(selectedPathId === path.id ? null : path.id)}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 1,
                mb: 0.5,
                borderRadius: 1,
                cursor: 'pointer',
                bgcolor: selectedPathId === path.id ? 'rgba(233, 30, 99, 0.15)' : 'rgba(0,0,0,0.02)',
                border: '2px solid',
                borderColor: selectedPathId === path.id ? '#e91e63' : 'transparent',
                transition: 'all 0.15s ease',
                '&:hover': { bgcolor: 'rgba(233, 30, 99, 0.08)' }
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: path.color }} />
                <Box>
                  <Typography variant="body2" sx={{ fontWeight: selectedPathId === path.id ? 600 : 400 }}>
                    {path.name}
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                    {path.points.length} points
                    {path.targetJoint && ` • ${path.targetJoint}`}
                  </Typography>
                </Box>
              </Box>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation()
                  setTargetPaths(prev => prev.filter(p => p.id !== path.id))
                  if (selectedPathId === path.id) setSelectedPathId(null)
                }}
                sx={{ width: 24, height: 24, color: '#999', '&:hover': { color: '#d32f2f' } }}
              >
                ×
              </IconButton>
            </Box>
          ))}
        </Box>
      ) : (
        <Box sx={{
          p: 2,
          mb: 2,
          borderRadius: 1,
          bgcolor: 'rgba(0,0,0,0.03)',
          border: '1px dashed rgba(0,0,0,0.2)',
          textAlign: 'center'
        }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
            No target paths yet
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
            Use <strong>Draw Path</strong> tool (T) to create one
          </Typography>
        </Box>
      )}

      {/* Joint selector */}
      {selectedPathId && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
            Joint to Optimize
          </Typography>
          <FormControl size="small" fullWidth>
            <Select
              value={selectedPath?.targetJoint || ''}
              onChange={(e) => {
                setTargetPaths(prev => prev.map(p =>
                  p.id === selectedPathId ? { ...p, targetJoint: e.target.value as string } : p
                ))
              }}
              displayEmpty
              sx={{ fontSize: '0.85rem' }}
            >
              <MenuItem value="" sx={{ fontSize: '0.85rem' }}>
                <em>Select joint...</em>
              </MenuItem>
              {joints
                .filter(j => j.type === 'Crank' || j.type === 'Revolute')
                .map(j => (
                  <MenuItem key={j.name} value={j.name} sx={{ fontSize: '0.85rem' }}>
                    {j.name} <Chip label={j.type} size="small" sx={{ ml: 1, height: 18, fontSize: '0.65rem' }} />
                  </MenuItem>
                ))
              }
            </Select>
          </FormControl>
        </Box>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          TRAJECTORY PREPROCESSING
          ═══════════════════════════════════════════════════════════════════════ */}
      {selectedPathId && selectedPath && (
        <>
          <Divider sx={{ my: 2 }} />

          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#00897b', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
            <span>🔄</span> Path Preprocessing
          </Typography>

          <Box sx={{
            p: 1.5,
            mb: 1.5,
            borderRadius: 1,
            bgcolor: 'rgba(0, 137, 123, 0.05)',
            border: '1px solid rgba(0, 137, 123, 0.2)'
          }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
              Current path: <strong>{selectedPath.points.length} points</strong>
              {preprocessResult && (
                <> • Processed from {preprocessResult.originalPoints} points</>
              )}
            </Typography>

            {/* Smoothing Section */}
            <Box sx={{ mb: 1.5 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={prepEnableSmooth}
                    onChange={(e) => setPrepEnableSmooth(e.target.checked)}
                    size="small"
                    color="primary"
                  />
                }
                label={<Typography variant="caption" sx={{ fontWeight: 500 }}>Enable Smoothing</Typography>}
              />

              {prepEnableSmooth && (
                <Box sx={{ pl: 1, mt: 0.5 }}>
                  <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                    <Box sx={{ flex: 1 }}>
                      <Tooltip title="Smoothing filter type. Savgol preserves peaks, Moving Avg is aggressive, Gaussian is natural." placement="top">
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                          Method ⓘ
                        </Typography>
                      </Tooltip>
                      <Select
                        size="small"
                        fullWidth
                        value={prepSmoothMethod}
                        onChange={(e) => setPrepSmoothMethod(e.target.value as SmoothMethod)}
                        sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                      >
                        <MenuItem value="savgol" sx={{ fontSize: '0.75rem' }}>Savitzky-Golay</MenuItem>
                        <MenuItem value="moving_avg" sx={{ fontSize: '0.75rem' }}>Moving Average</MenuItem>
                        <MenuItem value="gaussian" sx={{ fontSize: '0.75rem' }}>Gaussian</MenuItem>
                      </Select>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>
                      <Tooltip title="Window size. Larger = more smoothing. 2-4: light, 8-16: medium, 32+: heavy" placement="top">
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                          Window ⓘ
                        </Typography>
                      </Tooltip>
                      <Select
                        size="small"
                        fullWidth
                        value={prepSmoothWindow}
                        onChange={(e) => setPrepSmoothWindow(e.target.value as number)}
                        sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                      >
                        <MenuItem value={2} sx={{ fontSize: '0.75rem' }}>2 (Light)</MenuItem>
                        <MenuItem value={4} sx={{ fontSize: '0.75rem' }}>4 (Default)</MenuItem>
                        <MenuItem value={8} sx={{ fontSize: '0.75rem' }}>8 (Medium)</MenuItem>
                        <MenuItem value={16} sx={{ fontSize: '0.75rem' }}>16</MenuItem>
                        <MenuItem value={32} sx={{ fontSize: '0.75rem' }}>32 (Heavy)</MenuItem>
                        <MenuItem value={64} sx={{ fontSize: '0.75rem' }}>64 (Max)</MenuItem>
                      </Select>
                    </Box>

                    <Box sx={{ flex: 1 }}>
                      <Tooltip title="Polynomial order for Savgol. Must be < window. Higher = preserves peaks better." placement="top">
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                          Polyorder ⓘ
                        </Typography>
                      </Tooltip>
                      <Select
                        size="small"
                        fullWidth
                        value={prepSmoothPolyorder}
                        onChange={(e) => setPrepSmoothPolyorder(e.target.value as number)}
                        disabled={prepSmoothMethod !== 'savgol'}
                        sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                      >
                        <MenuItem value={1} sx={{ fontSize: '0.75rem' }}>1 (Linear)</MenuItem>
                        <MenuItem value={2} sx={{ fontSize: '0.75rem' }}>2</MenuItem>
                        <MenuItem value={3} sx={{ fontSize: '0.75rem' }}>3 (Default)</MenuItem>
                        <MenuItem value={4} sx={{ fontSize: '0.75rem' }}>4</MenuItem>
                        <MenuItem value={5} sx={{ fontSize: '0.75rem' }}>5</MenuItem>
                      </Select>
                    </Box>
                  </Box>
                </Box>
              )}
            </Box>

            {/* Resampling Section */}
            <Box sx={{ mb: 1.5 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={prepEnableResample}
                    onChange={(e) => setPrepEnableResample(e.target.checked)}
                    size="small"
                    color="primary"
                  />
                }
                label={<Typography variant="caption" sx={{ fontWeight: 500 }}>Enable Resampling</Typography>}
              />

              {prepEnableResample && (
                <Box sx={{ pl: 1, mt: 0.5 }}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Box sx={{ flex: 1 }}>
                      <Tooltip title={`Target number of points. Uses current Simulation Steps (${simulationSteps}) for optimization consistency.`} placement="top">
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                          Target Points ⓘ
                        </Typography>
                      </Tooltip>
                      <TextField
                        type="number"
                        size="small"
                        fullWidth
                        value={prepTargetNSteps}
                        onChange={(e) => {
                          const val = parseInt(e.target.value)
                          if (!isNaN(val) && val >= 4 && val <= 256) {
                            setPrepTargetNSteps(val)
                          }
                        }}
                        inputProps={{ min: 4, max: 256, step: 4 }}
                        helperText={simulationSteps !== prepTargetNSteps ? `Sim uses ${simulationSteps}` : undefined}
                        sx={{
                          '& .MuiInputBase-input': { fontSize: '0.75rem', py: 0.5 },
                          '& .MuiFormHelperText-root': { fontSize: '0.6rem', mt: 0.25, color: 'warning.main' }
                        }}
                      />
                    </Box>

                    <Box sx={{ flex: 1 }}>
                      <Tooltip title="Interpolation method. Parametric is best for closed curves." placement="top">
                        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                          Method ⓘ
                        </Typography>
                      </Tooltip>
                      <Select
                        size="small"
                        fullWidth
                        value={prepResampleMethod}
                        onChange={(e) => setPrepResampleMethod(e.target.value as ResampleMethod)}
                        sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                      >
                        <MenuItem value="parametric" sx={{ fontSize: '0.75rem' }}>Parametric</MenuItem>
                        <MenuItem value="cubic" sx={{ fontSize: '0.75rem' }}>Cubic</MenuItem>
                        <MenuItem value="linear" sx={{ fontSize: '0.75rem' }}>Linear</MenuItem>
                      </Select>
                    </Box>
                  </Box>
                </Box>
              )}
            </Box>

            {/* Preprocess Button */}
            <Button
              variant="outlined"
              fullWidth
              size="small"
              onClick={preprocessTrajectory}
              disabled={isPreprocessing || (!prepEnableSmooth && !prepEnableResample)}
              sx={{
                textTransform: 'none',
                fontSize: '0.8rem',
                color: '#00897b',
                borderColor: '#00897b',
                '&:hover': {
                  borderColor: '#00695c',
                  bgcolor: 'rgba(0, 137, 123, 0.08)'
                },
                '&.Mui-disabled': { borderColor: '#ccc' }
              }}
            >
              {isPreprocessing ? '⏳ Processing...' : '🔄 Apply Preprocessing'}
            </Button>

            {/* Preprocessing Result */}
            {preprocessResult && (
              <Box sx={{ mt: 1, p: 1, borderRadius: 0.5, bgcolor: 'rgba(0, 137, 123, 0.1)' }}>
                <Typography variant="caption" sx={{ display: 'block', color: '#00695c', fontWeight: 500 }}>
                  ✓ Processed successfully
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.65rem' }}>
                  {preprocessResult.originalPoints} → {preprocessResult.outputPoints} points
                  {preprocessResult.analysis && (
                    <>
                      {' • '}Path length: {(preprocessResult.analysis.total_path_length as number)?.toFixed(1)}
                      {preprocessResult.analysis.is_closed && ' • Closed curve'}
                    </>
                  )}
                </Typography>
              </Box>
            )}
          </Box>
        </>
      )}

      <Divider sx={{ my: 2 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          SIMULATION STEPS (N_STEPS)
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#6a1b9a', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
        <span>📊</span> Simulation Steps
      </Typography>

      <Box sx={{ mb: 2 }}>
        <Tooltip title="Number of trajectory points for simulation and optimization. Higher = more precision but slower. Should match preprocessed trajectory points." placement="right">
          <TextField
            type="number"
            size="small"
            fullWidth
            label="N_STEPS"
            value={simulationStepsInput}
            onChange={(e) => setSimulationStepsInput(e.target.value)}
            inputProps={{ min: MIN_SIMULATION_STEPS, max: MAX_SIMULATION_STEPS, step: 4 }}
            helperText={`Range: ${MIN_SIMULATION_STEPS}-${MAX_SIMULATION_STEPS}. Current: ${simulationSteps}`}
            sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
          />
        </Tooltip>

        {/* Sync button to match Target Points with Simulation Steps */}
        {prepTargetNSteps !== simulationSteps && (
          <Button
            size="small"
            variant="text"
            onClick={() => setPrepTargetNSteps(simulationSteps)}
            sx={{
              mt: 0.5,
              textTransform: 'none',
              fontSize: '0.7rem',
              color: '#6a1b9a'
            }}
          >
            ↻ Sync preprocessing target ({prepTargetNSteps}) to {simulationSteps}
          </Button>
        )}
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          OPTIMIZATION METHOD
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#1976d2', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
        <span>🔧</span> Method
      </Typography>

      <FormControl size="small" fullWidth sx={{ mb: 1 }}>
        <Select
          value={optMethod}
          onChange={(e) => setOptMethod(e.target.value as OptMethod)}
          sx={{ fontSize: '0.85rem' }}
        >
          <MenuItem value="pso">Particle Swarm (PSO)</MenuItem>
          <MenuItem value="pylinkage">Pylinkage PSO</MenuItem>
          <MenuItem value="scipy">L-BFGS-B (SciPy)</MenuItem>
          <MenuItem value="powell">Powell's Method</MenuItem>
          <MenuItem value="nelder-mead">Nelder-Mead Simplex</MenuItem>
        </Select>
      </FormControl>

      {/* Method description */}
      <Box sx={{
        p: 1.5,
        mb: 2,
        borderRadius: 1,
        bgcolor: 'rgba(25, 118, 210, 0.05)',
        border: '1px solid rgba(25, 118, 210, 0.2)'
      }}>
        <Typography variant="caption" sx={{ fontWeight: 600, color: '#1976d2', display: 'block' }}>
          {currentMethodInfo.name}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5 }}>
          {currentMethodInfo.description}
        </Typography>
        <Box sx={{ mt: 1, display: 'flex', gap: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" sx={{ color: '#2e7d32', fontWeight: 500 }}>✓ Pros</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontSize: '0.65rem' }}>
              {currentMethodInfo.pros}
            </Typography>
          </Box>
          <Box sx={{ flex: 1 }}>
            <Typography variant="caption" sx={{ color: '#d32f2f', fontWeight: 500 }}>✗ Cons</Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontSize: '0.65rem' }}>
              {currentMethodInfo.cons}
            </Typography>
          </Box>
        </Box>
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          HYPERPARAMETERS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#7b1fa2', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
        <span>⚙️</span> Hyperparameters
      </Typography>

      {/* PSO-specific parameters */}
      {(optMethod === 'pso' || optMethod === 'pylinkage') && (
        <>
          <Box sx={{ mb: 1.5 }}>
            <Tooltip title="Number of particles in the swarm. More particles = better exploration but slower. Typical: 20-50" placement="left">
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                Swarm Size (Particles) ⓘ
              </Typography>
            </Tooltip>
            <TextField
              type="number"
              size="small"
              fullWidth
              value={optNParticles}
              onChange={(e) => setOptNParticles(Math.max(5, Math.min(1024, parseInt(e.target.value) || 32)))}
              inputProps={{ min: 5, max: 1024, step: 16 }}
              sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
            />
          </Box>

          <Box sx={{ mb: 1.5 }}>
            <Tooltip title="Number of iterations for the swarm. More iterations = better convergence but slower. Typical: 256-1024, max 10000" placement="left">
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                Iterations ⓘ
              </Typography>
            </Tooltip>
            <TextField
              type="number"
              size="small"
              fullWidth
              value={optIterations}
              onChange={(e) => setOptIterations(Math.max(10, Math.min(10000, parseInt(e.target.value) || 512)))}
              inputProps={{ min: 10, max: 10000, step: 64 }}
              sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
            />
          </Box>
        </>
      )}

      {/* SciPy-specific parameters */}
      {(optMethod === 'scipy' || optMethod === 'powell' || optMethod === 'nelder-mead') && (
        <>
          <Box sx={{ mb: 1.5 }}>
            <Tooltip title="Maximum number of function evaluations. Prevents infinite loops. Typical: 100-1000" placement="left">
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                Max Iterations ⓘ
              </Typography>
            </Tooltip>
            <TextField
              type="number"
              size="small"
              fullWidth
              value={optMaxIterations}
              onChange={(e) => setOptMaxIterations(Math.max(10, Math.min(10000, parseInt(e.target.value) || 100)))}
              inputProps={{ min: 10, max: 10000, step: 50 }}
              sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
            />
          </Box>

          <Box sx={{ mb: 1.5 }}>
            <Tooltip title="Convergence tolerance. Smaller = more precise but slower. Typical: 1e-4 to 1e-8" placement="left">
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                Tolerance ⓘ
              </Typography>
            </Tooltip>
            <Select
              size="small"
              fullWidth
              value={optTolerance}
              onChange={(e) => setOptTolerance(e.target.value as number)}
              sx={{ fontSize: '0.85rem' }}
            >
              <MenuItem value={1e-4}>1e-4 (Fast, less precise)</MenuItem>
              <MenuItem value={1e-5}>1e-5</MenuItem>
              <MenuItem value={1e-6}>1e-6 (Default)</MenuItem>
              <MenuItem value={1e-7}>1e-7</MenuItem>
              <MenuItem value={1e-8}>1e-8 (Slow, very precise)</MenuItem>
            </Select>
          </Box>
        </>
      )}

      <Divider sx={{ my: 2 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          BOUNDS & CONSTRAINTS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#ed6c02', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
        <span>📐</span> Bounds
      </Typography>

      <Box sx={{ mb: 1.5 }}>
        <Tooltip title="How much link lengths can vary from initial values. Factor of 2.0 means lengths can be 0.5x to 2.0x original" placement="left">
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
            Bounds Factor ⓘ
          </Typography>
        </Tooltip>
        <Select
          size="small"
          fullWidth
          value={optBoundsFactor}
          onChange={(e) => setOptBoundsFactor(e.target.value as number)}
          sx={{ fontSize: '0.85rem' }}
        >
          <MenuItem value={1.25}>±25% (Conservative)</MenuItem>
          <MenuItem value={1.5}>±50%</MenuItem>
          <MenuItem value={2.0}>±100% (Default)</MenuItem>
          <MenuItem value={3.0}>±200% (Wide)</MenuItem>
          <MenuItem value={5.0}>±400% (Very Wide)</MenuItem>
        </Select>
      </Box>

      <Box sx={{ mb: 1.5 }}>
        <Tooltip title="Minimum allowed link length to prevent degenerate solutions" placement="left">
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
            Min Link Length ⓘ
          </Typography>
        </Tooltip>
        <TextField
          type="number"
          size="small"
          fullWidth
          value={optMinLength}
          onChange={(e) => setOptMinLength(Math.max(0.01, parseFloat(e.target.value) || 0.1))}
          inputProps={{ min: 0.01, max: 10, step: 0.1 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          RUN OPTIMIZATION
          ═══════════════════════════════════════════════════════════════════════ */}
      <Box sx={{ mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={optVerbose}
              onChange={(e) => setOptVerbose(e.target.checked)}
              size="small"
            />
          }
          label={<Typography variant="caption">Verbose logging</Typography>}
        />
      </Box>

      <Button
        variant="contained"
        fullWidth
        size="large"
        onClick={runOptimization}
        disabled={!canOptimize}
        sx={{
          textTransform: 'none',
          fontSize: '1rem',
          fontWeight: 600,
          py: 1.5,
          backgroundColor: '#e91e63',
          '&:hover': { backgroundColor: '#c2185b' },
          '&.Mui-disabled': { backgroundColor: '#e0e0e0' }
        }}
      >
        {isOptimizing ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <span>⏳</span> Optimizing...
          </Box>
        ) : (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <span>⚡</span> Run Optimization
          </Box>
        )}
      </Button>

      {/* Disabled reason */}
      {!canOptimize && !isOptimizing && (
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1, textAlign: 'center' }}>
          {!hasCrank ? 'Need a valid mechanism with Crank' :
           !selectedPath ? 'Select a target path above' :
           !selectedPath.targetJoint ? 'Select a joint to optimize' :
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
            <span>⚠️</span> Path has {selectedPath.points.length} points but simulation uses {simulationSteps}.
            Preprocess or adjust N_STEPS for best results.
          </Typography>
        </Box>
      )}

      {/* ═══════════════════════════════════════════════════════════════════════
          RESULTS
          ═══════════════════════════════════════════════════════════════════════ */}
      {optimizationResult && (
        <Box sx={{ mt: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: optimizationResult.success ? '#2e7d32' : '#d32f2f' }}>
              {optimizationResult.success ? '✓ Results' : '✗ Failed'}
            </Typography>
            {preOptimizationDoc && optimizationResult.success && (
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
                ↩ Revert
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
                {/* Error metrics */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>Initial Error</Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 600, color: '#ff7043' }}>
                    {optimizationResult.initialError.toFixed(4)}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>Final Error</Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 600, color: '#2e7d32' }}>
                    {optimizationResult.finalError.toFixed(4)}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary' }}>Improvement</Typography>
                  <Typography variant="caption" sx={{
                    fontWeight: 700,
                    color: optimizationResult.initialError > 0
                      ? ((1 - optimizationResult.finalError / optimizationResult.initialError) * 100 > 50 ? '#2e7d32' : '#ed6c02')
                      : '#666'
                  }}>
                    {optimizationResult.initialError > 0
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

                {/* Show dimension changes: Before → After */}
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
                              {(oldValue as number).toFixed(2)}
                            </Typography>
                            <Typography variant="caption" sx={{
                              color: changed ? '#1976d2' : 'text.secondary',
                              fontSize: '0.65rem',
                              textAlign: 'center'
                            }}>
                              {changed ? '→' : '='}
                            </Typography>
                            <Typography variant="caption" sx={{
                              fontFamily: 'monospace',
                              fontSize: '0.65rem',
                              textAlign: 'right',
                              color: '#2e7d32',
                              fontWeight: changed ? 600 : 400
                            }}>
                              {(newValue as number).toFixed(2)}
                            </Typography>
                          </React.Fragment>
                        )
                      })}
                    </Box>
                  </Box>
                )}

                {/* Canvas updated notice */}
                <Box sx={{
                  mt: 1.5,
                  p: 1,
                  bgcolor: 'rgba(25, 118, 210, 0.1)',
                  borderRadius: 1,
                  border: '1px solid rgba(25, 118, 210, 0.3)'
                }}>
                  <Typography variant="caption" sx={{ color: '#1976d2', fontWeight: 500 }}>
                    ✓ Canvas updated with optimized dimensions
                  </Typography>
                  {preOptimizationDoc && (
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5, fontSize: '0.65rem' }}>
                      Click "Revert" to restore original dimensions
                    </Typography>
                  )}
                </Box>
              </>
            ) : (
              <Typography variant="caption" sx={{ color: '#d32f2f' }}>
                {optimizationResult.message}
              </Typography>
            )}
          </Box>
        </Box>
      )}
    </Box>
  )
}

export default OptimizationToolbar
