/**
 * Target Trajectory Panel
 *
 * Manages target paths, achievable target generation, and path preprocessing.
 * Separated from optimization configuration for better maintainability.
 */

import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react'
import {
  Box, Typography, Button, Tooltip, Divider, FormControl, Select, MenuItem,
  TextField, FormControlLabel, Switch, IconButton, Chip, Dialog, DialogTitle,
  DialogContent, DialogActions, Accordion, AccordionSummary, AccordionDetails, Stack
} from '@mui/material'
import { Settings, Close, ExpandMore } from '@mui/icons-material'
import { canSimulate, type TrajectoryData } from '../../AnimateSimulate'
import type { PylinkJoint } from '../types'
import { DimensionVariationConfig } from './DimensionVariationConfig'
import { StaticJointMovementConfig } from './StaticJointMovementConfig'

// Types (avoid circular dependency by defining locally)
export type SmoothMethod = 'savgol' | 'moving_avg' | 'gaussian'
export type ResampleMethod = 'parametric' | 'cubic' | 'linear'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

export interface TargetPath {
  id: string
  name: string
  points: [number, number][]
  color: string
  targetJoint?: string
}

export interface PreprocessResult {
  originalPoints: number
  outputPoints: number
  analysis?: {
    total_path_length?: number
    is_closed?: boolean
  }
}

export interface MechVariationConfig {
  target_joint?: string
  dimension_variation: {
    default_variation_range: number
    default_enabled: boolean
    dimension_overrides: Record<string, [boolean, number, number]>
    exclude_dimensions: string[]
  }
  static_joint_movement: {
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
  max_attempts: number
  fallback_ranges: number[]
  random_seed: number | null
}

export interface DimensionInfo {
  names: string[]
  initial_values: number[]
  bounds: [number, number][]
  n_dimensions: number
}

export interface TargetTrajectoryPanelProps {
  // Joint data
  joints: PylinkJoint[]
  linkageDoc?: unknown
  trajectoryData?: TrajectoryData | null
  stretchingLinks?: string[]

  // Target paths
  targetPaths: TargetPath[]
  setTargetPaths: React.Dispatch<React.SetStateAction<TargetPath[]>>
  selectedPathId: string | null
  setSelectedPathId: (id: string | null) => void

  // Preprocessing
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

  // Simulation steps (for preprocessing sync)
  simulationSteps: number

  // Dimension info (fetched by parent after trajectory computation)
  dimensionInfo?: DimensionInfo | null
  isLoadingDimensions?: boolean
  dimensionInfoError?: string | null
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const TargetTrajectoryPanel: React.FC<TargetTrajectoryPanelProps> = ({
  joints,
  linkageDoc,
  trajectoryData,
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
  dimensionInfo: dimensionInfoProp,
  isLoadingDimensions: isLoadingDimensionsProp = false,
  dimensionInfoError: dimensionInfoErrorProp = null
}) => {
  const hasCrank = canSimulate(joints)
  const selectedPath = targetPaths.find(p => p.id === selectedPathId)

  // Single source of truth for target joint
  const [targetJoint, setTargetJoint] = useState<string | null>(null)

  // Achievable target config state
  const defaultAchievableTargetConfig: MechVariationConfig = {
    target_joint: '',
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

  const [achievableTargetConfig, setAchievableTargetConfig] = useState<MechVariationConfig>(defaultAchievableTargetConfig)
  const [isGeneratingAchievableTarget, setIsGeneratingAchievableTarget] = useState(false)
  const [showAchievableTargetConfigDialog, setShowAchievableTargetConfigDialog] = useState(false)
  const [dimensionSyncStates, setDimensionSyncStates] = useState<Record<string, boolean>>({})

  // Use dimension info from props (fetched by parent after trajectory computation)
  const dimensionInfo = dimensionInfoProp
  const isLoadingDimensions = isLoadingDimensionsProp
  const dimensionInfoError = dimensionInfoErrorProp

  // Create a stable key for the mechanism (for tracking fetched target joints)
  const mechanismKey = useMemo(() => {
    if (!linkageDoc) return null
    try {
      const doc = linkageDoc as { name?: string; linkage?: { joints?: Record<string, unknown>; edges?: Record<string, unknown> } }
      const keyParts = {
        name: doc.name,
        jointCount: Object.keys(doc.linkage?.joints || {}).length,
        edgeCount: Object.keys(doc.linkage?.edges || {}).length
      }
      return JSON.stringify(keyParts)
    } catch {
      return null
    }
  }, [linkageDoc])

  // Track which mechanism we've already selected a target joint for
  const targetJointSelectedForRef = useRef<string | null>(null)

  // Deterministic hash function for consistent joint selection
  const hashString = (str: string): number => {
    let hash = 0
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i)
      hash = ((hash << 5) - hash) + char
      hash = hash & hash // Convert to 32-bit integer
    }
    return Math.abs(hash)
  }

  // Select target joint deterministically based on mechanism
  const selectTargetJoint = useCallback(() => {
    if (!linkageDoc || !joints.length || !mechanismKey) return

    // Skip if we've already selected for this mechanism
    if (targetJointSelectedForRef.current === mechanismKey) {
      return
    }

    // Priority 1: Check if target_joint is stored in linkageDoc metadata (from demo load)
    const doc = linkageDoc as { meta?: { target_joint?: string } }
    if (doc.meta?.target_joint && joints.some(j => j.name === doc.meta.target_joint)) {
      setTargetJoint(doc.meta.target_joint)
      setAchievableTargetConfig(prev => ({ ...prev, target_joint: doc.meta.target_joint || '' }))
      targetJointSelectedForRef.current = mechanismKey
      console.log('[TargetJointSelect] Using target_joint from demo metadata:', doc.meta.target_joint)
      return
    }

    // Priority 2: Selected path target joint
    if (selectedPath?.targetJoint && joints.some(j => j.name === selectedPath.targetJoint)) {
      setTargetJoint(selectedPath.targetJoint)
      setAchievableTargetConfig(prev => ({ ...prev, target_joint: selectedPath.targetJoint || '' }))
      targetJointSelectedForRef.current = mechanismKey
      return
    }

    // Priority 3: Deterministic selection from non-default joints
    // Exclude: "crank", "static", and default names like "joint_1", "joint_2", "joint1", "joint2"
    const defaultNamePattern = /^(joint[_]?\d+|crank|static)$/i
    const candidateJoints = joints.filter(j =>
      (j.type === 'Crank' || j.type === 'Revolute') &&
      !defaultNamePattern.test(j.name)
    )

    if (candidateJoints.length > 0) {
      // Sort joints by name for consistency, then use hash of mechanism key to pick deterministically
      const sortedJoints = [...candidateJoints].sort((a, b) => a.name.localeCompare(b.name))
      const hash = hashString(mechanismKey)
      const selectedIndex = hash % sortedJoints.length
      const selectedJoint = sortedJoints[selectedIndex]
      setTargetJoint(selectedJoint.name)
      setAchievableTargetConfig(prev => ({ ...prev, target_joint: selectedJoint.name }))
      targetJointSelectedForRef.current = mechanismKey
      console.log('[TargetJointSelect] Deterministically selected:', selectedJoint.name, 'from', sortedJoints.length, 'candidates')
      return
    }

    // Fallback: any non-crank, non-static joint
    const fallbackJoints = joints.filter(j =>
      (j.type === 'Crank' || j.type === 'Revolute') &&
      j.name.toLowerCase() !== 'crank' &&
      j.name.toLowerCase() !== 'static'
    )

    if (fallbackJoints.length > 0) {
      const sortedJoints = [...fallbackJoints].sort((a, b) => a.name.localeCompare(b.name))
      const hash = hashString(mechanismKey)
      const selectedIndex = hash % sortedJoints.length
      const selectedJoint = sortedJoints[selectedIndex]
      setTargetJoint(selectedJoint.name)
      setAchievableTargetConfig(prev => ({ ...prev, target_joint: selectedJoint.name }))
      targetJointSelectedForRef.current = mechanismKey
      console.log('[TargetJointSelect] Fallback selection:', selectedJoint.name)
    }
  }, [linkageDoc, joints, mechanismKey, selectedPath?.targetJoint])

  // Select target joint only when trajectory is computed (mechanism is ready)
  useEffect(() => {
    // Only select when trajectory data exists (mechanism has been computed)
    if (!trajectoryData || !mechanismKey || !joints.length) {
      return
    }

    // Reset selection tracking when mechanism changes
    if (targetJointSelectedForRef.current !== mechanismKey) {
      selectTargetJoint()
    }
  }, [trajectoryData, mechanismKey, selectTargetJoint, joints.length])

  // Handle selectedPath?.targetJoint changes (user manually selects joint for a path)
  useEffect(() => {
    if (selectedPath?.targetJoint && joints.some(j => j.name === selectedPath.targetJoint)) {
      setTargetJoint(selectedPath.targetJoint)
      setAchievableTargetConfig(prev => ({ ...prev, target_joint: selectedPath.targetJoint || '' }))
    }
  }, [selectedPath?.targetJoint, joints])

  // Generate achievable target
  const handleGetAchievableTarget = async () => {
    if (!linkageDoc || !hasCrank) {
      return
    }

    setIsGeneratingAchievableTarget(true)
    try {
      const targetJointToSend = targetJoint || achievableTargetConfig.target_joint || ''

      const response = await fetch('/api/get-achievable-target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pylink_data: linkageDoc,
          target_joint: targetJointToSend,
          n_steps: simulationSteps,
          config: {
            dimension_variation: achievableTargetConfig.dimension_variation,
            static_joint_movement: achievableTargetConfig.static_joint_movement,
            topology_changes: achievableTargetConfig.topology_changes,
            max_attempts: achievableTargetConfig.max_attempts,
            fallback_ranges: achievableTargetConfig.fallback_ranges,
            random_seed: achievableTargetConfig.random_seed
          }
        })
      })

      if (response.ok) {
        const result = await response.json()
        if (result.status === 'success' && result.result && result.result.target) {
          const targetResult = result.result.target
          const newPath: TargetPath = {
            id: `target-${Date.now()}`,
            name: `Achievable Target (${targetResult.joint_name})`,
            points: targetResult.positions.map((p: number[]) => [p[0], p[1]] as [number, number]),
            color: '#4caf50',
            targetJoint: targetResult.joint_name
          }
          setTargetPaths(prev => [...prev, newPath])
          setSelectedPathId(newPath.id)
        } else {
          console.error('[ERROR] Failed to generate achievable target:', result.message || result)
        }
      }
    } catch (error) {
      console.error('[ERROR] Exception in handleGetAchievableTarget:', error)
    } finally {
      setIsGeneratingAchievableTarget(false)
    }
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      {/* Target Joint Selector */}
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#1976d2', mb: 1, display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.8rem' }}>
          <span>🎯</span> Target Joint
        </Typography>
        <FormControl size="small" fullWidth>
          <Select
            value={targetJoint || ''}
            onChange={(e) => {
              const newTargetJoint = e.target.value as string
              setTargetJoint(newTargetJoint)
              setAchievableTargetConfig(prev => ({
                ...prev,
                target_joint: newTargetJoint
              }))
              if (selectedPath) {
                setTargetPaths(prev => prev.map(p =>
                  p.id === selectedPathId ? { ...p, targetJoint: newTargetJoint } : p
                ))
              }
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
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5 }}>
          The joint whose trajectory will be optimized or used as target
        </Typography>
      </Box>

      {/* Target Paths List */}
      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#e91e63', mb: 1, display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.8rem' }}>
          <span>📍</span> Target Paths
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

        {/* Joint selector for selected path */}
        {selectedPathId && selectedPath && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
              Joint to Optimize
            </Typography>
            <FormControl size="small" fullWidth>
              <Select
                value={selectedPath.targetJoint || ''}
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
                {(() => {
                  // Deduplicate joints by name (keep first occurrence)
                  // This prevents showing duplicate joints if somehow multiple mechanisms are merged
                  const seen = new Set<string>()
                  const uniqueJoints = joints
                    .filter(j => j.type === 'Crank' || j.type === 'Revolute')
                    .filter(j => {
                      if (seen.has(j.name)) {
                        console.warn(`Duplicate joint name detected: ${j.name}`)
                        return false
                      }
                      seen.add(j.name)
                      return true
                    })

                  return uniqueJoints.map(j => (
                    <MenuItem key={j.name} value={j.name} sx={{ fontSize: '0.85rem' }}>
                      {j.name} <Chip label={j.type} size="small" sx={{ ml: 1, height: 18, fontSize: '0.65rem' }} />
                    </MenuItem>
                  ))
                })()}
              </Select>
            </FormControl>
          </Box>
        )}
      </Box>

      <Divider />

      {/* Achievable Target Generation */}
      {hasCrank && (
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#4caf50', mb: 1, display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.8rem' }}>
            <span>🎯</span> Achievable Target
          </Typography>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Button
              size="small"
              variant="outlined"
              startIcon={<Settings />}
              onClick={() => setShowAchievableTargetConfigDialog(true)}
              sx={{ fontSize: '0.75rem', textTransform: 'none' }}
            >
              Configure Achievable Target
            </Button>

            <Tooltip
              title={
                !hasCrank
                  ? "Need a valid mechanism with Crank joint"
                  : !targetJoint && !achievableTargetConfig.target_joint
                  ? "Select a target joint in Config dialog (or it will be auto-selected)"
                  : "Generate an achievable target trajectory using the configured settings"
              }
            >
              <span>
                <Button
                  size="small"
                  variant="contained"
                  color="success"
                  onClick={handleGetAchievableTarget}
                  disabled={isGeneratingAchievableTarget || !hasCrank}
                  sx={{ fontSize: '0.75rem', textTransform: 'none' }}
                >
                  {isGeneratingAchievableTarget ? 'Generating...' : 'Generate Achievable Target'}
                </Button>
              </span>
            </Tooltip>

            {targetJoint && (
              <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                Target joint: <strong>{targetJoint}</strong>
              </Typography>
            )}
          </Box>
        </Box>
      )}

      <Divider />

      {/* Path Preprocessing */}
      {selectedPathId && selectedPath && (
        <Box>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#00897b', mb: 1, display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.8rem' }}>
            <span>🔄</span> Path Preprocessing
          </Typography>

          <Box sx={{
            p: 1.5,
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
        </Box>
      )}

      {/* Achievable Target Config Dialog */}
      <Dialog
        open={showAchievableTargetConfigDialog}
        onClose={() => setShowAchievableTargetConfigDialog(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Achievable Target Config</Typography>
            <IconButton size="small" onClick={() => setShowAchievableTargetConfigDialog(false)}>
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
                    dimensionOverrides={achievableTargetConfig.dimension_variation.dimension_overrides}
                    dimensionSyncStates={dimensionSyncStates}
                    defaultEnabled={achievableTargetConfig.dimension_variation.default_enabled}
                    defaultVariationRange={achievableTargetConfig.dimension_variation.default_variation_range}
                    onOverrideChange={(name, override) => {
                      setAchievableTargetConfig(prev => ({
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
                      setAchievableTargetConfig(prev => ({
                        ...prev,
                        dimension_variation: { ...prev.dimension_variation, default_enabled: enabled }
                      }))
                    }}
                    onDefaultRangeChange={(range) => {
                      setAchievableTargetConfig(prev => ({
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
                  enabled={achievableTargetConfig.static_joint_movement.enabled}
                  defaultMaxX={achievableTargetConfig.static_joint_movement.max_x_movement}
                  defaultMaxY={achievableTargetConfig.static_joint_movement.max_y_movement}
                  jointOverrides={achievableTargetConfig.static_joint_movement.joint_overrides}
                  onEnabledChange={(enabled) => {
                    setAchievableTargetConfig(prev => ({
                      ...prev,
                      static_joint_movement: { ...prev.static_joint_movement, enabled }
                    }))
                  }}
                  onDefaultMaxXChange={() => {}}
                  onDefaultMaxYChange={() => {}}
                  onJointOverrideChange={(jointName, override) => {
                    setAchievableTargetConfig(prev => ({
                      ...prev,
                      static_joint_movement: {
                        ...prev.static_joint_movement,
                        joint_overrides: {
                          ...prev.static_joint_movement.joint_overrides,
                          [jointName]: override
                        }
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
                    Future functionality: Add/remove links and nodes during target generation.
                    This feature is not yet implemented.
                  </Typography>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={achievableTargetConfig.topology_changes.enabled}
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
                    value={achievableTargetConfig.max_attempts}
                    onChange={(e) => setAchievableTargetConfig(prev => ({
                      ...prev,
                      max_attempts: parseInt(e.target.value) || 128
                    }))}
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
                      value={achievableTargetConfig.fallback_ranges.join(', ')}
                      onChange={(e) => {
                        const ranges = e.target.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
                        if (ranges.length > 0) {
                          setAchievableTargetConfig(prev => ({ ...prev, fallback_ranges: ranges }))
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
                    value={achievableTargetConfig.random_seed || ''}
                    onChange={(e) => setAchievableTargetConfig(prev => ({
                      ...prev,
                      random_seed: e.target.value ? parseInt(e.target.value) : null
                    }))}
                    inputProps={{ min: 0 }}
                    helperText="Leave empty for random seed"
                  />
                </Stack>
              </AccordionDetails>
            </Accordion>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAchievableTargetConfigDialog(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => setShowAchievableTargetConfigDialog(false)}
            disabled={!targetJoint && !achievableTargetConfig.target_joint}
          >
            Done
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default TargetTrajectoryPanel
