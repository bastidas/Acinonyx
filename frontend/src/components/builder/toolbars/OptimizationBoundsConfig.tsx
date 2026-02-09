/**
 * Optimization Bounds Config
 *
 * Simple form-based component for configuring dimension variation bounds.
 * No accordions - just direct rendering when dimensions are available.
 */

import React, { useState, useRef } from 'react'
import { Box, Typography } from '@mui/material'
import { canSimulate, type TrajectoryData } from '../../AnimateSimulate'
import type { PylinkJoint } from '../types'
import { DimensionVariationConfig } from './DimensionVariationConfig'

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

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

export interface OptimizationBoundsConfigProps {
  linkageDoc?: unknown
  joints: PylinkJoint[]
  trajectoryData?: TrajectoryData | null
  stretchingLinks?: string[]
  optimizationConfig?: MechVariationConfig | Record<string, unknown>
  setOptimizationConfig?: (config: MechVariationConfig | Record<string, unknown>) => void
  dimensionInfo?: DimensionInfo | null
  isLoadingDimensions?: boolean
  dimensionInfoError?: string | null
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const OptimizationBoundsConfig: React.FC<OptimizationBoundsConfigProps> = ({
  linkageDoc,
  joints,
  trajectoryData,
  stretchingLinks = [],
  optimizationConfig: optimizationConfigProp,
  setOptimizationConfig: setOptimizationConfigProp,
  dimensionInfo: dimensionInfoProp,
  isLoadingDimensions: isLoadingDimensionsProp = false,
  dimensionInfoError: dimensionInfoErrorProp = null
}) => {
  const hasCrank = canSimulate(joints)

  // Use props for dimension info (fetched by parent)
  const dimensionInfo = dimensionInfoProp
  const isLoadingDimensions = isLoadingDimensionsProp
  const dimensionInfoError = dimensionInfoErrorProp

  // Default optimization config
  const defaultOptimizationConfig: MechVariationConfig = {
    dimension_variation: {
      default_variation_range: 2.0,
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

  // Internal state if prop not provided
  const [internalOptimizationConfig, setInternalOptimizationConfig] = useState<Record<string, unknown>>(defaultOptimizationConfig as Record<string, unknown>)
  const optimizationConfigState = optimizationConfigProp !== undefined ? (optimizationConfigProp as Record<string, unknown>) : internalOptimizationConfig

  // Use ref to track current state for wrapper function
  const optimizationConfigStateRef = useRef(optimizationConfigState)
  optimizationConfigStateRef.current = optimizationConfigState

  const setOptimizationConfigState = setOptimizationConfigProp
    ? (updater: Record<string, unknown> | ((prev: Record<string, unknown> | null | undefined) => Record<string, unknown>)) => {
        if (typeof updater === 'function') {
          const currentState = optimizationConfigStateRef.current || (defaultOptimizationConfig as Record<string, unknown>)
          const newValue = updater(currentState)
          setOptimizationConfigProp(newValue as MechVariationConfig | Record<string, unknown>)
        } else {
          setOptimizationConfigProp(updater as MechVariationConfig | Record<string, unknown>)
        }
      }
    : setInternalOptimizationConfig

  const [dimensionSyncStates, setDimensionSyncStates] = useState<Record<string, boolean>>({})


  // Extract config values
  const config = optimizationConfigState || (defaultOptimizationConfig as Record<string, unknown>)
  const dimVar = (config?.dimension_variation as {
    dimension_overrides?: Record<string, [boolean, number, number]>
    default_enabled?: boolean
    default_variation_range?: number
  } | undefined) || {}

  if (!hasCrank) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Need a valid mechanism with Crank joint
        </Typography>
      </Box>
    )
  }

  return (
    <Box>
      <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#ed6c02', mb: 1, display: 'flex', alignItems: 'center', gap: 1, fontSize: '0.8rem' }}>
        <span>📐</span> Optimization Bounds
      </Typography>

      {dimensionInfo ? (
        <Box sx={{ p: 2, border: '1px solid rgba(0,0,0,0.1)', borderRadius: 1, bgcolor: 'rgba(237, 108, 2, 0.02)' }}>
          <DimensionVariationConfig
            dimensionInfo={dimensionInfo}
            dimensionOverrides={dimVar.dimension_overrides || {}}
            dimensionSyncStates={dimensionSyncStates}
            defaultEnabled={dimVar.default_enabled ?? true}
            defaultVariationRange={dimVar.default_variation_range ?? 2.0}
            onOverrideChange={(name, override) => {
              setOptimizationConfigState((prev: Record<string, unknown> | null | undefined) => {
                const current = prev || optimizationConfigStateRef.current || defaultOptimizationConfig
                return {
                  ...current,
                  dimension_variation: {
                    ...(current.dimension_variation as Record<string, unknown> || {}),
                    dimension_overrides: {
                      ...((current.dimension_variation as { dimension_overrides?: Record<string, [boolean, number, number]> } | undefined)?.dimension_overrides || {}),
                      [name]: override
                    }
                  }
                }
              })
            }}
            onSyncStateChange={(name, synced) => {
              setDimensionSyncStates(prev => ({
                ...prev,
                [name]: synced
              }))
            }}
            onDefaultEnabledChange={(enabled) => {
              setOptimizationConfigState((prev: Record<string, unknown> | null | undefined) => {
                const current = prev || optimizationConfigStateRef.current || defaultOptimizationConfig
                return {
                  ...current,
                  dimension_variation: {
                    ...(current.dimension_variation as Record<string, unknown> || {}),
                    default_enabled: enabled
                  }
                }
              })
            }}
            onDefaultRangeChange={(range) => {
              setOptimizationConfigState((prev: Record<string, unknown> | null | undefined) => {
                const current = prev || optimizationConfigStateRef.current || defaultOptimizationConfig
                return {
                  ...current,
                  dimension_variation: {
                    ...(current.dimension_variation as Record<string, unknown> || {}),
                    default_variation_range: range
                  }
                }
              })
            }}
            compact={true}
            showTopControls={true}
          />
        </Box>
      ) : (
        <Box sx={{ p: 2, textAlign: 'center', border: '1px solid rgba(0,0,0,0.1)', borderRadius: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            {isLoadingDimensions ? 'Loading dimensions...' : dimensionInfoError || 'Dimensions not available'}
          </Typography>
        </Box>
      )}
    </Box>
  )
}

export default OptimizationBoundsConfig
