/**
 * useCanvasSettingsState
 *
 * Consolidates canvas/settings state for BuilderTab (Phase 6.3 — group 3).
 * Display preferences, simulation/settings panel state, and canvas dimensions.
 * Dark mode is initialized from cookie or system prefer-color-scheme and persisted to cookie on change.
 */

import { useState, useEffect } from 'react'
import type { ColorCycleType } from '../../../theme'
import { getEffectiveTheme, setThemePreference, getStoredSettings, setStoredSettings } from '../../../prefs'

export type CanvasBgColor = 'default' | 'white' | 'cream' | 'dark'
export type TrajectoryStyleOption = 'dots' | 'line' | 'both'
export type SelectionHighlightColorOption = 'blue' | 'orange' | 'green' | 'purple'

export interface CanvasDimensions {
  width: number
  height: number
}

export interface UseCanvasSettingsStateReturn {
  simulationSteps: number
  setSimulationSteps: React.Dispatch<React.SetStateAction<number>>
  simulationStepsInput: string
  setSimulationStepsInput: React.Dispatch<React.SetStateAction<string>>
  mechanismVersion: number
  setMechanismVersion: React.Dispatch<React.SetStateAction<number>>
  showTrajectory: boolean
  setShowTrajectory: React.Dispatch<React.SetStateAction<boolean>>
  autoSimulateDelayMs: number
  setAutoSimulateDelayMs: React.Dispatch<React.SetStateAction<number>>
  jointMergeRadius: number
  setJointMergeRadius: React.Dispatch<React.SetStateAction<number>>
  trajectoryColorCycle: ColorCycleType
  setTrajectoryColorCycle: React.Dispatch<React.SetStateAction<ColorCycleType>>
  darkMode: boolean
  setDarkMode: React.Dispatch<React.SetStateAction<boolean>>
  showGrid: boolean
  setShowGrid: React.Dispatch<React.SetStateAction<boolean>>
  showJointLabels: boolean
  setShowJointLabels: React.Dispatch<React.SetStateAction<boolean>>
  showLinkLabels: boolean
  setShowLinkLabels: React.Dispatch<React.SetStateAction<boolean>>
  canvasBgColor: CanvasBgColor
  setCanvasBgColor: React.Dispatch<React.SetStateAction<CanvasBgColor>>
  jointSize: number
  setJointSize: React.Dispatch<React.SetStateAction<number>>
  linkThickness: number
  setLinkThickness: React.Dispatch<React.SetStateAction<number>>
  trajectoryDotSize: number
  setTrajectoryDotSize: React.Dispatch<React.SetStateAction<number>>
  trajectoryDotOutline: boolean
  setTrajectoryDotOutline: React.Dispatch<React.SetStateAction<boolean>>
  trajectoryDotOpacity: number
  setTrajectoryDotOpacity: React.Dispatch<React.SetStateAction<number>>
  selectionHighlightColor: SelectionHighlightColorOption
  setSelectionHighlightColor: React.Dispatch<React.SetStateAction<SelectionHighlightColorOption>>
  trajectoryStyle: TrajectoryStyleOption
  setTrajectoryStyle: React.Dispatch<React.SetStateAction<TrajectoryStyleOption>>
  canvasDimensions: CanvasDimensions
  setCanvasDimensions: React.Dispatch<React.SetStateAction<CanvasDimensions>>
}

function getInitialSettings() {
  return getStoredSettings()
}

export function useCanvasSettingsState(): UseCanvasSettingsStateReturn {
  const initial = getInitialSettings()
  const [simulationSteps, setSimulationSteps] = useState(initial.simulationSteps)
  const [simulationStepsInput, setSimulationStepsInput] = useState(String(initial.simulationSteps))
  const [mechanismVersion, setMechanismVersion] = useState(0)
  const [showTrajectory, setShowTrajectory] = useState(true)
  const [autoSimulateDelayMs, setAutoSimulateDelayMs] = useState(initial.autoSimulateDelayMs)
  const [jointMergeRadius, setJointMergeRadius] = useState(initial.jointMergeRadius)
  const [trajectoryColorCycle, setTrajectoryColorCycle] = useState<ColorCycleType>(initial.trajectoryColorCycle)
  const [darkMode, setDarkMode] = useState(() => getEffectiveTheme() === 'dark')
  useEffect(() => {
    setThemePreference(darkMode ? 'dark' : 'light')
  }, [darkMode])
  const [showGrid, setShowGrid] = useState(initial.showGrid)
  const [showJointLabels, setShowJointLabels] = useState(initial.showJointLabels)
  const [showLinkLabels, setShowLinkLabels] = useState(initial.showLinkLabels)
  const [canvasBgColor, setCanvasBgColor] = useState<CanvasBgColor>(initial.canvasBgColor)
  const [jointSize, setJointSize] = useState(initial.jointSize)
  const [linkThickness, setLinkThickness] = useState(initial.linkThickness)
  const [trajectoryDotSize, setTrajectoryDotSize] = useState(initial.trajectoryDotSize)
  const [trajectoryDotOutline, setTrajectoryDotOutline] = useState(initial.trajectoryDotOutline)
  const [trajectoryDotOpacity, setTrajectoryDotOpacity] = useState(initial.trajectoryDotOpacity)
  const [selectionHighlightColor, setSelectionHighlightColor] = useState<SelectionHighlightColorOption>(initial.selectionHighlightColor)
  const [trajectoryStyle, setTrajectoryStyle] = useState<TrajectoryStyleOption>(initial.trajectoryStyle)
  const [canvasDimensions, setCanvasDimensions] = useState<CanvasDimensions>({ width: 1200, height: 700 })

  // Persist settings to cookie when any persisted value changes
  useEffect(() => {
    setStoredSettings({
      showGrid,
      showJointLabels,
      showLinkLabels,
      canvasBgColor,
      simulationSteps,
      autoSimulateDelayMs,
      trajectoryColorCycle,
      jointMergeRadius,
      jointSize,
      linkThickness,
      trajectoryDotSize,
      trajectoryDotOutline,
      trajectoryDotOpacity,
      selectionHighlightColor,
      trajectoryStyle
    })
  }, [
    showGrid,
    showJointLabels,
    showLinkLabels,
    canvasBgColor,
    simulationSteps,
    autoSimulateDelayMs,
    trajectoryColorCycle,
    jointMergeRadius,
    jointSize,
    linkThickness,
    trajectoryDotSize,
    trajectoryDotOutline,
    trajectoryDotOpacity,
    selectionHighlightColor,
    trajectoryStyle
  ])

  return {
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
    setSelectionHighlightColor,
    trajectoryStyle,
    setTrajectoryStyle,
    canvasDimensions,
    setCanvasDimensions
  }
}
