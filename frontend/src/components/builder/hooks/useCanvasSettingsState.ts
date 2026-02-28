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
  linkTransparency: number
  setLinkTransparency: React.Dispatch<React.SetStateAction<number>>
  linkColorMode: 'various' | 'z-level' | 'single'
  setLinkColorMode: React.Dispatch<React.SetStateAction<'various' | 'z-level' | 'single'>>
  linkColorSingle: string
  setLinkColorSingle: React.Dispatch<React.SetStateAction<string>>
  jointOutline: number
  setJointOutline: React.Dispatch<React.SetStateAction<number>>
  trajectoryDotSize: number
  setTrajectoryDotSize: React.Dispatch<React.SetStateAction<number>>
  trajectoryDotOutline: boolean
  setTrajectoryDotOutline: React.Dispatch<React.SetStateAction<boolean>>
  trajectoryDotOpacity: number
  setTrajectoryDotOpacity: React.Dispatch<React.SetStateAction<number>>
  showTrajectoryStepNumbers: boolean
  setShowTrajectoryStepNumbers: React.Dispatch<React.SetStateAction<boolean>>
  selectionHighlightColor: SelectionHighlightColorOption
  setSelectionHighlightColor: React.Dispatch<React.SetStateAction<SelectionHighlightColorOption>>
  trajectoryStyle: TrajectoryStyleOption
  setTrajectoryStyle: React.Dispatch<React.SetStateAction<TrajectoryStyleOption>>
  canvasDimensions: CanvasDimensions
  setCanvasDimensions: React.Dispatch<React.SetStateAction<CanvasDimensions>>
  exploreRadius: number
  setExploreRadius: React.Dispatch<React.SetStateAction<number>>
  exploreRadialSamples: number
  setExploreRadialSamples: React.Dispatch<React.SetStateAction<number>>
  exploreAzimuthalSamples: number
  setExploreAzimuthalSamples: React.Dispatch<React.SetStateAction<number>>
  exploreNMaxCombinatorial: number
  setExploreNMaxCombinatorial: React.Dispatch<React.SetStateAction<number>>
  exploreColormapEnabled: boolean
  setExploreColormapEnabled: React.Dispatch<React.SetStateAction<boolean>>
  exploreColormapType: 'rainbow' | 'twilight' | 'husl'
  setExploreColormapType: React.Dispatch<React.SetStateAction<'rainbow' | 'twilight' | 'husl'>>
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
  const [linkTransparency, setLinkTransparency] = useState(initial.linkTransparency)
  const [linkColorMode, setLinkColorMode] = useState(initial.linkColorMode)
  const [linkColorSingle, setLinkColorSingle] = useState(initial.linkColorSingle)
  const [jointOutline, setJointOutline] = useState(initial.jointOutline)
  const [trajectoryDotSize, setTrajectoryDotSize] = useState(initial.trajectoryDotSize)
  const [trajectoryDotOutline, setTrajectoryDotOutline] = useState(initial.trajectoryDotOutline)
  const [trajectoryDotOpacity, setTrajectoryDotOpacity] = useState(initial.trajectoryDotOpacity)
  const [showTrajectoryStepNumbers, setShowTrajectoryStepNumbers] = useState(initial.showTrajectoryStepNumbers)
  const [selectionHighlightColor, setSelectionHighlightColor] = useState<SelectionHighlightColorOption>(initial.selectionHighlightColor)
  const [trajectoryStyle, setTrajectoryStyle] = useState<TrajectoryStyleOption>(initial.trajectoryStyle)
  const [canvasDimensions, setCanvasDimensions] = useState<CanvasDimensions>({ width: 1200, height: 700 })
  const [exploreRadius, setExploreRadius] = useState(initial.exploreRadius)
  const [exploreRadialSamples, setExploreRadialSamples] = useState(initial.exploreRadialSamples)
  const [exploreAzimuthalSamples, setExploreAzimuthalSamples] = useState(initial.exploreAzimuthalSamples)
  const [exploreNMaxCombinatorial, setExploreNMaxCombinatorial] = useState(initial.exploreNMaxCombinatorial)
  const [exploreColormapEnabled, setExploreColormapEnabled] = useState(initial.exploreColormapEnabled)
  const [exploreColormapType, setExploreColormapType] = useState(initial.exploreColormapType)

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
      linkTransparency,
      linkColorMode,
      linkColorSingle,
      jointOutline,
      trajectoryDotSize,
      trajectoryDotOutline,
      trajectoryDotOpacity,
      showTrajectoryStepNumbers,
      selectionHighlightColor,
      trajectoryStyle,
      exploreRadius,
      exploreRadialSamples,
      exploreAzimuthalSamples,
      exploreNMaxCombinatorial,
      exploreColormapEnabled,
      exploreColormapType
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
    linkTransparency,
    linkColorMode,
    linkColorSingle,
    jointOutline,
    trajectoryDotSize,
    trajectoryDotOutline,
    trajectoryDotOpacity,
    showTrajectoryStepNumbers,
    selectionHighlightColor,
    trajectoryStyle,
    exploreRadius,
    exploreRadialSamples,
    exploreAzimuthalSamples,
    exploreNMaxCombinatorial,
    exploreColormapEnabled,
    exploreColormapType
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
    linkTransparency,
    setLinkTransparency,
    linkColorMode,
    setLinkColorMode,
    linkColorSingle,
    setLinkColorSingle,
    jointOutline,
    setJointOutline,
    trajectoryDotSize,
    setTrajectoryDotSize,
    trajectoryDotOutline,
    setTrajectoryDotOutline,
    trajectoryDotOpacity,
    setTrajectoryDotOpacity,
    showTrajectoryStepNumbers,
    setShowTrajectoryStepNumbers,
    selectionHighlightColor,
    setSelectionHighlightColor,
    trajectoryStyle,
    setTrajectoryStyle,
    canvasDimensions,
    setCanvasDimensions,
    exploreRadius,
    setExploreRadius,
    exploreRadialSamples,
    setExploreRadialSamples,
    exploreAzimuthalSamples,
    setExploreAzimuthalSamples,
    exploreNMaxCombinatorial,
    setExploreNMaxCombinatorial,
    exploreColormapEnabled,
    setExploreColormapEnabled,
    exploreColormapType,
    setExploreColormapType
  }
}
