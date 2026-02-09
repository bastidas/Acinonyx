/**
 * useToolSelectionState
 *
 * Consolidates tool and selection state for BuilderTab: current tool, open toolbars,
 * toolbar positions, hovered tool, selected joints/links, and hovered joint/link/polygon.
 * Replaces nine separate useState calls with one hook (Phase 6.3).
 */

import { useState } from 'react'
import type { ToolMode, ToolbarPosition } from '../../BuilderTools'

const DEFAULT_OPEN_TOOLBARS = new Set<string>(['tools', 'more'])

export interface UseToolSelectionStateReturn {
  toolMode: ToolMode
  setToolMode: React.Dispatch<React.SetStateAction<ToolMode>>
  openToolbars: Set<string>
  setOpenToolbars: React.Dispatch<React.SetStateAction<Set<string>>>
  toolbarPositions: Record<string, ToolbarPosition>
  setToolbarPositions: React.Dispatch<React.SetStateAction<Record<string, ToolbarPosition>>>
  hoveredTool: ToolMode | null
  setHoveredTool: React.Dispatch<React.SetStateAction<ToolMode | null>>
  selectedJoints: string[]
  setSelectedJoints: React.Dispatch<React.SetStateAction<string[]>>
  selectedLinks: string[]
  setSelectedLinks: React.Dispatch<React.SetStateAction<string[]>>
  hoveredJoint: string | null
  setHoveredJoint: React.Dispatch<React.SetStateAction<string | null>>
  hoveredLink: string | null
  setHoveredLink: React.Dispatch<React.SetStateAction<string | null>>
  hoveredPolygonId: string | null
  setHoveredPolygonId: React.Dispatch<React.SetStateAction<string | null>>
}

export function useToolSelectionState(): UseToolSelectionStateReturn {
  const [toolMode, setToolMode] = useState<ToolMode>('select')
  const [openToolbars, setOpenToolbars] = useState<Set<string>>(DEFAULT_OPEN_TOOLBARS)
  const [toolbarPositions, setToolbarPositions] = useState<Record<string, ToolbarPosition>>({})
  const [hoveredTool, setHoveredTool] = useState<ToolMode | null>(null)
  const [selectedJoints, setSelectedJoints] = useState<string[]>([])
  const [selectedLinks, setSelectedLinks] = useState<string[]>([])
  const [hoveredJoint, setHoveredJoint] = useState<string | null>(null)
  const [hoveredLink, setHoveredLink] = useState<string | null>(null)
  const [hoveredPolygonId, setHoveredPolygonId] = useState<string | null>(null)

  return {
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
  }
}
