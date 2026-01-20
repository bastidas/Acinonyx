/**
 * Tools Toolbar - Tool selection grid for the Builder
 */
import React from 'react'
import { Box, Typography, IconButton, Tooltip, Divider } from '@mui/material'
import { TOOLS, ToolMode, LinkCreationState, initialLinkCreationState } from '../../BuilderTools'

// Layout constants
const TOOL_BUTTON_SIZE = 48
const TOOLS_GRID_GAP = 6
const TOOLS_PADDING = 12
const TOOLS_BOX_WIDTH = (TOOL_BUTTON_SIZE * 3) + (TOOLS_GRID_GAP * 2) + (TOOLS_PADDING * 2)

export interface ToolsToolbarProps {
  toolMode: ToolMode
  setToolMode: (mode: ToolMode) => void
  hoveredTool: ToolMode | null
  setHoveredTool: (tool: ToolMode | null) => void
  linkCreationState: LinkCreationState
  setLinkCreationState: (state: LinkCreationState) => void
  setPreviewLine: (line: { start: [number, number]; end: [number, number] } | null) => void
}

export const ToolsToolbar: React.FC<ToolsToolbarProps> = ({
  toolMode,
  setToolMode,
  hoveredTool,
  setHoveredTool,
  linkCreationState,
  setLinkCreationState,
  setPreviewLine
}) => {
  return (
    <Box sx={{ p: `${TOOLS_PADDING}px`, width: TOOLS_BOX_WIDTH, boxSizing: 'border-box' }}>
      <Box sx={{
        display: 'grid',
        gridTemplateColumns: `repeat(3, ${TOOL_BUTTON_SIZE}px)`,
        gap: `${TOOLS_GRID_GAP}px`,
        justifyContent: 'center'
      }}>
        {TOOLS.map(tool => {
          const isActive = toolMode === tool.id
          const isHovered = hoveredTool === tool.id
          const isDelete = tool.id === 'delete'

          return (
            <Tooltip
              key={tool.id}
              title={
                <Box sx={{ p: 0.5 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {tool.label} {tool.shortcut && `(${tool.shortcut})`}
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.5, opacity: 0.9 }}>
                    {tool.description}
                  </Typography>
                </Box>
              }
              placement="bottom"
              arrow
            >
              <IconButton
                onClick={() => {
                  if (linkCreationState.isDrawing && tool.id !== 'draw_link') {
                    setLinkCreationState(initialLinkCreationState)
                    setPreviewLine(null)
                  }
                  setToolMode(tool.id)
                }}
                onMouseEnter={() => setHoveredTool(tool.id)}
                onMouseLeave={() => setHoveredTool(null)}
                sx={{
                  width: TOOL_BUTTON_SIZE,
                  height: TOOL_BUTTON_SIZE,
                  minWidth: TOOL_BUTTON_SIZE,
                  maxWidth: TOOL_BUTTON_SIZE,
                  borderRadius: 2,
                  fontSize: isDelete ? '1.2rem' : '1.5rem',
                  overflow: 'hidden',
                  backgroundColor: isActive
                    ? (isDelete ? '#d32f2f' : 'primary.main')
                    : (isHovered ? 'rgba(0,0,0,0.04)' : 'transparent'),
                  color: isActive ? 'white' : (isDelete && isHovered ? '#d32f2f' : 'text.primary'),
                  border: isActive ? 'none' : '1px solid transparent',
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    backgroundColor: isActive
                      ? (isDelete ? '#b71c1c' : 'primary.dark')
                      : (isDelete ? 'rgba(211, 47, 47, 0.08)' : 'rgba(0,0,0,0.06)'),
                    border: isActive ? 'none' : '1px solid rgba(0,0,0,0.12)'
                  }
                }}
              >
                <span style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%',
                  height: '100%',
                  lineHeight: 1
                }}>
                  {tool.icon}
                </span>
              </IconButton>
            </Tooltip>
          )
        })}
      </Box>

      <Divider sx={{ my: 1 }} />

      <Box sx={{ px: 0.5 }}>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
          {TOOLS.find(t => t.id === toolMode)?.label}
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
          {TOOLS.find(t => t.id === toolMode)?.description}
        </Typography>
      </Box>
    </Box>
  )
}

export default ToolsToolbar
