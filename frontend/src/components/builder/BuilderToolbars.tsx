/**
 * BuilderToolbars
 *
 * Renders the set of draggable floating toolbars (Tools, Links, Nodes, More, Optimize, Settings).
 * Position/dimension logic and content are provided by the parent (BuilderTab) via callbacks.
 */

import React from 'react'
import { DraggableToolbar, type ToolbarPosition, type ToolbarConfig } from '../BuilderTools'

export interface BuilderToolbarsProps {
  openToolbars: Set<string>
  toolbarConfigs: ToolbarConfig[]
  getToolbarPosition: (id: string) => ToolbarPosition
  getToolbarDimensions: (id: string) => { minWidth: number; maxHeight: number }
  onToggleToolbar: (id: string) => void
  onPositionChange: (id: string, position: ToolbarPosition) => void
  renderToolbarContent: (id: string) => React.ReactNode
  onInteract?: () => void
}

export function BuilderToolbars({
  openToolbars,
  toolbarConfigs,
  getToolbarPosition,
  getToolbarDimensions,
  onToggleToolbar,
  onPositionChange,
  renderToolbarContent,
  onInteract
}: BuilderToolbarsProps): JSX.Element {
  return (
    <>
      {Array.from(openToolbars).map(toolbarId => {
        const config = toolbarConfigs.find(c => c.id === toolbarId)
        if (!config) return null
        const dimensions = getToolbarDimensions(toolbarId)
        return (
          <DraggableToolbar
            key={toolbarId}
            id={toolbarId}
            title={config.title}
            icon={config.icon}
            initialPosition={getToolbarPosition(toolbarId)}
            onClose={() => onToggleToolbar(toolbarId)}
            onPositionChange={onPositionChange}
            onInteract={onInteract}
            minWidth={dimensions.minWidth}
            maxHeight={dimensions.maxHeight}
          >
            {renderToolbarContent(toolbarId)}
          </DraggableToolbar>
        )
      })}
    </>
  )
}
