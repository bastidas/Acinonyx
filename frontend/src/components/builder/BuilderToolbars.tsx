/**
 * BuilderToolbars
 *
 * Renders the set of draggable floating toolbars (Tools, Links, Nodes, More, Optimize, Settings).
 * Position/dimension logic and content are provided by the parent (BuilderTab) via callbacks.
 */

import React from 'react'
import {
  DraggableToolbar,
  type ToolbarPosition,
  type ToolbarConfig,
  type ToolbarDimensions,
  type ToolbarViewportBounds
} from '../BuilderTools'

export interface BuilderToolbarsProps {
  openToolbars: Set<string>
  toolbarConfigs: ToolbarConfig[]
  getToolbarPosition: (id: string) => ToolbarPosition
  getToolbarHeight: (id: string) => number | undefined
  getToolbarDimensions: (id: string) => ToolbarDimensions
  onToggleToolbar: (id: string) => void
  onPositionChange: (id: string, position: ToolbarPosition) => void
  onHeightChange: (id: string, height: number) => void
  renderToolbarContent: (id: string) => React.ReactNode
  viewportBounds: ToolbarViewportBounds
  onInteract?: () => void
}

export function BuilderToolbars({
  openToolbars,
  toolbarConfigs,
  getToolbarPosition,
  getToolbarHeight,
  getToolbarDimensions,
  onToggleToolbar,
  onPositionChange,
  onHeightChange,
  renderToolbarContent,
  viewportBounds,
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
            initialHeight={getToolbarHeight(toolbarId)}
            onClose={() => onToggleToolbar(toolbarId)}
            onPositionChange={onPositionChange}
            onHeightChange={onHeightChange}
            onInteract={onInteract}
            minWidth={dimensions.minWidth}
            minHeight={dimensions.minHeight}
            defaultHeight={dimensions.defaultHeight}
            maxHeight={dimensions.maxHeight}
            viewportBounds={viewportBounds}
          >
            {renderToolbarContent(toolbarId)}
          </DraggableToolbar>
        )
      })}
    </>
  )
}
