/**
 * BuilderCanvasArea
 *
 * Paper-wrapped canvas with CanvasRenderer, FooterToolbar, and ToolbarToggleButtons.
 * State and handlers are owned by the parent (BuilderTab); this component is presentational.
 * Optional children (e.g. DraggableToolbars) are rendered inside the Paper after the footer.
 */

import React from 'react'
import { Box, Paper, Button } from '@mui/material'
import {
  FooterToolbar,
  ToolbarToggleButtons,
  type ToolMode,
  type StatusMessage,
  type LinkCreationState,
  type PolygonDrawState,
  type MeasureState,
  type GroupSelectionState,
  type MergePolygonState,
  type PathDrawState
} from '../BuilderTools'
import { CanvasRenderer, SVGFilters } from './rendering'
import type { CanvasLayerRender } from './rendering'

export interface BuilderCanvasAreaOptimization {
  isOptimizedMechanism: boolean
  isSyncedToOptimizer: boolean
  syncToOptimizerResult: () => void
}

export interface BuilderCanvasAreaProps {
  canvasRef: React.RefObject<HTMLDivElement | null>
  canvasBgColor: string
  cursor: string
  optimization: BuilderCanvasAreaOptimization
  showGrid: boolean
  onMouseDown?: (event: React.MouseEvent<SVGSVGElement>) => void
  onMouseMove?: (event: React.MouseEvent<SVGSVGElement>) => void
  onMouseUp?: (event: React.MouseEvent<SVGSVGElement>) => void
  onMouseLeave?: (event: React.MouseEvent<SVGSVGElement>) => void
  onClick?: (event: React.MouseEvent<SVGSVGElement>) => void
  onDoubleClick?: (event: React.MouseEvent<SVGSVGElement>) => void
  renderGrid?: CanvasLayerRender
  renderDrawnObjects?: CanvasLayerRender
  renderLinks?: CanvasLayerRender
  renderPreviewLine?: CanvasLayerRender
  renderPolygonPreview?: CanvasLayerRender
  renderTargetPaths?: CanvasLayerRender
  renderPathPreview?: CanvasLayerRender
  renderTrajectories?: CanvasLayerRender
  renderJoints?: CanvasLayerRender
  renderSelectionBox?: CanvasLayerRender
  renderMeasurementMarkers?: CanvasLayerRender
  renderMeasurementLine?: CanvasLayerRender
  toolMode: ToolMode
  jointCount: number
  linkCount: number
  selectedJoints: string[]
  selectedLinks: string[]
  statusMessage: StatusMessage | null
  linkCreationState: LinkCreationState
  polygonDrawState?: PolygonDrawState
  measureState?: MeasureState
  groupSelectionState?: GroupSelectionState
  mergePolygonState?: MergePolygonState
  pathDrawState?: PathDrawState
  canvasWidth?: number
  onCancelAction?: () => void
  openToolbars: Set<string>
  onToggleToolbar: (id: string) => void
  onToolbarInteract?: () => void
  darkMode?: boolean
  children?: React.ReactNode
}

export function BuilderCanvasArea({
  canvasRef,
  canvasBgColor,
  cursor,
  optimization,
  showGrid,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onMouseLeave,
  onClick,
  onDoubleClick,
  renderGrid,
  renderDrawnObjects,
  renderLinks,
  renderPreviewLine,
  renderPolygonPreview,
  renderTargetPaths,
  renderPathPreview,
  renderTrajectories,
  renderJoints,
  renderSelectionBox,
  renderMeasurementMarkers,
  renderMeasurementLine,
  toolMode,
  jointCount,
  linkCount,
  selectedJoints,
  selectedLinks,
  statusMessage,
  linkCreationState,
  polygonDrawState,
  measureState,
  groupSelectionState,
  mergePolygonState,
  pathDrawState,
  canvasWidth,
  onCancelAction,
  openToolbars,
  onToggleToolbar,
  onToolbarInteract,
  darkMode = false,
  children
}: BuilderCanvasAreaProps): JSX.Element {
  return (
    <Paper
      ref={canvasRef}
      sx={{
        position: 'absolute',
        top: '12px',
        left: 0,
        right: 0,
        bottom: '5px',
        width: '100%',
        overflow: 'hidden',
        backgroundColor:
          canvasBgColor === 'default'
            ? 'var(--color-canvas-bg)'
            : canvasBgColor === 'white'
              ? '#ffffff'
              : canvasBgColor === 'cream'
                ? '#FAF3E1'
                : '#1a1a1a',
        border: '1px solid var(--color-border)',
        borderRadius: 2,
        cursor,
        pb: '42px'
      }}
    >
      {optimization.isOptimizedMechanism && (
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            right: 10,
            display: 'flex',
            flexDirection: 'column',
            gap: 1,
            zIndex: 1000
          }}
        >
          <Box
            sx={{
              bgcolor: optimization.isSyncedToOptimizer ? 'success.main' : 'warning.main',
              color: 'white',
              p: 1,
              borderRadius: 1,
              fontSize: '0.875rem',
              fontWeight: 500,
              boxShadow: 2
            }}
          >
            {optimization.isSyncedToOptimizer ? '✓ Synced to Optimizer' : '⚠ Out of Sync'}
          </Box>
          {!optimization.isSyncedToOptimizer && (
            <Button
              variant="contained"
              color="primary"
              size="small"
              onClick={optimization.syncToOptimizerResult}
              sx={{ boxShadow: 2, fontSize: '0.75rem', py: 0.5 }}
            >
              Sync to Optimizer Result
            </Button>
          )}
        </Box>
      )}

      <CanvasRenderer
        cursor={cursor}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        filters={<SVGFilters />}
        showGrid={showGrid}
        renderGrid={renderGrid}
        renderDrawnObjects={renderDrawnObjects}
        renderLinks={renderLinks}
        renderPreviewLine={renderPreviewLine}
        renderPolygonPreview={renderPolygonPreview}
        renderTargetPaths={renderTargetPaths}
        renderPathPreview={renderPathPreview}
        renderTrajectories={renderTrajectories}
        renderJoints={renderJoints}
        renderSelectionBox={renderSelectionBox}
        renderMeasurementMarkers={renderMeasurementMarkers}
        renderMeasurementLine={renderMeasurementLine}
      />

      <FooterToolbar
        toolMode={toolMode}
        jointCount={jointCount}
        linkCount={linkCount}
        selectedJoints={selectedJoints}
        selectedLinks={selectedLinks}
        statusMessage={statusMessage}
        linkCreationState={linkCreationState}
        polygonDrawState={polygonDrawState}
        measureState={measureState}
        groupSelectionState={groupSelectionState}
        mergePolygonState={mergePolygonState}
        pathDrawState={pathDrawState}
        canvasWidth={canvasWidth}
        onCancelAction={onCancelAction}
        darkMode={darkMode}
      />

      <ToolbarToggleButtons
        openToolbars={openToolbars}
        onToggleToolbar={onToggleToolbar}
        darkMode={darkMode}
        onInteract={onToolbarInteract}
      />

      {children}
    </Paper>
  )
}
