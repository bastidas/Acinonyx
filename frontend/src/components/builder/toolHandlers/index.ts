/**
 * Tool handlers module
 *
 * Infrastructure for per-tool canvas input handling. Enables extracting
 * handleCanvasMouseDown / handleCanvasMouseMove / handleCanvasClick etc.
 * from BuilderTab into dedicated handler modules per tool mode.
 */

import type { ToolMode } from '../../BuilderTools'
import type { ToolHandler } from './types'
import { baseToolHandler } from './BaseToolHandler'
import { selectToolHandler } from './selectToolHandler'
import { drawLinkToolHandler } from './drawLinkToolHandler'
import { deleteToolHandler } from './deleteToolHandler'
import { measureToolHandler } from './measureToolHandler'
import { groupSelectToolHandler } from './groupSelectToolHandler'
import { drawPolygonToolHandler } from './drawPolygonToolHandler'
import { mergeToolHandler } from './mergeToolHandler'
import { drawPathToolHandler } from './drawPathToolHandler'

export type { ToolHandler, ToolContext, CanvasPoint } from './types'
export type {
  JointWithPosition,
  LinkWithPosition,
  NearestJointResult,
  NearestLinkResult
} from './types'
export { baseToolHandler, createToolHandler, getPointFromContext } from './BaseToolHandler'
export { selectToolHandler } from './selectToolHandler'
export { drawLinkToolHandler } from './drawLinkToolHandler'
export { deleteToolHandler } from './deleteToolHandler'
export { measureToolHandler } from './measureToolHandler'
export { groupSelectToolHandler } from './groupSelectToolHandler'
export { drawPolygonToolHandler } from './drawPolygonToolHandler'
export { mergeToolHandler, handleMergeLinkClick, handleMergePolygonClick, type HandleMergePolygonClickParams } from './mergeToolHandler'
export { drawPathToolHandler } from './drawPathToolHandler'

/** Return the handler for a given tool mode. Other modes use base (no-op) handler. */
export function getToolHandler(mode: ToolMode): ToolHandler {
  switch (mode) {
    case 'select':
      return selectToolHandler
    case 'draw_link':
      return drawLinkToolHandler
    case 'delete':
      return deleteToolHandler
    case 'measure':
      return measureToolHandler
    case 'group_select':
      return groupSelectToolHandler
    case 'draw_polygon':
      return drawPolygonToolHandler
    case 'merge':
      return mergeToolHandler
    case 'draw_path':
      return drawPathToolHandler
    default:
      return baseToolHandler
  }
}
