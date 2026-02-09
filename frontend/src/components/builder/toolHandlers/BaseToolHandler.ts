/**
 * Base tool handler
 *
 * Default no-op implementation of ToolHandler. Concrete handlers can extend
 * or compose this and override only the events they need.
 */

import type { ToolHandler, ToolContext, CanvasPoint } from './types'

/** No-op handler: all methods return false (not handled). */
export const baseToolHandler: ToolHandler = {
  onMouseDown(): boolean {
    return false
  },
  onMouseMove(): boolean {
    return false
  },
  onMouseUp(): boolean {
    return false
  },
  onClick(): boolean {
    return false
  },
  onDoubleClick(): boolean {
    return false
  }
}

/**
 * Create a handler that only implements the provided methods; any omitted
 * event is treated as not handled (false).
 */
export function createToolHandler(partial: Partial<ToolHandler>): ToolHandler {
  return {
    ...baseToolHandler,
    ...partial
  }
}

/**
 * Get canvas point from event using context. Use when a handler needs to
 * recompute point (e.g. from a stored event). Returns null if canvas ref
 * is not ready.
 */
export function getPointFromContext(
  context: ToolContext,
  event: React.MouseEvent<SVGSVGElement>
): CanvasPoint | null {
  return context.getPointFromEvent(event)
}
