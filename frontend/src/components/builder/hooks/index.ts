/**
 * Builder Hooks Module
 *
 * Custom React hooks for the Builder component.
 * These hooks extract common functionality for better organization and reusability.
 *
 * Available Hooks:
 * - useKeyboardShortcuts: Keyboard event handling
 * - useCanvasInteraction: Canvas coordinates and element detection
 * - useStatusMessage: Status message display with auto-dismiss
 */

// Keyboard Shortcuts
export { useKeyboardShortcuts } from './useKeyboardShortcuts'
export type { KeyboardShortcutsConfig } from './useKeyboardShortcuts'

// Canvas Interaction
export { useCanvasInteraction } from './useCanvasInteraction'
export type {
  UseCanvasInteractionConfig,
  UseCanvasInteractionReturn,
  CanvasCoordinates,
  JointWithPosition,
  LinkWithPosition,
  NearestResult
} from './useCanvasInteraction'

// Status Message
export { useStatusMessage } from './useStatusMessage'
export type {
  StatusType,
  StatusMessage,
  UseStatusMessageReturn
} from './useStatusMessage'

// Move Group
export { useMoveGroup } from './useMoveGroup'
export type { UseMoveGroupParams, UseMoveGroupReturn, CanvasPoint as MoveGroupCanvasPoint } from './useMoveGroup'

// Modal State (Phase 6.2 — consolidated modal state)
export { useModalState } from './useModalState'
export type { UseModalStateReturn, DeleteConfirmDialogState as ModalDeleteConfirmState } from './useModalState'

// Tool + Selection State (Phase 6.3)
export { useToolSelectionState } from './useToolSelectionState'
export type { UseToolSelectionStateReturn } from './useToolSelectionState'

// Drawn / Path / Merge State (Phase 6.3)
export { useDrawnPathState } from './useDrawnPathState'
export type { UseDrawnPathStateReturn } from './useDrawnPathState'

// Tool Flows State (Phase 6.3 — link creation, drag, group select, polygon draw, measure, move group)
export { useToolFlowsState } from './useToolFlowsState'
export type { UseToolFlowsStateReturn, PreviewLine } from './useToolFlowsState'

// Status State (Phase 6.3 — group 7)
export { useStatusState } from './useStatusState'
export type { UseStatusStateReturn } from './useStatusState'

// Canvas/Settings State (Phase 6.3 — group 3)
export { useCanvasSettingsState } from './useCanvasSettingsState'
export type {
  UseCanvasSettingsStateReturn,
  CanvasBgColor as CanvasSettingsBgColor,
  CanvasDimensions as CanvasSettingsDimensions,
  TrajectoryStyleOption,
  SelectionHighlightColorOption
} from './useCanvasSettingsState'

// Document State (Phase 6 — group 4)
export { useDocumentState } from './useDocumentState'
export type { UseDocumentStateReturn } from './useDocumentState'

// Canvas layer renders (data prep + render wiring for BuilderCanvasArea)
export { useCanvasLayerRenders } from './useCanvasLayerRenders'
export type { UseCanvasLayerRendersParams, UseCanvasLayerRendersReturn } from './useCanvasLayerRenders'
