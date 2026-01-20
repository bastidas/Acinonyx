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
