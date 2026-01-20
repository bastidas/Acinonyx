/**
 * Type definitions for the Builder component
 *
 * These types are specific to the Builder UI component.
 * Core linkage types (Node, Edge, LinkageDocument) are in src/types/pylink.ts
 */

// Re-export core types for convenience
export type {
  LinkageDocument,
  Node,
  Edge,
  NodeId,
  EdgeId,
  Position,
  NodeRole,
  NodeMeta,
  EdgeMeta,
  HypergraphLinkage
} from '../../types'

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY TYPE ALIASES (for backward compatibility during migration)
// These will be removed once all code is migrated to use the new types directly
// ═══════════════════════════════════════════════════════════════════════════════

export interface JointRef {
  ref: string
}

export interface StaticJoint {
  type: 'Static'
  name: string
  x: number
  y: number
}

export interface CrankJoint {
  type: 'Crank'
  name: string
  joint0: JointRef
  distance: number
  angle: number
}

export interface RevoluteJoint {
  type: 'Revolute'
  name: string
  joint0: JointRef
  joint1: JointRef
  distance0: number
  distance1: number
}

export type PylinkJoint = StaticJoint | CrankJoint | RevoluteJoint

export interface PylinkageData {
  name: string
  joints: PylinkJoint[]
  solve_order: string[]
}

export interface JointMeta {
  color: string
  zlevel: number
  x?: number
  y?: number
  show_path?: boolean
}

export interface LinkMeta {
  color: string
  connects: string[]
  isGround?: boolean
}

export interface UIMeta {
  joints: Record<string, JointMeta>
  links: Record<string, LinkMeta>
}

/** Legacy document format - will be replaced with LinkageDocument */
export interface PylinkDocument {
  name: string
  pylinkage: PylinkageData
  meta: UIMeta
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANIMATION STATE
// ═══════════════════════════════════════════════════════════════════════════════

export interface AnimatedPositions {
  [jointName: string]: [number, number]
}

// TrajectoryData is defined in AnimateSimulate.tsx - import from there

// ═══════════════════════════════════════════════════════════════════════════════
// HIGHLIGHT & STYLING
// ═══════════════════════════════════════════════════════════════════════════════

export type HighlightType = 'none' | 'selected' | 'hovered' | 'move_group' | 'merge'
export type ObjectType = 'joint' | 'link' | 'polygon'

export interface HighlightStyle {
  stroke: string
  strokeWidth: number
  filter?: string
}
