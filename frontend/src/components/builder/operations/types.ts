/**
 * Types for builder operations
 *
 * These types define the inputs and outputs for pure operation functions.
 */

import { PylinkDocument, StaticJoint, CrankJoint, RevoluteJoint } from '../types'

// Re-export commonly used types
export type { PylinkDocument, StaticJoint, CrankJoint, RevoluteJoint }
export type PylinkJoint = StaticJoint | CrankJoint | RevoluteJoint

// ═══════════════════════════════════════════════════════════════════════════════
// OPERATION RESULTS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Result of a joint deletion operation
 */
export interface JointDeletionResult {
  newDoc: PylinkDocument
  deletedJoints: string[]
  deletedLinks: string[]
  message: string
}

/**
 * Result of a link deletion operation
 */
export interface LinkDeletionResult {
  newDoc: PylinkDocument
  deletedLinks: string[]
  orphanedJoints: string[]
  message: string
}

/**
 * Result of a joint move operation
 */
export interface JointMoveResult {
  newDoc: PylinkDocument
  movedJoint: string
}

/**
 * Result of a group translation operation
 */
export interface GroupTranslateResult {
  newDoc: PylinkDocument
  movedJoints: string[]
}

/**
 * Result of a joint merge operation
 */
export interface JointMergeResult {
  newDoc: PylinkDocument
  sourceJoint: string
  targetJoint: string
  deletedLinks: string[]
  message: string
}

/**
 * Result of a link creation operation
 */
export interface LinkCreationResult {
  newDoc: PylinkDocument
  linkName: string
  startJoint: string
  endJoint: string
  createdJoints: string[]
  message: string
}

/**
 * Result of a rename operation
 */
export interface RenameResult {
  newDoc: PylinkDocument
  oldName: string
  newName: string
  success: boolean
  error?: string
}

/**
 * Result of an update property operation
 */
export interface UpdatePropertyResult {
  newDoc: PylinkDocument
  success: boolean
  message?: string
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTION TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Function to get a joint's position
 */
export type GetJointPositionFn = (jointName: string) => [number, number] | null

/**
 * Function to calculate distance between two points
 */
export type CalculateDistanceFn = (p1: [number, number], p2: [number, number]) => number

/**
 * Function to find joints connected via links
 */
export type FindConnectedJointsFn = (jointName: string) => string[]
