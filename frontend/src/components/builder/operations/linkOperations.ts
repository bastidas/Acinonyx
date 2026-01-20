/**
 * Link Operations
 *
 * Pure functions for creating, updating, deleting, and manipulating links.
 * These functions take the current document state and return the new state.
 */

import {
  PylinkDocument,
  StaticJoint,
  RevoluteJoint,
  LinkDeletionResult,
  LinkCreationResult,
  RenameResult,
  UpdatePropertyResult,
  GetJointPositionFn,
  CalculateDistanceFn,
  FindConnectedJointsFn
} from './types'
import {
  calculateLinkDeletionResult,
  generateJointName,
  generateLinkName,
  getDefaultColor
} from '../../BuilderTools'

// ═══════════════════════════════════════════════════════════════════════════════
// DELETE LINK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Delete a link and any orphan joints
 *
 * @param doc - Current document state
 * @param linkName - Name of link to delete
 * @returns Result with new document and deletion details
 */
export function deleteLink(
  doc: PylinkDocument,
  linkName: string
): LinkDeletionResult {
  const result = calculateLinkDeletionResult(linkName, doc.meta.links)

  // Create new state
  const newLinks = { ...doc.meta.links }
  const newJoints = { ...doc.meta.joints }
  const newPylinkageJoints = doc.pylinkage.joints.filter(
    j => !result.jointsToDelete.includes(j.name)
  )
  const newSolveOrder = doc.pylinkage.solve_order.filter(
    name => !result.jointsToDelete.includes(name)
  )

  // Remove the link
  delete newLinks[linkName]

  // Remove orphan joint metadata
  result.jointsToDelete.forEach(jointName => {
    delete newJoints[jointName]
  })

  const newDoc: PylinkDocument = {
    ...doc,
    pylinkage: {
      ...doc.pylinkage,
      joints: newPylinkageJoints,
      solve_order: newSolveOrder
    },
    meta: {
      joints: newJoints,
      links: newLinks
    }
  }

  const orphanMsg = result.jointsToDelete.length > 0
    ? ` + ${result.jointsToDelete.length} orphan(s)`
    : ''

  return {
    newDoc,
    deletedLinks: [linkName],
    orphanedJoints: result.jointsToDelete,
    message: `Deleted ${linkName}${orphanMsg}`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CREATE LINK (with Static joints)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a new link between two points/joints using Static joints for new endpoints
 *
 * @param doc - Current document state
 * @param startPoint - Start position
 * @param endPoint - End position
 * @param startJointName - Existing joint name at start (or null to create new)
 * @param endJointName - Existing joint name at end (or null to create new)
 * @param calculateDistance - Distance calculation function
 * @returns Result with new document and creation details
 */
export function createLinkWithStaticJoints(
  doc: PylinkDocument,
  startPoint: [number, number],
  endPoint: [number, number],
  startJointName: string | null,
  endJointName: string | null,
  calculateDistance: CalculateDistanceFn
): LinkCreationResult {
  const existingJointNames = doc.pylinkage.joints.map(j => j.name)
  const existingLinkNames = Object.keys(doc.meta.links)
  const newPylinkageJoints = [...doc.pylinkage.joints]
  const createdJoints: string[] = []

  let actualStartJoint = startJointName
  let actualEndJoint = endJointName

  // Create start joint if needed
  if (!actualStartJoint) {
    actualStartJoint = generateJointName(existingJointNames)
    const newJoint: StaticJoint = {
      type: 'Static',
      name: actualStartJoint,
      x: startPoint[0],
      y: startPoint[1]
    }
    newPylinkageJoints.push(newJoint)
    existingJointNames.push(actualStartJoint)
    createdJoints.push(actualStartJoint)
  }

  // Create end joint if needed
  if (!actualEndJoint) {
    actualEndJoint = generateJointName(existingJointNames)
    const newJoint: StaticJoint = {
      type: 'Static',
      name: actualEndJoint,
      x: endPoint[0],
      y: endPoint[1]
    }
    newPylinkageJoints.push(newJoint)
    createdJoints.push(actualEndJoint)
  }

  // Create the link
  const linkName = generateLinkName(existingLinkNames)
  const linkColor = getDefaultColor(existingLinkNames.length)

  const newMetaLinks = { ...doc.meta.links }
  newMetaLinks[linkName] = {
    color: linkColor,
    connects: [actualStartJoint, actualEndJoint]
  }

  // Update solve order
  const newSolveOrder = [...doc.pylinkage.solve_order]
  if (!newSolveOrder.includes(actualStartJoint)) {
    newSolveOrder.push(actualStartJoint)
  }
  if (!newSolveOrder.includes(actualEndJoint)) {
    newSolveOrder.push(actualEndJoint)
  }

  const length = calculateDistance(startPoint, endPoint)

  return {
    newDoc: {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: newPylinkageJoints,
        solve_order: newSolveOrder
      },
      meta: {
        ...doc.meta,
        links: newMetaLinks
      }
    },
    linkName,
    startJoint: actualStartJoint,
    endJoint: actualEndJoint,
    createdJoints,
    message: `Created ${linkName} (${length.toFixed(1)} units)`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CREATE LINK (with Revolute default)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create a new link between two points/joints
 *
 * If user clicked on an existing joint, use it. Otherwise create a new joint.
 * New joints are Revolute if they connect to a non-static joint that has other connections,
 * otherwise Static.
 *
 * @param doc - Current document state
 * @param startPoint - Start position
 * @param endPoint - End position
 * @param startJointName - Existing joint at start (or null)
 * @param endJointName - Existing joint at end (or null)
 * @param getJointPosition - Position lookup function
 * @param calculateDistance - Distance calculation function
 * @param findConnectedJoints - Function to find joints connected via links
 * @returns Result with new document and creation details
 */
export function createLinkWithRevoluteDefault(
  doc: PylinkDocument,
  startPoint: [number, number],
  endPoint: [number, number],
  startJointName: string | null,
  endJointName: string | null,
  getJointPosition: GetJointPositionFn,
  calculateDistance: CalculateDistanceFn,
  findConnectedJoints: FindConnectedJointsFn
): LinkCreationResult {
  const existingJointNames = doc.pylinkage.joints.map(j => j.name)
  const existingLinkNames = Object.keys(doc.meta.links)
  let newJoints = [...doc.pylinkage.joints]
  const newMetaJoints = { ...doc.meta.joints }
  const createdJoints: string[] = []

  let actualStartJoint = startJointName
  let actualEndJoint = endJointName

  // Create start joint if user didn't click on an existing joint
  if (!actualStartJoint) {
    actualStartJoint = generateJointName(existingJointNames)
    existingJointNames.push(actualStartJoint)

    // Check if we can make this a Revolute joint
    let madeRevolute = false

    if (endJointName) {
      const endJoint = doc.pylinkage.joints.find(j => j.name === endJointName)

      if (endJoint && endJoint.type !== 'Static') {
        const connectedToEnd = findConnectedJoints(endJointName)
        const secondRef = connectedToEnd.find(j => {
          const joint = doc.pylinkage.joints.find(jt => jt.name === j)
          return joint && joint.type !== 'Static'
        }) || connectedToEnd[0]

        if (secondRef) {
          const endPos = getJointPosition(endJointName)
          const secondPos = getJointPosition(secondRef)

          if (endPos && secondPos) {
            const distance0 = calculateDistance(endPos, startPoint)
            const distance1 = calculateDistance(secondPos, startPoint)

            const newRevolute: RevoluteJoint = {
              type: 'Revolute',
              name: actualStartJoint,
              joint0: { ref: endJointName },
              joint1: { ref: secondRef },
              distance0,
              distance1
            }
            newJoints = [...newJoints, newRevolute]
            newMetaJoints[actualStartJoint] = {
              x: startPoint[0],
              y: startPoint[1],
              color: '',
              zlevel: 0
            }
            madeRevolute = true
          }
        }
      }
    }

    // Fallback to Static if we couldn't make a Revolute
    if (!madeRevolute) {
      const newStatic: StaticJoint = {
        type: 'Static',
        name: actualStartJoint,
        x: startPoint[0],
        y: startPoint[1]
      }
      newJoints = [...newJoints, newStatic]
    }

    createdJoints.push(actualStartJoint)
  }

  // Create end joint if user didn't click on an existing joint
  if (!actualEndJoint) {
    actualEndJoint = generateJointName(existingJointNames)

    // Similar logic for end joint
    let madeRevolute = false

    if (actualStartJoint) {
      const startJointData = newJoints.find(j => j.name === actualStartJoint)

      if (startJointData && startJointData.type !== 'Static') {
        const connectedToStart = findConnectedJoints(actualStartJoint)
        const secondRef = connectedToStart.find(j => {
          const joint = newJoints.find(jt => jt.name === j)
          return joint && joint.type !== 'Static'
        }) || connectedToStart[0]

        if (secondRef) {
          const startPos = getJointPosition(actualStartJoint) || startPoint
          const secondPos = getJointPosition(secondRef)

          if (secondPos) {
            const distance0 = calculateDistance(startPos, endPoint)
            const distance1 = calculateDistance(secondPos, endPoint)

            const newRevolute: RevoluteJoint = {
              type: 'Revolute',
              name: actualEndJoint,
              joint0: { ref: actualStartJoint },
              joint1: { ref: secondRef },
              distance0,
              distance1
            }
            newJoints = [...newJoints, newRevolute]
            newMetaJoints[actualEndJoint] = {
              x: endPoint[0],
              y: endPoint[1],
              color: '',
              zlevel: 0
            }
            madeRevolute = true
          }
        }
      }
    }

    // Fallback to Static
    if (!madeRevolute) {
      const newStatic: StaticJoint = {
        type: 'Static',
        name: actualEndJoint,
        x: endPoint[0],
        y: endPoint[1]
      }
      newJoints = [...newJoints, newStatic]
    }

    createdJoints.push(actualEndJoint)
  }

  // Create the link
  const linkName = generateLinkName(existingLinkNames)
  const linkColor = getDefaultColor(existingLinkNames.length)

  const newMetaLinks = { ...doc.meta.links }
  newMetaLinks[linkName] = {
    color: linkColor,
    connects: [actualStartJoint, actualEndJoint]
  }

  // Update solve order
  const newSolveOrder = [...doc.pylinkage.solve_order]
  if (!newSolveOrder.includes(actualStartJoint)) {
    newSolveOrder.push(actualStartJoint)
  }
  if (!newSolveOrder.includes(actualEndJoint)) {
    newSolveOrder.push(actualEndJoint)
  }

  const length = calculateDistance(startPoint, endPoint)

  return {
    newDoc: {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: newJoints,
        solve_order: newSolveOrder
      },
      meta: {
        ...doc.meta,
        joints: newMetaJoints,
        links: newMetaLinks
      }
    },
    linkName,
    startJoint: actualStartJoint,
    endJoint: actualEndJoint,
    createdJoints,
    message: `Created ${linkName} (${length.toFixed(1)} units)`
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENAME LINK
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Rename a link
 *
 * @param doc - Current document state
 * @param oldName - Current link name
 * @param newName - New link name
 * @returns Result with new document or error
 */
export function renameLink(
  doc: PylinkDocument,
  oldName: string,
  newName: string
): RenameResult {
  if (oldName === newName || !newName.trim()) {
    return { newDoc: doc, oldName, newName, success: false, error: 'Invalid name' }
  }

  if (doc.meta.links[newName]) {
    return { newDoc: doc, oldName, newName, success: false, error: `Link "${newName}" already exists` }
  }

  const newLinks = { ...doc.meta.links }
  newLinks[newName] = newLinks[oldName]
  delete newLinks[oldName]

  return {
    newDoc: { ...doc, meta: { ...doc.meta, links: newLinks } },
    oldName,
    newName,
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UPDATE LINK PROPERTY
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Update a link property (color, connects, isGround, etc.)
 *
 * @param doc - Current document state
 * @param linkName - Name of link to update
 * @param property - Property to update
 * @param value - New value
 * @returns Result with new document
 */
export function updateLinkProperty(
  doc: PylinkDocument,
  linkName: string,
  property: string,
  value: string | string[] | boolean
): UpdatePropertyResult {
  if (!doc.meta.links[linkName]) {
    return { newDoc: doc, success: false, message: 'Link not found' }
  }

  return {
    newDoc: {
      ...doc,
      meta: {
        ...doc.meta,
        links: {
          ...doc.meta.links,
          [linkName]: {
            ...doc.meta.links[linkName],
            [property]: value
          }
        }
      }
    },
    success: true
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BATCH DELETE LINKS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Delete multiple links at once
 *
 * @param doc - Current document state
 * @param linkNames - Names of links to delete
 * @returns Result with new document and deletion details
 */
export function deleteLinks(
  doc: PylinkDocument,
  linkNames: string[]
): LinkDeletionResult {
  let currentDoc = doc
  const allDeletedLinks: string[] = []
  const allOrphanedJoints: string[] = []

  for (const linkName of linkNames) {
    if (currentDoc.meta.links[linkName]) {
      const result = deleteLink(currentDoc, linkName)
      currentDoc = result.newDoc
      allDeletedLinks.push(...result.deletedLinks)
      allOrphanedJoints.push(...result.orphanedJoints)
    }
  }

  const orphanMsg = allOrphanedJoints.length > 0
    ? ` + ${allOrphanedJoints.length} orphan(s)`
    : ''

  return {
    newDoc: currentDoc,
    deletedLinks: allDeletedLinks,
    orphanedJoints: allOrphanedJoints,
    message: `Deleted ${allDeletedLinks.length} link(s)${orphanMsg}`
  }
}
