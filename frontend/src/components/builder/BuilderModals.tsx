/**
 * BuilderModals
 *
 * Delete Confirmation Dialog, Joint Edit Modal, and Link Edit Modal.
 * State and callbacks are owned by the parent (BuilderTab); this component is presentational.
 */

import React from 'react'
import {
  Box,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions
} from '@mui/material'
import { JointEditModal, JointData, LinkEditModal, LinkData } from '../BuilderTools'

export interface DeleteConfirmDialogState {
  open: boolean
  joints: string[]
  links: string[]
}

export interface BuilderModalsProps {
  deleteConfirmDialog: DeleteConfirmDialogState
  onCloseDeleteConfirm: () => void
  onConfirmDelete: () => void
  editingJointData: JointData | null
  onCloseJointEdit: () => void
  editingLinkData: LinkData | null
  onCloseLinkEdit: () => void
  renameJoint: (oldName: string, newName: string) => void
  renameLink: (oldName: string, newName: string) => void
  updateJointProperty: (jointName: string, property: string, value: string) => void
  updateLinkProperty: (linkName: string, property: string, value: string | string[] | boolean) => void
  onJointShowPathChange: (jointName: string, showPath: boolean) => void
  setEditingJointData: React.Dispatch<React.SetStateAction<JointData | null>>
  setEditingLinkData: React.Dispatch<React.SetStateAction<LinkData | null>>
  jointTypes: readonly string[]
  darkMode?: boolean
}

export function BuilderModals({
  deleteConfirmDialog,
  onCloseDeleteConfirm,
  onConfirmDelete,
  editingJointData,
  onCloseJointEdit,
  editingLinkData,
  onCloseLinkEdit,
  renameJoint,
  renameLink,
  updateJointProperty,
  updateLinkProperty,
  onJointShowPathChange,
  setEditingJointData,
  setEditingLinkData,
  jointTypes,
  darkMode = false
}: BuilderModalsProps): JSX.Element {
  return (
    <>
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={onCloseDeleteConfirm}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            onConfirmDelete()
          }
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>
          Confirm Delete
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete {deleteConfirmDialog.joints.length + deleteConfirmDialog.links.length} items?
          </DialogContentText>
          <Box sx={{ mt: 2 }}>
            {deleteConfirmDialog.joints.length > 0 && (
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                <strong>Joints:</strong> {deleteConfirmDialog.joints.join(', ')}
              </Typography>
            )}
            {deleteConfirmDialog.links.length > 0 && (
              <Typography variant="body2">
                <strong>Links:</strong> {deleteConfirmDialog.links.join(', ')}
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={onCloseDeleteConfirm}
            color="inherit"
          >
            Cancel
          </Button>
          <Button
            onClick={onConfirmDelete}
            color="error"
            variant="contained"
            autoFocus
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      <JointEditModal
        open={editingJointData !== null}
        onClose={onCloseJointEdit}
        jointData={editingJointData}
        jointTypes={jointTypes}
        onRename={renameJoint}
        onTypeChange={(jointName, newType) => updateJointProperty(jointName, 'type', newType)}
        onShowPathChange={(jointName, showPath) => {
          onJointShowPathChange(jointName, showPath)
          setEditingJointData(prev => prev ? { ...prev, showPath } : null)
        }}
        darkMode={darkMode}
      />

      <LinkEditModal
        open={editingLinkData !== null}
        onClose={onCloseLinkEdit}
        linkData={editingLinkData}
        onRename={renameLink}
        onColorChange={(linkName, color) => {
          updateLinkProperty(linkName, 'color', color)
          setEditingLinkData(prev => prev ? { ...prev, color } : null)
        }}
        onGroundChange={(linkName, isGround) => {
          updateLinkProperty(linkName, 'isGround', isGround)
          setEditingLinkData(prev => prev ? { ...prev, isGround } : null)
        }}
        darkMode={darkMode}
      />
    </>
  )
}
