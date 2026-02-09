/**
 * useModalState
 *
 * Consolidates modal-related state for BuilderTab: joint edit modal, link edit modal,
 * and delete confirmation dialog. Replaces three separate useState calls with one hook.
 */

import { useState } from 'react'
import type { JointData, LinkData } from '../../BuilderTools'

export interface DeleteConfirmDialogState {
  open: boolean
  joints: string[]
  links: string[]
}

const initialDeleteConfirm: DeleteConfirmDialogState = {
  open: false,
  joints: [],
  links: []
}

export interface UseModalStateReturn {
  editingJointData: JointData | null
  setEditingJointData: React.Dispatch<React.SetStateAction<JointData | null>>
  editingLinkData: LinkData | null
  setEditingLinkData: React.Dispatch<React.SetStateAction<LinkData | null>>
  deleteConfirmDialog: DeleteConfirmDialogState
  setDeleteConfirmDialog: React.Dispatch<React.SetStateAction<DeleteConfirmDialogState>>
}

export function useModalState(): UseModalStateReturn {
  const [editingJointData, setEditingJointData] = useState<JointData | null>(null)
  const [editingLinkData, setEditingLinkData] = useState<LinkData | null>(null)
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState<DeleteConfirmDialogState>(initialDeleteConfirm)

  return {
    editingJointData,
    setEditingJointData,
    editingLinkData,
    setEditingLinkData,
    deleteConfirmDialog,
    setDeleteConfirmDialog
  }
}
