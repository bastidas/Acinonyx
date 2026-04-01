/**
 * More Toolbar - File Operations, Demos, Canvas Images
 *
 * Note: Animation controls have been moved to AnimateToolbar
 */
import React from 'react'
import {
  Box,
  Typography,
  Button,
  Tooltip,
  Divider,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Slider,
  List,
  ListItem,
  ListItemText,
  IconButton
} from '@mui/material'
import ImageIcon from '@mui/icons-material/Image'
import EditIcon from '@mui/icons-material/Edit'
import DeleteIcon from '@mui/icons-material/Delete'
import type { CanvasImageData } from '../../../types'
import { useFileInput } from '../hooks/useFileInput'

export interface MoreToolbarProps {
  // Demo operations
  loadDemo4Bar: () => void
  loadDemoLeg: () => void
  loadDemoWalker: () => void
  loadDemoComplex: () => void

  // File operations
  loadPylinkGraphLast: () => void
  /** Trigger load file picker; when user selects a .json file, this is called with the File */
  onLoadFileSelected: (file: File) => void
  savePylinkGraph: () => void
  /** Suggested default name for Save As dialog (e.g. doc name or acinonyx-YYYYMMDD) */
  suggestedSaveAsName: string
  /** Called when user confirms Save As with the chosen filename (without .json is ok) */
  onSaveAs: (filename: string) => void | Promise<void>
  /** Called when user confirms Clear all (reset to empty document) */
  onClearAll: () => void

  // Canvas image overlays
  canvases: CanvasImageData[]
  setCanvases: React.Dispatch<React.SetStateAction<CanvasImageData[]>>
  /** When set, the edit dialog opens for this canvas (e.g. from list Edit or from handle double-click) */
  editingCanvasId: string | null
  setEditingCanvasId: (id: string | null) => void
  showStatus?: (message: string, severity: 'success' | 'error' | 'warning' | 'info' | 'action', duration?: number) => void
}

function generateCanvasId(): string {
  return `canvas-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export const MoreToolbar: React.FC<MoreToolbarProps> = ({
  loadDemo4Bar,
  loadDemoLeg,
  loadDemoWalker,
  loadDemoComplex,
  loadPylinkGraphLast,
  onLoadFileSelected,
  savePylinkGraph,
  suggestedSaveAsName,
  onSaveAs,
  onClearAll,
  canvases,
  setCanvases,
  editingCanvasId,
  setEditingCanvasId,
  showStatus
}) => {
  const [editDraft, setEditDraft] = React.useState<{ position: [number, number]; scale: number; alpha: number } | null>(null)

  const editingCanvas = editingCanvasId ? canvases.find(c => c.id === editingCanvasId) ?? null : null

  const handleCanvasFileSelected = React.useCallback(
    (file: File) => {
      const reader = new FileReader()
      reader.onload = () => {
        const dataUrl = reader.result as string
        const img = new Image()
        img.onload = () => {
          const newCanvas: CanvasImageData = {
            id: generateCanvasId(),
            dataUrl,
            position: [0, 0],
            scale: 1,
            alpha: 0.7,
            naturalWidth: img.naturalWidth,
            naturalHeight: img.naturalHeight
          }
          setCanvases(prev => [...prev, newCanvas])
          showStatus?.('Canvas added. Use list below to edit position, scale, or transparency.', 'success', 4000)
          setEditingCanvasId(newCanvas.id)
        }
        img.onerror = () => showStatus?.('Failed to load image dimensions', 'error', 3000)
        img.src = dataUrl
      }
      reader.onerror = () => showStatus?.('Failed to read file', 'error', 3000)
      reader.readAsDataURL(file)
    },
    [setCanvases, setEditingCanvasId, showStatus]
  )

  const canvasFileInput = useFileInput('.jpg,.jpeg,.png', handleCanvasFileSelected)
  const loadFileInput = useFileInput('.json,application/json', onLoadFileSelected)

  const [saveAsOpen, setSaveAsOpen] = React.useState(false)
  const [saveAsName, setSaveAsName] = React.useState('')
  const [clearConfirmOpen, setClearConfirmOpen] = React.useState(false)

  const handleClearConfirm = React.useCallback(() => {
    onClearAll()
    setClearConfirmOpen(false)
  }, [onClearAll])

  const openSaveAsDialog = React.useCallback(() => {
    setSaveAsName(suggestedSaveAsName)
    setSaveAsOpen(true)
  }, [suggestedSaveAsName])

  const closeSaveAsDialog = React.useCallback(() => {
    setSaveAsOpen(false)
  }, [])

  const handleSaveAsConfirm = React.useCallback(() => {
    const name = saveAsName.trim()
    if (name) {
      onSaveAs(name)
      closeSaveAsDialog()
    }
  }, [saveAsName, onSaveAs, closeSaveAsDialog])

  React.useEffect(() => {
    if (!editingCanvasId) {
      setEditDraft(null)
      return
    }
    const c = canvases.find(can => can.id === editingCanvasId)
    if (c) setEditDraft({ position: [...c.position], scale: c.scale, alpha: c.alpha })
    // Only sync when dialog opens (editingCanvasId changes), not when canvases change (e.g. drag)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editingCanvasId])

  const openEdit = (c: CanvasImageData) => {
    setEditingCanvasId(c.id)
  }

  const closeEdit = () => {
    setEditingCanvasId(null)
    setEditDraft(null)
  }

  const MIN_SCALE = 0.0001

  const applyEdit = () => {
    if (!editingCanvas || !editDraft) return
    const scale = Math.max(MIN_SCALE, Number(editDraft.scale)) || MIN_SCALE
    setCanvases(prev =>
      prev.map(c =>
        c.id === editingCanvas.id
          ? {
              ...c,
              position: editDraft.position,
              scale,
              alpha: editDraft.alpha
            }
          : c
      )
    )
    closeEdit()
  }

  const removeCanvas = (id: string) => {
    setCanvases(prev => prev.filter(c => c.id !== id))
    if (editingCanvasId === id) setEditingCanvasId(null)
  }

  return (
    <Box sx={{ p: 1.5 }}>
      <input
        ref={canvasFileInput.ref as React.RefObject<HTMLInputElement>}
        type="file"
        accept={canvasFileInput.accept}
        style={{ display: 'none' }}
        onChange={canvasFileInput.onChange}
      />
      <input
        ref={loadFileInput.ref as React.RefObject<HTMLInputElement>}
        type="file"
        accept={loadFileInput.accept}
        style={{ display: 'none' }}
        onChange={loadFileInput.onChange}
      />
      {/* ═══════════════════════════════════════════════════════════════════════
          FILE OPERATIONS (first - most used)
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        File Operations
      </Typography>
      <Box sx={{ display: 'flex', gap: 0.5, mt: 1, mb: 1 }}>
        <Tooltip title="Open a mechanism from a .json file (file picker)" placement="top" leaveDelay={0}>
          <Button
            variant="outlined"
            size="small"
            onClick={loadFileInput.trigger}
            sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
          >
            ↑ Load
          </Button>
        </Tooltip>
        <Tooltip title="Load the most recently saved file from the working directory" placement="top" leaveDelay={0}>
          <Button
            variant="outlined"
            size="small"
            onClick={loadPylinkGraphLast}
            sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
          >
            ↑ Last
          </Button>
        </Tooltip>
      </Box>
      <Box sx={{ display: 'flex', gap: 0.5, mb: 1.5 }}>
        <Tooltip title="Save current mechanism as acinonyx-YYYYMMDD.json (or default name)" placement="top" leaveDelay={0}>
          <Button
            variant="outlined"
            size="small"
            onClick={savePylinkGraph}
            sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
          >
            ↓ Save
          </Button>
        </Tooltip>
        <Tooltip title="Save current mechanism with a name you choose" placement="top" leaveDelay={0}>
          <Button
            variant="outlined"
            size="small"
            onClick={openSaveAsDialog}
            sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
          >
            ↓ Save As
          </Button>
        </Tooltip>
      </Box>

      {/* Save As dialog */}
      <Dialog open={saveAsOpen} onClose={closeSaveAsDialog} maxWidth="xs" fullWidth>
        <DialogTitle>Save As</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            label="Filename"
            value={saveAsName}
            onChange={e => setSaveAsName(e.target.value)}
            inputProps={{
              onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => {
                if (e.key !== 'Enter') return
                e.preventDefault()
                e.stopPropagation()
                handleSaveAsConfirm()
              }
            }}
            size="small"
            sx={{ mt: 1 }}
            helperText=".json will be added if omitted"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={closeSaveAsDialog}>Cancel</Button>
          <Button variant="contained" onClick={handleSaveAsConfirm} disabled={!saveAsName.trim()}>
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Tooltip title="Clear all links and joints; start with an empty mechanism" placement="top" leaveDelay={0}>
        <Button
          variant="outlined"
          color="error"
          fullWidth
          size="small"
          onClick={() => setClearConfirmOpen(true)}
          sx={{ textTransform: 'none', fontSize: '0.7rem', mb: 1 }}
        >
          Clear all
        </Button>
      </Tooltip>

      {/* Clear all confirmation - Enter confirms */}
      <Dialog
        open={clearConfirmOpen}
        onClose={() => setClearConfirmOpen(false)}
        onKeyDown={e => {
          if (e.key === 'Enter') {
            e.preventDefault()
            handleClearConfirm()
          }
        }}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Clear all?</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary">
            Clear all links and joints? This cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setClearConfirmOpen(false)}>Cancel</Button>
          <Button variant="contained" color="error" onClick={handleClearConfirm} autoFocus>
            Clear all
          </Button>
        </DialogActions>
      </Dialog>

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          CANVAS (reference images)
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Canvas
      </Typography>
      <Tooltip
        title="Insert a JPEG or PNG as a reference layer (position, scale, transparency editable)"
        placement="top"
        leaveDelay={0}
      >
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={canvasFileInput.trigger}
          startIcon={<ImageIcon sx={{ fontSize: '1rem' }} />}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem', mt: 1, mb: 0.5 }}
        >
          Insert Canvas
        </Button>
      </Tooltip>
      {canvases.length > 0 && (
        <List dense sx={{ py: 0, maxHeight: 120, overflow: 'auto' }}>
          {canvases.map((c, i) => (
            <ListItem
              key={c.id}
              secondaryAction={
                <>
                  <IconButton size="small" onClick={() => openEdit(c)} aria-label="Edit canvas">
                    <EditIcon fontSize="small" />
                  </IconButton>
                  <IconButton size="small" onClick={() => removeCanvas(c.id)} aria-label="Remove canvas">
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                </>
              }
              sx={{ py: 0, minHeight: 36 }}
            >
              <ListItemText primary={`Canvas ${i + 1}`} primaryTypographyProps={{ fontSize: '0.75rem' }} />
            </ListItem>
          ))}
        </List>
      )}

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          DEMOS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Demos
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 1, mb: 1.5 }}>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemo4Bar}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Four Bar
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemoLeg}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Leg
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemoWalker}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Walker
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemoComplex}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Complex
        </Button>
      </Box>

      {/* Edit Canvas Dialog */}
      <Dialog open={!!editingCanvas} onClose={closeEdit} maxWidth="xs" fullWidth>
        <DialogTitle>Edit Canvas</DialogTitle>
        <DialogContent>
          {editingCanvas && editDraft && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
              <TextField
                label="Position X"
                type="number"
                size="small"
                value={editDraft.position[0]}
                onChange={e => setEditDraft(d => (d ? { ...d, position: [Number(e.target.value), d.position[1]] } : d))}
                inputProps={{ step: 1 }}
              />
              <TextField
                label="Position Y"
                type="number"
                size="small"
                value={editDraft.position[1]}
                onChange={e => setEditDraft(d => (d ? { ...d, position: [d.position[0], Number(e.target.value)] } : d))}
                inputProps={{ step: 1 }}
              />
              <TextField
                label="Scale"
                type="number"
                size="small"
                value={editDraft.scale}
                onChange={e => setEditDraft(d => (d ? { ...d, scale: Number(e.target.value) ?? 1 } : d))}
                inputProps={{ step: 0.001, min: 0.0001 }}
                helperText="1 = natural size (min 0.0001)"
              />
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Transparency
              </Typography>
              <Slider
                value={editDraft.alpha}
                min={0}
                max={1}
                step={0.05}
                valueLabelDisplay="auto"
                valueLabelFormat={v => `${Math.round(v * 100)}%`}
                onChange={(_, value) => setEditDraft(d => (d ? { ...d, alpha: value as number } : d))}
              />
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          {editingCanvas && (
            <Button color="error" onClick={() => removeCanvas(editingCanvas.id)}>
              Remove
            </Button>
          )}
          <Button onClick={closeEdit}>Cancel</Button>
          <Button variant="contained" onClick={applyEdit}>
            Apply
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}

export default MoreToolbar
