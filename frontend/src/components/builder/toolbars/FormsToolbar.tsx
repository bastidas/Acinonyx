/**
 * Forms Toolbar - Polygon forms for CNC: padding, create rigid/link forms, list and edit
 */
import React, { useRef, useState } from 'react'
import {
  Box,
  Typography,
  Button,
  Tooltip,
  Divider,
  Slider,
  FormControlLabel,
  Switch,
  Collapse,
  TextField
} from '@mui/material'
import type { DrawnObject } from '../rendering/types'

export interface ZLevelRow {
  z: number
  color: string
}

/** Mirrors backend ZLevelHeuristicConfig for API request body */
export interface ZLevelHeuristicConfig {
  weight_reduce_deltas?: number
  weight_reduce_height?: number
  min_z?: number
  crank_z?: number | null
  weight_crank?: number
  hard_pins?: Record<string, number>
  soft_pins?: Record<string, [number, number]>
}

export const DEFAULT_Z_LEVEL_CONFIG: ZLevelHeuristicConfig = {
  weight_reduce_deltas: 1,
  weight_reduce_height: 0.3,
  min_z: 0,
  crank_z: 1,
  weight_crank: 1
}

export interface FormsToolbarProps {
  /** Padding (margin) in model units for bounding polygons */
  formPaddingUnits: number
  onFormPaddingChange: (value: number) => void
  createRigidForms: boolean
  setCreateRigidForms: (v: boolean) => void
  createLinkForms: boolean
  setCreateLinkForms: (v: boolean) => void
  computeZLevelsAfterCreate: boolean
  setComputeZLevelsAfterCreate: (v: boolean) => void
  onCreateForms: () => void | Promise<void>
  onComputeLinkZLevels: () => void | Promise<void>
  /** Polygon drawn objects (forms) */
  objects: DrawnObject[]
  selectedIds: string[]
  onSelectForms: (ids: string[]) => void
  openFormEdit: (formId: string) => void
  showStatus?: (message: string, severity: 'success' | 'error' | 'warning' | 'info' | 'action', duration?: number) => void
  /** When true, use light text for dark backgrounds */
  darkMode?: boolean
  /** Document-derived z-level rows for color editor (built from links + forms) */
  zLevelRows?: ZLevelRow[]
  /** Called when user changes a z-level color; parent updates links/forms and sets link color mode to z-level */
  onZLevelColorChange?: (z: number, color: string) => void
  /** Called when user drags a form to a new position; parent sets form z_level, z_level_fixed=true, then recomputes */
  onReorderForms?: (formId: string, newZLevel: number) => void | Promise<void>
  /** Called when user toggles pin z-level on a form; parent updates form z_level_fixed */
  onFormZLevelFixedChange?: (formId: string, fixed: boolean) => void
  /** Z-level heuristic config for compute and create; when set, sent to API */
  zLevelConfig?: ZLevelHeuristicConfig
  onZLevelConfigChange?: (config: ZLevelHeuristicConfig) => void
  /** Delete all polygon forms (with confirmation) */
  onDeleteAllForms?: () => void | Promise<void>
}

export const FormsToolbar: React.FC<FormsToolbarProps> = ({
  formPaddingUnits,
  onFormPaddingChange,
  createRigidForms,
  setCreateRigidForms,
  createLinkForms,
  setCreateLinkForms,
  computeZLevelsAfterCreate,
  setComputeZLevelsAfterCreate,
  onCreateForms,
  onComputeLinkZLevels,
  objects,
  selectedIds,
  onSelectForms,
  openFormEdit,
  showStatus,
  darkMode = false,
  zLevelRows = [],
  onZLevelColorChange,
  onReorderForms,
  onFormZLevelFixedChange,
  zLevelConfig,
  onZLevelConfigChange,
  onDeleteAllForms
}) => {
  const formClickTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [draggedFormId, setDraggedFormId] = useState<string | null>(null)
  const [dropTargetIndex, setDropTargetIndex] = useState<number | null>(null)
  const [zConfigOpen, setZConfigOpen] = useState(false)
  const config = { ...DEFAULT_Z_LEVEL_CONFIG, ...(zLevelConfig ?? {}) }

  const polygons = Array.isArray(objects)
    ? objects.filter((o): o is DrawnObject => o && typeof o === 'object' && o.type === 'polygon' && Array.isArray(o.points) && o.points.length >= 3)
    : []
  const sortedPolygons = [...polygons].sort((a, b) => {
    const za = typeof a.z_level === 'number' && Number.isFinite(a.z_level) ? a.z_level : 0
    const zb = typeof b.z_level === 'number' && Number.isFinite(b.z_level) ? b.z_level : 0
    return za - zb
  })
  const safeZLevelRows: ZLevelRow[] = Array.isArray(zLevelRows)
    ? zLevelRows.filter(
        (row): row is ZLevelRow =>
          row != null &&
          typeof row === 'object' &&
          typeof row.z === 'number' &&
          Number.isFinite(row.z) &&
          typeof row.color === 'string'
      )
    : []

  // Theme-aware text colors (app uses body class for dark mode, not MUI palette)
  const textPrimary = darkMode ? '#e8e4dc' : undefined
  const textSecondary = darkMode ? '#b8b4ac' : undefined
  const textMuted = darkMode ? '#888888' : undefined

  const handleCreateClick = () => {
    if (!createRigidForms && !createLinkForms) {
      showStatus?.('Turn on "Create rigid forms" and/or "Create link forms"', 'warning', 3000)
      return
    }
    void onCreateForms()
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', p: 2, ...(textPrimary ? { color: textPrimary } : {}) }}>
      <Typography variant="caption" sx={{ fontWeight: 600, color: textSecondary ?? 'text.secondary' }}>
        Padding (units) {formPaddingUnits}
      </Typography>
      <Slider
        size="small"
        value={formPaddingUnits}
        onChange={(_, value) => onFormPaddingChange(value as number)}
        min={1}
        max={30}
        step={1}
        valueLabelDisplay="auto"
        sx={{ mt: 0.5, mb: 1 }}
      />

      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={createRigidForms}
            onChange={(e) => setCreateRigidForms(e.target.checked)}
          />
        }
        label={<Typography variant="caption" sx={textPrimary ? { color: textPrimary } : undefined}>Create rigid forms</Typography>}
        sx={{ mb: 0.5 }}
      />
      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={createLinkForms}
            onChange={(e) => setCreateLinkForms(e.target.checked)}
          />
        }
        label={<Typography variant="caption" sx={textPrimary ? { color: textPrimary } : undefined}>Create link forms</Typography>}
        sx={{ mb: 0.5 }}
      />

      <FormControlLabel
        control={
          <Switch
            size="small"
            checked={computeZLevelsAfterCreate}
            onChange={(e) => setComputeZLevelsAfterCreate(e.target.checked)}
          />
        }
        label={<Typography variant="caption" sx={textPrimary ? { color: textPrimary } : undefined}>Compute z-levels after create</Typography>}
        sx={{ mb: 1 }}
      />

      <Tooltip
        title="Create polygons from rigid groups and/or one per link; uses padding above. We do not create new forms on links that already have forms attached."
        placement="left"
        leaveDelay={0}
        disableInteractive
        enterDelay={400}
      >
        <Button
          variant="contained"
          size="small"
          fullWidth
          onClick={handleCreateClick}
          onKeyDown={(e) => { if (e.key === ' ' || e.code === 'Space') e.preventDefault() }}
          sx={{ textTransform: 'none', fontSize: '0.75rem', mb: 1 }}
        >
          Create forms
        </Button>
      </Tooltip>

      <Tooltip
        title="Assign z-levels to links and forms by overlap; colors by layer"
        placement="left"
        leaveDelay={0}
        disableInteractive
        enterDelay={400}
      >
        <Button
          variant="outlined"
          size="small"
          fullWidth
          onClick={() => void onComputeLinkZLevels()}
          onKeyDown={(e) => { if (e.key === ' ' || e.code === 'Space') e.preventDefault() }}
          sx={{ textTransform: 'none', fontSize: '0.75rem', mb: 1 }}
        >
          Compute Z-levels
        </Button>
      </Tooltip>

      {onDeleteAllForms && (
        <Tooltip
          title="Remove all polygon forms"
          placement="left"
          leaveDelay={0}
          disableInteractive
          enterDelay={400}
        >
          <Button
            variant="outlined"
            size="small"
            fullWidth
            onClick={() => void onDeleteAllForms()}
            onKeyDown={(e) => { if (e.key === ' ' || e.code === 'Space') e.preventDefault() }}
            sx={{ textTransform: 'none', fontSize: '0.75rem', mb: 1 }}
          >
            Delete all forms
          </Button>
        </Tooltip>
      )}

      {onZLevelConfigChange && (
        <>
          <Button
            size="small"
            fullWidth
            onClick={() => setZConfigOpen((o) => !o)}
            sx={{ textTransform: 'none', fontSize: '0.7rem', mb: 0.5, color: textSecondary ?? 'text.secondary' }}
          >
            {zConfigOpen ? 'Hide' : 'Z-level config'}…
          </Button>
          <Collapse in={zConfigOpen}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mb: 1, pl: 0.5 }}>
              <Tooltip
                title="Weight for preferring adjacent-level connections (N↔N±1)"
                placement="left"
                leaveDelay={0}
                disableInteractive
                enterDelay={300}
              >
                <TextField
                  size="small"
                  label="Adjacent-level weight"
                  type="number"
                  value={config.weight_reduce_deltas ?? 1}
                  onChange={(e) => onZLevelConfigChange({ ...(zLevelConfig ?? {}), weight_reduce_deltas: Number(e.target.value) || 0 })}
                  inputProps={{ min: 0, step: 0.1 }}
                  sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem' } }}
                />
              </Tooltip>
              <Tooltip
                title="Weight for preferring smaller total z span (max−min)"
                placement="left"
                leaveDelay={0}
                disableInteractive
                enterDelay={300}
              >
                <TextField
                  size="small"
                  label="Height weight"
                  type="number"
                  value={config.weight_reduce_height ?? 0.3}
                  onChange={(e) => onZLevelConfigChange({ ...(zLevelConfig ?? {}), weight_reduce_height: Number(e.target.value) || 0 })}
                  inputProps={{ min: 0, step: 0.1 }}
                  sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem' } }}
                />
              </Tooltip>
              <Tooltip
                title="Minimum allowed z-level; all assignments must have z ≥ this."
                placement="left"
                leaveDelay={0}
                disableInteractive
                enterDelay={300}
              >
                <TextField
                  size="small"
                  label="Min z"
                  type="number"
                  value={config.min_z ?? 0}
                  onChange={(e) => onZLevelConfigChange({ ...(zLevelConfig ?? {}), min_z: Number(e.target.value) || 0 })}
                  inputProps={{ step: 1 }}
                  sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem' } }}
                />
              </Tooltip>
              <Tooltip
                title="Preferred z for crank-incident (root) entity; empty = no role-based preference."
                placement="left"
                leaveDelay={0}
                disableInteractive
                enterDelay={300}
              >
                <TextField
                  size="small"
                  label="Crank z"
                  type="number"
                  value={config.crank_z ?? ''}
                  onChange={(e) => {
                    const v = e.target.value
                    onZLevelConfigChange({ ...(zLevelConfig ?? {}), crank_z: v === '' || v === 'null' ? null : Number(v) })
                  }}
                  placeholder="none"
                  inputProps={{ step: 1 }}
                  sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem' } }}
                />
              </Tooltip>
              <Tooltip
                title="Weight for preferring root entity at crank_z (when crank_z is set)."
                placement="left"
                leaveDelay={0}
                disableInteractive
                enterDelay={300}
              >
                <TextField
                  size="small"
                  label="Crank weight"
                  type="number"
                  value={config.weight_crank ?? 1}
                  onChange={(e) => onZLevelConfigChange({ ...(zLevelConfig ?? {}), weight_crank: Number(e.target.value) || 0 })}
                  inputProps={{ min: 0, step: 0.1 }}
                  sx={{ '& .MuiInputBase-input': { fontSize: '0.75rem' } }}
                />
              </Tooltip>
            </Box>
          </Collapse>
        </>
      )}

      {safeZLevelRows.length > 0 && onZLevelColorChange && (
        <>
          <Typography variant="caption" sx={{ fontWeight: 600, color: textSecondary ?? 'text.secondary', mt: 1 }}>
            Z levels
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mb: 1 }}>
            {safeZLevelRows.map(({ z, color }) => (
              <Box key={z} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Typography variant="caption" sx={{ minWidth: 20, color: textPrimary ?? 'text.primary' }}>{String(z)}:</Typography>
                <input
                  type="color"
                  value={color}
                  onChange={(e) => onZLevelColorChange(z, e.target.value)}
                  style={{
                    width: 28,
                    height: 24,
                    padding: 0,
                    border: '1px solid var(--color-border, #ccc)',
                    borderRadius: 4,
                    cursor: 'pointer',
                    flexShrink: 0
                  }}
                  aria-label={`Color for z-level ${z}`}
                />
              </Box>
            ))}
          </Box>
        </>
      )}

      <Divider sx={{ my: 1 }} />

      <Typography variant="caption" sx={{ fontWeight: 600, color: textSecondary ?? 'text.secondary' }}>
        Forms ({polygons.length})
      </Typography>
      <Box sx={{ overflow: 'auto', flex: 1, minHeight: 0 }}>
        {sortedPolygons.length === 0 ? (
          <Box sx={{ p: 2, textAlign: 'center' }}>
            <Typography variant="caption" sx={{ color: textSecondary ?? 'text.secondary' }}>No forms yet</Typography>
            <Typography variant="caption" display="block" sx={{ color: textMuted ?? 'text.disabled' }}>Use Create forms above</Typography>
          </Box>
        ) : (
          sortedPolygons.map((obj, index) => {
            const isSelected = selectedIds.includes(obj.id)
            const isDragging = draggedFormId === obj.id
            const isDropTarget = dropTargetIndex === index
            const handleClick = (e: React.MouseEvent) => {
              if (formClickTimeoutRef.current) {
                clearTimeout(formClickTimeoutRef.current)
                formClickTimeoutRef.current = null
              }
              if (e.detail === 2) {
                openFormEdit(obj.id)
              } else {
                formClickTimeoutRef.current = setTimeout(() => {
                  onSelectForms([obj.id])
                  formClickTimeoutRef.current = null
                }, 200)
              }
            }

            const handleDragStart = (e: React.DragEvent) => {
              setDraggedFormId(obj.id)
              e.dataTransfer.effectAllowed = 'move'
              e.dataTransfer.setData('text/plain', obj.id)
              e.dataTransfer.setData('application/x-form-id', obj.id)
            }
            const handleDragEnd = () => {
              setDraggedFormId(null)
              setDropTargetIndex(null)
            }
            const handleDragOver = (e: React.DragEvent) => {
              e.preventDefault()
              e.dataTransfer.dropEffect = 'move'
              if (draggedFormId && draggedFormId !== obj.id) setDropTargetIndex(index)
            }
            const handleDragLeave = () => {
              if (dropTargetIndex === index) setDropTargetIndex(null)
            }
            const handleDrop = (e: React.DragEvent) => {
              e.preventDefault()
              setDraggedFormId(null)
              setDropTargetIndex(null)
              const droppedId = e.dataTransfer.getData('application/x-form-id') || e.dataTransfer.getData('text/plain')
              if (!onReorderForms || droppedId === obj.id) return
              const droppedIndex = sortedPolygons.findIndex((p) => p.id === droppedId)
              if (droppedIndex < 0) return
              const targetZ = obj.z_level ?? 0
              void onReorderForms(droppedId, targetZ)
            }

            return (
              <Box
                key={obj.id}
                onClick={handleClick}
                draggable={!!onReorderForms}
                onDragStart={handleDragStart}
                onDragEnd={handleDragEnd}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                sx={{
                  py: 0.75,
                  px: 1.5,
                  cursor: onReorderForms ? 'grab' : 'pointer',
                  backgroundColor: isDropTarget
                    ? (darkMode ? 'rgba(255,255,255,0.12)' : 'action.selected')
                    : isSelected
                      ? 'primary.main'
                      : 'transparent',
                  color: isSelected ? 'primary.contrastText' : (textPrimary ?? 'text.primary'),
                  borderLeft: `3px solid ${obj.fillColor}`,
                  opacity: isDragging ? 0.6 : 1,
                  '&:hover': {
                    backgroundColor: isSelected ? 'primary.dark' : 'action.hover'
                  },
                  '&:active': onReorderForms ? { cursor: 'grabbing' } : undefined,
                  transition: 'background-color 0.15s ease'
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flex: 1, minWidth: 0 }}>
                  <Typography sx={{ fontSize: '0.75rem', fontWeight: 500, flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>{String(obj.name || obj.id || '')}</Typography>
                  {onFormZLevelFixedChange && (
                    <Tooltip title={obj.z_level_fixed ? 'Z-level pinned (unpin to allow recompute)' : 'Pin z-level (exclude from recompute)'}>
                      <Switch
                        size="small"
                        checked={!!obj.z_level_fixed}
                        onChange={(e) => {
                          e.stopPropagation()
                          onFormZLevelFixedChange(obj.id, e.target.checked)
                        }}
                        onClick={(e) => e.stopPropagation()}
                        sx={{
                          ml: 0.5,
                          py: 0,
                          '& .MuiSwitch-switchBase': { py: 0.25 },
                          '& .MuiSwitch-thumb': { width: 14, height: 14 },
                          '& .MuiSwitch-track': { borderRadius: 7, height: 16 }
                        }}
                        aria-label={obj.z_level_fixed ? 'Unpin z-level' : 'Pin z-level'}
                      />
                    </Tooltip>
                  )}
                </Box>
                <Typography variant="caption" sx={{ fontSize: '0.65rem', opacity: isSelected ? 0.9 : 0.7 }}>
                  z: {typeof obj.z_level === 'number' && Number.isFinite(obj.z_level) ? obj.z_level : '—'}
                </Typography>
              </Box>
            )
          })
        )}
      </Box>
    </Box>
  )
}

export default FormsToolbar
