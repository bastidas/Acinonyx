/**
 * Nodes Toolbar - List of joints with selection and edit capabilities
 */
import React, { useRef } from 'react'
import { Box, Typography, Chip } from '@mui/material'
import { jointColors } from '../../../theme'
import type { PylinkJoint } from '../types'

export interface NodesToolbarProps {
  joints: PylinkJoint[]
  selectedJoints: string[]
  setSelectedJoints: (joints: string[]) => void
  setSelectedLinks: (links: string[]) => void
  hoveredJoint: string | null
  setHoveredJoint: (joint: string | null) => void
  selectionColor: string
  getJointPosition: (name: string) => [number, number] | null
  openJointEditModal: (jointName: string) => void
}

export const NodesToolbar: React.FC<NodesToolbarProps> = ({
  joints,
  selectedJoints,
  setSelectedJoints,
  setSelectedLinks,
  hoveredJoint,
  setHoveredJoint,
  selectionColor,
  getJointPosition,
  openJointEditModal
}) => {
  // Track click timing for distinguishing single vs double click
  const jointClickTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  return (
    <Box sx={{ overflow: 'auto', maxHeight: 350 }}>
      {joints.length === 0 ? (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">No joints yet</Typography>
          <Typography variant="caption" display="block" color="text.disabled">Use Draw Link tool (L)</Typography>
        </Box>
      ) : (
        joints.map((joint) => {
          const pos = getJointPosition(joint.name)
          const isSelected = selectedJoints.includes(joint.name)
          const isHovered = hoveredJoint === joint.name
          const typeColor = joint.type === 'Static' ? jointColors.static : joint.type === 'Crank' ? jointColors.crank : jointColors.pivot

          const handleClick = (e: React.MouseEvent) => {
            // Clear any pending single-click action
            if (jointClickTimeoutRef.current) {
              clearTimeout(jointClickTimeoutRef.current)
              jointClickTimeoutRef.current = null
            }

            if (e.detail === 2) {
              // Double click - open modal
              openJointEditModal(joint.name)
            } else {
              // Single click - delay selection to allow for double-click
              jointClickTimeoutRef.current = setTimeout(() => {
                setSelectedJoints([joint.name])
                setSelectedLinks([])
                jointClickTimeoutRef.current = null
              }, 200)
            }
          }

          return (
            <Box
              key={joint.name}
              onMouseEnter={() => setHoveredJoint(joint.name)}
              onMouseLeave={() => setHoveredJoint(null)}
              onClick={handleClick}
              sx={{
                py: 0.75, px: 1.5, cursor: 'pointer',
                backgroundColor: isSelected ? `${selectionColor}14` : (isHovered ? `${selectionColor}1f` : 'transparent'),
                borderLeft: `3px solid ${typeColor}`,
                '&:hover': { backgroundColor: `${selectionColor}1f` },
                transition: 'background-color 0.15s ease'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography sx={{ fontSize: '0.75rem', fontWeight: isHovered ? 600 : 500 }}>{joint.name}</Typography>
                <Chip
                  label={joint.type}
                  size="small"
                  sx={{
                    height: 14, fontSize: '0.5rem',
                    backgroundColor: joint.type === 'Static' ? '#ffebee' : joint.type === 'Crank' ? '#fff3e0' : '#e3f2fd'
                  }}
                />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                {pos ? `(${pos[0].toFixed(1)}, ${pos[1].toFixed(1)})` : ''}
              </Typography>
            </Box>
          )
        })
      )}
    </Box>
  )
}

export default NodesToolbar
