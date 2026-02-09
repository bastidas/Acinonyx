/**
 * Links Toolbar - List of links with selection and edit capabilities
 */
import React, { useRef } from 'react'
import { Box, Typography } from '@mui/material'
import { calculateDistance, getDefaultColor } from '../../BuilderTools'
import type { LinkMeta } from '../types'
import type { LinkageDocument } from '../../../types'

export interface LinksToolbarProps {
  links: Record<string, LinkMeta>
  linkageDoc: LinkageDocument  // Add LinkageDocument to access edge distances
  selectedLinks: string[]
  setSelectedLinks: (links: string[]) => void
  setSelectedJoints: (joints: string[]) => void
  hoveredLink: string | null
  setHoveredLink: (link: string | null) => void
  selectionColor: string
  getJointPosition: (name: string) => [number, number] | null
  openLinkEditModal: (linkName: string) => void
}

export const LinksToolbar: React.FC<LinksToolbarProps> = ({
  links,
  linkageDoc,
  selectedLinks,
  setSelectedLinks,
  setSelectedJoints,
  hoveredLink,
  setHoveredLink,
  selectionColor,
  getJointPosition,
  openLinkEditModal
}) => {
  // Track click timing for distinguishing single vs double click
  const linkClickTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  return (
    <Box sx={{ overflow: 'auto', maxHeight: 350 }}>
      {Object.entries(links).length === 0 ? (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">No links yet</Typography>
          <Typography variant="caption" display="block" color="text.disabled">Use Create Link tool (C)</Typography>
        </Box>
      ) : (
        Object.entries(links).map(([linkName, linkMeta], index) => {
          // CRITICAL FIX: Read distance from LinkageDocument edge, not from visual positions
          // This ensures we show the actual optimized edge distance, not a calculated value
          const edge = linkageDoc.linkage?.edges?.[linkName]
          const distance = edge?.distance

          // Fallback to calculated distance only if edge distance is not available
          const pos0 = getJointPosition(linkMeta.connects[0])
          const pos1 = getJointPosition(linkMeta.connects[1])
          const calculatedLength = pos0 && pos1 ? calculateDistance(pos0, pos1) : null
          const length = distance !== undefined && distance !== null ? distance : calculatedLength
          const isSelected = selectedLinks.includes(linkName)
          const isHovered = hoveredLink === linkName

          const handleClick = (e: React.MouseEvent) => {
            // Clear any pending single-click action
            if (linkClickTimeoutRef.current) {
              clearTimeout(linkClickTimeoutRef.current)
              linkClickTimeoutRef.current = null
            }

            if (e.detail === 2) {
              // Double click - open modal
              openLinkEditModal(linkName)
            } else {
              // Single click - delay selection to allow for double-click
              linkClickTimeoutRef.current = setTimeout(() => {
                setSelectedLinks([linkName])
                setSelectedJoints([])
                linkClickTimeoutRef.current = null
              }, 200)
            }
          }

          return (
            <Box
              key={linkName}
              onMouseEnter={() => setHoveredLink(linkName)}
              onMouseLeave={() => setHoveredLink(null)}
              onClick={handleClick}
              sx={{
                py: 0.75, px: 1.5, cursor: 'pointer',
                backgroundColor: isSelected ? `${selectionColor}14` : (isHovered ? `${selectionColor}1f` : 'transparent'),
                borderLeft: `3px solid ${linkMeta.color || getDefaultColor(index)}`,
                '&:hover': { backgroundColor: `${selectionColor}1f` },
                transition: 'background-color 0.15s ease'
              }}
            >
              <Typography sx={{ fontSize: '0.75rem', fontWeight: isHovered ? 600 : 500 }}>{linkName}</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                {linkMeta.connects.join(' → ')}{length ? ` • ${length.toFixed(1)}` : ''}
              </Typography>
            </Box>
          )
        })
      )}
    </Box>
  )
}

export default LinksToolbar
