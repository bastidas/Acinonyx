import React, { useState, useEffect } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent
} from '@mui/material'
import cheetahGif2 from '../assets/cheetah_run2.gif'
import acinonyxLogo from '../assets/acinonyx_logo.png'
import worldIcon from '../assets/social/world.png'
import githubIcon from '../assets/social/github.png'

const StatusAboutTab: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Auto-run status check on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch('/api/status')
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        await response.json() // Just verify we get valid JSON
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Connection failed')
      } finally {
        setLoading(false)
      }
    }
    checkStatus()
  }, [])

  return (
    <Box sx={{
      py: 3,
      px: 2,
      maxWidth: 900,
      margin: '0 auto'
    }}>
      {/* About Card - Now at top */}
      <Card sx={{
        mb: 3,
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border-light)',
        transition: 'all 0.25s ease'
      }}>
        <CardContent>
          <Typography
            variant="h6"
            gutterBottom
            sx={{
              fontWeight: 600,
              color: 'var(--color-text-primary)'
            }}
          >
            About Acinonyx
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            Acinonyx is an open-source platform for designing and simulating planar mechanical linkages. It addresses a fundamental challenge in mechanism synthesis: <em>given a desired path, how do we construct a multi-link mechanism that traces that path while satisfying design constraints?</em>
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            A <strong>mechanical linkage</strong> is an assembly of rigid bodies (links) connected by joints that transforms input motion into desired output motion. Linkages are everywhere—from walking robots and grabbers to automotive suspensions and industrial machinery. The simplest closed-chain linkage is the <strong>four-bar linkage</strong>, but complex mechanisms combine multiple linkages to achieve sophisticated motion paths.
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Key Features:
          </Typography>
          <Typography
            variant="body2"
            component="ul"
            sx={{
              pl: 2,
              color: 'var(--color-text-secondary)',
              '& li': { mb: 0.5 }
            }}
          >
            <li><strong>Interactive Design</strong> — Build linkages visually by creating joints and links on a canvas</li>
            <li><strong>Real-time Simulation</strong> — Animate mechanisms to visualize joint trajectories and motion paths</li>
            <li><strong>Path Synthesis</strong> — Draw target trajectories and optimize link dimensions to match them</li>
            <li><strong>Multi-link Support</strong> — Design complex mechanisms beyond simple four-bars</li>
            <li><strong>Save & Load</strong> — Persist your designs and share them as JSON files</li>
          </Typography>
        </CardContent>
      </Card>

      {/* Keyboard Shortcuts Card */}
      <Card sx={{
        mb: 3,
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border-light)',
        transition: 'all 0.25s ease'
      }}>
        <CardContent>
          <Typography
            variant="h6"
            gutterBottom
            sx={{
              fontWeight: 600,
              color: 'var(--color-text-primary)'
            }}
          >
            Keyboard Shortcuts
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              mt: 2,
              color: 'var(--color-text-primary)'
            }}
          >
            Tool Selection
          </Typography>
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: 'auto 1fr',
            gap: 1,
            mb: 2,
            '& kbd': {
              backgroundColor: 'var(--color-bg-tan)',
              border: '1px solid var(--color-border-light)',
              borderRadius: '4px',
              padding: '2px 8px',
              fontFamily: 'monospace',
              fontWeight: 600,
              fontSize: '0.85rem',
              minWidth: '28px',
              textAlign: 'center',
              display: 'inline-block'
            }
          }}>
            <kbd>C</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Create Link — click two points to create a new link</Typography>
            <kbd>S</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Select — click to select joints or links, drag to move</Typography>
            <kbd>X</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Delete — click a joint or link to delete it</Typography>
            <kbd>G</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Group Select — drag a box to select multiple elements</Typography>
            <kbd>M</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Select Mechanism — click to select entire connected mechanism</Typography>
            <kbd>R</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Measure — click two points to measure distance</Typography>
            <kbd>P</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Draw Polygon — click to add vertices, double-click to close; Shift+click to start a circle (center), then set radius</Typography>
            <kbd>E</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Merge — merge polygon/link/forms; Shift+click merged form to unmerge</Typography>
            <kbd>T</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Draw Path — draw a target trajectory for optimization</Typography>
            <kbd>Y</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Explore Trajectories — click a joint to see valid/invalid positions in a circle; click a valid (green) dot to apply that position</Typography>
          </Box>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Node Type Conversion (with joint selected)
          </Typography>
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: 'auto 1fr',
            gap: 1,
            mb: 2,
            '& kbd': {
              backgroundColor: 'var(--color-bg-tan)',
              border: '1px solid var(--color-border-light)',
              borderRadius: '4px',
              padding: '2px 8px',
              fontFamily: 'monospace',
              fontWeight: 600,
              fontSize: '0.85rem',
              minWidth: '28px',
              textAlign: 'center',
              display: 'inline-block'
            }
          }}>
            <kbd>Q</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Convert to Revolute</Typography>
            <kbd>W</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Convert to Static</Typography>
            <kbd>A</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Convert to Crank</Typography>
          </Box>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            General
          </Typography>
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: 'auto 1fr',
            gap: 1,
            '& kbd': {
              backgroundColor: 'var(--color-bg-tan)',
              border: '1px solid var(--color-border-light)',
              borderRadius: '4px',
              padding: '2px 8px',
              fontFamily: 'monospace',
              fontWeight: 600,
              fontSize: '0.85rem',
              minWidth: '28px',
              textAlign: 'center',
              display: 'inline-block'
            }
          }}>
            <kbd>Space</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Play/pause animation (or run simulation if none)</Typography>
            <kbd>Escape</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Cancel current action</Typography>
            <kbd>Delete</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Delete selected elements</Typography>
            <kbd>Enter</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Complete path drawing</Typography>
            <kbd>~</kbd> <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)' }}>Toggle log viewer</Typography>
            <Typography variant="body2" sx={{ color: 'var(--color-text-secondary)', gridColumn: '1 / -1', mt: 1 }}><strong>Tip:</strong> Middle-click drag or <kbd>Shift</kbd> + drag to pan the canvas.</Typography>
          </Box>
        </CardContent>
      </Card>

      {/* Why Doesn't My Mechanism Work Card */}
      <Card sx={{
        mb: 3,
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border-light)',
        transition: 'all 0.25s ease'
      }}>
        <CardContent>
          <Typography
            variant="h6"
            gutterBottom
            sx={{
              fontWeight: 600,
              color: 'var(--color-text-primary)'
            }}
          >
            Why Doesn't My Mechanism Work?
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            There are countless ways to create a poorly-formed mechanism that doesn't meet the requirements for trajectory calculation in this implementation. It's possible to create fully constrained systems—think of a rigid cube—that obviously can't be animated.
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Understanding Node Types
          </Typography>
          <Typography
            variant="body2"
            component="ul"
            sx={{
              pl: 2,
              mb: 2,
              color: 'var(--color-text-secondary)',
              '& li': { mb: 0.5 }
            }}
          >
            <li><strong>Static</strong> — A fixed anchor point that doesn't move. It serves as a ground reference for the mechanism. Press <code>W</code> to convert a node to Static. Notice Static nodes are are marked with an underline are red.</li>
            <li><strong>Crank</strong> — A node that rotates around a fixed point (its parent Static node) at a constant radius. The crank provides the input motion that drives the mechanism. Press <code>A</code> to convert a node to Crank. Notice Crank nodes are are marked with a triangle and are orange.</li>
            <li><strong>Revolute</strong> — A pivot joint whose position is determined by constraints from two parent nodes. Most nodes in a mechanism are Revolute joints. Press <code>Q</code> to convert a node to Revolute. Revolute nodes are not styled in any particular way.</li>
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            The Crank Requirement
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            In this implementation, there must be at least one Crank node, and that crank must be able to make a full revolution. It's <em>very easy</em> to accidentally make even the simplest system over-constrained by making a single link too long or too short! You can see a minimal working example in the Demo section by clicking "Four Bar." If you're having problems with your mechanism, try shortening or lengthening a link.
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Grashof's Law
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            If you're wondering why your four-bar linkage won't complete a full rotation, consider <strong>Grashof's Law</strong>: For four-bar linkages, this law predicts whether continuous rotation is possible based on link lengths. Let <em>s</em> = shortest link, <em>l</em> = longest link, and <em>p</em>, <em>q</em> = the other two links. If <em>s</em> + <em>l</em> ≤ <em>p</em> + <em>q</em>, the linkage permits continuous rotation (crank-rocker or double-crank). Otherwise, it's a non-Grashof linkage limited to oscillation (double-rocker). Note: The current implementation doesn't support double-rocker oscillation.
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Under-Constrained Mechanisms
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            Just as we can over-constrain and lock a mechanism, we can also under-constrain them. If you make a triangle of links, it's locked in shape and will compute correctly. However, if you make a square of links, it would collapse into a parallelogram since it has an extra degree of freedom. You need to fully constrain shapes by adding additional links. For example, if you add a square of links to a working mechanism, trajectory simulation will fail—to fix this, add a diagonal cross-link to triangulate and rigidify the square.
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Summary: Making Your Mechanisms Work
          </Typography>
          <Typography
            variant="body2"
            component="ul"
            sx={{
              pl: 2,
              color: 'var(--color-text-secondary)',
              '& li': { mb: 0.5 }
            }}
          >
            <li>Mark appropriate nodes as Static, Crank, or Revolute</li>
            <li>Shorten or lengthen links as necessary to satisfy Grashof's Law</li>
            <li>Fully constrain open shapes by triangulating them (add diagonal links)</li>
            <li>Avoid dangling or hanging links that aren't part of a closed chain</li>
          </Typography>
        </CardContent>
      </Card>

      {/* Footer with Cheetah GIF, Logo, and Status */}
      <Box sx={{
        mt: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2
      }}>
        <img
          src={cheetahGif2}
          alt="Running Cheetah"
          style={{
            maxWidth: '50%',
            height: 'auto',
            opacity: 0.8
          }}
        />
        <img
          src={acinonyxLogo}
          alt="Acinonyx"
          style={{
            width: '48px',
            height: '48px',
            objectFit: 'contain',
            borderRadius: '8px',
            opacity: 0.7
          }}
        />
        <Typography
          variant="caption"
          align="center"
          display="block"
          sx={{
            color: 'var(--color-text-muted)',
            fontSize: '0.75rem'
          }}
        >
          Attribution-ShareAlike 4.0 License
          <br />
          <a
            href="https://github.com/bastidas/automata"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: 'var(--color-primary)',
              textDecoration: 'none'
            }}
          >
            github.com/bastidas/automata
          </a>
        </Typography>

        {/* Tiny backend status indicator */}
        <Typography
          variant="caption"
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            color: 'var(--color-text-muted)',
            fontSize: '0.7rem',
            mt: 1
          }}
        >
          <span style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            backgroundColor: loading ? '#999' : error ? '#f44336' : '#4caf50',
            display: 'inline-block'
          }} />
          Backend: {loading ? 'checking...' : error ? 'offline' : 'online'}
        </Typography>

        {/* Social Links */}
        <Box sx={{
          display: 'flex',
          gap: 1.5,
          mt: 2,
          alignItems: 'center'
        }}>
          <a
            href="http://www.alexanderbastidasfry.com"
            target="_blank"
            rel="noopener noreferrer"
            style={{ lineHeight: 0 }}
          >
            <img
              src={worldIcon}
              alt="website"
              style={{
                width: '36px',
                height: '36px',
                objectFit: 'contain',
                display: 'block'
              }}
            />
          </a>
          <a
            href="https://github.com/bastidas/Game-Theory"
            target="_blank"
            rel="noopener noreferrer"
            style={{ lineHeight: 0 }}
          >
            <img
              src={githubIcon}
              alt="github"
              style={{
                width: '36px',
                height: '36px',
                objectFit: 'contain',
                display: 'block'
              }}
            />
          </a>
        </Box>
      </Box>
    </Box>
  )
}

export default StatusAboutTab
