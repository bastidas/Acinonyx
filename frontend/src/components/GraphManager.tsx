import React, { useState, useCallback } from 'react'
import { Link, Node, Connection } from '../response_models'

// ============================================================================
// Pure Utility Functions
// ============================================================================

/**
 * Calculate Euclidean distance between two points in unit coordinates
 */
export const calculateLength = (point1: [number, number], point2: [number, number]): number => {
  const dx = point2[0] - point1[0]
  const dy = point2[1] - point1[1]
  return Math.sqrt(dx * dx + dy * dy)
}

// ============================================================================
// Graph State & Types
// ============================================================================

// Graph management utilities
export interface GraphNode extends Node {
  connections: string[] // Array of connection IDs this node participates in
}

// GraphConnection extends Connection with an internal id for tracking
export interface GraphConnection extends Connection {
  id: string // Internal connection tracking id
}

export interface GraphState {
  nodes: GraphNode[]
  connections: GraphConnection[]
  links: Link[] // Single source of truth for all link data
}

// ============================================================================
// Link Deletion Helper Type
// ============================================================================

export interface DeleteItem {
  type: 'link' | 'node'
  id: string
  name: string
}

// Hook for managing graph state
export const useGraphManager = () => {
  const [graphState, setGraphState] = useState<GraphState>({
    nodes: [],
    connections: [],
    links: []
  })

  // ============================================================================
  // Helper: Look up link by meta.id (single source of truth)
  // ============================================================================
  const getLinkById = useCallback((linkId: string): Link | undefined => {
    return graphState.links.find(link => link.meta.id === linkId)
  }, [graphState.links])

  // Helper to get link for a connection
  const getLinkForConnection = useCallback((conn: GraphConnection): Link | undefined => {
    return graphState.links.find(link => link.meta.id === conn.link_id)
  }, [graphState.links])

  // Add a new node
  const addNode = useCallback((x: number, y: number, id?: string): GraphNode => {
    // Generate unique ID if not provided, ensuring no collisions
    let nodeId = id
    if (!nodeId) {
      const timestamp = Date.now()
      const random = Math.random().toString(36).substr(2, 9)
      nodeId = `node_${timestamp}_${random}`
      
      // Double-check for uniqueness (very unlikely but safe)
      while (graphState.nodes.some(node => node.id === nodeId)) {
        nodeId = `node_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      }
    } else {
      // Ensure provided ID is unique
      while (graphState.nodes.some(node => node.id === nodeId)) {
        const suffix = Math.random().toString(36).substr(2, 4)
        nodeId = `${id}_${suffix}`
      }
    }
    
    const newNode: GraphNode = {
      // Required fields from backend
      name: nodeId, // Use ID as name by default
      n_iterations: 24, // Default to match backend expectation
      init_pos: [x, y], // Set initial position
      // Frontend required fields
      id: nodeId,
      pos: [x, y],
      // Optional fields with defaults
      fixed: false,
      fixed_loc: undefined,
      // GraphNode extension
      connections: []
    }
    
    setGraphState(prev => ({
      ...prev,
      nodes: [...prev.nodes, newNode]
    }))
    
    return newNode
  }, [])

  // Find node at position (within tolerance in units)
  const findNodeAt = useCallback((x: number, y: number, tolerance: number = 5.0): GraphNode | null => {
    const foundNode = graphState.nodes.find(node => {
      const distance = Math.sqrt(
        Math.pow(x - node.pos[0], 2) + Math.pow(y - node.pos[1], 2)
      )
      return distance <= tolerance
    }) || null
    
    return foundNode
  }, [graphState.nodes])

  // Add a connection between two nodes with a link
  const addConnection = useCallback((fromNodeId: string, toNodeId: string, link: Link): string => {
    const connectionId = `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    setGraphState(prev => {
      // Find the connected nodes to check if any are fixed
      const fromNode = prev.nodes.find(n => n.id === fromNodeId)
      const toNode = prev.nodes.find(n => n.id === toNodeId)
      const hasFixedNode = (fromNode?.fixed || false) || (toNode?.fixed || false)
      
      // Update link with correct has_fixed field
      const updatedLink: Link = {
        ...link,
        has_fixed: hasFixedNode
      }
      
      // Connection now only references link by id
      const newConnection: GraphConnection = {
        id: connectionId,
        from_node: fromNodeId,
        to_node: toNodeId,
        link_id: updatedLink.meta.id // Reference only, not embedded
      }

      // Update nodes to include this connection
      const updatedNodes = prev.nodes.map(node => {
        if (node.id === fromNodeId || node.id === toNodeId) {
          return {
            ...node,
            connections: [...node.connections, connectionId]
          }
        }
        return node
      })

      return {
        ...prev,
        nodes: updatedNodes,
        connections: [...prev.connections, newConnection],
        links: [...prev.links, updatedLink]
      }
    })

    return connectionId
  }, [])

  // Find connections that start or end at a specific point
  const findConnectionsAt = useCallback((x: number, y: number, tolerance: number = 10): GraphConnection[] => {
    return graphState.connections.filter(conn => {
      const fromNode = graphState.nodes.find(n => n.id === conn.from_node)
      const toNode = graphState.nodes.find(n => n.id === conn.to_node)
      
      if (!fromNode || !toNode) return false
      
      const distToStart = Math.sqrt(Math.pow(x - fromNode.pos[0], 2) + Math.pow(y - fromNode.pos[1], 2))
      const distToEnd = Math.sqrt(Math.pow(x - toNode.pos[0], 2) + Math.pow(y - toNode.pos[1], 2))
      
      return distToStart <= tolerance || distToEnd <= tolerance
    })
  }, [graphState.nodes, graphState.connections])

  // Get node by ID
  const getNode = useCallback((nodeId: string): GraphNode | null => {
    return graphState.nodes.find(node => node.id === nodeId) || null
  }, [graphState.nodes])

  // Update a link's properties (links are the single source of truth)
  const updateLink = useCallback((linkId: string, updates: Partial<Link>) => {
    setGraphState(prev => ({
      ...prev,
      links: prev.links.map(link => {
        if (link.meta.id === linkId) {
          // Handle meta updates separately to preserve structure
          if (updates.meta) {
            return { 
              ...link, 
              ...updates,
              meta: { ...link.meta, ...updates.meta }
            }
          }
          return { 
            ...link, 
            ...updates
          }
        }
        return link
      })
    }))
  }, [])

  // Update node position and all connected link endpoints (including recalculating length)
  const updateNodePosition = useCallback((nodeId: string, newX: number, newY: number) => {
    setGraphState(prev => {
      // Update the node position
      const updatedNodes = prev.nodes.map(node => 
        node.id === nodeId ? { ...node, pos: [newX, newY] as [number, number] } : node
      )
      
      // Find all connections involving this node
      const connectionsInvolving = prev.connections.filter(conn => 
        conn.from_node === nodeId || conn.to_node === nodeId
      )
      
      // Update link endpoints and recalculate length for all connected links
      const updatedLinks = prev.links.map(link => {
        const connection = connectionsInvolving.find(conn => conn.link_id === link.meta.id)
        if (!connection) return link
        
        // Start with current link values from meta
        let newStartPoint = link.meta.start_point
        let newEndPoint = link.meta.end_point
        
        // Update start_point if this node is the from_node
        if (connection.from_node === nodeId) {
          newStartPoint = [newX, newY] as [number, number]
        }
        
        // Update end_point if this node is the to_node  
        if (connection.to_node === nodeId) {
          newEndPoint = [newX, newY] as [number, number]
        }
        
        // Recalculate length from the new start and end points
        let newLength = link.length
        if (newStartPoint && newEndPoint) {
          newLength = calculateLength(newStartPoint, newEndPoint)
        }
        
        return { 
          ...link, 
          length: newLength,
          meta: {
            ...link.meta,
            start_point: newStartPoint,
            end_point: newEndPoint
          }
        }
      })
      
      return {
        ...prev,
        nodes: updatedNodes,
        links: updatedLinks
      }
    })
  }, [])

  // Update a connection to point to different nodes
  const updateConnection = useCallback((connectionId: string, newFromNode: string, newToNode: string) => {
    setGraphState(prev => {
      const updatedConnections = prev.connections.map(conn => 
        conn.id === connectionId 
          ? { ...conn, from_node: newFromNode, to_node: newToNode }
          : conn
      )
      
      return {
        ...prev,
        connections: updatedConnections
      }
    })
  }, [])

  // Delete a link and its connections
  const deleteLink = useCallback((linkId: string) => {
    setGraphState(prev => {
      // Find connections that use this link
      const connectionsToRemove = prev.connections.filter(conn => conn.link_id === linkId)
      const connectionIdsToRemove = connectionsToRemove.map(conn => conn.id)
      
      // Collect node IDs from the removed connections to check for orphaned nodes
      const nodeIdsToCheck = new Set<string>()
      connectionsToRemove.forEach(conn => {
        nodeIdsToCheck.add(conn.from_node)
        nodeIdsToCheck.add(conn.to_node)
      })
      
      // Update nodes to remove these connections
      let updatedNodes = prev.nodes.map(node => ({
        ...node,
        connections: node.connections.filter(connId => !connectionIdsToRemove.includes(connId))
      }))
      
      // Check each node that was connected to the deleted link
      // If a node has no remaining connections, delete it
      const nodeIdsToDelete = new Set<string>()
      nodeIdsToCheck.forEach(nodeId => {
        const node = updatedNodes.find(n => n.id === nodeId)
        if (node && node.connections.length === 0) {
          nodeIdsToDelete.add(nodeId)
        }
      })
      
      // Remove orphaned nodes
      if (nodeIdsToDelete.size > 0) {
        updatedNodes = updatedNodes.filter(node => !nodeIdsToDelete.has(node.id))
      }
      
      return {
        ...prev,
        nodes: updatedNodes,
        connections: prev.connections.filter(conn => conn.link_id !== linkId),
        links: prev.links.filter(link => link.meta.id !== linkId)
      }
    })
  }, [])

  // Delete a node and all its connections
  const deleteNode = useCallback((nodeId: string) => {
    setGraphState(prev => {
      // Find connections that involve this node
      const connectionsToRemove = prev.connections.filter(conn => 
        conn.from_node === nodeId || conn.to_node === nodeId
      )
      const connectionIdsToRemove = connectionsToRemove.map(conn => conn.id)
      const linkIdsToRemove = connectionsToRemove.map(conn => conn.link_id)
      
      // Update remaining nodes to remove these connections
      const updatedNodes = prev.nodes
        .filter(node => node.id !== nodeId)
        .map(node => ({
          ...node,
          connections: node.connections.filter(connId => !connectionIdsToRemove.includes(connId))
        }))
      
      return {
        ...prev,
        nodes: updatedNodes,
        connections: prev.connections.filter(conn => 
          conn.from_node !== nodeId && conn.to_node !== nodeId
        ),
        links: prev.links.filter(link => !linkIdsToRemove.includes(link.meta.id))
      }
    })
  }, [])

  // Merge two nodes - transfer all connections from source to target node
  const mergeNodes = useCallback((sourceNodeId: string, targetNodeId: string) => {
    setGraphState(prev => {
      const sourceNode = prev.nodes.find(node => node.id === sourceNodeId)
      const targetNode = prev.nodes.find(node => node.id === targetNodeId)
      
      if (!sourceNode || !targetNode) return prev
      
      // Update all connections that reference the source node to reference the target node
      const updatedConnections = prev.connections.map(conn => {
        if (conn.from_node === sourceNodeId) {
          return { ...conn, from_node: targetNodeId }
        }
        if (conn.to_node === sourceNodeId) {
          return { ...conn, to_node: targetNodeId }
        }
        return conn
      })
      
      // Update links to use target node position and recalculate length
      const updatedLinks = prev.links.map(link => {
        const connection = updatedConnections.find(conn => conn.link_id === link.meta.id)
        if (!connection) return link
        
        // Start with current link values from meta
        let newStartPoint = link.meta.start_point
        let newEndPoint = link.meta.end_point
        
        if (connection.from_node === targetNodeId && link.meta.start_point) {
          newStartPoint = [targetNode.pos[0], targetNode.pos[1]] as [number, number]
        }
        if (connection.to_node === targetNodeId && link.meta.end_point) {
          newEndPoint = [targetNode.pos[0], targetNode.pos[1]] as [number, number]
        }
        
        // Recalculate length from the new start and end points
        let newLength = link.length
        if (newStartPoint && newEndPoint) {
          newLength = calculateLength(newStartPoint, newEndPoint)
        }
        
        return { 
          ...link, 
          length: newLength,
          meta: {
            ...link.meta,
            start_point: newStartPoint,
            end_point: newEndPoint
          }
        }
      })
      
      // Merge connection IDs and remove the source node
      const mergedConnectionIds = [...new Set([...sourceNode.connections, ...targetNode.connections])]
      const updatedNodes = prev.nodes
        .filter(node => node.id !== sourceNodeId)
        .map(node => {
          if (node.id === targetNodeId) {
            return { ...node, connections: mergedConnectionIds }
          }
          return node
        })
      
      return {
        ...prev,
        nodes: updatedNodes,
        connections: updatedConnections,
        links: updatedLinks
      }
    })
  }, [])

  // Clear all graph data
  const clearGraph = useCallback(() => {
    setGraphState({
      nodes: [],
      connections: [],
      links: []
    })
  }, [])

  // Get graph structure for backend - connections are lightweight references
  const getGraphStructure = useCallback(() => {
    return {
      nodes: graphState.nodes.map(node => ({
        id: node.id,
        pos: node.pos,
        fixed: node.fixed || false,
        fixed_loc: node.fixed_loc
      })),
      // Connections include link_id AND link_name for robust backend matching
      connections: graphState.connections.map(conn => {
        const link = graphState.links.find(l => l.meta.id === conn.link_id)
        return {
          from_node: conn.from_node,
          to_node: conn.to_node,
          link_id: conn.link_id,
          link_name: link?.name  // Include name for fallback matching
        }
      }),
      // Links are the single source of truth
      links: graphState.links
    }
  }, [graphState])

  // Get unique z-levels from links
  const getUniqueZLevels = useCallback((): number[] => {
    const zlevels = graphState.links.map(link => link.zlevel || 0)
    return [...new Set(zlevels)].sort((a, b) => a - b)
  }, [graphState.links])

  // Toggle node fixed state and update connected links
  const toggleNodeFixed = useCallback((nodeId: string, fixed: boolean) => {
    setGraphState(prev => {
      // Update the node
      const updatedNodes = prev.nodes.map(node => {
        if (node.id === nodeId) {
          const newFixedLoc = fixed ? node.pos : undefined
          return {
            ...node,
            fixed,
            fixed_loc: newFixedLoc,
            init_pos: fixed ? node.pos : node.init_pos // Update init_pos when fixing
          }
        }
        return node
      })
      
      // Find all connections involving this node
      const connectionsInvolving = prev.connections.filter(conn => 
        conn.from_node === nodeId || conn.to_node === nodeId
      )
      
      // Update has_fixed field and fixed_loc on connected links
      const updatedLinks = prev.links.map(link => {
        const connection = connectionsInvolving.find(conn => conn.link_id === link.meta.id)
        if (!connection) return link
        
        // Check if either connected node is fixed
        const fromNode = updatedNodes.find(n => n.id === connection.from_node)
        const toNode = updatedNodes.find(n => n.id === connection.to_node)
        const hasFixedNode = (fromNode?.fixed || false) || (toNode?.fixed || false)
        
        // Set fixed_loc on the link if a connected node is fixed
        let linkFixedLoc = link.fixed_loc
        if (hasFixedNode) {
          // Use the position of the fixed node as the link's fixed_loc
          if (fromNode?.fixed && fromNode.fixed_loc) {
            linkFixedLoc = fromNode.fixed_loc
          } else if (toNode?.fixed && toNode.fixed_loc) {
            linkFixedLoc = toNode.fixed_loc
          }
        } else {
          // Clear fixed_loc if no nodes are fixed
          linkFixedLoc = undefined
        }
        
        return {
          ...link,
          has_fixed: hasFixedNode,
          fixed_loc: linkFixedLoc
        }
      })
      
      return {
        ...prev,
        nodes: updatedNodes,
        links: updatedLinks
      }
    })
  }, [])

  // Utility function to sync has_fixed fields on all links based on connected nodes
  const syncLinkFixedFields = useCallback(() => {
    setGraphState(prev => {
      const updatedLinks = prev.links.map(link => {
        // Find connection for this link
        const connection = prev.connections.find(conn => conn.link_id === link.meta.id)
        if (!connection) return link
        
        // Check if either connected node is fixed
        const fromNode = prev.nodes.find(n => n.id === connection.from_node)
        const toNode = prev.nodes.find(n => n.id === connection.to_node)
        const hasFixedNode = (fromNode?.fixed || false) || (toNode?.fixed || false)
        
        return {
          ...link,
          has_fixed: hasFixedNode
        }
      })
      
      return {
        ...prev,
        links: updatedLinks
      }
    })
  }, [])

  return {
    graphState,
    addNode,
    findNodeAt,
    addConnection,
    findConnectionsAt,
    getNode,
    getLinkById,
    getLinkForConnection,
    updateLink,
    updateNodePosition,
    deleteLink,
    deleteNode,
    mergeNodes,
    updateConnection,
    clearGraph,
    getGraphStructure,
    getUniqueZLevels,
    toggleNodeFixed,
    syncLinkFixedFields
  }
}

// GraphManager component for visual debugging (optional)
export const GraphManager: React.FC<{
  graphState: GraphState
  getLinkForConnection: (conn: GraphConnection) => Link | undefined
  onNodeClick?: (node: GraphNode) => void
  onConnectionClick?: (connection: GraphConnection) => void
}> = ({ graphState, getLinkForConnection, onNodeClick: _onNodeClick, onConnectionClick: _onConnectionClick }) => {
  return (
    <div style={{ padding: '10px', border: '1px solid #ccc', margin: '10px' }}>
      <h4>Graph State Debug</h4>
      <div>
        <strong>Nodes ({graphState.nodes.length}):</strong>
        <ul>
          {graphState.nodes.map(node => (
            <li key={node.id}>
              {node.id} at ({node.pos[0].toFixed(0)}, {node.pos[1].toFixed(0)}) 
              - {node.connections.length} connections
            </li>
          ))}
        </ul>
      </div>
      <div>
        <strong>Connections ({graphState.connections.length}):</strong>
        <ul>
          {graphState.connections.map(conn => {
            const link = getLinkForConnection(conn)
            return (
              <li key={conn.id}>
                {conn.from_node} → {conn.to_node} via {link?.name || 'unknown'}
              </li>
            )
          })}
        </ul>
      </div>
    </div>
  )
}
