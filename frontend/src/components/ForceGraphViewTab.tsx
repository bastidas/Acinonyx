import React, { useState, useRef, useEffect } from 'react'
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Alert,
  CircularProgress,
  Collapse,
  IconButton
} from '@mui/material'
import VisibilityIcon from '@mui/icons-material/Visibility'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import * as d3 from 'd3'
import { graphColors, colors } from '../theme'

interface GraphData {
  nodes: any[]
  links: any[]
  connections: any[]
}

interface ForceGraphProps {
  data: GraphData
}

const ForceGraph: React.FC<ForceGraphProps> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!data || !svgRef.current) return

    // Clear previous content
    d3.select(svgRef.current).selectAll("*").remove()

    const svg = d3.select(svgRef.current)
    const width = 1200
    const height = 800

    // Use the data structure directly as it comes from force.json
    // The data already has the correct format: nodes with id, links with source/target
    const nodes = data.nodes.map(node => ({ ...node }))
    const links = data.links.map(link => ({ ...link }))

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d: any) => d.id))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2))

    // Add links - with color coding for driven links
    const link = svg.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .style("fill", "none")
      .style("stroke", (d: any) => {
        // Check if this link is driven (blue for driven, dark for others)
        const isDrivern = d.link && d.link.is_driven
        return isDrivern ? graphColors[0] : colors.textPrimary
      })
      .style("stroke-width", (d: any) => {
        // Make driven links slightly thicker
        const isDrivern = d.link && d.link.is_driven
        return isDrivern ? "2px" : "1px"
      })

    // Add link labels
    const linkLabels = svg.append("g")
      .attr("class", "link-labels")
      .selectAll("text")
      .data(links)
      .enter().append("text")
      .style("font-size", "10px")
      .style("fill", "#666")
      .style("text-anchor", "middle")
      .style("pointer-events", "none")
      .style("user-select", "none")

    // Add nodes - matching the original force.css styling
    const node = svg.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 5)
      .style("cursor", "pointer")
      .style("fill", colors.textPrimary)
      .style("stroke", colors.surface)
      .style("stroke-width", "1px")
      .call(d3.drag<SVGCircleElement, any>()
        .on("start", (event, d: any) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on("drag", (event, d: any) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on("end", (event, d: any) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        }))

    // Add node labels
    const nodeLabels = svg.append("g")
      .attr("class", "node-labels")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .style("font-size", "12px")
      .style("fill", "#333")
      .style("text-anchor", "middle")
      .style("pointer-events", "none")
      .style("user-select", "none")
      .style("font-weight", "bold")
      .text((d: any) => d.name || d.id)

    // Add tooltips
    node.append("title")
      .text((d: any) => d.name || d.id)

    // Set up the simulation
    simulation
      .nodes(nodes)
      .on("tick", ticked)

    const linkForce = simulation.force("link") as d3.ForceLink<any, any>
    if (linkForce) {
      linkForce.links(links)
    }

    function ticked() {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y)

      // Update link labels with positions and angles
      linkLabels
        .attr("x", (d: any) => (d.source.x + d.target.x) / 2)
        .attr("y", (d: any) => (d.source.y + d.target.y) / 2)
        .attr("transform", (d: any) => {
          const dx = d.target.x - d.source.x
          const dy = d.target.y - d.source.y
          const angle = Math.atan2(dy, dx) * 180 / Math.PI
          return `rotate(${angle}, ${(d.source.x + d.target.x) / 2}, ${(d.source.y + d.target.y) / 2})`
        })
        .text((d: any) => {
          //const dx = d.target.x - d.source.x
          //const dy = d.target.y - d.source.y
          //const angle = Math.atan2(dy, dx) * 180 / Math.PI
          const linkName = d.link ? d.link.name : 'link'
          return `${linkName}`
        })

      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y)

      // Update node labels
      nodeLabels
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y - 8) // Position above the node
    }

    return () => {
      simulation.stop()
    }
  }, [data])

  return (
    <svg
      ref={svgRef}
      width={1200}
      height={800}
      style={{
        border: '1px solid #ccc',
        borderRadius: '4px',
        backgroundColor: '#fff'
      }}
    />
  )
}

const GraphViewTab: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showJsonData, setShowJsonData] = useState(false)

  const loadGraphData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('api/load-last-force-graph', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      })
      
      if (!response.ok) {
        throw new Error(`Failed to load graph data: ${response.statusText}`)
      }
      
      const data = await response.json()
      
      // Check if the response contains an error
      if (data.error) {
        throw new Error(data.error)
      }
      
      setGraphData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{ p: 2 }}>
      <Typography variant="h4" component="h2" gutterBottom>
        Force Graph View
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Load and visualize mechanical linkage graphs from saved configurations.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Load Force Graph Data
          </Typography>
          
          <Button
            variant="contained"
            startIcon={loading ? <CircularProgress size={20} /> : <VisibilityIcon />}
            onClick={loadGraphData}
            disabled={loading}
            sx={{ mb: 2 }}
          >
            {loading ? 'Loading...' : 'Load Most Recent Force Graph'}
          </Button>
          
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          {graphData && (
            <Alert severity="success" sx={{ mb: 2 }}>
              Graph data loaded successfully!
            </Alert>
          )}
        </CardContent>
      </Card>

      {graphData && (
        <Box>
          {/* Main Graph Visualization */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Graph Visualization
              </Typography>
              
              <Typography variant="body2" sx={{ mb: 2 }}>
                Interactive force-directed layout. Drag nodes to rearrange the visualization.
              </Typography>
              
              <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                <ForceGraph data={graphData} />
              </Box>
            </CardContent>
          </Card>

          {/* Collapsible Graph Data Section */}
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ flexGrow: 1 }}>
                  Graph Data
                </Typography>
                <IconButton
                  onClick={() => setShowJsonData(!showJsonData)}
                  aria-label="toggle graph data visibility"
                >
                  {showJsonData ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Box>
              
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Nodes:</strong> {graphData.nodes?.length || 0}
              </Typography>
              
              <Typography variant="body2" sx={{ mb: 1 }}>
                <strong>Links:</strong> {graphData.links?.length || 0}
              </Typography>
              
              <Typography variant="body2" sx={{ mb: 2 }}>
                <strong>Connections:</strong> {graphData.connections?.length || 0}
              </Typography>
              
              <Collapse in={showJsonData}>
                <Box sx={{ 
                  backgroundColor: '#f5f5f5', 
                  p: 2, 
                  borderRadius: 1, 
                  maxHeight: '400px', 
                  overflow: 'auto' 
                }}>
                  <pre style={{ margin: 0, fontSize: '10px' }}>
                    {JSON.stringify(graphData, null, 2)}
                  </pre>
                </Box>
              </Collapse>
            </CardContent>
          </Card>
        </Box>
      )}
    </Box>
  )
}

export default GraphViewTab