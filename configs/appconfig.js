// Application Configuration
// Centralized configuration for all ports and endpoints in JavaScript format
// This can be imported by both frontend and backend (when needed)

export const AppConfig = {
  // Port Configuration
  FRONTEND_PORT: 5173,
  BACKEND_PORT: 8021,
  
  // URLs (derived from ports)
  get FRONTEND_URL() {
    return `http://localhost:${this.FRONTEND_PORT}`
  },
  
  get BACKEND_URL() {
    return `http://localhost:${this.BACKEND_PORT}`
  },
  
  // API Configuration
  API_PREFIX: '/api'
}

// Named exports for convenience
export const FRONTEND_PORT = AppConfig.FRONTEND_PORT
export const BACKEND_PORT = AppConfig.BACKEND_PORT
export const FRONTEND_URL = AppConfig.FRONTEND_URL
export const BACKEND_URL = AppConfig.BACKEND_URL