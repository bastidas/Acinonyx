# Application Configuration
# Centralized configuration for all ports and endpoints

class AppConfig:
    """Centralized application configuration"""
    
    # Port Configuration
    FRONTEND_PORT = 5173
    BACKEND_PORT = 8021
    
    # URLs (derived from ports)
    FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"
    BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
    
    # API Configuration
    API_PREFIX = "/api"
    
    @classmethod
    def get_frontend_url(cls):
        return cls.FRONTEND_URL
    
    @classmethod
    def get_backend_url(cls):
        return cls.BACKEND_URL
    
    @classmethod
    def get_api_base_url(cls):
        return f"{cls.BACKEND_URL}{cls.API_PREFIX}"

# For backward compatibility and easy imports
FRONTEND_PORT = AppConfig.FRONTEND_PORT
BACKEND_PORT = AppConfig.BACKEND_PORT
FRONTEND_URL = AppConfig.FRONTEND_URL
BACKEND_URL = AppConfig.BACKEND_URL