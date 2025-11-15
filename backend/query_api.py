import copy
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional


from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain.schema.document import Document
from pydantic import BaseModel
from automata.conigs.config_models import AutoConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan should instantiate code will be executed once, before the application
    starts receiving requests.

    Because this code is executed before the application starts taking requests, and
    right after it finishes handling requests, it covers the whole application lifespan
    """
    # Load the ML model - QueryHandler now expects QueryHandlerConfig object
    config: AutoConfig = get_config()

    lifespan_vars["config"] = config  

    yield
    # Clean up the ML models and release the resources
    lifespan_vars.clear()


app = FastAPI(
    title="Tangent Query API",
    description="API for Automata frontend",
    version="0.9.0",
    lifespan=lifespan
)


class StatusRequest(BaseModel):
    """Request model for query endpoint."""

    query: str
    streaming: bool = False


class StatusResponse(BaseModel):
    """Response model for query endpoint."""

    response: str



@app.post("/status")
async def handle_query(request: QueryRequest):
    """Handle status request.

  
    """
   
    try:
        if request.streaming:
            return StreamingResponse(
                stream_query_response(request.query), media_type="text/plain"
            )
        else:
            response = lifespan_vars["query_handler"].handle_query(request.query)
            return {"response": response.response, "sources": response.sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
