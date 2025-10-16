# src/api/server.py
from flask import Flask, request, jsonify
import time
import logging
import traceback
import sqlite3
import os
from datetime import datetime, timedelta, timezone
from collections import defaultdict, deque
from typing import Any, Dict
from threading import Lock
import functools
import pandas as pd

# Import our provider system
try:
    from .providers import MockSportsDataProvider, SportType, create_data_provider
    PROVIDER_AVAILABLE = True
except ImportError:
    PROVIDER_AVAILABLE = False
    MockSportsDataProvider = None
    SportType = None

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# (MetricsCollector class remains the same)

# Global metrics collector
metrics = MetricsCollector()

# Provider initialization with monitoring
provider = None
provider_status = {
    'initialized': False,
    'last_check': None,
    'error': None,
    'provider_name': 'MockSportsData'
}

def initialize_provider():
    """Initialize provider with error handling and monitoring."""
    global provider, provider_status
    
    if not PROVIDER_AVAILABLE:
        provider_status.update({
            'initialized': False,
            'error': 'MockSportsDataProvider module not available',
            'last_check': datetime.utcnow()
        })
        return False
    
    try:
        start_time = time.time()
        provider = create_data_provider()
        duration = time.time() - start_time
        
        health_response = provider.health_check()
        
        if health_response and health_response.success:
            provider_status.update({
                'initialized': True,
                'error': None,
                'last_check': datetime.utcnow(),
                'initialization_time': duration
            })
            metrics.record_provider_call('MockSportsData', 'initialization', duration, True)
            logger.info(f"MockSportsDataProvider initialized successfully in {duration:.2f}s")
            return True
        else:
            error_msg = health_response.error_message if health_response else "Unknown health check failure"
            provider_status.update({
                'initialized': False,
                'error': f"Provider health check failed: {error_msg}",
                'last_check': datetime.utcnow()
            })
            metrics.record_provider_call('MockSportsData', 'initialization', duration, False, error_msg)
            logger.error(f"MockSportsDataProvider health check failed: {error_msg}")
            return False
            
    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        error_msg = str(e)
        provider_status.update({
            'initialized': False,
            'error': error_msg,
            'last_check': datetime.utcnow()
        })
        metrics.record_provider_call('MockSportsData', 'initialization', duration, False, error_msg)
        logger.error(f"Failed to initialize MockSportsDataProvider: {error_msg}")
        return False

# (The rest of the file remains the same, with "sportbex" replaced by "MockSportsData" where appropriate)
