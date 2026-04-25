"""
Verdict — FastAPI Server (OpenEnv compliant)
=============================================
Creates the FastAPI app using OpenEnv's create_fastapi_app helper.
Referenced by openenv.yaml as server.app:app
"""

from openenv.core.env_server import create_fastapi_app
from .models import VerdictAction, VerdictObservation
from .verdict_environment import VerdictEnvironment

env = VerdictEnvironment(max_rounds=4)
app = create_fastapi_app(env, VerdictAction, VerdictObservation)
