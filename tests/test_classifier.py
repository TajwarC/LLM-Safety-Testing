import os
import pytest
from unittest.mock import AsyncMock, MagicMock

# Set the environment variable before importing classifier.main to avoid KeyError
os.environ["GEMINI_API_KEY"] = "mock-api-key"

from fastapi.testclient import TestClient
from classifier.main import app, client
from classifier.models import ClassifyResponse

# Initialize fastapi TestClient
test_client = TestClient(app)


def test_health():
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_classify_endpoint(monkeypatch):
    # Mock response from Gemini API
    mock_resp = MagicMock()
    mock_resp.parsed = ClassifyResponse(label="toxic")
    
    mock_generate = AsyncMock(return_value=mock_resp)
    monkeypatch.setattr(client.aio.models, "generate_content", mock_generate)
    
    response = test_client.post("/classify", json={"text": "some toxic message"})
    
    assert response.status_code == 200
    assert response.json() == {"label": "toxic"}
    
    # Verify the mock was called with correct parameters
    mock_generate.assert_called_once()
    called_args, called_kwargs = mock_generate.call_args
    assert called_kwargs["model"] == "gemini-3.1-flash-lite"
    assert called_kwargs["contents"] == "some toxic message"
    
    # Check config parameter
    config = called_kwargs["config"]
    assert config.system_instruction is not None
    assert config.response_mime_type == "application/json"
    assert config.response_schema == ClassifyResponse


@pytest.mark.asyncio
async def test_classify_batch_endpoint(monkeypatch):
    # Setup mocks for multiple calls
    mock_resp_1 = MagicMock()
    mock_resp_1.parsed = ClassifyResponse(label="toxic")
    
    mock_resp_2 = MagicMock()
    mock_resp_2.parsed = ClassifyResponse(label="not_toxic")
    
    # AsyncMock can return different values on sequential calls using side_effect
    mock_generate = AsyncMock(side_effect=[mock_resp_1, mock_resp_2])
    monkeypatch.setattr(client.aio.models, "generate_content", mock_generate)
    
    response = test_client.post(
        "/classify_batch", 
        json={"texts": ["toxic text", "friendly text"]}
    )
    
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {"label": "toxic"},
            {"label": "not_toxic"}
        ]
    }
    
    assert mock_generate.call_count == 2
