import os
import pytest
from unittest.mock import AsyncMock, MagicMock

# Set the environment variable before importing classifier.main to avoid KeyError
os.environ["OPENAI_API_KEY"] = "mock-api-key"

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
    # Mock response from OpenAI chat completion
    mock_choice = MagicMock()
    mock_choice.message.content = '{"label": "toxic"}'
    
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    mock_create = AsyncMock(return_value=mock_completion)
    monkeypatch.setattr(client.chat.completions, "create", mock_create)
    
    response = test_client.post("/classify", json={"text": "some toxic message"})
    
    assert response.status_code == 200
    assert response.json() == {"label": "toxic"}
    
    # Verify the mock was called with correct parameters
    mock_create.assert_called_once()
    called_args, called_kwargs = mock_create.call_args
    assert called_kwargs["model"] == "gpt-5.4-nano"
    assert called_kwargs["messages"][1]["content"] == "some toxic message"


@pytest.mark.asyncio
async def test_classify_batch_endpoint(monkeypatch):
    # Setup mocks for multiple calls
    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = '{"label": "toxic"}'
    mock_completion_1 = MagicMock()
    mock_completion_1.choices = [mock_choice_1]
    
    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = '{"label": "not_toxic"}'
    mock_completion_2 = MagicMock()
    mock_completion_2.choices = [mock_choice_2]
    
    # AsyncMock can return different values on sequential calls using side_effect
    mock_create = AsyncMock(side_effect=[mock_completion_1, mock_completion_2])
    monkeypatch.setattr(client.chat.completions, "create", mock_create)
    
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
    
    assert mock_create.call_count == 2
