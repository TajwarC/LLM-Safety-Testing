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


from classifier.openai import OpenAIClassifier
from classifier.huggingface import HuggingFaceClassifier

@pytest.mark.asyncio
async def test_openai_classifier(monkeypatch):
    mock_parse = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.parsed = ClassifyResponse(label="toxic")
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    mock_parse.return_value = mock_resp

    clf = OpenAIClassifier(api_key="mock-key", model_name="gpt-4o-mini")
    monkeypatch.setattr(clf.client.beta.chat.completions, "parse", mock_parse)

    resp = await clf.classify("some input")
    assert resp.label == "toxic"

    mock_parse.assert_called_once()
    called_kwargs = mock_parse.call_args[1]
    assert called_kwargs["model"] == "gpt-4o-mini"
    assert called_kwargs["response_format"] == ClassifyResponse


@pytest.mark.asyncio
async def test_huggingface_classifier_standard(monkeypatch):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"label": "toxic", "score": 0.99}]
    
    # Mock pipeline function call in toxic_bert module
    monkeypatch.setattr("classifier.toxic_bert.pipeline", lambda *args, **kwargs: mock_pipe)

    clf = HuggingFaceClassifier(model_name_or_path="unitary/toxic-bert", device="cpu")
    
    resp = await clf.classify("some input")
    assert resp.label == "toxic"
    mock_pipe.assert_called_with("some input")

    # Test batch
    mock_pipe.return_value = [
        {"label": "toxic", "score": 0.99},
        {"label": "not_toxic", "score": 0.01}
    ]
    resp_batch = await clf.classify_batch(["input1", "input2"])
    assert resp_batch.results[0].label == "toxic"
    assert resp_batch.results[1].label == "not_toxic"


@pytest.mark.asyncio
async def test_huggingface_classifier_llamaguard(monkeypatch):
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "unsafe\n01,02"}]
    
    # Mock pipeline function call in llamaguard module
    monkeypatch.setattr("classifier.llamaguard.pipeline", lambda *args, **kwargs: mock_pipe)

    clf = HuggingFaceClassifier(model_name_or_path="meta-llama/LlamaGuard-7b", device="cpu")
    
    resp = await clf.classify("some input")
    assert resp.label == "toxic"

    # Test batch
    mock_pipe.return_value = [
        {"generated_text": "unsafe\n01"},
        {"generated_text": "safe"}
    ]
    resp_batch = await clf.classify_batch(["input1", "input2"])
    assert resp_batch.results[0].label == "toxic"
    assert resp_batch.results[1].label == "not_toxic"

