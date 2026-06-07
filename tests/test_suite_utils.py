import os
import json
import pytest
import asyncio
import httpx
from unittest.mock import MagicMock, patch

from evaluation_suite.utils import process_batch
from evaluation_suite.eval_robustness import calculate_mcc
from evaluation_suite.data_loader import load_bias_data, load_robustness_data, load_correctness_data
from evaluation_suite.reporter import report_correctness, report_robustness, report_bias


# --- Tests for evaluation_suite/utils.py (process_batch) ---

@pytest.mark.asyncio
async def test_process_batch_success():
    # Mock httpx response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [{"label": "toxic"}, {"label": "not_toxic"}]
    }
    
    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = mock_post
    
    sem = asyncio.Semaphore(1)
    batch = ["toxic text", "friendly text"]
    
    result = await process_batch(mock_client, batch, "http://mock-api/classify_batch", sem)
    assert result == ["toxic", "not_toxic"]


@pytest.mark.asyncio
async def test_process_batch_api_error():
    # Mock httpx response with error status
    mock_response = MagicMock()
    mock_response.status_code = 500
    
    async def mock_post(*args, **kwargs):
        return mock_response

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = mock_post
    
    sem = asyncio.Semaphore(1)
    batch = ["text1", "text2"]
    
    result = await process_batch(mock_client, batch, "http://mock-api/classify_batch", sem)
    assert result == ["error", "error"]


@pytest.mark.asyncio
async def test_process_batch_exception():
    # Mock post raising exception
    async def mock_post(*args, **kwargs):
        raise httpx.RequestError("Connection failed")

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = mock_post
    
    sem = asyncio.Semaphore(1)
    batch = ["text1"]
    
    result = await process_batch(mock_client, batch, "http://mock-api/classify_batch", sem)
    assert result == ["error"]


# --- Tests for evaluation_suite/eval_robustness.py (calculate_mcc) ---

def test_calculate_mcc_perfect_prediction():
    true = ["toxic", "not_toxic", "toxic", "not_toxic"]
    pred = ["toxic", "not_toxic", "toxic", "not_toxic"]
    assert calculate_mcc(true, pred) == 1.0


def test_calculate_mcc_worst_prediction():
    true = ["toxic", "not_toxic", "toxic", "not_toxic"]
    pred = ["not_toxic", "toxic", "not_toxic", "toxic"]
    assert calculate_mcc(true, pred) == -1.0


def test_calculate_mcc_zero_denominator():
    # When denominator is zero (e.g. all predictions are "toxic")
    true = ["toxic", "not_toxic", "toxic", "not_toxic"]
    pred = ["toxic", "toxic", "toxic", "toxic"]
    # tp = 2, fp = 2, fn = 0, tn = 0
    # Denominator: math.sqrt((2+2)*(2+0)*(0+2)*(0+0)) = math.sqrt(4 * 2 * 2 * 0) = 0
    assert calculate_mcc(true, pred) == 0.0


# --- Tests for evaluation_suite/data_loader.py ---

@patch("evaluation_suite.data_loader.load_dataset")
def test_load_bias_data(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.__getitem__.return_value = "mock_train_dataset"
    mock_load_dataset.return_value = mock_ds
    
    result = load_bias_data()
    assert result == "mock_train_dataset"
    mock_load_dataset.assert_called_once_with("TajwarC/Social-Media-Counterfactuals")


@patch("evaluation_suite.data_loader.load_dataset")
def test_load_robustness_data(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.__getitem__.return_value = ["sample1", "sample2"]
    mock_load_dataset.return_value = mock_ds
    
    result = load_robustness_data()
    assert result == ["sample1", "sample2"]
    mock_load_dataset.assert_called_once_with("TajwarC/mteb_toxic_conversations_2.5k_robustness")


@patch("evaluation_suite.data_loader.load_dataset")
def test_load_correctness_data(mock_load_dataset):
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 100
    mock_load_dataset.return_value = mock_ds
    
    load_correctness_data(sample_size=10)
    mock_load_dataset.assert_called_once_with("mteb/toxic_conversations_50k", split="train")
    mock_ds.select.assert_called_once_with(range(10))


# --- Tests for evaluation_suite/reporter.py ---

def test_report_correctness(tmp_path):
    output_dir = tmp_path / "reports"
    metrics = {
        "tp": 10, "fp": 2, "fn": 1, "tn": 87,
        "precision": 0.8333, "recall": 0.9091, "mcc": 0.84,
        "mcc_threshold": 0.50, "precision_threshold": 0.80,
        "passed": True
    }
    
    report_correctness(metrics, output_dir=str(output_dir))
    
    report_file = output_dir / "correctness_report.json"
    assert report_file.exists()
    
    with open(report_file, "r") as f:
        data = json.load(f)
    assert data["tp"] == 10
    assert data["passed"] is True
    assert "timestamp" in data


def test_report_robustness(tmp_path):
    output_dir = tmp_path / "reports"
    metrics = {
        "invariance_threshold": 0.90,
        "base_mcc": 0.85,
        "perturbations": [
            {
                "type": "typo",
                "robust_mcc": 0.84,
                "invariance_rate": 0.95,
                "passed": True
            }
        ],
        "passed": True
    }
    
    report_robustness(metrics, output_dir=str(output_dir))
    
    report_file = output_dir / "robustness_report.json"
    assert report_file.exists()
    
    with open(report_file, "r") as f:
        data = json.load(f)
    assert data["base_mcc"] == 0.85
    assert data["passed"] is True


def test_report_bias(tmp_path):
    output_dir = tmp_path / "reports"
    metrics = {
        "disparity_threshold": 0.05,
        "passed": True,
        "control": {
            "term": "someone",
            "fpr": 0.02,
            "fnr": 0.03
        },
        "categories": {
            "race": {
                "delta_fpr": 0.01,
                "delta_fnr": 0.02,
                "passed": True,
                "terms": [
                    {
                        "term": "black",
                        "fpr": 0.025,
                        "fnr": 0.035,
                        "fps": 5,
                        "benign_total": 200,
                        "fns": 7,
                        "toxic_total": 200
                    }
                ]
            }
        }
    }
    
    report_bias(metrics, output_dir=str(output_dir))
    
    report_file = output_dir / "bias_report.json"
    assert report_file.exists()
    
    with open(report_file, "r") as f:
        data = json.load(f)
    assert data["control"]["term"] == "someone"
    assert data["passed"] is True
