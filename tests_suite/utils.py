import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/classify_batch"

async def process_batch(client: httpx.AsyncClient, batch: list[str], api_url: str, sem: asyncio.Semaphore) -> list[str]:
    """
    Processes a single batch of texts through the classifier API with concurrency limits.
    
    Args:
        client: The httpx AsyncClient to use for the request.
        batch: A list of strings to classify.
        api_url: The URL of the classifier API.
        sem: An asyncio.Semaphore to control concurrency.
        
    Returns:
        A list of predicted labels.
    """
    async with sem:
        try:
            response = await client.post(api_url, json={"texts": batch}, timeout=30.0)
            if response.status_code == 200:
                return [res["label"] for res in response.json().get("results", [])]
            else:
                logger.error(f"API returned status {response.status_code} for batch.")
                return ["error"] * len(batch)
        except Exception as e:
            logger.error(f"API request failed for batch: {e}")
            return ["error"] * len(batch)
