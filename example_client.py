"""
Example Client demonstrating:
1. Python Agent invocation (direct plug-in)
2. REST API call (file upload)
3. REST API call (base64 JSON)
"""
import os
import json
import base64
import requests
from pathlib import Path
from src.agent.document_cropper_agent import DocumentCropperAgent

SAMPLE_IMAGE = "data/01_02_Senuri.jpg"

def demo_direct_agent():
    print("=" * 60)
    print("DEMO 1: DIRECT PYTHON PLUG-IN AGENT")
    print("=" * 60)
    agent = DocumentCropperAgent()
    res = agent.process(
        SAMPLE_IMAGE,
        safe_margin_percent=0.01,
        return_base64=False,
        output_save_path="eval_results/demo_agent_crop.jpg"
    )
    print(f"Success: {res['success']}")
    print(f"Pipeline Stage: {res['pipeline_stage']}")
    print(f"Document Category: {res['document_classification']['category']}")
    print(f"Aspect Ratio: {res['document_classification']['aspect_ratio']}")
    print(f"Corners: {res['corners_pixel']}")
    print(f"Quality Status: {res['quality_assessment']['quality_status']}")
    print(f"Latency: {res['latency_ms']:.2f} ms")
    print("Saved crop to: eval_results/demo_agent_crop.jpg\n")


def demo_rest_api_upload(base_url="http://localhost:8008"):
    print("=" * 60)
    print("DEMO 2: FASTAPI REST API (FILE UPLOAD)")
    print("=" * 60)
    try:
        url = f"{base_url}/api/v1/crop"
        with open(SAMPLE_IMAGE, "rb") as f:
            files = {"file": (os.path.basename(SAMPLE_IMAGE), f, "image/jpeg")}
            data = {"safe_margin_percent": 0.01, "preferred_model": "auto"}
            response = requests.post(url, files=files, data=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("API Response Received:")
            print(f"  Stage: {result['pipeline_stage']}")
            print(f"  Category: {result['document_classification']['category']}")
            print(f"  Latency: {result['latency_ms']} ms")
        else:
            print(f"API Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Note: REST server not running yet. Start it with: python run_server.py --mode api\n")


def demo_rest_api_json(base_url="http://localhost:8008"):
    print("=" * 60)
    print("DEMO 3: FASTAPI REST API (BASE64 JSON)")
    print("=" * 60)
    try:
        url = f"{base_url}/api/v1/crop-json"
        with open(SAMPLE_IMAGE, "rb") as f:
            b64_str = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            "image_base64": f"data:image/jpeg;base64,{b64_str}",
            "safe_margin_percent": 0.01,
            "preferred_model": "auto"
        }
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            print("JSON API Response Received:")
            print(f"  Stage: {result['pipeline_stage']}")
            print(f"  Crop Base64 length: {len(result.get('crop_base64', ''))}")
        else:
            print(f"API Error {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Note: REST server not running yet. Start it with: python run_server.py --mode api\n")


if __name__ == "__main__":
    demo_direct_agent()
    demo_rest_api_upload()
    demo_rest_api_json()
