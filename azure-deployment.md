# Enterprise Azure Deployment Guide
## Smart Document Cropper & Orientation Microservice

This guide covers deploying the Smart Document Cropper engine to Microsoft Azure for production microservice architectures.

---

### Architectural Overview
- **Runtime:** Python 3.10 / 3.11 with ONNX Runtime (CPU-optimized, zero GPU required).
- **Inference Latency:** 250ms–650ms on standard 2 vCPU / 4GB RAM instances.
- **Protocol:** REST API (FastAPI / Uvicorn) with Swagger OpenAPI docs (`/docs`), OpenAPI JSON, health checks (`/api/v1/health`), and streaming endpoints.

---

### Deployment Option 1: Azure Container Apps (Recommended - Serverless & Cost Effective)
Azure Container Apps is ideal for enterprise workloads because it supports **scale-to-zero** (zero idle cost) and scales horizontally under request load.

#### Step 1: Login and Create Azure Container Registry (ACR)
```bash
az login
az group create --name rg-smart-cropper --location southeastasia
az acr create --resource-group rg-smart-cropper --name acrsmartcropper --sku Basic --admin-enabled true
```

#### Step 2: Build and Push Docker Image to ACR
```bash
# Build the container image directly in Azure (no local Docker daemon required)
az acr build --registry acrsmartcropper --image smart-document-cropper:v1.0 .
```

#### Step 3: Deploy to Azure Container Apps
```bash
# Create the Container Apps environment
az containerapp env create \
  --name env-smart-cropper \
  --resource-group rg-smart-cropper \
  --location southeastasia

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name acrsmartcropper --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name acrsmartcropper --query "passwords[0].value" -o tsv)

# Deploy the microservice
az containerapp create \
  --name app-smart-cropper \
  --resource-group rg-smart-cropper \
  --environment env-smart-cropper \
  --image acrsmartcropper.azurecr.io/smart-document-cropper:v1.0 \
  --target-port 8000 \
  --ingress external \
  --cpu 2.0 --memory 4.0Gi \
  --min-replicas 1 \
  --max-replicas 10 \
  --registry-server acrsmartcropper.azurecr.io \
  --registry-username $ACR_USERNAME \
  --registry-password $ACR_PASSWORD
```

Your service will now be available with an HTTPS endpoint:
`https://app-smart-cropper.<region>.azurecontainerapps.io/docs`

---

### Deployment Option 2: Azure App Service (Linux Web App)

#### Step 1: Create an App Service Plan
```bash
az appservice plan create \
  --name plan-smart-cropper \
  --resource-group rg-smart-cropper \
  --sku B2 \
  --is-linux
```

#### Step 2: Deploy Container Web App
```bash
az webapp create \
  --resource-group rg-smart-cropper \
  --plan plan-smart-cropper \
  --name webapp-smart-cropper \
  --deployment-container-image-name acrsmartcropper.azurecr.io/smart-document-cropper:v1.0

# Configure port
az webapp config appsettings set \
  --resource-group rg-smart-cropper \
  --name webapp-smart-cropper \
  --settings WEBSITES_PORT=8000
```

---

### Deployment Option 3: Direct Integration into Another Python Project (SDK)
If your other Azure solution is already a Python service (e.g. Django, Flask, FastAPI, Celery worker, Azure Function), you can simply copy the `src/` and `weights/` folders into your repository:

```python
from src.agent.document_cropper_agent import DocumentCropperAgent

cropper = DocumentCropperAgent()

# 1. Direct one-line crop & auto-orient
cropped_bgr = cropper.crop(image_bytes_or_path, safe_margin_percent=0.02, auto_orient=True)

# 2. Extract specific semantic field (e.g., Doctor Name, Total, NIC)
field = cropper.extract_field(image_bytes_or_path, query="Total", is_key_value=True)
```

---

### Enterprise Production Best Practices
1. **Health Check Probes:** Point liveness and readiness probes to `/api/v1/health`.
2. **CPU Optimization:** Keep ONNX Runtime thread count configured to match container vCPU count (already tuned automatically).
3. **Payload Limits:** For large 20MB+ mobile camera images, client requests should upload multipart form files or compressed JPEGs.
4. **CORS:** Configure specific allowed origins in `src/api/app.py` for frontend security.
