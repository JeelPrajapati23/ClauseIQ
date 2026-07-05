#!/usr/bin/env bash
# One-time provisioning of the Azure backend: resource group, ACR, Container Apps
# environment, and the Container App itself. Safe to re-run — every `az ... create`
# below is idempotent (Azure no-ops if the resource already exists with the same
# settings). After this runs once, day-to-day deploys go through
# .github/workflows/deploy-backend.yml instead of this script.
#
# Prereqs: `az login` already run, and the `containerapp` extension installed
# (this script installs/upgrades it for you).
#
# Usage: fill in the CONFIG block below, then: bash deploy/azure/bootstrap.sh
set -euo pipefail

# ── CONFIG ──────────────────────────────────────────────────────────────────
RESOURCE_GROUP="clauseiq-rg"
LOCATION="eastus"                      # pick a region close to you / your Neon+Qdrant Cloud region
ACR_NAME="clauseiqacr$RANDOM"          # must be globally unique, alphanumeric only — change or accept the random suffix
ENVIRONMENT_NAME="clauseiq-env"
APP_NAME="clauseiq-backend"
IMAGE_NAME="clauseiq-backend"

# Secrets / config for the running app — fill these in before running.
GROQ_API_KEY="${GROQ_API_KEY:?set GROQ_API_KEY in your shell env before running}"
COHERE_API_KEY="${COHERE_API_KEY:?set COHERE_API_KEY in your shell env before running}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:?set JWT_SECRET_KEY in your shell env before running}"
DATABASE_URL="${DATABASE_URL:?set DATABASE_URL to your Neon connection string before running}"
QDRANT_URL="${QDRANT_URL:?set QDRANT_URL to your Qdrant Cloud cluster URL before running}"
QDRANT_API_KEY="${QDRANT_API_KEY:?set QDRANT_API_KEY to your Qdrant Cloud API key before running}"
APP_FRONTEND_URL="${APP_FRONTEND_URL:?set APP_FRONTEND_URL to your Vercel URL before running}"
CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS:-$APP_FRONTEND_URL}"
RAG_SYSTEM_PROMPT_FILE="${RAG_SYSTEM_PROMPT_FILE:-system_prompt_v3.txt}"
# ─────────────────────────────────────────────────────────────────────────────

az extension add --name containerapp --upgrade -y
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

echo "== Resource group =="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "== Container Registry ($ACR_NAME) =="
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true --output none
ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)"

echo "== Building and pushing the backend image via ACR Tasks (no local Docker needed) =="
az acr build --registry "$ACR_NAME" --image "$IMAGE_NAME:latest" .

echo "== Container Apps environment =="
az containerapp env create \
  --name "$ENVIRONMENT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --output none

echo "== Container App (first revision) =="
ACR_USERNAME="$(az acr credential show --name "$ACR_NAME" --query username -o tsv)"
ACR_PASSWORD="$(az acr credential show --name "$ACR_NAME" --query 'passwords[0].value' -o tsv)"

az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$ENVIRONMENT_NAME" \
  --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest" \
  --registry-server "$ACR_LOGIN_SERVER" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --target-port 8000 \
  --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets \
    groq-api-key="$GROQ_API_KEY" \
    cohere-api-key="$COHERE_API_KEY" \
    jwt-secret-key="$JWT_SECRET_KEY" \
    database-url="$DATABASE_URL" \
    qdrant-api-key="$QDRANT_API_KEY" \
  --env-vars \
    ENVIRONMENT=production \
    RAG_SYSTEM_PROMPT_FILE="$RAG_SYSTEM_PROMPT_FILE" \
    QDRANT_URL="$QDRANT_URL" \
    APP_FRONTEND_URL="$APP_FRONTEND_URL" \
    CORS_ALLOWED_ORIGINS="$CORS_ALLOWED_ORIGINS" \
    GROQ_API_KEY=secretref:groq-api-key \
    COHERE_API_KEY=secretref:cohere-api-key \
    JWT_SECRET_KEY=secretref:jwt-secret-key \
    DATABASE_URL=secretref:database-url \
    QDRANT_API_KEY=secretref:qdrant-api-key \
  --output none

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"

echo "== Setting APP_URL to the app's own HTTPS endpoint =="
az containerapp update \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --set-env-vars APP_URL="https://$FQDN" \
  --output none

cat <<EOF

Done.

Backend URL:        https://$FQDN
ACR login server:   $ACR_LOGIN_SERVER
Resource group:      $RESOURCE_GROUP

Next steps:
1. Point Vercel's VITE_API_URL build env var at https://$FQDN
2. Add these GitHub Actions secrets for ongoing CI/CD deploys (see
   .github/workflows/deploy-backend.yml):
     AZURE_CREDENTIALS   (service principal JSON, scoped to $RESOURCE_GROUP)
     ACR_NAME             = $ACR_NAME
     RESOURCE_GROUP       = $RESOURCE_GROUP
     CONTAINER_APP_NAME   = $APP_NAME
3. min-replicas is 0 to conserve your Azure student credit — the first request
   after idle will be slow (cold start: container boot + embedding model load).
   Set --min-replicas 1 on the Container App if you'd rather keep it warm.
4. Do not raise --max-replicas above 1: app/database.py's BM25 cache is
   per-process in-memory, so a second replica would serve a stale cache.
EOF
