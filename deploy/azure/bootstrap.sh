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
# Secrets live in deploy/azure/.env.azure (gitignored — see
# deploy/azure/.env.azure.example for the template), NOT in this file. Never
# paste real keys into this script: it's tracked by git, and every
# ${VAR:?message} below just prints `message` as an error if VAR is unset —
# it is not a default value.
#
# Usage: cp deploy/azure/.env.azure.example deploy/azure/.env.azure, fill it
# in, then: bash deploy/azure/bootstrap.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, so `az acr build .` picks up the right Dockerfile

ENV_FILE="deploy/azure/.env.azure"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# ── CONFIG ──────────────────────────────────────────────────────────────────
RESOURCE_GROUP="clauseiq-rg"
# This subscription has an explicit Azure Policy ("Allowed resource deployment regions")
# restricting deployments to exactly: eastasia, koreacentral, centralindia, austriaeast,
# southeastasia (check via `az policy assignment list` if this ever changes). Override by
# setting LOCATION in deploy/azure/.env.azure to pick a different one of those five.
LOCATION="${LOCATION:-centralindia}"
ACR_NAME="clauseiqacr$RANDOM"          # must be globally unique, alphanumeric only — change or accept the random suffix
ENVIRONMENT_NAME="clauseiq-env"
APP_NAME="clauseiq-backend"
IMAGE_NAME="clauseiq-backend"

# Secrets / config for the running app — set these in deploy/azure/.env.azure.
GROQ_API_KEY="${GROQ_API_KEY:?set GROQ_API_KEY in deploy/azure/.env.azure}"
COHERE_API_KEY="${COHERE_API_KEY:?set COHERE_API_KEY in deploy/azure/.env.azure}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:?set JWT_SECRET_KEY in deploy/azure/.env.azure}"
DATABASE_URL="${DATABASE_URL:?set DATABASE_URL (Neon connection string) in deploy/azure/.env.azure}"
QDRANT_URL="${QDRANT_URL:?set QDRANT_URL (Qdrant Cloud cluster URL) in deploy/azure/.env.azure}"
QDRANT_API_KEY="${QDRANT_API_KEY:?set QDRANT_API_KEY (Qdrant Cloud API key) in deploy/azure/.env.azure}"
APP_FRONTEND_URL="${APP_FRONTEND_URL:?set APP_FRONTEND_URL (your Vercel URL) in deploy/azure/.env.azure}"
CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS:-$APP_FRONTEND_URL}"
RAG_SYSTEM_PROMPT_FILE="${RAG_SYSTEM_PROMPT_FILE:-system_prompt_v3.txt}"
# Optional — same as locally, leave blank and password-reset links get printed to the
# Container App's logs instead of emailed (see app/auth/email_utils.py's fallback).
SMTP_HOST="${SMTP_HOST:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASS="${SMTP_PASS:-}"
FROM_EMAIL="${FROM_EMAIL:-noreply@clauseiq.local}"
# ─────────────────────────────────────────────────────────────────────────────

az extension add --name containerapp --upgrade -y
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait

echo "== Resource group =="
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none

echo "== Container Registry ($ACR_NAME) =="
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --admin-enabled true --output none
ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)"

echo "== Building and pushing the backend image via local Docker =="
# ACR Tasks (`az acr build`) is disabled on student/trial subscriptions (TasksOperationsNotAllowed),
# so build locally instead — needs Docker Desktop (or another local Docker) running.
az acr login --name "$ACR_NAME"
docker build -t "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest" .
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"

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
    smtp-user="$SMTP_USER" \
    smtp-pass="$SMTP_PASS" \
  --env-vars \
    ENVIRONMENT=production \
    RAG_SYSTEM_PROMPT_FILE="$RAG_SYSTEM_PROMPT_FILE" \
    QDRANT_URL="$QDRANT_URL" \
    APP_FRONTEND_URL="$APP_FRONTEND_URL" \
    CORS_ALLOWED_ORIGINS="$CORS_ALLOWED_ORIGINS" \
    SMTP_HOST="$SMTP_HOST" \
    SMTP_PORT="$SMTP_PORT" \
    FROM_EMAIL="$FROM_EMAIL" \
    GROQ_API_KEY=secretref:groq-api-key \
    COHERE_API_KEY=secretref:cohere-api-key \
    JWT_SECRET_KEY=secretref:jwt-secret-key \
    DATABASE_URL=secretref:database-url \
    QDRANT_API_KEY=secretref:qdrant-api-key \
    SMTP_USER=secretref:smtp-user \
    SMTP_PASS=secretref:smtp-pass \
  --output none

FQDN="$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)"

echo "== Setting APP_URL to the app's own HTTPS endpoint =="
# ARM sometimes hasn't finished propagating the just-created app to the endpoint this
# call hits, causing a spurious "does not exist" — retry a few times before giving up.
for attempt in 1 2 3 4 5; do
  if az containerapp update \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --set-env-vars APP_URL="https://$FQDN" \
    --output none; then
    break
  fi
  if [[ "$attempt" -eq 5 ]]; then
    echo "Still failing after 5 attempts. The app exists at https://$FQDN — set APP_URL manually with:" >&2
    echo "  az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --set-env-vars APP_URL=https://$FQDN" >&2
    exit 1
  fi
  echo "  not yet visible for update, retrying in 5s (attempt $attempt/5)..."
  sleep 5
done

cat <<EOF

Done.

Backend URL:        https://$FQDN
ACR login server:   $ACR_LOGIN_SERVER
Resource group:      $RESOURCE_GROUP

Next steps:
1. Frontend/vercel.json proxies /api/* on the Vercel domain to this backend —
   update its "destination" to https://$FQDN if it doesn't match already.
   Leave VITE_API_URL unset in Vercel (production builds default to the
   relative "/api" path); only set it if you need to bypass the proxy.
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
