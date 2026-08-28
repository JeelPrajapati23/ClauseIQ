#!/usr/bin/env bash
# One-time provisioning of the GCP backend: Artifact Registry repo, Secret
# Manager secrets, the Cloud Run service itself, and a service account for
# GitHub Actions to deploy with. Safe to re-run — every `gcloud ... create`
# below either no-ops or is guarded with an existence check. After this runs
# once, day-to-day deploys go through .github/workflows/deploy-backend.yml
# instead of this script.
#
# Prereqs:
#   - gcloud CLI installed and `gcloud auth login` already run.
#   - A GCP project already created with a billing account linked (project
#     creation + billing link are a one-time console step — see
#     https://console.cloud.google.com/projectcreate and
#     https://console.cloud.google.com/billing — free-tier usage stays $0,
#     but Cloud Run/Artifact Registry require billing to be enabled at all).
#   - Docker running locally (used to build the image, same as the app's
#     existing Dockerfile).
#
# Secrets live in deploy/gcp/.env.gcp (gitignored — see
# deploy/gcp/.env.gcp.example for the template), NOT in this file. Never
# paste real keys into this script: it's tracked by git, and every
# ${VAR:?message} below just prints `message` as an error if VAR is unset —
# it is not a default value.
#
# Usage: cp deploy/gcp/.env.gcp.example deploy/gcp/.env.gcp, fill it in,
# then: bash deploy/gcp/bootstrap.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root, so `docker build .` picks up the right Dockerfile

ENV_FILE="deploy/gcp/.env.gcp"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

# ── CONFIG ──────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:?set PROJECT_ID in deploy/gcp/.env.gcp}"
REGION="${REGION:-us-central1}"
REPO_NAME="clauseiq-backend"
SERVICE_NAME="clauseiq-backend"
DEPLOYER_SA_NAME="github-actions-deployer"

# Secrets / config for the running app — set these in deploy/gcp/.env.gcp.
GROQ_API_KEY="${GROQ_API_KEY:?set GROQ_API_KEY in deploy/gcp/.env.gcp}"
COHERE_API_KEY="${COHERE_API_KEY:?set COHERE_API_KEY in deploy/gcp/.env.gcp}"
HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN:?set HUGGINGFACEHUB_API_TOKEN in deploy/gcp/.env.gcp}"
JWT_SECRET_KEY="${JWT_SECRET_KEY:?set JWT_SECRET_KEY in deploy/gcp/.env.gcp}"
DATABASE_URL="${DATABASE_URL:?set DATABASE_URL (Neon connection string) in deploy/gcp/.env.gcp}"
QDRANT_URL="${QDRANT_URL:?set QDRANT_URL (Qdrant Cloud cluster URL) in deploy/gcp/.env.gcp}"
QDRANT_API_KEY="${QDRANT_API_KEY:?set QDRANT_API_KEY (Qdrant Cloud API key) in deploy/gcp/.env.gcp}"
APP_FRONTEND_URL="${APP_FRONTEND_URL:?set APP_FRONTEND_URL (your Vercel URL) in deploy/gcp/.env.gcp}"
CORS_ALLOWED_ORIGINS="${CORS_ALLOWED_ORIGINS:-$APP_FRONTEND_URL}"
RAG_SYSTEM_PROMPT_FILE="${RAG_SYSTEM_PROMPT_FILE:-system_prompt_v3.txt}"
# Optional — same as locally, leave blank and password-reset links get printed to the
# Cloud Run service's logs instead of emailed (see app/auth/email_utils.py's fallback).
SMTP_HOST="${SMTP_HOST:-}"
SMTP_PORT="${SMTP_PORT:-587}"
SMTP_USER="${SMTP_USER:-}"
SMTP_PASS="${SMTP_PASS:-}"
FROM_EMAIL="${FROM_EMAIL:-noreply@clauseiq.local}"
# Optional — LangSmith tracing. Leave LANGCHAIN_TRACING_V2 unset in
# .env.gcp to keep tracing off (see app/generator.py's silently-off fallback).
LANGCHAIN_TRACING_V2="${LANGCHAIN_TRACING_V2:-}"
LANGCHAIN_API_KEY="${LANGCHAIN_API_KEY:-}"
LANGCHAIN_PROJECT="${LANGCHAIN_PROJECT:-clauseiq-rag}"
# ─────────────────────────────────────────────────────────────────────────────

gcloud config set project "$PROJECT_ID" --quiet

echo "== Enabling required APIs =="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com \
  --quiet

echo "== Artifact Registry repo ($REPO_NAME) =="
if ! gcloud artifacts repositories describe "$REPO_NAME" --location "$REGION" &>/dev/null; then
  gcloud artifacts repositories create "$REPO_NAME" \
    --repository-format=docker \
    --location="$REGION" \
    --description="ClauseIQ backend images"
fi
IMAGE_REPO="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
IMAGE="$IMAGE_REPO/$SERVICE_NAME"

echo "== Building and pushing the backend image via local Docker =="
gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
docker build -t "$IMAGE:latest" .
docker push "$IMAGE:latest"

echo "== Secret Manager secrets =="
create_or_update_secret() {
  local name="$1" value="$2"
  if gcloud secrets describe "$name" &>/dev/null; then
    printf '%s' "$value" | gcloud secrets versions add "$name" --data-file=- --quiet >/dev/null
  else
    printf '%s' "$value" | gcloud secrets create "$name" --data-file=- --replication-policy=automatic --quiet >/dev/null
  fi
}
create_or_update_secret groq-api-key "$GROQ_API_KEY"
create_or_update_secret cohere-api-key "$COHERE_API_KEY"
create_or_update_secret hf-api-token "$HUGGINGFACEHUB_API_TOKEN"
create_or_update_secret jwt-secret-key "$JWT_SECRET_KEY"
create_or_update_secret database-url "$DATABASE_URL"
create_or_update_secret qdrant-api-key "$QDRANT_API_KEY"
create_or_update_secret smtp-user "$SMTP_USER"
create_or_update_secret smtp-pass "$SMTP_PASS"
create_or_update_secret langchain-api-key "$LANGCHAIN_API_KEY"

echo "== Granting the Cloud Run runtime service account access to those secrets =="
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="$PROJECT_NUMBER-compute@developer.gserviceaccount.com"
for secret in groq-api-key cohere-api-key hf-api-token jwt-secret-key database-url qdrant-api-key smtp-user smtp-pass langchain-api-key; do
  gcloud secrets add-iam-policy-binding "$secret" \
    --member="serviceAccount:$RUNTIME_SA" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null
done

echo "== Cloud Run service (first revision) =="
# Single instance cap: app/database.py's BM25 cache is per-process in-memory,
# so a second concurrent instance would serve a stale cache — don't raise
# max-instances without moving that cache to a shared store (e.g. Redis) first.
gcloud run deploy "$SERVICE_NAME" \
  --image="$IMAGE:latest" \
  --region="$REGION" \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --cpu=1 \
  --memory=2Gi \
  --min-instances=0 \
  --max-instances=1 \
  --set-env-vars="ENVIRONMENT=production,RAG_SYSTEM_PROMPT_FILE=$RAG_SYSTEM_PROMPT_FILE,QDRANT_URL=$QDRANT_URL,APP_FRONTEND_URL=$APP_FRONTEND_URL,CORS_ALLOWED_ORIGINS=$CORS_ALLOWED_ORIGINS,SMTP_HOST=$SMTP_HOST,SMTP_PORT=$SMTP_PORT,FROM_EMAIL=$FROM_EMAIL,LANGCHAIN_TRACING_V2=$LANGCHAIN_TRACING_V2,LANGCHAIN_PROJECT=$LANGCHAIN_PROJECT" \
  --set-secrets="GROQ_API_KEY=groq-api-key:latest,COHERE_API_KEY=cohere-api-key:latest,HUGGINGFACEHUB_API_TOKEN=hf-api-token:latest,JWT_SECRET_KEY=jwt-secret-key:latest,DATABASE_URL=database-url:latest,QDRANT_API_KEY=qdrant-api-key:latest,SMTP_USER=smtp-user:latest,SMTP_PASS=smtp-pass:latest,LANGCHAIN_API_KEY=langchain-api-key:latest" \
  --quiet

SERVICE_URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"

echo "== Setting APP_URL to the service's own HTTPS endpoint =="
gcloud run services update "$SERVICE_NAME" \
  --region="$REGION" \
  --update-env-vars="APP_URL=$SERVICE_URL" \
  --quiet

echo "== Service account for GitHub Actions CI/CD =="
DEPLOYER_SA_EMAIL="$DEPLOYER_SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$DEPLOYER_SA_EMAIL" &>/dev/null; then
  gcloud iam service-accounts create "$DEPLOYER_SA_NAME" \
    --display-name="GitHub Actions deployer for clauseiq-backend"
fi
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$DEPLOYER_SA_EMAIL" \
    --role="$role" \
    --quiet >/dev/null
done
KEY_FILE="deploy/gcp/github-actions-deployer-key.json"
gcloud iam service-accounts keys create "$KEY_FILE" \
  --iam-account="$DEPLOYER_SA_EMAIL"

cat <<EOF

Done.

Backend URL:          $SERVICE_URL
Artifact Registry:    $IMAGE_REPO
GCP project:          $PROJECT_ID ($REGION)

Next steps:
1. Frontend/vercel.json proxies /api/* on the Vercel domain to this backend —
   update its "destination" to $SERVICE_URL if it doesn't match already.
   Leave VITE_API_URL unset in Vercel (production builds default to the
   relative "/api" path); only set it if you need to bypass the proxy.
2. Add these GitHub Actions secrets for ongoing CI/CD deploys (see
   .github/workflows/deploy-backend.yml):
     GCP_SA_KEY       = contents of $KEY_FILE (the whole JSON file, as-is)
     GCP_PROJECT_ID   = $PROJECT_ID
     GCP_REGION       = $REGION
   Then DELETE $KEY_FILE locally — it's a live credential and must never be
   committed (it's gitignored, but don't leave it lying around either).
3. min-instances is 0 to stay in the always-free tier — the first request
   after idle will be slow (cold start: container boot + embedding model
   load). Set --min-instances 1 on the Cloud Run service if you'd rather
   keep it warm (this will incur cost — no longer free).
4. Do not raise --max-instances above 1: app/database.py's BM25 cache is
   per-process in-memory, so a second instance would serve a stale cache.
EOF
