// Backend base URL. Production builds default to the relative "/api" path,
// which Vercel rewrites (Frontend/vercel.json) proxies through to the Azure
// backend so the browser sees same-origin requests — required for the auth
// cookie to survive strict cross-site tracking protections (Brave Shields,
// Safari ITP, Firefox strict mode), which block SameSite=None cookies
// outright. `npm run dev` still defaults to hitting the local API directly.
// VITE_API_URL overrides either default if set at build time.
export const API_BASE_URL =
  import.meta.env.VITE_API_URL || (import.meta.env.PROD ? "/api" : "http://localhost:8000");

// Error response bodies aren't uniform: FastAPI's HTTPException gives
// {detail: "..."}, Pydantic validation errors give {detail: [{msg, ...}]},
// and slowapi's rate-limit handler gives {error: "..."} — never assume
// `detail` is a plain string.
export function extractErrorMessage(body, fallback) {
  if (!body) return fallback;
  const { detail, error } = body;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => (typeof d === "string" ? d : d.msg || JSON.stringify(d))).join(" ");
  }
  if (typeof error === "string" && error) return error;
  return fallback;
}

// The production backend runs on Render's free tier, which sleeps when idle and
// takes ~30-60s to wake. A request that arrives mid-wake fails outright or comes
// back as a 5xx gateway page (HTML, not JSON) from the proxy. fetchJson treats
// either as "still waking": it calls onWaking (so the UI can say so), waits for
// /health to answer, then retries the request once.
const WAKE_TIMEOUT_MS = 90_000;
const WAKE_POLL_MS = 3_000;

class BackendWakingError extends Error {}

async function waitForBackend() {
  const deadline = Date.now() + WAKE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      if ((await fetch(`${API_BASE_URL}/health`, { cache: "no-store" })).ok) return true;
    } catch { /* still waking */ }
    await new Promise((resolve) => setTimeout(resolve, WAKE_POLL_MS));
  }
  return false;
}

async function attemptJson(url, options) {
  let res;
  try {
    res = await fetch(url, options);
  } catch {
    throw new BackendWakingError();
  }
  if ([502, 503, 504].includes(res.status)) throw new BackendWakingError();
  let data = {};
  try {
    data = await res.json();
  } catch {
    if (!res.ok) throw new BackendWakingError();
  }
  return { res, data };
}

// Returns {res, data}. Throws only if the backend still hasn't answered after
// WAKE_TIMEOUT_MS — show BACKEND_UNREACHABLE_MESSAGE in that case.
export async function fetchJson(url, options, onWaking) {
  try {
    return await attemptJson(url, options);
  } catch (err) {
    if (!(err instanceof BackendWakingError)) throw err;
    onWaking?.();
    if (!(await waitForBackend())) throw err;
    return attemptJson(url, options);
  }
}

export const BACKEND_WAKING_MESSAGE =
  "The server is waking up after being idle — this can take up to a minute. Hang tight…";
export const BACKEND_UNREACHABLE_MESSAGE =
  "Couldn't reach the server. Please check your connection and try again in a minute.";
