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
