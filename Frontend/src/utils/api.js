// Backend base URL. Set VITE_API_URL at build time for staging/production
// (e.g. VITE_API_URL=https://api.clauseiq.example); defaults to the local
// dev API for `npm run dev`.
export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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
