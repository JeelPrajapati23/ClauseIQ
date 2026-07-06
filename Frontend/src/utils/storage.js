let idCounter = 0;
export const nextId = () => `id-${Date.now()}-${idCounter++}`;

export function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

export function relativeTime(ts) {
  const d = Date.now() - ts;
  if (d < 60000) return "just now";
  if (d < 3600000) return `${Math.floor(d / 60000)}m ago`;
  if (d < 86400000) return `${Math.floor(d / 3600000)}h ago`;
  if (d < 604800000) return `${Math.floor(d / 86400000)}d ago`;
  return new Date(ts).toLocaleDateString();
}

// Sessions are namespaced per user so chat history never leaks across
// accounts sharing the same browser (localStorage has no server-side scoping).
const storageKeyFor = (userKey) => `clauseiq_sessions_${userKey}`;

export function loadSessions(userKey) {
  try { return JSON.parse(localStorage.getItem(storageKeyFor(userKey)) || "[]"); }
  catch { return []; }
}
export function saveSessions(userKey, arr) {
  try { localStorage.setItem(storageKeyFor(userKey), JSON.stringify(arr)); } catch { /* quota exceeded */ }
}

// Tracks in-flight upload-indexing jobs (see /upload-jobs/{id}) so a page reload
// mid-processing resumes polling instead of losing track of the upload — status
// lives on the backend, this is just enough to know which jobs to ask about.
const jobsKeyFor = (userKey) => `clauseiq_pending_jobs_${userKey}`;

export function loadPendingJobs(userKey) {
  try { return JSON.parse(localStorage.getItem(jobsKeyFor(userKey)) || "[]"); }
  catch { return []; }
}
export function savePendingJobs(userKey, arr) {
  try { localStorage.setItem(jobsKeyFor(userKey), JSON.stringify(arr)); } catch { /* quota exceeded */ }
}
