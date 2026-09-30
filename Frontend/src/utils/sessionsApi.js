import { API_BASE_URL, extractErrorMessage, fetchJson } from "./api.js";

// Server-side chat session/message persistence. Mirrors the fetch conventions
// already used inline in chat.jsx: credentials: "include" for the httponly
// auth cookie, 401 -> onSessionExpired, errors surfaced via extractErrorMessage.

async function handle(res, onSessionExpired, fallbackError) {
  if (res.status === 401) {
    onSessionExpired?.();
    throw new Error("Session expired.");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(extractErrorMessage(body, fallbackError));
  }
  return res.json();
}

// The first call after page load — goes through fetchJson so a sleeping backend
// is waited for instead of failing (which used to leave an empty sidebar).
export async function fetchSessions(onSessionExpired) {
  const { res, data } = await fetchJson(`${API_BASE_URL}/sessions`, { credentials: "include" });
  if (res.status === 401) {
    onSessionExpired?.();
    throw new Error("Session expired.");
  }
  if (!res.ok) throw new Error(extractErrorMessage(data, "Failed to load chat history."));
  return data;
}

export function createSession(payload, onSessionExpired) {
  return fetch(`${API_BASE_URL}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify(payload),
  }).then((res) => handle(res, onSessionExpired, "Failed to create chat session."));
}

export function fetchSession(id, onSessionExpired) {
  return fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(id)}`, { credentials: "include" })
    .then((res) => handle(res, onSessionExpired, "Failed to load chat session."));
}

export function renameSession(id, name, onSessionExpired) {
  return fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ name }),
  }).then((res) => handle(res, onSessionExpired, "Failed to rename chat session."));
}

export function updateSessionFiles(id, uploadedFiles, onSessionExpired) {
  return fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ uploaded_files: uploadedFiles }),
  }).then((res) => handle(res, onSessionExpired, "Failed to update chat session."));
}

export function deleteSession(id, onSessionExpired) {
  return fetch(`${API_BASE_URL}/sessions/${encodeURIComponent(id)}`, {
    method: "DELETE",
    credentials: "include",
  }).then((res) => handle(res, onSessionExpired, "Failed to delete chat session."));
}
