import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { nextId, formatBytes, relativeTime, loadPendingJobs as loadPendingJobsForUser, savePendingJobs as savePendingJobsForUser } from "./utils/storage.js";
import { API_BASE_URL, extractErrorMessage } from "./utils/api.js";
import { fetchSessions, createSession, fetchSession, renameSession as renameSessionApi, updateSessionFiles, deleteSession as deleteSessionApi } from "./utils/sessionsApi.js";
import { IcPlus, IcSearch, IcUpload, IcArrow, IcDoc, IcClose, IcMenu, IcPage, IcCheck, IcAlert, IcChat, IcCopy, IcCopied, IcDots, IcPencil, IcTrash } from "./components/icons.jsx";
import { FileChip, TypingDots } from "./components/FileChip.jsx";
import AdminPanel from "./components/AdminPanel.jsx";
import ResetPasswordModal from "./components/ResetPasswordModal.jsx";
import clauseiqMark from "./assets/clauseiq-mark.svg";

// ── Server <-> frontend session/message shape mapping ────────────────────────
// The backend returns sessions/messages in snake_case with raw retrieval
// metadata (see app/main.py _persist_chat_turn); the UI works with the same
// shapes handleSend/handleCompare build up live during streaming.
const mapSession = (s) => ({
  id: s.id,
  name: s.name || "",
  uploadedFiles: s.uploaded_files || [],
  createdAt: new Date(s.created_at).getTime(),
});

const sourcesToAskCitations = (sources) => (sources || []).map((s) => ({
  id: nextId(),
  source: s.file,
  pages: s.pages || (s.page != null ? [s.page] : []),
  pageRange: s.page_range || (s.page != null ? `p. ${s.page}` : ""),
  section: s.section || "",
  page: s.page ?? s.pages?.[0] ?? null,
  preview: s.content_preview || null,
}));

// Claim-level citations (app/citations.py): one verified claim, the verbatim quote
// supporting it, and the exact page it was found on.
const toClaimCitations = (items) => (items || []).map((c) => ({
  id: nextId(),
  claim: c.claim,
  quote: c.quote,
  source: c.file,
  page: c.page ?? null,
  pageRange: c.page != null ? `p. ${c.page}` : "",
  section: c.section || "",
  exact: !!c.exact,
}));

const sourcesToCompareCitations = (sources) => (sources || []).map((s) => ({
  id: nextId(), source: s.file, page: s.page, preview: s.content_preview || null,
}));

const mapMessage = (m) => {
  if (m.role !== "assistant") {
    return { id: m.id, role: m.role, content: m.content };
  }
  if (m.extra?.isComparison) {
    const coverage = m.verification?.coverage || {};
    return {
      id: m.id, role: "assistant", content: m.content,
      citations: sourcesToCompareCitations(m.citations),
      weakEvidence: Object.entries(coverage).filter(([, ok]) => !ok).map(([doc]) => doc),
      isComparison: true,
      comparedDocs: m.extra.comparedDocs || [],
    };
  }
  return {
    id: m.id, role: "assistant", content: m.content,
    citations: sourcesToAskCitations(m.citations),
    claimCitations: toClaimCitations(m.extra?.claim_citations),
    verification: m.verification ? {
      verdict: m.verification.verdict,
      score: m.verification.score,
      totalClaims: m.verification.total_claims || 0,
      unverifiedClaims: m.verification.unverified_claims || [],
    } : null,
  };
};

export default function Chat({ authUser, onLogout, onSessionExpired }) {
  // Pending-upload-job tracking stays client-side/local-storage (see utils/storage.js)
  // — it's just enough to resume polling after a refresh, not chat history.
  const userKey = authUser?.id ?? authUser?.email ?? "anon";
  const loadPendingJobs = () => loadPendingJobsForUser(userKey);
  const savePendingJobs = (arr) => savePendingJobsForUser(userKey, arr);

  // ── Sessions (chat history) — persisted server-side, see utils/sessionsApi.js ──
  const [sessions, setSessions] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [searchQuery, setSearchQuery] = useState("");

  // ── Chat working state ────────────────────────────────────────────────────
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [attached, setAttached] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [view, setView] = useState("empty");
  const [procState, setProcState] = useState({ name: "", progress: 0 });
  const [isNarrow, setIsNarrow] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [showAdmin, setShowAdmin] = useState(false);
  const [avatarMenuOpen, setAvatarMenuOpen] = useState(false);
  const [showResetPassword, setShowResetPassword] = useState(false);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editingName, setEditingName] = useState("");
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [sessionMenu, setSessionMenu] = useState(null); // { id, x, y }
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [selectedDocs, setSelectedDocs] = useState(new Set());
  const [isComparing, setIsComparing] = useState(false);
  const [docSelectorOpen, setDocSelectorOpen] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // ── Refs ──────────────────────────────────────────────────────────────────
  const fileInputRef      = useRef(null);
  const dropzoneInputRef  = useRef(null);
  const textareaRef       = useRef(null);
  const scrollRef         = useRef(null);
  const procTimerRef      = useRef(null);
  const sessionsRef       = useRef(sessions);
  const messagesRef       = useRef(messages);
  const renameInputRef    = useRef(null);

  useEffect(() => { sessionsRef.current = sessions; }, [sessions]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);
  useEffect(() => {
    if (editingSessionId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [editingSessionId]);

  // ── Load chat history from the server on mount ───────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await fetchSessions(onSessionExpired);
        if (cancelled) return;
        const mapped = list.map(mapSession);
        setSessions(mapped);
        if (mapped.length > 0) {
          const full = await fetchSession(mapped[0].id, onSessionExpired);
          if (cancelled) return;
          const msgs = (full.messages || []).map(mapMessage);
          setActiveSessionId(mapped[0].id);
          setMessages(msgs);
          setView(msgs.length > 0 || mapped[0].uploadedFiles.length > 0 ? "chat" : "empty");
        }
      } catch {
        // Failed to reach the backend for history — leave the empty-state view.
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── matchMedia ────────────────────────────────────────────────────────────
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 1080px)");
    const update = () => setIsNarrow(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  // ── Textarea auto-height ──────────────────────────────────────────────────
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 140) + "px";
  }, [input]);

  // ── Progress bar simulation ───────────────────────────────────────────────
  useEffect(() => {
    if (view !== "processing") return;
    const interval = setInterval(() => {
      setProcState((p) => {
        if (p.progress >= 90) { clearInterval(interval); return p; }
        return { ...p, progress: Math.min(90, p.progress + Math.round(Math.random() * 14 + 7)) };
      });
    }, 280);
    procTimerRef.current = interval;
    return () => clearInterval(interval);
  }, [view]);

  // ── Settle: upload done → complete bar → chat (or error state) ───────────
  useEffect(() => {
    if (view !== "processing") return;
    const allSettled = attached.length > 0 && attached.every((f) => f.status === "ready" || f.status === "error");
    if (!allSettled) return;
    clearInterval(procTimerRef.current);
    const hasSuccess = attached.some((f) => f.status === "ready");
    const t = setTimeout(() => {
      if (hasSuccess) {
        setProcState((p) => ({ ...p, progress: 100 }));
        setTimeout(() => { setView("chat"); setAttached([]); }, 600);
      } else {
        setProcState((p) => ({ ...p, error: true }));
      }
    }, 0);
    return () => clearTimeout(t);
  }, [attached, view]);

  // ── Session management ────────────────────────────────────────────────────
  // Sessions/messages are the server's source of truth (see utils/sessionsApi.js)
  // — switching a session fetches its messages fresh rather than reading a
  // locally-cached copy, since the backend already persisted the last exchange
  // the moment /ask or /compare completed.
  const loadSessionInto = async (id) => {
    const full = await fetchSession(id, onSessionExpired);
    const msgs = (full.messages || []).map(mapMessage);
    setActiveSessionId(id);
    setMessages(msgs);
    setView(msgs.length > 0 || (full.uploaded_files || []).length > 0 ? "chat" : "empty");
  };

  const switchSession = async (id) => {
    if (id === activeSessionId) { setSidebarOpen(false); return; }
    setSidebarOpen(false);
    setSelectedCitation(null);
    setInput("");
    setAttached([]);
    setSelectedDocs(new Set());
    setCompareMode(false);
    try {
      await loadSessionInto(id);
    } catch (err) {
      console.error("Failed to load chat session:", err.message);
    }
  };

  const newChat = () => {
    // Reuse current session if it is already empty (avoid clutter). No server
    // session is created until the first message/upload actually lands (see
    // handleSend/handleCompare/handleFiles), so an empty "new chat" is free.
    const current = sessions.find((s) => s.id === activeSessionId);
    if (!current || (messages.length === 0 && !current.uploadedFiles?.length)) {
      setView("empty"); setSidebarOpen(false); setInput(""); setAttached([]);
      setSelectedDocs(new Set()); setCompareMode(false);
      return;
    }
    setActiveSessionId(null);
    setMessages([]);
    setSelectedCitation(null);
    setInput("");
    setAttached([]);
    setSidebarOpen(false);
    setView("empty");
  };

  const deleteSession = async (id) => {
    try {
      await deleteSessionApi(id, onSessionExpired);
    } catch (err) {
      console.error("Failed to delete chat session:", err.message);
      return;
    }
    const remaining = sessions.filter((s) => s.id !== id);
    setSessions(remaining);
    if (id === activeSessionId) {
      setSelectedCitation(null);
      if (remaining.length > 0) {
        try {
          await loadSessionInto(remaining[0].id);
          return;
        } catch (err) {
          console.error("Failed to load chat session:", err.message);
        }
      }
      setActiveSessionId(null);
      setMessages([]);
      setView("empty");
    }
  };

  const requestDelete = (id) => {
    setDeleteConfirmId(id);
    setSessionMenu(null);
  };

  const startRename = (id, currentName) => {
    setDeleteConfirmId(null);
    setSessionMenu(null);
    setEditingSessionId(id);
    setEditingName(currentName || "");
  };

  const openSessionMenu = (id, e) => {
    e.stopPropagation();
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.min(rect.right - 144, window.innerWidth - 154);
    setSessionMenu({ id, x, y: rect.bottom + 5 });
  };

  const saveRename = async (id) => {
    const trimmed = editingName.trim();
    setEditingSessionId(null);
    setEditingName("");
    const session = sessions.find((s) => s.id === id);
    const name = trimmed || session?.name || "Untitled chat";
    setSessions((prev) => prev.map((s) => (s.id === id ? { ...s, name } : s)));
    try {
      await renameSessionApi(id, name, onSessionExpired);
    } catch (err) {
      console.error("Failed to rename chat session:", err.message);
    }
  };

  // ── Upload ────────────────────────────────────────────────────────────────
  // Upload returns almost immediately (202 + job id) — parsing/chunking/embedding
  // happens in a background task on the server, so this never risks Vercel's
  // ~120s proxy timeout no matter how long embedding actually takes. We poll
  // /upload-jobs/{id} until the backend reports "ready" or "failed".
  const POLL_INTERVAL_MS = 3000;

  const addPendingJob = (job) => savePendingJobs([...loadPendingJobs(), job]);
  const removePendingJob = (jobId) => savePendingJobs(loadPendingJobs().filter((j) => j.jobId !== jobId));

  const pollJobStatus = (jobId, fileId, targetSessionId, filename) => {
    const tick = async () => {
      let res;
      try {
        res = await fetch(`${API_BASE_URL}/upload-jobs/${jobId}`, { credentials: "include" });
      } catch {
        setTimeout(tick, POLL_INTERVAL_MS); // transient network blip — keep trying
        return;
      }
      if (res.status === 401) { onSessionExpired?.(); removePendingJob(jobId); return; }
      if (!res.ok) {
        removePendingJob(jobId);
        if (fileId) setAttached((prev) => prev.map((f) => (f.id === fileId ? { ...f, status: "error", errorMessage: "Lost track of this upload. Please try again." } : f)));
        return;
      }
      const job = await res.json();
      if (job.status === "processing") {
        setTimeout(tick, POLL_INTERVAL_MS);
        return;
      }
      removePendingJob(jobId);
      if (job.status === "ready") {
        if (fileId) setAttached((prev) => prev.map((f) => (f.id === fileId ? { ...f, status: "ready" } : f)));
        // Re-fetch the session's authoritative file list before appending, rather
        // than trusting local `sessions` state, so a concurrent/refresh-resumed
        // poll can't clobber files another tab already recorded server-side.
        try {
          const current = await fetchSession(targetSessionId, onSessionExpired);
          const nextFiles = [...(current.uploaded_files || []), filename];
          const updated = await updateSessionFiles(targetSessionId, nextFiles, onSessionExpired);
          const mapped = mapSession(updated);
          setSessions((prev) => prev.some((s) => s.id === targetSessionId)
            ? prev.map((s) => (s.id === targetSessionId ? mapped : s))
            : [mapped, ...prev]);
        } catch (err) {
          console.error("Failed to save uploaded document to session:", err.message);
        }
      } else {
        const message = job.error || "Failed to process the uploaded document.";
        if (fileId) {
          setAttached((prev) => prev.map((f) => (f.id === fileId ? { ...f, status: "error", errorMessage: message } : f)));
        } else {
          setUploadError(`"${filename}" failed to process: ${message}`);
          setTimeout(() => setUploadError(null), 6000);
        }
      }
    };
    tick();
  };

  // Resume polling for any upload that was still processing when the page was
  // last closed/refreshed — status lives on the backend, so nothing is lost.
  useEffect(() => {
    loadPendingJobs().forEach((job) => pollJobStatus(job.jobId, null, job.sessionId, job.filename));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const uploadFile = async (fileObj, targetSessionId) => {
    setAttached((prev) => prev.map((f) => (f.id === fileObj.id ? { ...f, status: "uploading" } : f)));
    try {
      const form = new FormData();
      form.append("file", fileObj.raw);
      const res = await fetch(`${API_BASE_URL}/upload-and-index`, { method: "POST", body: form, credentials: "include" });
      if (!res.ok) {
        if (res.status === 401) { onSessionExpired?.(); throw new Error("Session expired."); }
        const errBody = await res.json().catch(() => ({}));
        throw new Error(extractErrorMessage(errBody, "Upload failed"));
      }
      const { job_id, filename } = await res.json();
      setAttached((prev) => prev.map((f) => (f.id === fileObj.id ? { ...f, status: "processing" } : f)));
      addPendingJob({ jobId: job_id, filename, sessionId: targetSessionId });
      pollJobStatus(job_id, fileObj.id, targetSessionId, filename);
    } catch (err) {
      setAttached((prev) => prev.map((f) => (f.id === fileObj.id ? { ...f, status: "error", errorMessage: err.message } : f)));
    }
  };

  const MAX_FILE_BYTES = 20 * 1024 * 1024; // 20 MB

  const handleFiles = async (fileList) => {
    const all = Array.from(fileList);
    const oversized = all.filter((f) => f.size > MAX_FILE_BYTES);
    if (oversized.length > 0) {
      setUploadError(
        `"${oversized[0].name}" is too large (${formatBytes(oversized[0].size)}). Max allowed: 20 MB.`
      );
      setTimeout(() => setUploadError(null), 5000);
    }
    const allowed = all.filter((f) => f.size <= MAX_FILE_BYTES);
    if (allowed.length === 0) return;
    const newFiles = allowed.map((f) => ({
      id: nextId(), name: f.name, size: formatBytes(f.size), raw: f, status: "pending",
    }));
    setAttached((prev) => [...prev, ...newFiles]);
    if (newFiles.length === 0) return;

    setProcState({ name: newFiles[0].name, progress: 0 });
    setView("processing");

    // Sessions are created lazily on the backend — this is the first thing
    // (upload or message) to happen in a fresh "new chat", so create it now.
    let targetSessionId = activeSessionId;
    if (!targetSessionId) {
      try {
        const created = await createSession({ uploaded_files: [] }, onSessionExpired);
        targetSessionId = created.id;
        setSessions((prev) => [mapSession(created), ...prev]);
        setActiveSessionId(created.id);
        setMessages([]);
      } catch (err) {
        console.error("Failed to create chat session:", err.message);
        const failedIds = new Set(newFiles.map((f) => f.id));
        setAttached((prev) => prev.map((f) =>
          failedIds.has(f.id) ? { ...f, status: "error", errorMessage: "Could not start a new chat session." } : f
        ));
        return;
      }
    }

    newFiles.forEach((f) => uploadFile(f, targetSessionId));
  };

  const onFileInputChange     = (e) => { if (e.target.files?.length) handleFiles(e.target.files); e.target.value = ""; };
  const onDropzoneInputChange = (e) => { if (e.target.files?.length) handleFiles(e.target.files); e.target.value = ""; };
  const removeAttached = (id) => setAttached((prev) => prev.filter((f) => f.id !== id));
  const retryUpload    = (id) => { const f = attached.find((f) => f.id === id); if (f) uploadFile(f, activeSessionId); };

  const deleteDocument = async (filename) => {
    try {
      const res = await fetch(
        `${API_BASE_URL}/documents/${encodeURIComponent(filename)}`,
        { method: "DELETE", credentials: "include" }
      );
      if (!res.ok) {
        if (res.status === 401) { onSessionExpired?.(); throw new Error("Session expired."); }
        const body = await res.json().catch(() => ({}));
        throw new Error(extractErrorMessage(body, "Delete failed"));
      }
      const remaining = (activeSess?.uploadedFiles || []).filter((f) => f !== filename);
      await updateSessionFiles(activeSessionId, remaining, onSessionExpired);
      setSessions((prev) => prev.map((s) =>
        s.id === activeSessionId ? { ...s, uploadedFiles: remaining } : s
      ));
      // If last doc removed and no conversation yet, go back to dropzone
      if (remaining.length === 0 && messagesRef.current.length === 0) {
        setView("empty");
      }
    } catch (err) {
      console.error("Delete document failed:", err.message);
    }
  };

  // ── Send ──────────────────────────────────────────────────────────────────
  const handleSend = async () => {
    if (sendDisabled) return;
    const text = input.trim();

    // Capture session id + history snapshot at send time (async-safe)
    const sessionIdAtSend = activeSessionId;
    const userMsg    = { id: nextId(), role: "user", content: text, files: attached };
    const assistantId = nextId();

    const history = messages
      .filter((m) => m.content)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }));

    // Track messages locally so we can save the exact exchange at the end,
    // independent of whether the user switches sessions mid-stream.
    let localMessages = [
      ...messages,
      userMsg,
      { id: assistantId, role: "assistant", content: "", citations: [], verification: null },
    ];

    setInput("");
    setAttached([]);
    setIsLoading(true);
    setMessages([...localMessages]);

    try {
      const res = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({
          question: text, history, document_filter: activeSess?.uploadedFiles || [],
          session_id: sessionIdAtSend,
        }),
      });

      if (!res.ok) {
        if (res.status === 401) { onSessionExpired?.(); throw new Error("Session expired."); }
        const errBody = await res.json().catch(() => ({}));
        throw new Error(extractErrorMessage(errBody, "Request failed"));
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === "token") {
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + data.content } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "done") {
            const citations = sourcesToAskCitations(data.sources);
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, citations } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "verification") {
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? {
                ...m,
                verification: {
                  verdict: data.verdict,
                  score: data.score,
                  totalClaims: data.total_claims || 0,
                  unverifiedClaims: data.unverified_claims || [],
                },
              } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "citations") {
            const claimCitations = toClaimCitations(data.citations);
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, claimCitations } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "error") {
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, content: data.detail || "Something went wrong.", isError: true } : m
            );
            setMessages([...localMessages]);
          }
        }
      }
    } catch {
      localMessages = localMessages.map((m) =>
        m.id === assistantId
          ? { ...m, content: "Something went wrong reaching the backend. Check that the API is running and try again.", isError: true }
          : m
      );
      setMessages([...localMessages]);
    } finally {
      setIsLoading(false);
      // The backend already persisted this exchange as part of /ask — just
      // name the session locally (and on the server) if it's still unnamed.
      const session = sessions.find((s) => s.id === sessionIdAtSend);
      if (session && !session.name) {
        const firstUser = localMessages.find((m) => m.role === "user");
        const name = firstUser?.content?.slice(0, 48).trim() || "New chat";
        setSessions((prev) => prev.map((s) => (s.id === sessionIdAtSend ? { ...s, name } : s)));
        renameSessionApi(sessionIdAtSend, name, onSessionExpired).catch((err) =>
          console.error("Failed to name chat session:", err.message)
        );
      }
    }
  };

  const handleLogout = async () => {
    await fetch(`${API_BASE_URL}/auth/logout`, { method: "POST", credentials: "include" });
    onLogout();
  };

  const copyText = (id, text) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const toggleDoc = (name) => {
    setSelectedDocs(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  };

  const handleCompare = async () => {
    if (selectedDocs.size < 2 || !input.trim() || isComparing) return;
    const query = input.trim();
    const docIds = Array.from(selectedDocs);
    const sessionIdAtSend = activeSessionId;
    const userMsg = { id: nextId(), role: "user", content: query };
    const assistantId = nextId();

    let localMessages = [
      ...messages,
      userMsg,
      { id: assistantId, role: "assistant", content: "", citations: [], weakEvidence: [], isComparison: true, comparedDocs: docIds },
    ];

    setInput("");
    setIsComparing(true);
    setMessages([...localMessages]);

    try {
      const res = await fetch(`${API_BASE_URL}/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ doc_ids: docIds, query, session_id: sessionIdAtSend }),
      });

      if (!res.ok) {
        if (res.status === 401) { onSessionExpired?.(); throw new Error("Session expired."); }
        const errBody = await res.json().catch(() => ({}));
        throw new Error(extractErrorMessage(errBody, "Comparison failed."));
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === "verification") {
            const weakEvidence = Object.entries(data.coverage)
              .filter(([, ok]) => !ok)
              .map(([doc]) => doc);
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, weakEvidence } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "token") {
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, content: m.content + data.content } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "done") {
            const citations = sourcesToCompareCitations(data.sources);
            localMessages = localMessages.map((m) =>
              m.id === assistantId ? { ...m, citations } : m
            );
            setMessages([...localMessages]);
          } else if (data.type === "error") {
            localMessages = localMessages.map((m) =>
              m.id === assistantId
                ? { ...m, content: data.detail || "Comparison failed.", isError: true }
                : m
            );
            setMessages([...localMessages]);
          }
        }
      }
    } catch (err) {
      localMessages = localMessages.map((m) =>
        m.id === assistantId
          ? { ...m, content: err.message || "Comparison failed. Check the API is running.", isError: true }
          : m
      );
      setMessages([...localMessages]);
    } finally {
      setIsComparing(false);
      // The backend already persisted this exchange as part of /compare — just
      // name the session locally (and on the server) if it's still unnamed.
      const session = sessions.find((s) => s.id === sessionIdAtSend);
      if (session && !session.name) {
        const firstUser = localMessages.find((m) => m.role === "user");
        const name = firstUser?.content?.slice(0, 48).trim() || "Comparison";
        setSessions((prev) => prev.map((s) => (s.id === sessionIdAtSend ? { ...s, name } : s)));
        renameSessionApi(sessionIdAtSend, name, onSessionExpired).catch((err) =>
          console.error("Failed to name chat session:", err.message)
        );
      }
    }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (compareMode) handleCompare(); else handleSend();
    }
  };

  // ── Active session metadata ───────────────────────────────────────────────
  const activeSess = sessions.find((s) => s.id === activeSessionId);

  const sendDisabled =
    isLoading ||
    !input.trim() ||
    !(activeSess?.uploadedFiles?.length) ||
    attached.some((f) => f.status === "uploading" || f.status === "pending" || f.status === "error");

  const compareDisabled = isComparing || !input.trim() || selectedDocs.size < 2;
  const canCompare = (activeSess?.uploadedFiles?.length || 0) >= 2;

  // ── Theme tokens ──────────────────────────────────────────────────────────
  const T = {
    bg: "#0a0b0d", panel: "#101216", panel2: "#15181d", field: "#0c0e11",
    ink: "#eef0f2", ink2: "#b6bbc2", muted: "#7c828c",
    border: "#22262d", border2: "#2c313a",
    accent: "#c6f24a", accentInk: "#0c1003", accentGlow: "rgba(198,242,74,0.16)",
    sans: "'Space Grotesk', sans-serif",
    mono: "'IBM Plex Mono', monospace",
  };

  // ── Derived sidebar data ──────────────────────────────────────────────────
  const filteredSessions = sessions.filter((s) =>
    !searchQuery ||
    (s.name || "").toLowerCase().includes(searchQuery.toLowerCase()) ||
    (s.uploadedFiles?.[0] || "").toLowerCase().includes(searchQuery.toLowerCase())
  );


  // ── Source panel inner ────────────────────────────────────────────────────
  const sourcePanelInner = selectedCitation ? (
    <>
      <div style={{ height: 56, flexShrink: 0, borderBottom: `1px solid ${T.border}`,
        display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 18px 0 22px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 9, minWidth: 0 }}>
          <span style={{ color: T.accent, flexShrink: 0 }}><IcPage /></span>
          <span style={{ fontFamily: T.mono, fontSize: 11, color: T.muted, letterSpacing: "0.06em",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {selectedCitation.source}
          </span>
        </div>
        <button onClick={() => setSelectedCitation(null)} className="bf-rail"
          style={{ background: "none", border: "none", color: T.muted, cursor: "pointer",
            padding: 6, borderRadius: 7, display: "grid", placeItems: "center", transition: "all .14s", flexShrink: 0 }}>
          <IcClose />
        </button>
      </div>
      <div style={{ flex: 1, overflowY: "auto", overflowX: "hidden", padding: "20px 22px 28px" }}>
        {/* Claim-level citation: the claim plus the verbatim quote supporting it */}
        {selectedCitation.quote ? (
          <>
            {selectedCitation.section && (
              <div style={{ fontFamily: T.mono, fontSize: 11, color: T.accent,
                letterSpacing: "0.06em", marginBottom: 12, wordBreak: "break-word" }}>
                {selectedCitation.section}
              </div>
            )}
            {selectedCitation.pageRange && (
              <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 14 }}>
                <span style={{ color: T.accent, flexShrink: 0 }}><IcPage s={12} /></span>
                <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 500,
                  color: T.accent, letterSpacing: "0.06em" }}>{selectedCitation.pageRange.toUpperCase()}</span>
              </div>
            )}
            <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted, letterSpacing: "0.06em", marginBottom: 6 }}>
              CLAIM
            </div>
            <p style={{ fontSize: 13, lineHeight: 1.7, color: T.ink, margin: "0 0 16px",
              wordBreak: "break-word", overflowWrap: "break-word" }}>
              {selectedCitation.claim}
            </p>
            <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted, letterSpacing: "0.06em", marginBottom: 6 }}>
              SOURCE TEXT
            </div>
            <blockquote style={{ margin: 0, padding: "2px 0 2px 12px", borderLeft: `2px solid ${T.accent}`,
              fontSize: 13, lineHeight: 1.85, color: T.ink, fontStyle: "italic",
              wordBreak: "break-word", overflowWrap: "break-word" }}>
              “{selectedCitation.quote}”
            </blockquote>
          </>
        ) : selectedCitation.id ? (() => {
          const { section, pageRange, page, preview } = selectedCitation;
          const rangeLabel = pageRange || (page != null ? `p. ${page}` : null);
          return (
            <>
              {section && (
                <div style={{ fontFamily: T.mono, fontSize: 11, color: T.accent,
                  letterSpacing: "0.06em", marginBottom: 12, wordBreak: "break-word" }}>
                  {section}
                </div>
              )}
              {rangeLabel && (
                <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 14 }}>
                  <span style={{ color: T.accent, flexShrink: 0 }}><IcPage s={12} /></span>
                  <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 500,
                    color: T.accent, letterSpacing: "0.06em" }}>{rangeLabel.toUpperCase()}</span>
                </div>
              )}
              {preview ? (
                <p style={{ fontSize: 13, lineHeight: 1.85, color: T.ink, margin: 0,
                  wordBreak: "break-word", overflowWrap: "break-word" }}>
                  {preview.replace(/\.\.\.$/, "").slice(0, 350)}
                </p>
              ) : (
                <p style={{ fontSize: 12.5, color: T.muted, lineHeight: 1.7, margin: 0 }}>
                  No preview available.
                </p>
              )}
            </>
          );
        })() : (
          /* Legacy format: group object with pages array of {page, preview} */
          (selectedCitation.pages || []).map((p, i) => (
            <div key={p.page ?? i} style={{ marginBottom: i < (selectedCitation.pages?.length ?? 1) - 1 ? 20 : 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 8 }}>
                <span style={{ color: T.accent, flexShrink: 0 }}><IcPage s={12} /></span>
                <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 500, color: T.accent,
                  letterSpacing: "0.06em" }}>PAGE {p.page}</span>
              </div>
              {p.preview ? (
                <p style={{ fontSize: 13, lineHeight: 1.85, color: T.ink, margin: 0,
                  wordBreak: "break-word", overflowWrap: "break-word" }}>
                  {p.preview.slice(0, 300)}
                </p>
              ) : (
                <p style={{ fontSize: 12.5, color: T.muted, lineHeight: 1.7, margin: 0 }}>
                  No preview available.
                </p>
              )}
              {i < (selectedCitation.pages?.length ?? 1) - 1 && (
                <div style={{ marginTop: 18, borderBottom: `1px dashed ${T.border2}` }} />
              )}
            </div>
          ))
        )}
        <div style={{ marginTop: 22, paddingTop: 16, borderTop: `1px dashed ${T.border2}`,
          display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: T.accent }}><IcCheck /></span>
          <span style={{ fontFamily: T.mono, fontSize: 11, color: T.muted }}>
            {selectedCitation.quote
              ? (selectedCitation.exact ? "Quoted verbatim from the document" : "Closely matches the document text")
              : "Cited from retrieved context"}
          </span>
        </div>
      </div>
    </>
  ) : null;

  // ── Chat history sidebar content ──────────────────────────────────────────
  const sidebarContent = (
    <>
      {/* New chat button */}
      <button onClick={newChat} className="bf-new"
        style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8, width: "100%",
          padding: "10px", borderRadius: 10, border: `1px solid ${T.border2}`, background: T.panel,
          color: T.ink, fontFamily: T.sans, fontWeight: 600, fontSize: 13.5, cursor: "pointer",
          marginBottom: 12, transition: "all .14s", whiteSpace: "nowrap", flexShrink: 0 }}>
        <IcPlus s={16} /> New chat
      </button>

      {/* Search */}
      <div style={{ position: "relative", marginBottom: 14, flexShrink: 0 }}>
        <span style={{ position: "absolute", left: 9, top: "50%", transform: "translateY(-50%)",
          color: T.muted, display: "flex", pointerEvents: "none" }}>
          <IcSearch s={13} />
        </span>
        <input
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search chats…"
          style={{ width: "100%", boxSizing: "border-box", background: T.panel2,
            border: `1px solid ${T.border}`, borderRadius: 8, padding: "7px 10px 7px 28px",
            color: T.ink, fontFamily: T.mono, fontSize: 11.5, outline: "none",
            transition: "border-color .14s" }}
          onFocus={(e) => (e.target.style.borderColor = T.accent)}
          onBlur={(e) => (e.target.style.borderColor = T.border)}
        />
      </div>

      {/* Label row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 4px 10px", flexShrink: 0 }}>
        <span style={{ fontFamily: T.mono, fontSize: 11, letterSpacing: "0.12em",
          textTransform: "uppercase", color: T.muted }}>Chats</span>
        <span style={{ fontFamily: T.mono, fontSize: 11, color: T.muted }}>
          {String(sessions.length).padStart(2, "0")}
        </span>
      </div>

      {/* Sessions list */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 3 }}>
        {filteredSessions.length === 0 ? (
          <p style={{ fontFamily: T.mono, fontSize: 11, color: T.muted, lineHeight: 1.7, padding: "0 4px" }}>
            {searchQuery ? "No chats match your search." : "Upload a document to start your first chat."}
          </p>
        ) : (
          filteredSessions.map((s) => {
            const isActive = s.id === activeSessionId;
            const isEditing = editingSessionId === s.id;
            return (
              <div key={s.id}
                onClick={isEditing ? undefined : () => switchSession(s.id)}
                className={`bf-thread${isActive ? " bf-thread-active" : ""}`}
                style={{ borderRadius: 9, padding: "9px 10px", position: "relative",
                  cursor: isEditing ? "default" : "pointer",
                  background: isActive ? T.panel2 : "transparent",
                  border: `1px solid ${isActive ? T.border2 : "transparent"}`,
                  transition: "all .12s" }}>
                <>
                    {/* Chat icon + name or rename input */}
                    <div style={{ display: "flex", alignItems: "center", gap: 7, paddingRight: isEditing ? 0 : 28 }}>
                      {!isEditing && (
                        <span style={{ color: isActive ? T.accent : T.muted, flexShrink: 0, display: "flex" }}>
                          <IcChat s={13} />
                        </span>
                      )}
                      {isEditing ? (
                        <input
                          ref={renameInputRef}
                          value={editingName}
                          onChange={(e) => setEditingName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === "Enter") { e.preventDefault(); saveRename(s.id); }
                            if (e.key === "Escape") { setEditingSessionId(null); }
                          }}
                          onBlur={() => saveRename(s.id)}
                          onClick={(e) => e.stopPropagation()}
                          style={{ flex: 1, background: T.field, border: `1px solid ${T.accent}`,
                            borderRadius: 6, padding: "3px 8px", color: T.ink,
                            fontFamily: T.sans, fontSize: 12.5, outline: "none" }}
                        />
                      ) : (
                        <span style={{ fontSize: 12.5, fontWeight: isActive ? 600 : 500,
                          color: isActive ? T.ink : T.ink2,
                          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {s.name || "Untitled chat"}
                        </span>
                      )}
                    </div>

                    {/* Doc name + time */}
                    {!isEditing && (
                      <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted,
                        marginTop: 4, paddingLeft: 20, display: "flex", gap: 5, alignItems: "center" }}>
                        {s.uploadedFiles?.[0] && (
                          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                            maxWidth: 110, flexShrink: 1 }} title={s.uploadedFiles[0]}>{s.uploadedFiles[0]}</span>
                        )}
                        {s.uploadedFiles?.[0] && <span style={{ opacity: 0.5, flexShrink: 0 }}>·</span>}
                        <span style={{ flexShrink: 0, opacity: 0.8 }}>{relativeTime(s.createdAt)}</span>
                      </div>
                    )}

                    {/* 3-dot menu button — revealed on hover */}
                    {!isEditing && (
                      <button onClick={(e) => openSessionMenu(s.id, e)}
                        className="bf-menu-btn"
                        style={{ position: "absolute", right: 6, top: "50%", transform: "translateY(-50%)",
                          background: T.panel2, border: `1px solid ${T.border2}`, color: T.muted,
                          cursor: "pointer", padding: "3px 5px", borderRadius: 6,
                          display: "flex", alignItems: "center", opacity: 0,
                          transition: "opacity .12s, color .12s" }}>
                        <IcDots s={13} />
                      </button>
                    )}
                </>
              </div>
            );
          })
        )}
      </div>

      {/* Footer */}
      <div style={{ paddingTop: 12, borderTop: `1px solid ${T.border}`, flexShrink: 0 }}>
        <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted, marginBottom: 4,
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={authUser?.email}>
          {authUser?.email}
        </div>
        <div style={{ fontFamily: T.mono, fontSize: 10, color: T.muted, opacity: 0.6 }}>
          RAG pipeline · local inference
        </div>
      </div>
    </>
  );

  // ── Rail nav icons ─────────────────────────────────────────────────────────
  const railItems = [IcPlus];

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div style={{ position: "fixed", inset: 0, display: "flex", overflow: "hidden",
      background: T.bg, color: T.ink, fontFamily: T.sans, fontSize: 14 }}>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        @keyframes bf-bounce { 0%, 80%, 100% { transform: translateY(0); opacity: .5; } 40% { transform: translateY(-5px); opacity: 1; } }
        @keyframes bf-spin   { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        ::-webkit-scrollbar       { width: 9px; height: 9px; }
        ::-webkit-scrollbar-thumb { background: #2c313a; border-radius: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        input::placeholder        { color: #7c828c; }
        .bf-new:hover        { border-color: #c6f24a !important; }
        .bf-thread:hover     { background: #15181d !important; border-color: #2c313a !important; }
        .bf-thread:hover .bf-menu-btn { opacity: 1 !important; }
        .bf-menu-btn:hover { color: #eef0f2 !important; border-color: #3a3f4a !important; }
        .bf-thread-active:hover { border-color: #c6f24a !important; }
        .bf-rail:hover       { background: #15181d !important; color: #eef0f2 !important; }
        .bf-sug:hover        { border-color: #c6f24a !important; color: #c6f24a !important; }
        .bf-cite:hover       { background: #c6f24a !important; color: #0c1003 !important; border-color: #c6f24a !important; }
        .bf-send:hover       { filter: brightness(1.08); }
        .bf-composer:focus-within { border-color: #c6f24a !important; }
        .bf-msg-user:hover .bf-copy-btn  { opacity: 1 !important; }
        .bf-copy-btn:hover   { color: #eef0f2 !important; }
      `}</style>

      {/* ── Desktop left rail (56px) ─────────────────────────────────── */}
      <nav style={{ width: 56, flexShrink: 0, background: T.panel, borderRight: `1px solid ${T.border}`,
        display: "flex", flexDirection: "column", alignItems: "center", padding: "16px 0", gap: 6 }}
        className="hidden-mobile">
        <div onClick={newChat} title="ClauseIQ"
          style={{ width: 32, height: 32, borderRadius: 9, display: "grid",
            placeItems: "center", marginBottom: 12, overflow: "hidden",
            cursor: "pointer", userSelect: "none" }}>
          <img src={clauseiqMark} alt="" width={32} height={32} style={{ display: "block" }} />
        </div>
        {/* Sidebar collapse toggle */}
        <button onClick={() => setSidebarCollapsed(v => !v)} className="bf-rail"
          title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          style={{ width: 38, height: 38, borderRadius: 9, display: "grid", placeItems: "center",
            color: sidebarCollapsed ? T.accent : T.ink2, cursor: "pointer", transition: "all .14s",
            background: "none", border: "none" }}>
          <IcMenu s={18} />
        </button>
        {railItems.map((Ic, i) => (
          <button key={i} className="bf-rail"
            onClick={i === 0 ? newChat : undefined}
            style={{ width: 38, height: 38, borderRadius: 9, display: "grid", placeItems: "center",
              color: i === 0 ? T.ink : T.muted, cursor: "pointer", transition: "all .14s",
              background: "none", border: "none" }}><Ic /></button>
        ))}
        <div style={{ flex: 1 }} />

        {/* Avatar + sign-out popover */}
        <div style={{ position: "relative" }}>
          {avatarMenuOpen && (
            <>
              {/* Invisible backdrop to close on outside click */}
              <div onClick={() => setAvatarMenuOpen(false)}
                style={{ position: "fixed", inset: 0, zIndex: 99 }} />
              <div style={{
                position: "absolute", bottom: 38, left: 4, transform: "none",
                background: T.panel2, border: `1px solid ${T.border2}`, borderRadius: 10,
                minWidth: 190, boxShadow: "0 8px 32px rgba(0,0,0,0.5)", zIndex: 100,
                overflow: "hidden",
              }}>
                <div style={{ padding: "10px 14px 8px", borderBottom: `1px solid ${T.border}` }}>
                  <div style={{ fontFamily: T.mono, fontSize: 9.5, color: T.muted,
                    letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 3 }}>signed in as</div>
                  <div style={{ fontFamily: T.mono, fontSize: 11.5, color: T.ink,
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {authUser?.email}
                  </div>
                  {authUser?.role === "admin" && (
                    <div style={{ fontFamily: T.mono, fontSize: 10, color: T.accent, marginTop: 3 }}>admin</div>
                  )}
                </div>
                <button onClick={() => { setAvatarMenuOpen(false); setShowResetPassword(true); }}
                  style={{ width: "100%", padding: "9px 14px", background: "none", border: "none",
                    color: T.ink2, fontFamily: T.mono, fontSize: 12, cursor: "pointer",
                    textAlign: "left", transition: "background .12s", borderBottom: `1px solid ${T.border}` }}
                  onMouseEnter={e => e.currentTarget.style.background = T.panel}
                  onMouseLeave={e => e.currentTarget.style.background = "none"}>
                  Reset password
                </button>
                <button onClick={handleLogout}
                  style={{ width: "100%", padding: "9px 14px", background: "none", border: "none",
                    color: "#f87171", fontFamily: T.mono, fontSize: 12, cursor: "pointer",
                    textAlign: "left", transition: "background .12s" }}
                  onMouseEnter={e => e.currentTarget.style.background = "rgba(248,113,113,0.08)"}
                  onMouseLeave={e => e.currentTarget.style.background = "none"}>
                  Sign out
                </button>
              </div>
            </>
          )}
          <div onClick={() => setAvatarMenuOpen(v => !v)}
            style={{ width: 30, height: 30, borderRadius: "50%", background: T.panel2,
              border: `1px solid ${avatarMenuOpen ? T.accent : T.border}`,
              display: "grid", placeItems: "center", cursor: "pointer",
              fontSize: 11, fontWeight: 600, fontFamily: T.mono, color: T.ink2,
              transition: "border-color .14s" }}
            title={authUser?.email}>
            {authUser?.email ? authUser.email.slice(0, 2).toUpperCase() : "??"}
          </div>
        </div>
      </nav>

      {/* ── Desktop chat history sidebar (collapsible, 270px) ───────── */}
      <aside
        className="hidden-mobile"
        style={{
          width: sidebarCollapsed ? 0 : 270,
          minWidth: sidebarCollapsed ? 0 : 270,
          overflow: "hidden",
          flexShrink: 0,
          background: T.bg,
          borderRight: sidebarCollapsed ? "none" : `1px solid ${T.border}`,
          display: "flex",
          flexDirection: "column",
          padding: sidebarCollapsed ? 0 : 14,
          transition: "width .22s ease, min-width .22s ease, padding .22s ease",
        }}>
        {!sidebarCollapsed && sidebarContent}
      </aside>

      {/* ── Mobile sidebar overlay ─────────────────────────────────────── */}
      {sidebarOpen && (
        <div onClick={() => setSidebarOpen(false)}
          style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 50, display: "flex" }}>
          <div onClick={(e) => e.stopPropagation()}
            style={{ display: "flex", height: "100%", boxShadow: "0 24px 60px -20px rgba(0,0,0,0.7)" }}>
            <nav style={{ width: 56, flexShrink: 0, background: T.panel, borderRight: `1px solid ${T.border}`,
              display: "flex", flexDirection: "column", alignItems: "center", padding: "16px 0", gap: 6 }}>
              <div style={{ width: 32, height: 32, borderRadius: 9, display: "grid",
                placeItems: "center", marginBottom: 12, overflow: "hidden" }}>
                <img src={clauseiqMark} alt="" width={32} height={32} style={{ display: "block" }} />
              </div>
            </nav>
            <aside style={{ width: 270, background: T.bg, borderRight: `1px solid ${T.border}`,
              display: "flex", flexDirection: "column", padding: 14 }}>
              {sidebarContent}
            </aside>
          </div>
        </div>
      )}

      {/* ── Main column ──────────────────────────────────────────────── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>

        {/* Header */}
        <header style={{ height: 60, flexShrink: 0, borderBottom: `1px solid ${T.border}`,
          display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0 28px", gap: 12 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, minWidth: 0 }}>
            {/* Mobile menu */}
            <button onClick={() => setSidebarOpen(true)} className="bf-rail show-mobile"
              style={{ background: "none", border: "none", color: T.ink, cursor: "pointer", padding: 4,
                display: "none", gridTemplateColumns: "1fr", placeItems: "center" }}>
              <IcMenu />
            </button>
            <span style={{ fontSize: 15, fontWeight: 600, letterSpacing: "-0.01em", color: T.ink,
              whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
              {activeSess?.name || "ClauseIQ"}
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            {authUser?.role === "admin" && (
              <button onClick={() => setShowAdmin(true)} className="bf-new"
                style={{ display: "flex", alignItems: "center", gap: 6, padding: "7px 12px", borderRadius: 8,
                  border: `1px solid rgba(198,242,74,0.4)`, background: "rgba(198,242,74,0.06)", color: T.accent,
                  fontFamily: T.mono, fontSize: 11, cursor: "pointer", transition: "all .14s" }}>
                Admin
              </button>
            )}
            <button onClick={() => fileInputRef.current?.click()} className="bf-new"
              style={{ display: "flex", alignItems: "center", gap: 7, padding: "7px 12px", borderRadius: 8,
                border: `1px solid ${T.border2}`, background: "transparent", color: T.ink2,
                fontFamily: T.mono, fontSize: 11.5, cursor: "pointer", transition: "all .14s" }}>
              <IcUpload s={15} /> Upload
            </button>
            <input ref={fileInputRef} type="file" multiple accept=".pdf" onChange={onFileInputChange} style={{ display: "none" }} />
          </div>
        </header>

        {/* ── VIEW: empty (dropzone) ──────────────────────────────────── */}
        {view === "empty" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", padding: "32px clamp(20px,6vw,48px)", overflowY: "auto" }}>
            <div style={{ width: "100%", maxWidth: 560 }}>
              <div style={{ fontFamily: T.mono, fontSize: 11.5, color: T.accent, letterSpacing: "0.14em",
                marginBottom: 14, textAlign: "center" }}>NEW REVIEW</div>
              <h1 style={{ fontFamily: T.sans, fontSize: 30, fontWeight: 700, color: T.ink,
                letterSpacing: "-0.02em", textAlign: "center", margin: "0 0 10px", lineHeight: 1.15 }}>
                Ask anything about your document.
              </h1>
              <p style={{ fontSize: 14.5, color: T.muted, textAlign: "center", lineHeight: 1.6, margin: "0 0 30px" }}>
                Upload a legal or business document and get answers grounded in its exact text — every claim cites the clause it came from.
              </p>

              {/* Dropzone */}
              <div
                onClick={() => dropzoneInputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => { e.preventDefault(); if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files); }}
                style={{ border: "1.5px dashed #2c313a", borderRadius: 16, background: T.panel,
                  padding: "46px 40px", textAlign: "center", cursor: "pointer", transition: "all .16s" }}
                onMouseEnter={(e) => { e.currentTarget.style.borderColor = T.accent; e.currentTarget.style.background = T.accentGlow; }}
                onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#2c313a"; e.currentTarget.style.background = T.panel; }}
              >
                <div style={{ width: 52, height: 52, borderRadius: 13, background: T.accent, color: T.accentInk,
                  display: "grid", placeItems: "center", margin: "0 auto 18px" }}><IcUpload s={24} /></div>
                <div style={{ fontFamily: T.sans, fontSize: 16, fontWeight: 600, color: T.ink, marginBottom: 6 }}>
                  Drop a PDF to start
                </div>
                <div style={{ fontSize: 13, color: T.muted, lineHeight: 1.6 }}>
                  Click to browse or drag & drop · PDF · max 150 pages · 20 MB
                </div>
              </div>
              <input ref={dropzoneInputRef} type="file" multiple accept=".pdf" onChange={onDropzoneInputChange} style={{ display: "none" }} />
            </div>
          </div>
        )}

        {/* ── VIEW: processing ──────────────────────────────────────────── */}
        {view === "processing" && (
          <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center",
            justifyContent: "center", padding: 40 }}>
            {procState.error ? (
              <div style={{ width: "100%", maxWidth: 420, textAlign: "center" }}>
                <div style={{ width: 52, height: 52, borderRadius: 13,
                  background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.3)",
                  display: "grid", placeItems: "center", margin: "0 auto 18px", color: "#f87171" }}>
                  <IcAlert s={24} />
                </div>
                <h2 style={{ fontFamily: T.sans, fontSize: 20, fontWeight: 700, color: T.ink,
                  margin: "0 0 8px" }}>Upload failed</h2>
                <p style={{ fontFamily: T.mono, fontSize: 12, color: T.muted, margin: "0 0 24px", lineHeight: 1.6 }}>
                  {attached.find((f) => f.status === "error")?.errorMessage
                    || "The file could not be indexed. Check the error on the chip above and try again."}
                </p>
                <button
                  onClick={() => { setView("empty"); setAttached([]); setProcState({ name: "", progress: 0 }); }}
                  style={{ padding: "9px 22px", borderRadius: 9,
                    border: "1px solid rgba(248,113,113,0.4)", background: "rgba(248,113,113,0.08)",
                    color: "#f87171", fontFamily: T.mono, fontSize: 12, cursor: "pointer" }}>
                  Try again
                </button>
              </div>
            ) : (
              <div style={{ width: "100%", maxWidth: 420 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 22 }}>
                  <div style={{ width: 44, height: 44, borderRadius: 11, background: T.panel2,
                    border: `1px solid ${T.border2}`, display: "grid", placeItems: "center",
                    color: T.accent, flexShrink: 0 }}><IcDoc s={20} /></div>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontFamily: T.mono, fontSize: 13, color: T.ink,
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{procState.name}</div>
                    <div style={{ fontFamily: T.mono, fontSize: 11, color: T.muted, marginTop: 3 }}>
                      Indexing · building embeddings · can take a couple of minutes for large PDFs
                    </div>
                  </div>
                </div>
                <div style={{ height: 8, borderRadius: 5, background: T.panel2, overflow: "hidden",
                  border: `1px solid ${T.border}` }}>
                  <div style={{ height: "100%", width: procState.progress + "%", background: T.accent,
                    borderRadius: 5, transition: "width .3s ease" }} />
                </div>
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: 12 }}>
                  <span style={{ fontFamily: T.mono, fontSize: 11, color: T.muted }}>
                    Extracting chunks · building citations
                  </span>
                  <span style={{ fontFamily: T.mono, fontSize: 11, color: T.accent }}>{procState.progress}%</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── VIEW: chat ────────────────────────────────────────────────── */}
        {view === "chat" && (
          <>
            {/* Messages */}
            <div ref={scrollRef} style={{ flex: 1, overflowY: "auto", overflowX: "hidden" }}>
              <div style={{ maxWidth: 760, margin: "0 auto", padding: "30px clamp(20px,6vw,56px) 12px",
                display: "flex", flexDirection: "column", gap: 24 }}>

                {messages.length === 0 ? (
                  <div style={{ textAlign: "center", paddingTop: 56 }}>
                    <div style={{ width: 52, height: 52, borderRadius: 13, background: T.accent, color: T.accentInk,
                      display: "grid", placeItems: "center", margin: "0 auto 18px" }}><IcDoc s={24} /></div>
                    {activeSess?.uploadedFiles?.length > 0 ? (
                      <>
                        <h1 style={{ fontFamily: T.sans, fontSize: 26, fontWeight: 700, color: T.ink,
                          letterSpacing: "-0.02em", margin: "0 0 10px", lineHeight: 1.2 }}>
                          Your document is ready.
                        </h1>
                        <p style={{ fontSize: 14.5, color: T.muted, lineHeight: 1.6, margin: "0 0 20px" }}>
                          Ask anything — every answer will be grounded in the exact text of your file.
                        </p>
                        <div style={{ display: "inline-flex", flexDirection: "column", gap: 6, alignItems: "center" }}>
                          {activeSess.uploadedFiles.map((name) => (
                            <div key={name} style={{ display: "flex", alignItems: "center", gap: 7,
                              fontFamily: T.mono, fontSize: 11.5, color: T.ink2,
                              border: `1px solid ${T.border2}`, borderRadius: 8, padding: "5px 12px",
                              background: T.panel }}>
                              <span style={{ color: T.accent }}><IcDoc s={13} /></span>
                              <span style={{ flex: 1 }}>{name}</span>
                              <button
                                onClick={() => deleteDocument(name)}
                                title="Remove document"
                                style={{ background: "none", border: "none", cursor: "pointer",
                                  color: T.muted, padding: "0 0 0 6px", display: "flex",
                                  alignItems: "center", lineHeight: 1, transition: "color .14s" }}
                                onMouseEnter={e => e.currentTarget.style.color = "#f87171"}
                                onMouseLeave={e => e.currentTarget.style.color = T.muted}>
                                <IcClose s={11} />
                              </button>
                            </div>
                          ))}
                        </div>
                      </>
                    ) : (
                      <>
                        <h1 style={{ fontFamily: T.sans, fontSize: 26, fontWeight: 700, color: T.ink,
                          letterSpacing: "-0.02em", margin: "0 0 10px", lineHeight: 1.2 }}>
                          Ready to help.
                        </h1>
                        <p style={{ fontSize: 14.5, color: T.muted, lineHeight: 1.6, margin: "0 0 20px" }}>
                          Upload a document using the button above to get started.
                        </p>
                      </>
                    )}
                  </div>
                ) : (
                  messages.map((m) =>
                    m.role === "user" ? (
                      <div key={m.id} className="bf-msg-user"
                        style={{ alignSelf: "flex-end", maxWidth: "74%", position: "relative" }}>
                        <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted,
                          textAlign: "right", marginBottom: 6, letterSpacing: "0.08em" }}>YOU</div>
                        {m.files && m.files.length > 0 && (
                          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8, justifyContent: "flex-end" }}>
                            {m.files.map((f) => <FileChip key={f.id} name={f.name} size={f.size} />)}
                          </div>
                        )}
                        <div style={{ background: T.panel2, border: `1px solid ${T.border}`,
                          padding: "13px 16px", borderRadius: 12, fontSize: 14.5, lineHeight: 1.55, color: T.ink }}>
                          {m.content}
                        </div>
                        {m.content && (
                          <button onClick={() => copyText(m.id, m.content)}
                            className="bf-copy-btn"
                            title="Copy"
                            style={{ position: "absolute", bottom: -22, right: 2,
                              display: "flex", alignItems: "center", gap: 4,
                              opacity: 0, background: "none", border: "none", cursor: "pointer",
                              color: copiedId === m.id ? "#c6f24a" : T.muted,
                              fontFamily: T.mono, fontSize: 10.5, transition: "opacity .14s, color .14s" }}>
                            {copiedId === m.id ? <IcCopied s={12} /> : <IcCopy s={12} />}
                            {copiedId === m.id ? "copied" : "copy"}
                          </button>
                        )}
                      </div>
                    ) : (
                      <div key={m.id} style={{ alignSelf: "flex-start", maxWidth: "84%" }}>
                        <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.accent,
                          marginBottom: 8, letterSpacing: "0.08em" }}>ClauseIQ</div>

                        {m.isComparison && m.comparedDocs?.length > 0 && (
                          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
                            <div style={{ display: "inline-flex", alignItems: "center", gap: 6,
                              fontFamily: T.mono, fontSize: 10.5, color: T.muted,
                              border: `1px solid ${T.border2}`, borderRadius: 6,
                              padding: "3px 9px", background: T.panel2 }}>
                              <IcDoc s={11} />
                              {m.comparedDocs.length === 2
                                ? `${m.comparedDocs[0]} vs ${m.comparedDocs[1]}`
                                : `${m.comparedDocs.length} documents compared`}
                            </div>
                            {m.weakEvidence?.map((doc) => (
                              <div key={doc} style={{ display: "inline-flex", alignItems: "center", gap: 5,
                                fontFamily: T.mono, fontSize: 10.5, color: "#fbbf24",
                                border: "1px solid rgba(251,191,36,0.3)", borderRadius: 6,
                                padding: "3px 9px", background: "rgba(251,191,36,0.06)" }}>
                                <IcAlert s={11} /> No evidence in {doc}
                              </div>
                            ))}
                          </div>
                        )}

                        {!m.content && !m.isError ? (
                          m.isComparison ? (
                            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                              <TypingDots />
                              <span style={{ fontFamily: T.mono, fontSize: 11, color: T.muted }}>
                                Analyzing documents…
                              </span>
                            </div>
                          ) : <TypingDots />
                        ) : m.content ? (
                          m.isError ? (
                            <div style={{ fontSize: 14.5, lineHeight: 1.75, color: "#f87171" }}>{m.content}</div>
                          ) : (
                            <div style={{ fontSize: 15, lineHeight: 1.75, color: T.ink }}
                              className="prose prose-invert max-w-none
                                prose-p:text-[0.9rem] prose-p:leading-7 prose-p:my-2
                                prose-strong:text-[#c6f24a]
                                prose-code:bg-[#15181d] prose-code:text-[#c6f24a] prose-code:text-[0.8em] prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:font-medium
                                prose-ul:pl-5 prose-li:my-1 prose-li:leading-7
                                prose-headings:text-[#eef0f2] prose-headings:font-bold
                                prose-blockquote:border-l-[#c6f24a] prose-blockquote:text-[#b6bbc2]
                                prose-table:w-full prose-table:text-[0.82rem]
                                prose-thead:border-b prose-thead:border-[#2c313a]
                                prose-th:text-[#c6f24a] prose-th:font-semibold prose-th:py-2 prose-th:px-3 prose-th:text-left
                                prose-td:py-2 prose-td:px-3 prose-td:border-b prose-td:border-[#22262d] prose-td:text-[#b6bbc2]
                                prose-tr:border-none">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                            </div>
                          )
                        ) : null}

                        {/* Backward compat: old localStorage messages */}
                        {!m.verification && m.isFaithful === false && (
                          <div style={{ display: "inline-flex", alignItems: "center", gap: 5,
                            fontFamily: T.mono, fontSize: 10.5, color: "#fbbf24",
                            border: "1px solid rgba(251,191,36,0.25)", background: "rgba(251,191,36,0.05)",
                            borderRadius: 6, padding: "3px 8px", marginTop: 10 }}>
                            <IcAlert s={11} /> May not be fully grounded in retrieved context
                          </div>
                        )}
                        {/* Verification badge — below the answer, theme-matched */}
                        {m.verification && (() => {
                          const { verdict, totalClaims, unverifiedClaims } = m.verification;
                          const verifiedCount = totalClaims - unverifiedClaims.length;
                          if (verdict === "PASS") return (
                            <div style={{ display: "inline-flex", alignItems: "center", gap: 5,
                              fontFamily: T.mono, fontSize: 10.5, color: T.muted, marginTop: 10 }}>
                              <span style={{ color: T.accent }}><IcCheck s={11} /></span>
                              {totalClaims > 0 ? `${totalClaims}/${totalClaims} claims verified` : "Verified"}
                            </div>
                          );
                          if (verdict === "PARTIAL") return (
                            <div style={{ display: "inline-flex", alignItems: "center", gap: 5,
                              fontFamily: T.mono, fontSize: 10.5, color: "#fbbf24",
                              border: "1px solid rgba(251,191,36,0.25)", background: "rgba(251,191,36,0.05)",
                              borderRadius: 6, padding: "3px 8px", marginTop: 10 }}>
                              <IcAlert s={11} />
                              {verifiedCount}/{totalClaims} claims verified
                            </div>
                          );
                          return (
                            <div style={{ display: "inline-flex", alignItems: "center", gap: 5,
                              fontFamily: T.mono, fontSize: 10.5, color: "#fbbf24",
                              border: "1px solid rgba(251,191,36,0.25)", background: "rgba(251,191,36,0.05)",
                              borderRadius: 6, padding: "3px 8px", marginTop: 10 }}>
                              <IcAlert s={11} /> Unverified — may not be grounded in retrieved context
                            </div>
                          );
                        })()}

                        {/* Claim-level evidence — supersedes the source chips once verification has located each claim's quote */}
                        {m.claimCitations?.length > 0 && (
                          <div style={{ marginTop: 12, display: "flex", flexDirection: "column", gap: 6 }}>
                            <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted, letterSpacing: "0.06em" }}>
                              EVIDENCE
                            </div>
                            {m.claimCitations.map((c, i) => {
                              const isActive = selectedCitation?.id === c.id;
                              return (
                                <button key={c.id}
                                  onClick={() => setSelectedCitation(isActive ? null : c)}
                                  className="bf-cite"
                                  style={{ display: "flex", gap: 10, alignItems: "flex-start", textAlign: "left",
                                    width: "100%", padding: "7px 10px", borderRadius: 8, cursor: "pointer",
                                    border: `1px solid ${isActive ? T.accent : T.border2}`,
                                    background: isActive ? "rgba(198,242,74,0.06)" : "transparent",
                                    transition: "all .14s", minWidth: 0 }}>
                                  <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600,
                                    color: T.accent, flexShrink: 0, lineHeight: 1.6 }}>[{i + 1}]</span>
                                  <span style={{ display: "flex", flexDirection: "column", gap: 2, minWidth: 0 }}>
                                    <span style={{ fontFamily: T.sans, fontSize: 12.5, color: T.ink,
                                      lineHeight: 1.5, wordBreak: "break-word" }}>{c.claim}</span>
                                    <span style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted,
                                      overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                                      §{c.source}{c.section ? ` — ${c.section}` : ""}{c.pageRange ? ` · ${c.pageRange}` : ""}
                                    </span>
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        )}

                        {!(m.claimCitations?.length > 0) && m.citations && m.citations.length > 0 && (() => {
                          return (
                            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 12 }}>
                              {m.citations.map((c) => {
                                const isActive = selectedCitation?.id === c.id;
                                const displayRange = c.pageRange || (c.page != null ? `p. ${c.page}` : "");
                                return (
                                  <button key={c.id}
                                    onClick={() => setSelectedCitation(isActive ? null : c)}
                                    className="bf-cite"
                                    style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 500,
                                      color: isActive ? T.accentInk : T.accent,
                                      border: `1px solid ${isActive ? T.accent : T.border2}`,
                                      padding: "3px 10px", borderRadius: 6,
                                      cursor: "pointer", transition: "all .14s", userSelect: "none",
                                      background: isActive ? T.accent : "transparent",
                                      display: "inline-flex", alignItems: "center", gap: 4,
                                      whiteSpace: "nowrap" }}>
                                    <span>§{c.source}</span>
                                    {c.section && (
                                      <span style={{ maxWidth: 140, overflow: "hidden",
                                        textOverflow: "ellipsis" }}>
                                        — {c.section}
                                      </span>
                                    )}
                                    {displayRange && <span>· {displayRange}</span>}
                                  </button>
                                );
                              })}
                            </div>
                          );
                        })()}

                        {m.content && !m.isError && (
                          <button onClick={() => copyText(m.id, m.content)}
                            title="Copy response"
                            style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 10,
                              background: "none", border: "none", cursor: "pointer",
                              color: copiedId === m.id ? "#c6f24a" : T.muted,
                              fontFamily: T.mono, fontSize: 10.5, transition: "color .14s" }}
                            onMouseEnter={e => { if (copiedId !== m.id) e.currentTarget.style.color = "#eef0f2"; }}
                            onMouseLeave={e => { if (copiedId !== m.id) e.currentTarget.style.color = "#7c828c"; }}>
                            {copiedId === m.id ? <IcCopied s={13} /> : <IcCopy s={13} />}
                            {copiedId === m.id ? "Copied" : "Copy"}
                          </button>
                        )}
                      </div>
                    )
                  )
                )}
              </div>
            </div>

            {/* Composer */}
            <div style={{ padding: `0 clamp(20px,6vw,56px) 26px` }}>

              {/* Ask / Compare mode toggle — only when ≥ 2 docs uploaded */}
              {canCompare && (
                <div style={{ display: "flex", gap: 3, marginBottom: 10,
                  background: T.panel2, borderRadius: 9, padding: 3,
                  border: `1px solid ${T.border}`, alignSelf: "flex-start",
                  width: "fit-content" }}>
                  {["ask", "compare"].map((m) => (
                    <button key={m}
                      onClick={() => { setCompareMode(m === "compare"); setSelectedDocs(new Set()); setDocSelectorOpen(false); }}
                      style={{ padding: "5px 14px", borderRadius: 7, border: "none",
                        background: (m === "compare") === compareMode ? T.accent : "transparent",
                        color: (m === "compare") === compareMode ? T.accentInk : T.muted,
                        fontFamily: T.mono, fontSize: 11, fontWeight: 500, cursor: "pointer",
                        transition: "all .14s" }}>
                      {m === "ask" ? "Ask" : "Compare"}
                    </button>
                  ))}
                </div>
              )}

              {/* Doc selector (compare mode only) */}
              {compareMode && (
                <div style={{ marginBottom: 8 }}>
                  {/* Toggle button */}
                  <button
                    onClick={() => setDocSelectorOpen((o) => !o)}
                    style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: docSelectorOpen ? 6 : 0,
                      background: "none", border: `1px solid ${T.border2}`, borderRadius: 7,
                      padding: "5px 12px", cursor: "pointer", transition: "border-color .14s" }}
                    onMouseEnter={e => e.currentTarget.style.borderColor = T.accent}
                    onMouseLeave={e => e.currentTarget.style.borderColor = T.border2}>
                    <IcDoc s={12} color={selectedDocs.size >= 2 ? T.accent : T.muted} />
                    <span style={{ fontFamily: T.mono, fontSize: 11,
                      color: selectedDocs.size >= 2 ? T.accent : T.muted }}>
                      {selectedDocs.size >= 2
                        ? `${selectedDocs.size} docs selected`
                        : "Select documents"}
                    </span>
                    <span style={{ fontFamily: T.mono, fontSize: 10, color: T.muted, marginLeft: 2 }}>
                      {docSelectorOpen ? "▲" : "▼"}
                    </span>
                  </button>

                  {/* Collapsible panel */}
                  {docSelectorOpen && (
                    <div style={{ background: T.panel2, border: `1px solid ${T.border2}`,
                      borderRadius: 10, padding: "10px 14px" }}>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                        {activeSess?.uploadedFiles?.map((filename) => {
                          const checked = selectedDocs.has(filename);
                          return (
                            <div key={filename} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                              <button onClick={() => toggleDoc(filename)}
                                style={{ display: "flex", alignItems: "center", gap: 7,
                                  padding: "5px 10px", borderRadius: 7, cursor: "pointer",
                                  border: `1px solid ${checked ? T.accent : T.border2}`,
                                  background: checked ? "rgba(198,242,74,0.08)" : "transparent",
                                  color: checked ? T.accent : T.ink2,
                                  fontFamily: T.mono, fontSize: 11.5, transition: "all .14s" }}>
                                <span style={{ width: 13, height: 13, borderRadius: 3, flexShrink: 0,
                                  border: `1.5px solid ${checked ? T.accent : T.border2}`,
                                  background: checked ? T.accent : "transparent",
                                  display: "grid", placeItems: "center", transition: "all .14s" }}>
                                  {checked && <span style={{ width: 5, height: 5, background: T.accentInk, borderRadius: 1, display: "block" }} />}
                                </span>
                                {filename}
                              </button>
                              <button
                                onClick={() => deleteDocument(filename)}
                                title="Remove document"
                                style={{ background: "none", border: "none", cursor: "pointer",
                                  color: T.muted, padding: 4, display: "flex",
                                  alignItems: "center", borderRadius: 5, transition: "color .14s" }}
                                onMouseEnter={e => e.currentTarget.style.color = "#f87171"}
                                onMouseLeave={e => e.currentTarget.style.color = T.muted}>
                                <IcClose s={11} />
                              </button>
                            </div>
                          );
                        })}
                      </div>
                      {selectedDocs.size < 2 && (
                        <div style={{ fontFamily: T.mono, fontSize: 10, color: T.muted, marginTop: 7 }}>
                          Select at least 2 documents to compare
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* No-doc banner — shown when conversation exists but all docs removed */}
              {messages.length > 0 && !(activeSess?.uploadedFiles?.length) && (
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  gap: 12, marginBottom: 8, padding: "9px 14px", borderRadius: 9,
                  background: "rgba(198,242,74,0.06)", border: `1px solid rgba(198,242,74,0.22)`,
                }}>
                  <span style={{ fontFamily: T.mono, fontSize: 11.5, color: T.ink2, lineHeight: 1.4 }}>
                    No documents attached — upload one to keep asking questions.
                  </span>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 600,
                      color: T.accentInk, background: T.accent, border: "none",
                      borderRadius: 6, padding: "5px 14px", cursor: "pointer",
                      whiteSpace: "nowrap", flexShrink: 0 }}>
                    Upload
                  </button>
                </div>
              )}

              {/* Indexed document pills — these are the real deletable docs */}
              {(activeSess?.uploadedFiles?.length > 0) && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                  {activeSess.uploadedFiles.map((name) => (
                    <div key={name} style={{
                      display: "flex", alignItems: "center", gap: 6,
                      fontFamily: T.mono, fontSize: 11, color: T.ink2,
                      border: `1px solid ${T.border2}`, borderRadius: 7,
                      padding: "3px 8px 3px 10px", background: T.panel2,
                    }}>
                      <span style={{ color: T.accent, display: "flex" }}><IcDoc s={11} /></span>
                      <span style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                      <button
                        onClick={() => deleteDocument(name)}
                        title="Remove document"
                        style={{ background: "none", border: "none", cursor: "pointer",
                          color: T.muted, padding: "0 0 0 2px", display: "flex",
                          alignItems: "center", transition: "color .14s", flexShrink: 0 }}
                        onMouseEnter={e => e.currentTarget.style.color = "#f87171"}
                        onMouseLeave={e => e.currentTarget.style.color = T.muted}>
                        <IcClose s={10} />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* File chips — in-progress uploads only */}
              {attached.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
                  {attached.map((f) => (
                    <FileChip key={f.id} name={f.name} size={f.size} status={f.status} errorMessage={f.errorMessage}
                      onRemove={() => removeAttached(f.id)} onRetry={() => retryUpload(f.id)} />
                  ))}
                </div>
              )}

              <div className="bf-composer"
                style={{ border: `1px solid ${compareMode ? "rgba(198,242,74,0.35)" : T.border2}`,
                  borderRadius: 12, background: T.field,
                  padding: "6px 6px 6px 14px", display: "flex", alignItems: "center", gap: 10,
                  transition: "border-color .14s" }}>
                <span style={{ color: T.muted, display: "grid", placeItems: "center", flexShrink: 0 }}>
                  <IcUpload />
                </span>
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={onKeyDown}
                  placeholder={compareMode ? "Compare termination clauses…" : "query this document…"}
                  rows={1}
                  maxLength={4000}
                  style={{ flex: 1, border: "none", outline: "none", fontFamily: T.mono, fontSize: 13.5,
                    color: T.ink, background: "transparent", minWidth: 0, resize: "none", lineHeight: 1.55,
                    padding: "6px 0", maxHeight: 140 }}
                />
                <button
                  onClick={compareMode ? handleCompare : handleSend}
                  disabled={compareMode ? compareDisabled : sendDisabled}
                  className="bf-send"
                  style={{ height: 38, padding: "0 16px", borderRadius: 8,
                    background: (compareMode ? compareDisabled : sendDisabled) ? "#2c313a" : T.accent,
                    border: "none",
                    color: (compareMode ? compareDisabled : sendDisabled) ? T.muted : T.accentInk,
                    fontFamily: T.sans, fontWeight: 600, fontSize: 13.5,
                    cursor: (compareMode ? compareDisabled : sendDisabled) ? "default" : "pointer",
                    display: "flex", alignItems: "center", gap: 7,
                    transition: "filter .14s, background .14s", flexShrink: 0 }}>
                  {isComparing ? "Analyzing…" : compareMode ? "Compare" : "Send"} {!isComparing && <IcArrow />}
                </button>
              </div>
              <div style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted, marginTop: 9, textAlign: "center" }}>
                {compareMode
                  ? selectedDocs.size < 2 ? "Select at least 2 documents to compare" : "Enter to compare · Shift+Enter for a new line"
                  : "Enter to send · Shift+Enter for a new line"}
              </div>
            </div>
          </>
        )}
      </main>

      {/* ── Source / Citation panel (sidebar on wide, overlay on narrow) ── */}
      {selectedCitation && (
        isNarrow ? (
          <div onClick={() => setSelectedCitation(null)}
            style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 45,
              display: "flex", justifyContent: "flex-end" }}>
            <aside onClick={(e) => e.stopPropagation()}
              style={{ width: "min(360px, 100vw)", background: T.panel, borderLeft: `1px solid ${T.border}`,
                display: "flex", flexDirection: "column", boxShadow: "-8px 0 32px rgba(0,0,0,0.4)" }}>
              {sourcePanelInner}
            </aside>
          </div>
        ) : (
          <aside style={{ width: 360, flexShrink: 0, background: T.panel, borderLeft: `1px solid ${T.border}`,
            display: "flex", flexDirection: "column" }}>
            {sourcePanelInner}
          </aside>
        )
      )}

      {/* Mobile-only CSS */}
      <style>{`
        @media (max-width: 760px) {
          .hidden-mobile { display: none !important; }
          .show-mobile   { display: grid !important; }
        }
      `}</style>

      {showAdmin && <AdminPanel onClose={() => setShowAdmin(false)} />}
      {showResetPassword && <ResetPasswordModal onClose={() => setShowResetPassword(false)} />}

      {/* ── Session context menu ─────────────────────────────────────── */}
      {sessionMenu && (
        <>
          <div onClick={() => setSessionMenu(null)}
            style={{ position: "fixed", inset: 0, zIndex: 148 }} />
          <div style={{
            position: "fixed", left: sessionMenu.x, top: sessionMenu.y, zIndex: 149,
            background: "#13161b", border: `1px solid #2a2f38`, borderRadius: 11,
            minWidth: 162, boxShadow: "0 12px 36px rgba(0,0,0,0.65)", overflow: "hidden",
            padding: "4px",
          }}>
            {/* Rename */}
            <button
              onClick={() => startRename(sessionMenu.id, sessions.find(s => s.id === sessionMenu.id)?.name)}
              style={{ width: "100%", padding: "9px 12px", background: "none", border: "none",
                borderRadius: 7, color: T.ink2, fontFamily: T.sans, fontWeight: 500, fontSize: 13,
                cursor: "pointer", textAlign: "left", display: "flex", alignItems: "center", gap: 10,
                transition: "background .1s, color .1s" }}
              onMouseEnter={e => { e.currentTarget.style.background = "#1e2229"; e.currentTarget.style.color = T.ink; }}
              onMouseLeave={e => { e.currentTarget.style.background = "none"; e.currentTarget.style.color = T.ink2; }}>
              <span style={{ color: T.muted, display: "flex", flexShrink: 0 }}><IcPencil s={14} /></span>
              Rename
            </button>

            {/* Divider */}
            <div style={{ height: 1, background: "#1e2229", margin: "3px 0" }} />

            {/* Delete */}
            <button
              onClick={() => requestDelete(sessionMenu.id)}
              style={{ width: "100%", padding: "9px 12px", background: "none", border: "none",
                borderRadius: 7, color: "#f87171", fontFamily: T.sans, fontWeight: 500, fontSize: 13,
                cursor: "pointer", textAlign: "left", display: "flex", alignItems: "center", gap: 10,
                transition: "background .1s" }}
              onMouseEnter={e => e.currentTarget.style.background = "rgba(248,113,113,0.1)"}
              onMouseLeave={e => e.currentTarget.style.background = "none"}>
              <span style={{ display: "flex", flexShrink: 0 }}><IcTrash s={14} /></span>
              Delete
            </button>
          </div>
        </>
      )}

      {/* ── Delete confirmation modal ────────────────────────────────── */}
      {deleteConfirmId && (
        <div onClick={() => setDeleteConfirmId(null)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)", zIndex: 200,
            display: "flex", alignItems: "center", justifyContent: "center", fontFamily: T.sans }}>
          <div onClick={(e) => e.stopPropagation()}
            style={{ width: "100%", maxWidth: 360, margin: "0 20px",
              background: T.panel, border: `1px solid ${T.border2}`, borderRadius: 14,
              padding: "28px 24px 22px", boxShadow: "0 24px 64px rgba(0,0,0,0.7)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
              <span style={{ width: 34, height: 34, borderRadius: 9, flexShrink: 0,
                background: "rgba(248,113,113,0.1)", border: "1px solid rgba(248,113,113,0.25)",
                display: "grid", placeItems: "center", color: "#f87171" }}>
                <IcTrash s={16} />
              </span>
              <span style={{ fontFamily: T.sans, fontWeight: 700, fontSize: 16, color: T.ink }}>
                Delete this chat?
              </span>
            </div>
            <div style={{ fontFamily: T.mono, fontSize: 12, color: T.muted, lineHeight: 1.6, marginBottom: 24 }}>
              {(() => {
                const sess = sessions.find(s => s.id === deleteConfirmId);
                return sess?.name
                  ? <><span style={{ color: T.ink2 }}>"{sess.name}"</span> will be permanently removed.</>
                  : "This chat will be permanently removed.";
              })()}
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={() => setDeleteConfirmId(null)}
                style={{ flex: 1, padding: "10px 0", borderRadius: 9, background: "none",
                  border: `1px solid ${T.border2}`, color: T.muted,
                  fontFamily: T.sans, fontWeight: 500, fontSize: 13.5, cursor: "pointer" }}>
                Cancel
              </button>
              <button onClick={() => { deleteSession(deleteConfirmId); setDeleteConfirmId(null); }}
                style={{ flex: 1, padding: "10px 0", borderRadius: 9,
                  background: "rgba(248,113,113,0.12)", border: "1px solid rgba(248,113,113,0.4)",
                  color: "#f87171", fontFamily: T.sans, fontWeight: 600, fontSize: 13.5, cursor: "pointer" }}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Upload error toast ───────────────────────────────────────── */}
      {uploadError && (
        <div style={{
          position: "fixed", bottom: 28, left: "50%", transform: "translateX(-50%)",
          zIndex: 300, background: "#1e1215", border: "1px solid rgba(248,113,113,0.4)",
          borderRadius: 10, padding: "12px 18px 12px 14px",
          display: "flex", alignItems: "center", gap: 10,
          boxShadow: "0 8px 28px rgba(0,0,0,0.6)", maxWidth: "min(480px, 90vw)",
        }}>
          <span style={{ color: "#f87171", flexShrink: 0, display: "flex" }}><IcAlert s={15} /></span>
          <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: 12.5, color: "#ffb3b3", lineHeight: 1.5 }}>
            {uploadError}
          </span>
          <button onClick={() => setUploadError(null)}
            style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer",
              color: "#7c828c", flexShrink: 0, display: "flex", padding: 2 }}
            onMouseEnter={e => e.currentTarget.style.color = "#f87171"}
            onMouseLeave={e => e.currentTarget.style.color = "#7c828c"}>
            <IcClose s={12} />
          </button>
        </div>
      )}
    </div>
  );
}
