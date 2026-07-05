import { useState } from "react";
import { API_BASE_URL, extractErrorMessage } from "../utils/api.js";
import { IcMail, IcLock, IcEye, IcEyeOff, IcArrow } from "./icons.jsx";
import clauseiqMark from "../assets/clauseiq-mark.svg";

const T = {
  bg: "#0a0b0d", panel: "#101216", panel2: "#15181d", field: "#0c0e11",
  ink: "#eef0f2", ink2: "#b6bbc2", muted: "#7c828c",
  border: "#22262d", border2: "#2c313a",
  accent: "#c6f24a", accentInk: "#0c1003", glow: "rgba(198,242,74,0.16)",
  sans: "'Space Grotesk', sans-serif",
  mono: "'IBM Plex Mono', monospace",
};

// Icon-prefixed text input with optional trailing action (e.g. password toggle)
function Field({ label, hint, icon, trailing, ...inputProps }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{
        display: "block", fontFamily: T.mono, fontSize: 11, letterSpacing: "0.08em",
        textTransform: "uppercase", color: T.muted, marginBottom: 8,
      }}>
        {label} {hint && <span style={{ opacity: 0.6, textTransform: "none", letterSpacing: 0 }}>{hint}</span>}
      </label>
      <div className="auth-input-wrap" style={{
        display: "flex", alignItems: "center", gap: 10,
        border: `1px solid ${T.border2}`, borderRadius: 10, background: T.field, padding: "0 12px",
        transition: "border-color .15s, box-shadow .15s",
      }}>
        <span style={{ color: T.muted, display: "grid", placeItems: "center", flexShrink: 0 }}>{icon}</span>
        <input {...inputProps} className="auth-input"
          style={{ flex: 1, minWidth: 0, border: "none", outline: "none", background: "transparent",
            color: T.ink, fontFamily: T.mono, fontSize: 13.5, padding: "11px 0" }} />
        {trailing}
      </div>
    </div>
  );
}

export default function Auth({ onLogin, initialResetToken, initialNotice }) {
  const [mode, setMode] = useState(() => initialResetToken ? "reset" : "login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [notice, setNotice] = useState(initialNotice || "");

  const switchMode = (next) => {
    setMode(next);
    setError("");
    setSuccess("");
    setNotice("");
    setPassword("");
    setNewPassword("");
    setConfirmPassword("");
    setShowPw(false);
  };

  // ── Login / Register ──────────────────────────────────────────────────────
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setNotice("");
    if (!email.trim() || !password) { setError("Email and password are required."); return; }
    if (mode === "register" && password.length < 8) { setError("Password must be at least 8 characters."); return; }
    setLoading(true);
    try {
      const endpoint = mode === "login" ? "/auth/login" : "/auth/register";
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: email.trim(), password }),
      });
      const data = await res.json();
      if (!res.ok) { setError(extractErrorMessage(data, "Something went wrong.")); return; }
      if (mode === "register") {
        const loginRes = await fetch(`${API_BASE_URL}/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
          body: JSON.stringify({ email: email.trim(), password }),
        });
        if (loginRes.ok) {
          const loginData = await loginRes.json();
          onLogin({ id: loginData.id, email: loginData.email, role: loginData.role });
        } else {
          // Account was created, but the immediate follow-up sign-in failed (rate
          // limit, transient error, ...). Land on the login tab with the same
          // credentials already filled in rather than leaving the user stuck.
          const loginErrBody = await loginRes.json().catch(() => ({}));
          setError(extractErrorMessage(loginErrBody, "Account created, but automatic sign-in failed. Please sign in below."));
          setMode("login");
        }
      } else {
        onLogin({ id: data.id, email: data.email, role: data.role });
      }
    } catch {
      setError("Cannot reach the backend. Make sure the API is running.");
    } finally {
      setLoading(false);
    }
  };

  // ── Forgot password ───────────────────────────────────────────────────────
  const handleForgot = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setNotice("");
    if (!email.trim()) { setError("Please enter your email address."); return; }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/auth/forgot-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email.trim() }),
      });
      const data = await res.json();
      if (!res.ok) { setError(extractErrorMessage(data, "Something went wrong.")); return; }
      setSuccess(data.message);
      setEmail("");
    } catch {
      setError("Cannot reach the backend. Make sure the API is running.");
    } finally {
      setLoading(false);
    }
  };

  // ── Reset password (via email link token) ─────────────────────────────────
  const handleReset = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setNotice("");
    if (!newPassword || !confirmPassword) { setError("Please fill in both password fields."); return; }
    if (newPassword !== confirmPassword) { setError("Passwords do not match."); return; }
    if (newPassword.length < 8) { setError("Password must be at least 8 characters."); return; }
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token: initialResetToken, new_password: newPassword }),
      });
      const data = await res.json();
      if (!res.ok) { setError(extractErrorMessage(data, "Something went wrong.")); return; }
      setSuccess(data.message);
      setNewPassword("");
      setConfirmPassword("");
      // Clear the token from the URL and switch to login
      window.history.replaceState({}, "", "/");
      setTimeout(() => switchMode("login"), 1800);
    } catch {
      setError("Cannot reach the backend. Make sure the API is running.");
    } finally {
      setLoading(false);
    }
  };

  const isLoginRegister = mode === "login" || mode === "register";

  const head = {
    login:    { kicker: "Welcome back",      title: "Sign in to ClauseIQ",       sub: "Pick up your document reviews where you left off." },
    register: { kicker: "Get started",       title: "Create your ClauseIQ account", sub: "Upload contracts and start asking questions in minutes." },
    forgot:   { kicker: "Account recovery",  title: "Forgot your password?",     sub: "Enter your email and we'll send you a reset link." },
    reset:    { kicker: "Account recovery",  title: "Set a new password",        sub: "Enter and confirm your new password below." },
  }[mode];

  const btnPrimaryStyle = {
    width: "100%", padding: "13px 0", borderRadius: 10, border: "none",
    background: loading ? T.panel2 : T.accent, color: loading ? T.muted : T.accentInk,
    fontFamily: T.sans, fontWeight: 600, fontSize: 15,
    cursor: loading ? "default" : "pointer", display: "flex", alignItems: "center",
    justifyContent: "center", gap: 8, transition: "filter .15s",
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: T.bg, fontFamily: T.sans, overflow: "auto" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        .auth-input-wrap:focus-within { border-color: #c6f24a !important; box-shadow: 0 0 0 3px rgba(198,242,74,0.16); }
        .auth-btn-primary:hover:not(:disabled) { filter: brightness(1.07); }
        .auth-tab:hover { color: #eef0f2 !important; }
        .auth-link:hover { color: #c6f24a !important; text-decoration: underline; }
        .auth-ghost-btn:hover { color: #eef0f2 !important; }
        .auth-stage { display: grid; grid-template-columns: 1.05fr 1fr; min-height: 100vh; }
        .auth-mobile-logo { display: none; }
        @media (max-width: 900px) {
          .auth-stage { grid-template-columns: 1fr; }
          .auth-brand { display: none !important; }
          .auth-mobile-logo { display: flex !important; }
          .auth-formwrap { padding: 32px 22px !important; }
        }
      `}</style>

      <div className="auth-stage">
        {/* ── Brand panel ──────────────────────────────────────────────── */}
        <section className="auth-brand" style={{
          position: "relative", overflow: "hidden", padding: "48px 56px",
          display: "flex", flexDirection: "column", justifyContent: "space-between",
          background: `radial-gradient(120% 90% at 12% 8%, rgba(198,242,74,0.07), transparent 55%),
                       linear-gradient(180deg, #0c0e11, #0a0b0d 60%)`,
          borderRight: `1px solid ${T.border}`,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <img src={clauseiqMark} alt="" width={38} height={38} style={{ borderRadius: 10, display: "block" }} />
            <div style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.01em", color: T.ink }}>
              Clause<span style={{ color: T.accent }}>IQ</span>
            </div>
          </div>

          <div style={{ maxWidth: 460 }}>
            <div style={{ fontFamily: T.mono, fontSize: 12, letterSpacing: "0.16em", textTransform: "uppercase",
              color: T.accent, marginBottom: 22 }}>
              Legal document intelligence
            </div>
            <h1 style={{ fontSize: "clamp(30px, 3.4vw, 46px)", lineHeight: 1.08, letterSpacing: "-0.025em",
              fontWeight: 700, margin: "0 0 20px", color: T.ink }}>
              Read contracts at the speed of a question.
            </h1>
            <p style={{ fontSize: 15.5, lineHeight: 1.65, color: T.ink2, margin: 0, maxWidth: 400 }}>
              Upload an agreement and ask anything. ClauseIQ answers in plain language — and cites the exact clause every time.
            </p>

            <div style={{ marginTop: 30, border: `1px solid ${T.border}`, borderRadius: 14,
              background: "rgba(21,24,29,0.7)", backdropFilter: "blur(6px)", padding: "18px 20px", maxWidth: 440 }}>
              <div style={{ fontSize: 14, color: T.ink2, marginBottom: 12 }}>
                <span style={{ fontFamily: T.mono, fontSize: 10.5, color: T.muted, letterSpacing: "0.08em" }}>YOU&nbsp;&nbsp;</span>
                When can either party terminate for convenience?
              </div>
              <div style={{ fontSize: 14.5, lineHeight: 1.6, color: T.ink }}>
                <span style={{ fontFamily: T.mono, fontSize: 10.5, color: T.accent, letterSpacing: "0.08em" }}>CLAUSEIQ&nbsp;&nbsp;</span>
                Either party may terminate with <strong style={{ color: T.accent, fontWeight: 600 }}>60 days' written notice</strong>
                <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 500, color: T.accent,
                  border: `1px solid ${T.border2}`, padding: "1px 6px", borderRadius: 6, margin: "0 2px", whiteSpace: "nowrap" }}>§12.2</span>,
                and accrued fees remain payable
                <span style={{ fontFamily: T.mono, fontSize: 11, fontWeight: 500, color: T.accent,
                  border: `1px solid ${T.border2}`, padding: "1px 6px", borderRadius: 6, margin: "0 2px", whiteSpace: "nowrap" }}>§12.4</span>.
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {["Legal & paralegal teams", "Compliance officers", "Independent professionals"].map((role) => (
              <div key={role} style={{ fontFamily: T.mono, fontSize: 11, color: T.muted,
                border: `1px solid ${T.border}`, borderRadius: 20, padding: "6px 12px" }}>{role}</div>
            ))}
          </div>
        </section>

        {/* ── Form panel ───────────────────────────────────────────────── */}
        <section className="auth-formwrap" style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "48px 40px" }}>
          <div style={{ width: "100%", maxWidth: 392 }}>
            <div className="auth-mobile-logo" style={{ alignItems: "center", gap: 11, justifyContent: "center", marginBottom: 34 }}>
              <img src={clauseiqMark} alt="" width={34} height={34} style={{ borderRadius: 9, display: "block" }} />
              <div style={{ fontSize: 20, fontWeight: 700, color: T.ink }}>Clause<span style={{ color: T.accent }}>IQ</span></div>
            </div>

            <div style={{ marginBottom: 28 }}>
              <div style={{ fontFamily: T.mono, fontSize: 11.5, letterSpacing: "0.14em", textTransform: "uppercase",
                color: T.muted, marginBottom: 12 }}>{head.kicker}</div>
              <h2 style={{ fontSize: 26, fontWeight: 700, letterSpacing: "-0.02em", margin: "0 0 8px", color: T.ink }}>{head.title}</h2>
              <p style={{ fontSize: 14, color: T.muted, margin: 0 }}>{head.sub}</p>
            </div>

            {/* ── Tabs (login / register only) ── */}
            {isLoginRegister && (
              <div style={{ display: "flex", gap: 4, marginBottom: 22, borderBottom: `1px solid ${T.border}`, paddingBottom: 16 }}>
                {["login", "register"].map((tab) => (
                  <button key={tab} onClick={() => switchMode(tab)} className="auth-tab"
                    style={{
                      flex: 1, padding: "8px 0", borderRadius: 8, border: "none",
                      background: mode === tab ? T.panel2 : "transparent",
                      color: mode === tab ? T.ink : T.muted,
                      fontFamily: T.sans, fontWeight: mode === tab ? 600 : 400, fontSize: 13.5,
                      cursor: "pointer", transition: "all .14s",
                      borderBottom: mode === tab ? `2px solid ${T.accent}` : "2px solid transparent",
                    }}>
                    {tab === "login" ? "Sign In" : "Create Account"}
                  </button>
                ))}
              </div>
            )}

            {/* ── Messages ── */}
            {notice && (
              <div style={{
                background: T.panel2, border: `1px solid ${T.border2}`,
                borderRadius: 8, padding: "10px 14px", marginBottom: 16,
                fontFamily: T.mono, fontSize: 12, color: T.ink2,
              }}>{notice}</div>
            )}
            {error && (
              <div style={{
                background: "rgba(248,113,113,0.08)", border: "1px solid rgba(248,113,113,0.25)",
                borderRadius: 8, padding: "10px 14px", marginBottom: 16,
                fontFamily: T.mono, fontSize: 12, color: "#f87171",
              }}>{error}</div>
            )}
            {success && (
              <div style={{
                background: "rgba(198,242,74,0.08)", border: "1px solid rgba(198,242,74,0.25)",
                borderRadius: 8, padding: "10px 14px", marginBottom: 16,
                fontFamily: T.mono, fontSize: 12, color: T.accent,
              }}>{success}</div>
            )}

            {/* ── Login / Register form ── */}
            {isLoginRegister && (
              <form onSubmit={handleSubmit}>
                <Field label="Email" icon={<IcMail s={17} />} type="email" value={email}
                  onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email" />
                <Field label="Password" hint={mode === "register" ? "(min 8 chars)" : undefined}
                  icon={<IcLock s={17} />} type={showPw ? "text" : "password"} value={password}
                  onChange={(e) => setPassword(e.target.value)} placeholder="••••••••"
                  autoComplete={mode === "login" ? "current-password" : "new-password"}
                  trailing={
                    <button type="button" onClick={() => setShowPw(v => !v)} className="auth-ghost-btn"
                      aria-label={showPw ? "Hide password" : "Show password"}
                      style={{ background: "none", border: "none", color: T.muted, cursor: "pointer",
                        padding: 4, display: "grid", placeItems: "center", flexShrink: 0 }}>
                      {showPw ? <IcEyeOff s={17} /> : <IcEye s={17} />}
                    </button>
                  } />

                {mode === "login" && (
                  <div style={{ textAlign: "right", marginBottom: 20, marginTop: -4 }}>
                    <button type="button" onClick={() => switchMode("forgot")} className="auth-link"
                      style={{ background: "none", border: "none", cursor: "pointer",
                        fontFamily: T.sans, fontSize: 13, color: T.accent, padding: 0 }}>
                      Forgot password?
                    </button>
                  </div>
                )}

                <button type="submit" disabled={loading} className="auth-btn-primary"
                  style={{ ...btnPrimaryStyle, marginTop: mode === "register" ? 6 : 0 }}>
                  {loading ? "Please wait…" : mode === "login" ? "Sign In" : "Create Account"}
                  {!loading && <IcArrow s={15} />}
                </button>
              </form>
            )}

            {/* ── Forgot password form ── */}
            {mode === "forgot" && (
              <form onSubmit={handleForgot}>
                <Field label="Email" icon={<IcMail s={17} />} type="email" value={email}
                  onChange={(e) => setEmail(e.target.value)} placeholder="you@example.com" autoComplete="email" />
                <button type="submit" disabled={loading} className="auth-btn-primary" style={btnPrimaryStyle}>
                  {loading ? "Sending…" : "Send Reset Link"}
                  {!loading && <IcArrow s={15} />}
                </button>
                <button type="button" onClick={() => switchMode("login")}
                  style={{ marginTop: 12, width: "100%", padding: "10px 0", borderRadius: 10,
                    background: "none", border: `1px solid ${T.border2}`,
                    color: T.muted, fontFamily: T.sans, fontWeight: 500, fontSize: 13.5,
                    cursor: "pointer" }}>
                  Back to Sign In
                </button>
              </form>
            )}

            {/* ── Reset password form ── */}
            {mode === "reset" && (
              <form onSubmit={handleReset}>
                <Field label="New Password" hint="(min 8 chars)" icon={<IcLock s={17} />}
                  type={showPw ? "text" : "password"} value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)} placeholder="••••••••" autoComplete="new-password"
                  trailing={
                    <button type="button" onClick={() => setShowPw(v => !v)} className="auth-ghost-btn"
                      aria-label={showPw ? "Hide password" : "Show password"}
                      style={{ background: "none", border: "none", color: T.muted, cursor: "pointer",
                        padding: 4, display: "grid", placeItems: "center", flexShrink: 0 }}>
                      {showPw ? <IcEyeOff s={17} /> : <IcEye s={17} />}
                    </button>
                  } />
                <Field label="Confirm Password" icon={<IcLock s={17} />}
                  type={showPw ? "text" : "password"} value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)} placeholder="••••••••" autoComplete="new-password" />
                <button type="submit" disabled={loading} className="auth-btn-primary" style={btnPrimaryStyle}>
                  {loading ? "Saving…" : "Set New Password"}
                  {!loading && <IcArrow s={15} />}
                </button>
              </form>
            )}

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12,
              marginTop: 40, fontFamily: T.mono, fontSize: 11, color: T.muted }}>
              <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                <IcLock s={13} /> Secure sign-in
              </span>
              <span>© 2026 ClauseIQ</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
