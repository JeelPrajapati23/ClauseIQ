import { useState, useEffect } from "react";
import Chat from "./chat.jsx";
import Auth from "./components/Auth.jsx";
import { API_BASE_URL } from "./utils/api.js";
import clauseiqMark from "./assets/clauseiq-mark.svg";

function LoadingScreen() {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "#0a0b0d",
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <img src={clauseiqMark} alt="" width={40} height={40}
        style={{ borderRadius: 10, animation: "pulse 1.4s ease-in-out infinite" }} />
      <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }`}</style>
    </div>
  );
}

export default function App() {
  // null = still loading, false = not authenticated, object = authenticated user
  const [authUser, setAuthUser] = useState(null);
  const [authNotice, setAuthNotice] = useState("");
  const [resetToken] = useState(() => {
    const p = new URLSearchParams(window.location.search);
    return p.get("reset_token") || null;
  });

  useEffect(() => {
    fetch(`${API_BASE_URL}/auth/me`, { credentials: "include" })
      .then((r) => (r.ok ? r.json() : null))
      .then((user) => setAuthUser(user || false))
      .catch(() => setAuthUser(false));
  }, []);

  const handleLogin = (user) => { setAuthNotice(""); setAuthUser(user); };
  const handleLogout = () => { setAuthUser(false); };
  // Distinct from a normal sign-out: the session died mid-use (expired cookie,
  // 401 from an API call), so the login screen explains why the user is back here.
  const handleSessionExpired = () => {
    setAuthUser(false);
    setAuthNotice("Your session has expired. Please sign in again.");
  };

  // If a reset_token is in the URL, always show Auth in reset mode
  if (resetToken) return <Auth onLogin={handleLogin} initialResetToken={resetToken} />;
  if (authUser === null) return <LoadingScreen />;
  if (!authUser) return <Auth onLogin={handleLogin} initialNotice={authNotice} />;
  return <Chat authUser={authUser} onLogout={handleLogout} onSessionExpired={handleSessionExpired} />;
}
