import { IcSpin, IcAlert, IcDoc, IcClose } from "./icons.jsx";

export function TypingDots() {
  return (
    <div style={{ display: "flex", gap: 5, alignItems: "center", padding: "4px 0" }}>
      {[0, 1, 2].map((i) => (
        <span key={i} style={{ width: 7, height: 7, borderRadius: "50%", background: "#7c828c",
          animation: `bf-bounce 1.2s ${i * 0.16}s infinite ease-in-out`, display: "block" }} />
      ))}
    </div>
  );
}

export function FileChip({ name, size, status, errorMessage, onRemove, onRetry }) {
  const isUploading = status === "uploading" || status === "pending" || status === "processing";
  const isError = status === "error";
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6, borderRadius: 8,
      padding: "3px 8px", border: `1px solid ${isError ? "#5c2b2e" : "#2c313a"}`,
      background: isError ? "rgba(92,43,46,0.3)" : "#15181d",
      fontFamily: "'IBM Plex Mono', monospace", fontSize: 11,
      color: isError ? "#ff8a80" : "#b6bbc2", maxWidth: 200,
    }}>
      {isUploading ? <IcSpin s={12} /> : isError ? <IcAlert s={12} /> : <IcDoc s={12} />}
      <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
        title={isError && errorMessage ? errorMessage : name}>{name}</span>
      {size && !isError && <span style={{ opacity: 0.6, flexShrink: 0, fontSize: 10 }}>{size}</span>}
      {isError && onRetry && (
        <button onClick={onRetry} style={{ fontSize: 10, textDecoration: "underline", color: "#ff8a80", flexShrink: 0, background: "none", border: "none", cursor: "pointer" }}>retry</button>
      )}
      {onRemove && (
        <button onClick={onRemove} style={{ display: "flex", flexShrink: 0, opacity: 0.6, background: "none", border: "none", cursor: "pointer", color: "inherit" }}
          onMouseEnter={e => e.currentTarget.style.opacity = "1"} onMouseLeave={e => e.currentTarget.style.opacity = "0.6"}>
          <IcClose s={10} />
        </button>
      )}
    </div>
  );
}
