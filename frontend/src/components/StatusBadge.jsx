/**
 * StatusBadge — shows the current backend connection status
 * with an animated indicator dot.
 */

import PropTypes from "prop-types";
import "./StatusBadge.css";

const STATUS_LABELS = {
  loading: "Connecting…",
  connected: "Backend Connected",
  error: "Backend Unreachable",
};

export function StatusBadge({ status, detail, error }) {
  const label = STATUS_LABELS[status] ?? "Unknown";

  return (
    <div className={`status-badge status-badge--${status}`}>
      <span className="status-badge__dot" aria-hidden="true" />
      <span className="status-badge__label">{label}</span>
      {status === "connected" && detail && (
        <span className="status-badge__meta">
          v{detail.service?.split("-").pop() || "?"} · {new Date(detail.timestamp).toLocaleTimeString()}
        </span>
      )}
      {status === "error" && error && (
        <span className="status-badge__error" title={error}>
          ⚠ {error.slice(0, 60)}{error.length > 60 ? "…" : ""}
        </span>
      )}
    </div>
  );
}

StatusBadge.propTypes = {
  status: PropTypes.oneOf(["loading", "connected", "error"]).isRequired,
  detail: PropTypes.object,
  error: PropTypes.string,
};
