/**
 * useBackendStatus — React hook
 *
 * Polls the /api/health endpoint once on mount and returns
 * the connection status to the rest of the app.
 */

import { useState, useEffect } from "react";
import { getHealth } from "../services/api";

/**
 * @returns {{ status: "idle"|"loading"|"connected"|"error", detail: object|null, error: string|null }}
 */
export function useBackendStatus() {
  const [status, setStatus] = useState("loading");
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function check() {
      try {
        const data = await getHealth();
        if (!cancelled) {
          setDetail(data);
          setStatus("connected");
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          setStatus("error");
        }
      }
    }

    check();
    return () => {
      cancelled = true;
    };
  }, []);

  return { status, detail, error };
}
