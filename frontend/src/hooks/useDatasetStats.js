/**
 * useDatasetStats — fetches live dataset statistics from /api/dataset/stats
 */

import { useState, useEffect, useCallback } from "react";
import { getDatasetStats } from "../services/api";

export function useDatasetStats() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getDatasetStats();
      setStats(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { stats, loading, error, refresh };
}
