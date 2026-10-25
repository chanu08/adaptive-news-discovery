/**
 * API service — centralised axios-free fetch wrapper.
 * All backend communication goes through this module.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

/**
 * Generic fetch helper with error handling.
 * @param {string} endpoint  - path relative to BASE_URL (e.g. "/api/health")
 * @param {RequestInit} opts - standard fetch options
 */
async function apiFetch(endpoint, opts = {}) {
  const url = `${BASE_URL}${endpoint}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...opts.headers },
    ...opts,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }

  return res.json();
}

// ── Health ─────────────────────────────────────────────────────────────────

/**
 * Check if the backend API is reachable.
 * @returns {Promise<{status: string, timestamp: string, service: string}>}
 */
export async function getHealth() {
  return apiFetch("/api/health");
}

// ── Articles ───────────────────────────────────────────────────────────────

/**
 * Fetch a paginated list of articles.
 * @param {number} skip
 * @param {number} limit
 */
export async function getArticles(skip = 0, limit = 20) {
  return apiFetch(`/api/articles/?skip=${skip}&limit=${limit}`);
}

/**
 * Fetch a single article by ID.
 * @param {number} id
 */
export async function getArticle(id) {
  return apiFetch(`/api/articles/${id}`);
}

export async function getArticleRepresentation(mindId) {
  return apiFetch(`/api/articles/mind/${mindId}/representation`);
}

// ── Search ─────────────────────────────────────────────────────────────────

/**
 * Search articles by query string (placeholder — Phase 2).
 * @param {string} query
 * @param {string|null} category
 * @param {number} limit
 */
export async function searchArticles(query, category = null, limit = 10) {
  const params = new URLSearchParams({ q: query, limit });
  if (category) params.append("category", category);
  return apiFetch(`/api/search/?${params}`);
}

// ── Dataset ────────────────────────────────────────────────────────────────

/**
 * Fetch live dataset statistics from the backend.
 * All values are computed from the real database — never hardcoded.
 */
export async function getDatasetStats() {
  return apiFetch("/api/dataset/stats");
}

/**
 * Trigger MIND ingestion for a split directory.
 * @param {string} splitDir  - path to MIND data directory or zip
 * @param {string} split     - "train" | "dev" | "test"
 */
export async function triggerIngestion(splitDir, split) {
  return apiFetch("/api/dataset/ingest", {
    method: "POST",
    body: JSON.stringify({ split_dir: splitDir, split }),
  });
}

/**
 * Fetch ingestion run history.
 */
export async function getIngestionRuns() {
  return apiFetch("/api/dataset/runs");
}

// ── Users ──────────────────────────────────────────────────────────────────

export async function getUserProfile(mindUserId) {
  return apiFetch(`/api/users/mind/${mindUserId}/profile`);
}

export async function getUserDrift(mindUserId) {
  return apiFetch(`/api/users/mind/${mindUserId}/drift`);
}
