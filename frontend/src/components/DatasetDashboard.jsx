/**
 * DatasetDashboard — Phase 2 admin section
 *
 * Shows real dataset statistics fetched live from the backend.
 * Includes an ingestion panel for triggering MIND data loading.
 */

import { useState } from "react";
import { useDatasetStats } from "../hooks/useDatasetStats";
import { triggerIngestion } from "../services/api";
import "./DatasetDashboard.css";

// ── Stat card ────────────────────────────────────────────────────────────────

function StatCard({ icon, label, value, sub }) {
  return (
    <div className="stat-card">
      <div className="stat-card__icon">{icon}</div>
      <div className="stat-card__body">
        <div className="stat-card__value">{value}</div>
        <div className="stat-card__label">{label}</div>
        {sub && <div className="stat-card__sub">{sub}</div>}
      </div>
    </div>
  );
}

// ── Category bar ─────────────────────────────────────────────────────────────

function CategoryBar({ categories }) {
  if (!categories || Object.keys(categories).length === 0) return null;
  const total = Object.values(categories).reduce((a, b) => a + b, 0);
  const sorted = Object.entries(categories).sort((a, b) => b[1] - a[1]);

  return (
    <div className="category-bar">
      <h3 className="dashboard__section-title">Articles by Category</h3>
      <div className="category-bar__list">
        {sorted.map(([cat, count]) => (
          <div key={cat} className="category-row">
            <span className="category-row__name">{cat}</span>
            <div className="category-row__track">
              <div
                className="category-row__fill"
                style={{ width: `${(count / total) * 100}%` }}
              />
            </div>
            <span className="category-row__count">{count.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Ingestion panel ───────────────────────────────────────────────────────────

function IngestionPanel({ onIngested }) {
  const [splitDir, setSplitDir] = useState("");
  const [split, setSplit]       = useState("train");
  const [status, setStatus]     = useState(null);  // null | "loading" | "ok" | "error"
  const [message, setMessage]   = useState("");

  async function handleIngest(e) {
    e.preventDefault();
    if (!splitDir.trim()) return;
    setStatus("loading");
    setMessage("");
    try {
      const res = await triggerIngestion(splitDir.trim(), split);
      setStatus("ok");
      setMessage(res.message || "Ingestion started.");
      onIngested?.();
    } catch (err) {
      setStatus("error");
      setMessage(err.message);
    }
  }

  return (
    <div className="ingest-panel">
      <h3 className="dashboard__section-title">Trigger MIND Ingestion</h3>
      <p className="ingest-panel__hint">
        Point to the directory containing <code>news.tsv</code> and{" "}
        <code>behaviors.tsv</code>, or a <code>.zip</code> file.
        <br />
        Example: <code>../data/raw/mind/large/train</code>
      </p>
      <form className="ingest-panel__form" onSubmit={handleIngest}>
        <input
          id="ingest-split-dir"
          className="ingest-panel__input"
          type="text"
          placeholder="Path to MIND split directory or zip…"
          value={splitDir}
          onChange={(e) => setSplitDir(e.target.value)}
        />
        <select
          id="ingest-split-select"
          className="ingest-panel__select"
          value={split}
          onChange={(e) => setSplit(e.target.value)}
        >
          <option value="train">train</option>
          <option value="dev">dev</option>
          <option value="test">test</option>
        </select>
        <button
          id="ingest-submit-btn"
          className="ingest-panel__btn"
          type="submit"
          disabled={status === "loading"}
        >
          {status === "loading" ? "⏳ Starting…" : "▶ Ingest"}
        </button>
      </form>
      {message && (
        <div className={`ingest-panel__result ingest-panel__result--${status}`}>
          {status === "ok" ? "✓" : "⚠"} {message}
        </div>
      )}
    </div>
  );
}

// ── Main dashboard ────────────────────────────────────────────────────────────

export function DatasetDashboard() {
  const { stats, loading, error, refresh } = useDatasetStats();

  const fmt = (n) => (typeof n === "number" ? n.toLocaleString() : "—");

  return (
    <section className="dashboard" id="dataset-dashboard">
      <div className="dashboard__header">
        <div>
          <h2 className="dashboard__title">MIND Dataset Status</h2>
          <p className="dashboard__subtitle">
            Microsoft News Dataset · Real statistics from database
          </p>
        </div>
        <button
          id="dataset-refresh-btn"
          className="dashboard__refresh-btn"
          onClick={refresh}
          disabled={loading}
        >
          {loading ? "⏳" : "↻"} Refresh
        </button>
      </div>

      {error && (
        <div className="dashboard__error">
          ⚠ Could not fetch statistics: {error}
        </div>
      )}

      {loading && !stats && (
        <div className="dashboard__loading">Loading statistics from database…</div>
      )}

      {stats && (
        <>
          {/* ── Top-line stats ─────────────────────────────────────────── */}
          <div className="dashboard__stat-grid">
            <StatCard
              icon="📰"
              label="Total Articles"
              value={fmt(stats.total_articles)}
              sub={`${stats.unique_categories} categories`}
            />
            <StatCard
              icon="👥"
              label="Total Users"
              value={fmt(stats.total_users)}
            />
            <StatCard
              icon="📋"
              label="Behavior Logs"
              value={fmt(stats.total_behavior_logs)}
              sub={`${fmt(stats.total_impression_items)} impression items`}
            />
            <StatCard
              icon="👆"
              label="Total Clicks"
              value={fmt(stats.total_clicks)}
              sub={`CTR: ${(stats.click_through_rate * 100).toFixed(1)}%`}
            />
            <StatCard
              icon="📝"
              label="With Abstract"
              value={fmt(stats.articles_with_abstract)}
              sub={`${fmt(stats.articles_without_abstract)} missing`}
            />
            <StatCard
              icon="📅"
              label="Behavior Range"
              value={stats.earliest_behavior
                ? new Date(stats.earliest_behavior).toLocaleDateString()
                : "—"}
              sub={stats.latest_behavior
                ? `→ ${new Date(stats.latest_behavior).toLocaleDateString()}`
                : ""}
            />
          </div>

          {/* ── Split breakdown ────────────────────────────────────────── */}
          <div className="dashboard__split-section">
            <h3 className="dashboard__section-title">Articles per Split</h3>
            <div className="split-pills">
              {Object.entries(stats.articles_per_split || {}).map(([sp, cnt]) => (
                <div key={sp} className={`split-pill split-pill--${sp}`}>
                  <span className="split-pill__name">{sp}</span>
                  <span className="split-pill__count">{fmt(cnt)}</span>
                </div>
              ))}
            </div>

            <h3 className="dashboard__section-title" style={{ marginTop: "16px" }}>Behavior Logs per Split</h3>
            <div className="split-pills">
              {Object.entries(stats.behavior_logs_per_split || {}).map(([sp, cnt]) => (
                <div key={sp} className={`split-pill split-pill--${sp}`}>
                  <span className="split-pill__name">{sp}</span>
                  <span className="split-pill__count">{fmt(cnt)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── Category breakdown ─────────────────────────────────────── */}
          <CategoryBar categories={stats.articles_per_category} />

          {/* ── Advanced metrics ───────────────────────────────────────── */}
          <div className="dashboard__meta-grid">
            <div className="meta-item">
              <span className="meta-item__label">Avg impressions/log</span>
              <span className="meta-item__value">{stats.avg_impressions_per_log}</span>
            </div>
            <div className="meta-item">
              <span className="meta-item__label">Avg history length</span>
              <span className="meta-item__value">{stats.avg_history_length}</span>
            </div>
            <div className="meta-item">
              <span className="meta-item__label">Orphan impression refs</span>
              <span className="meta-item__value">{fmt(stats.orphan_impression_article_ids)}</span>
            </div>
            <div className="meta-item">
              <span className="meta-item__label">Last ingestion</span>
              <span className={`meta-item__value meta-item__value--${stats.last_ingestion_status}`}>
                {stats.last_ingestion_status || "never"}
              </span>
            </div>
          </div>

          {/* ── Ingestion panel ────────────────────────────────────────── */}
          <IngestionPanel onIngested={refresh} />

          <p className="dashboard__timestamp">
            Statistics computed at: {new Date(stats.generated_at).toLocaleString()}
          </p>
        </>
      )}

      {stats?.total_articles === 0 && !loading && (
        <div className="dashboard__empty">
          <p>📂 No articles ingested yet.</p>
          <p>
            Download MINDlarge_train.zip and MINDlarge_dev.zip from{" "}
            <a href="https://msnews.github.io" target="_blank" rel="noreferrer">
              msnews.github.io
            </a>
            , extract them into <code>data/raw/mind/large/</code>, then use the ingestion panel above.
          </p>
        </div>
      )}
    </section>
  );
}
