/**
 * Home page — Phase 1 landing / status dashboard.
 */

import { StatusBadge } from "../components/StatusBadge";
import { ArchitectureDiagram } from "../components/ArchitectureDiagram";
import { useBackendStatus } from "../hooks/useBackendStatus";
import "./Home.css";

export function Home() {
  const { status, detail, error } = useBackendStatus();

  return (
    <main className="home">
      {/* ── Hero ──────────────────────────────────────────────── */}
      <section className="home__hero">
        <div className="home__hero-badge">Information Retrieval · Phase 1</div>
        <h1 className="home__title">
          Adaptive News Discovery
        </h1>
        <p className="home__tagline">
          Using Interest Drift &amp; Diversity-Aware IR
        </p>
        <p className="home__description">
          A college-level Information Retrieval project that combines{" "}
          <strong>TF-IDF retrieval</strong>, <strong>user interest modelling</strong>,{" "}
          <strong>interest drift detection</strong>, and{" "}
          <strong>MMR diversity re-ranking</strong> to personalise news recommendations.
        </p>

        <div className="home__status-row">
          <StatusBadge status={status} detail={detail} error={error} />
        </div>

        {status === "connected" && (
          <div className="home__connected-detail">
            <p>
              ✓ Backend API is online at{" "}
              <code>{import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"}</code>
            </p>
            {detail && (
              <p className="home__connected-meta">
                Service: <strong>{detail.service}</strong> · Timestamp:{" "}
                {new Date(detail.timestamp).toLocaleString()}
              </p>
            )}
          </div>
        )}

        {status === "error" && (
          <div className="home__error-detail">
            <p>⚠ Cannot reach the backend. Make sure it is running:</p>
            <pre>cd backend && uvicorn app.main:app --reload</pre>
          </div>
        )}
      </section>

      {/* ── Content ───────────────────────────────────────────── */}
      <section className="home__content">
        <ArchitectureDiagram />

        <div className="home__info-cards">
          <div className="info-card">
            <div className="info-card__icon">🏗️</div>
            <h3>Phase 1 Complete</h3>
            <p>Project foundation, FastAPI backend, React frontend, and SQLite connection layer are all set up.</p>
          </div>
          <div className="info-card">
            <div className="info-card__icon">📊</div>
            <h3>Phase 2 Next</h3>
            <p>TF-IDF index construction, news article ingestion, and the text preprocessing pipeline.</p>
          </div>
          <div className="info-card">
            <div className="info-card__icon">🎯</div>
            <h3>Core IR System</h3>
            <p>Built from scratch — no LLM replacements. Every component implements classical IR algorithms.</p>
          </div>
        </div>
      </section>
    </main>
  );
}
