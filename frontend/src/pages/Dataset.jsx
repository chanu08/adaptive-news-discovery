/**
 * Dataset page — Phase 2
 * Shows the MIND dataset ingestion status and live statistics.
 */

import { DatasetDashboard } from "../components/DatasetDashboard";
import "./Dataset.css";

export function Dataset() {
  return (
    <main className="dataset-page">
      <div className="dataset-page__header">
        <div className="dataset-page__badge">Phase 2 · MIND Dataset</div>
        <h1 className="dataset-page__title">Dataset Administration</h1>
        <p className="dataset-page__desc">
          Inspect and manage the ingested{" "}
          <strong>Microsoft MIND (MINDlarge)</strong> dataset.
          All statistics are computed in real time from the SQLite database.
        </p>
      </div>
      <DatasetDashboard />
    </main>
  );
}
