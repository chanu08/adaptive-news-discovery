/**
 * ArchitectureDiagram — visual representation of the IR system pipeline.
 * Phase 1: shows the full planned architecture with "coming soon" badges.
 */

import "./ArchitectureDiagram.css";

const PIPELINE = [
  { id: "news",       label: "News Articles",          icon: "📰", phase: 1, active: true  },
  { id: "preprocess", label: "Text Preprocessing",     icon: "🔤", phase: 2, active: false },
  {
    id: "split", label: "", icon: "", phase: 2, active: false, isSplit: true,
    branches: [
      { id: "tfidf",  label: "TF-IDF Index",      icon: "📊" },
      { id: "topics", label: "Topic Extraction",   icon: "🏷️" },
    ],
  },
  { id: "db",         label: "Article Database",        icon: "🗄️", phase: 2, active: false },
  { id: "ui",         label: "User Interaction",        icon: "👤", phase: 3, active: false },
  { id: "profile",    label: "User Interest Model",     icon: "🧠", phase: 3, active: false },
  { id: "drift",      label: "Interest Drift",          icon: "📈", phase: 4, active: false },
  { id: "retrieval",  label: "Candidate Retrieval",     icon: "🔍", phase: 4, active: false },
  { id: "ranking",    label: "Similarity Ranking",      icon: "⚡", phase: 4, active: false },
  { id: "diversity",  label: "Diversity Re-ranking",    icon: "🎯", phase: 5, active: false },
  { id: "explain",    label: "Explanation Generator",   icon: "💬", phase: 5, active: false },
  { id: "recs",       label: "Final Recommendations",   icon: "✨", phase: 5, active: false },
  { id: "feedback",   label: "User Feedback → Update",  icon: "🔄", phase: 6, active: false },
];

export function ArchitectureDiagram() {
  return (
    <div className="arch" aria-label="IR system architecture pipeline">
      <h2 className="arch__title">System Architecture</h2>
      <p className="arch__subtitle">Information Retrieval Pipeline</p>
      <div className="arch__pipeline">
        {PIPELINE.map((step) => {
          if (step.isSplit) {
            return (
              <div key={step.id} className="arch__split">
                {step.branches.map((b) => (
                  <div key={b.id} className="arch__step arch__step--branch arch__step--upcoming">
                    <span className="arch__icon">{b.icon}</span>
                    <span className="arch__label">{b.label}</span>
                    <span className="arch__phase-badge">Ph. 2</span>
                  </div>
                ))}
              </div>
            );
          }
          return (
            <div
              key={step.id}
              className={`arch__step ${step.active ? "arch__step--active" : "arch__step--upcoming"}`}
            >
              <span className="arch__icon">{step.icon}</span>
              <span className="arch__label">{step.label}</span>
              {!step.active && (
                <span className="arch__phase-badge">Ph. {step.phase}</span>
              )}
              {step.active && (
                <span className="arch__phase-badge arch__phase-badge--current">Phase 1 ✓</span>
              )}
              {step.id !== "feedback" && <div className="arch__arrow">↓</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
