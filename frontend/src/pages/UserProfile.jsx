import React, { useState } from 'react';
import { getUserProfile, getUserDrift } from '../services/api';
import './UserProfile.css';

// Phase 10: Unified Recommendation Pipeline
const fetchRecommendationsAPI = async (userId, candParams) => {
  const { candidate_pool_size, split, alpha, apply_mmr, final_k, mmr_lambda, include_explanations } = candParams;
  const params = new URLSearchParams({ candidate_pool_size, split, alpha });
  if (apply_mmr) {
    params.append("apply_mmr", "true");
    if (final_k) params.append("final_k", final_k);
    if (mmr_lambda) params.append("mmr_lambda", mmr_lambda);
  } else {
    params.append("apply_mmr", "false");
    if (final_k) params.append("final_k", final_k);
  }
  if (include_explanations !== undefined) {
    params.append("include_explanations", include_explanations);
  }
  const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
  const res = await fetch(`${BASE_URL}/api/users/mind/${userId}/recommendations?${params}`);
  if (!res.ok) throw new Error(`Error ${res.status}: ${await res.text()}`);
  return res.json();
};

function UserProfile() {
  const [userId, setUserId] = useState('');
  const [profile, setProfile] = useState(null);
  const [drift, setDrift] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [candidates, setCandidates] = useState(null);
  const [candLoading, setCandLoading] = useState(false);
  const [candError, setCandError] = useState(null);
  const [candParams, setCandParams] = useState({ candidate_pool_size: 50, final_k: 10, split: 'all', alpha: 0.8, apply_mmr: true, mmr_lambda: 0.7, include_explanations: true });

  // Phase 9: per-article feedback state { mind_id: 'recorded' | 'error' | null }
  const [feedbackStates, setFeedbackStates] = useState({});

  const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

  const submitFeedback = async (mindId, eventType) => {
    setFeedbackStates(prev => ({ ...prev, [mindId]: 'loading' }));
    try {
      const res = await fetch(`${BASE_URL}/api/users/mind/${userId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mind_id: mindId, event_type: eventType, source: 'recommendation' }),
      });
      if (!res.ok) throw new Error(await res.text());
      setFeedbackStates(prev => ({ ...prev, [mindId]: `OK ${eventType}` }));
    } catch (err) {
      setFeedbackStates(prev => ({ ...prev, [mindId]: 'error' }));
    }
  };

  const fetchProfile = async (e) => {
    e.preventDefault();
    if (!userId) return;
    
    setLoading(true);
    setError(null);
    setProfile(null);
    setDrift(null);
    setCandidates(null);
    setCandError(null);
    
    try {
      const data = await getUserProfile(userId);
      setProfile(data);
      try {
          const driftData = await getUserDrift(userId);
          setDrift(driftData);
      } catch (driftErr) {
          console.warn("Drift data not available:", driftErr);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadCandidates = async () => {
    if (!userId) return;
    setCandLoading(true);
    setCandError(null);
    try {
      const data = await fetchRecommendationsAPI(userId, candParams);
      setCandidates(data);
    } catch (err) {
      setCandError(err.message);
    } finally {
      setCandLoading(false);
    }
  };

  const renderInterests = (title, profileData) => {
    if (!profileData) return <div className="profile-card"><p>No {title} data available.</p></div>;
    
    const { categories, subcategories, updated_at } = profileData;
    
    return (
      <div className="profile-card">
        <h3>{title} Interests</h3>
        <span className="updated-time">Updated: {new Date(updated_at).toLocaleString()}</span>
        
        <div className="features-grid">
          <div className="feature-column">
            <h4>Top Categories</h4>
            <ul>
              {Object.entries(categories || {})
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([cat, weight]) => (
                  <li key={cat}>
                    <span className="feature-name">{cat}</span>
                    <span className="feature-weight">{weight.toFixed(4)}</span>
                  </li>
                ))}
            </ul>
          </div>
          
          <div className="feature-column">
            <h4>Top Subcategories</h4>
            <ul>
              {Object.entries(subcategories || {})
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([cat, weight]) => (
                  <li key={cat}>
                    <span className="feature-name">{cat}</span>
                    <span className="feature-weight">{weight.toFixed(4)}</span>
                  </li>
                ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  const renderDrift = () => {
    if (!drift) return null;
    
    return (
      <div className="drift-card">
        <h3>INTEREST DRIFT (Phase 4)</h3>
        <div className="drift-metrics">
          <div className="metric-box">
            <h4>JSD Score</h4>
            <span className="metric-value">{drift.jsd_score.toFixed(4)}</span>
          </div>
          <div className="metric-box">
            <h4>Evidence Level</h4>
            <span className={`metric-value evidence-${drift.evidence_level.toLowerCase()}`}>{drift.evidence_level}</span>
          </div>
        </div>
        
        <div className="drift-evidence-stats">
           <p><strong>Total Historical Clicks:</strong> {drift.evidence.historical_clicks}</p>
           <p><strong>Timestamped Clicks:</strong> {drift.evidence.timestamped_clicks}</p>
           <p><strong>Recent Timestamped:</strong> {drift.evidence.recent_timestamped_clicks}</p>
        </div>

        <div className="drift-lists">
          <div className="drift-column emerging">
            <h4>Emerging</h4>
            <ul>
              {drift.emerging.slice(0, 5).map((d, i) => (
                <li key={i}>{d.feature} <span>(Δ +{d.delta.toFixed(3)})</span></li>
              ))}
            </ul>
          </div>
          <div className="drift-column declining">
            <h4>Declining</h4>
            <ul>
              {drift.declining.slice(0, 5).map((d, i) => (
                <li key={i}>{d.feature} <span>(Δ {d.delta.toFixed(3)})</span></li>
              ))}
            </ul>
          </div>
          <div className="drift-column stable">
            <h4>Stable</h4>
            <ul>
              {drift.stable.slice(0, 5).map((d, i) => (
                <li key={i}>{d.feature}</li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    );
  };

  const renderCandidates = () => {
    return (
      <div className="candidates-card">
        <h3>RECOMMENDATIONS (Phase 10 Pipeline)</h3>
        
        <div className="cand-controls">
          <label>
            Candidates Pool:
            <input type="number" min="1" max="500" value={candParams.candidate_pool_size} onChange={e => setCandParams({...candParams, candidate_pool_size: parseInt(e.target.value) || 50})} />
          </label>
          <label>
            Final K:
            <input type="number" min="1" max="100" value={candParams.final_k || 10} onChange={e => setCandParams({...candParams, final_k: parseInt(e.target.value) || 10})} />
          </label>
          <label>
            Split:
            <select value={candParams.split} onChange={e => setCandParams({...candParams, split: e.target.value})}>
              <option value="all">All</option>
              <option value="train">Train</option>
              <option value="dev">Dev</option>
              <option value="test">Test</option>
            </select>
          </label>
          <label className="checkbox-label">
            <input type="checkbox" checked={candParams.apply_mmr} onChange={e => setCandParams({...candParams, apply_mmr: e.target.checked})} />
            Apply MMR Re-ranking
          </label>
          {candParams.apply_mmr && (
            <label>
              MMR Lambda (Relevance Wt):
              <input type="number" step="0.1" min="0" max="1" value={candParams.mmr_lambda || 0.7} onChange={e => setCandParams({...candParams, mmr_lambda: parseFloat(e.target.value) || 0.7})} />
            </label>
          )}
          <label className="checkbox-label">
            <input type="checkbox" checked={candParams.include_explanations} onChange={e => setCandParams({...candParams, include_explanations: e.target.checked})} />
            Include Explanations
          </label>
          <button onClick={loadCandidates} disabled={candLoading}>
            {candLoading ? 'Generating...' : 'Get Recommendations'}
          </button>
        </div>

        {candError && <div className="error-message">{candError}</div>}

        {candidates && (
          <div className="candidates-results">
            <div className="cand-meta">
              <p><strong>Pipeline Status:</strong> OK (Latency: {candidates.diagnostics.latency_ms}ms)</p>
              <p><strong>Profiles:</strong> LT={candidates.profile.long_term_available ? 'Yes' : 'No'}, RT={candidates.profile.recent_available ? 'Yes' : 'No'}, JSD={candidates.drift.jsd.toFixed(4)}</p>
            </div>

            {candidates.recommendations && candidates.recommendations.length > 0 && (
              <table className="cand-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Relevance</th>
                    {candidates.diagnostics.mmr_applied && <th>MMR Score</th>}
                    <th>Title</th>
                    <th>Category</th>
                    {candidates.diagnostics.explanations_generated && <th>Explanation</th>}
                    <th>Feedback</th>
                  </tr>
                </thead>
                <tbody>
                  {candidates.recommendations.map(c => (
                    <tr key={c.mind_id}>
                      <td>{c.rank}</td>
                      <td className="cand-sim">{c.relevance_score.toFixed(4)}</td>
                      {candidates.diagnostics.mmr_applied && <td>{c.mmr_score.toFixed(4)}</td>}
                      <td>{c.title}</td>
                      <td>{c.category} / {c.subcategory}</td>
                      {c.explanation && (
                        <td>
                          <div style={{fontSize: '0.9em', padding: '5px', backgroundColor: '#2a2a2a', borderRadius: '4px', borderLeft: '3px solid #646cff'}}>
                            <div style={{color: '#999', fontSize: '0.85em', textTransform: 'uppercase', marginBottom: '4px'}}>Why this article?</div>
                            <div style={{color: '#e0e0e0', fontStyle: 'italic', marginBottom: '4px'}}>"{c.explanation.primary_reason}"</div>
                            {c.explanation.supporting_reasons?.length > 0 && (
                              <div style={{display: 'flex', gap: '4px', flexWrap: 'wrap'}}>
                                {c.explanation.supporting_reasons.map(r => (
                                  <span key={r} style={{fontSize: '0.75em', padding: '2px 6px', backgroundColor: '#333', borderRadius: '8px', color: '#aaa'}}>{r}</span>
                                ))}
                              </div>
                            )}
                          </div>
                        </td>
                      )}
                      <td>
                        {feedbackStates[c.mind_id] && feedbackStates[c.mind_id] !== 'loading'
                          ? <span style={{color: feedbackStates[c.mind_id].startsWith('✓') ? '#4caf50' : '#f44336', fontSize: '0.85em'}}>{feedbackStates[c.mind_id]}</span>
                          : <span style={{display: 'flex', gap: '4px'}}>
                              <button
                                onClick={() => submitFeedback(c.mind_id, 'LIKE')}
                                disabled={feedbackStates[c.mind_id] === 'loading'}
                                style={{padding: '2px 8px', background: '#1e3a1e', border: '1px solid #4caf50', color: '#4caf50', borderRadius: '4px', cursor: 'pointer', fontSize: '0.85em'}}
                                title="Like this article"
                              >👍 Like</button>
                              <button
                                onClick={() => submitFeedback(c.mind_id, 'DISLIKE')}
                                disabled={feedbackStates[c.mind_id] === 'loading'}
                                style={{padding: '2px 8px', background: '#3a1e1e', border: '1px solid #f44336', color: '#f44336', borderRadius: '4px', cursor: 'pointer', fontSize: '0.85em'}}
                                title="Dislike this article"
                              >👎 Dislike</button>
                            </span>
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {candidates.recommendations && candidates.recommendations.length === 0 && (
              <p>No candidates found matching the criteria.</p>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="user-profile-page">
      <header className="page-header">
        <h1>User Interest Profiling</h1>
        <p>Inspect the generated long-term and recent interest profiles for any MIND user.</p>
      </header>

      <div className="search-section">
        <form onSubmit={fetchProfile} className="search-form">
          <input 
            type="text" 
            placeholder="Enter MIND User ID (e.g., U536528)" 
            value={userId}
            onChange={(e) => setUserId(e.target.value.trim())}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Building / Loading...' : 'Inspect Profile'}
          </button>
        </form>
        {error && <div className="error-message">{error}</div>}
      </div>

      {profile && (
        <div className="profiles-container">
          {renderDrift()}
          {renderCandidates()}
          {renderInterests('Long-Term', profile.long_term)}
          {renderInterests('Recent', profile.recent)}
          
          <div className="profile-info-panel">
            <h4>Profile Construction Rules (Phase 3 & 6)</h4>
            <ul>
              <li><strong>Positive Signal:</strong> Clicked articles only (`clicked=1`).</li>
              <li><strong>Temporal Weighting:</strong> Exponential decay (`exp(-lambda * age_days)`).</li>
              <li><strong>Long-Term Profile:</strong> Built from complete chronological history (`history` + `impressions`). Lambda = 0.01.</li>
              <li><strong>Recent Profile:</strong> Built from the most recent 20 clicks. Lambda = 0.1.</li>
              <li><strong>Retrieval Strategy:</strong> Sparse TF-IDF Cosine Similarity over user vectors and 130K documents. Excludes clicked articles.</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default UserProfile;
