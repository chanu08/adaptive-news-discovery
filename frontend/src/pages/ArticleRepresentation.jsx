import React, { useState } from 'react';
import { getArticleRepresentation } from '../services/api';
import './ArticleRepresentation.css';

function ArticleRepresentation() {
  const [mindId, setMindId] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchRepresentation = async (e) => {
    e.preventDefault();
    if (!mindId) return;
    
    setLoading(true);
    setError(null);
    setData(null);
    
    try {
      const result = await getArticleRepresentation(mindId);
      setData(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="article-rep-page">
      <header className="page-header">
        <h1>IR Article Representation (Phase 5)</h1>
        <p>Inspect the TF-IDF and NMF topic representation for any canonical MIND article.</p>
      </header>

      <div className="search-section">
        <form onSubmit={fetchRepresentation} className="search-form">
          <input 
            type="text" 
            placeholder="Enter MIND Article ID (e.g., N55528)" 
            value={mindId}
            onChange={(e) => setMindId(e.target.value.trim())}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Loading...' : 'Inspect Article'}
          </button>
        </form>
        {error && <div className="error-message">{error}</div>}
      </div>

      {data && (
        <div className="rep-container">
          <div className="rep-card">
            <h3>Metadata</h3>
            <p><strong>MIND ID:</strong> {data.mind_id}</p>
            <p><strong>Title:</strong> {data.title}</p>
            <p><strong>Category:</strong> {data.category} / {data.subcategory}</p>
            <p><strong>Preprocessing Version:</strong> {data.preprocessing_version}</p>
            <p><strong>Indexed At:</strong> {new Date(data.indexed_at).toLocaleString()}</p>
          </div>

          <div className="rep-card">
            <h3>Dominant Topic: {data.dominant_topic}</h3>
            <h4>Top Terms</h4>
            <div className="terms-container">
              {data.top_terms.map((term, i) => (
                <span key={i} className="term-badge">{term}</span>
              ))}
            </div>
            
            <h4>Top 3 Topic Weights</h4>
            <ul className="topic-weights">
              {Object.entries(data.topic_weights)
                .sort((a, b) => b[1] - a[1])
                .map(([topic, weight]) => (
                  <li key={topic}>Topic {topic}: <span>{weight.toFixed(4)}</span></li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

export default ArticleRepresentation;
