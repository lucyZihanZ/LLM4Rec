import React, { useState } from "react";

// Point to FastAPI backend (default: localhost:8000)
const baseUrl = "http://localhost:8000";

function RecommendApp() {
  // React state hooks
  const [query, setQuery] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Form submit handler
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) {
      setError("Please enter a query.");
      return;
    }
    setLoading(true);
    setError("");
    setRecommendations([]);
    try {
      const response = await fetch(
        `${baseUrl}/recommend?query=${encodeURIComponent(query)}`
      );
      if (!response.ok) throw new Error("API request failed");
      const data = await response.json();
      setRecommendations(data.recommendations);
      if (data.recommendations.length === 0) {
        setError("No recommendations found. Try another query!");
      }
    } catch (err) {
      setError("Recommendation failed, please check your backend or API status.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: "40px auto", fontFamily: "sans-serif", padding: 24 }}>
      <h2>Product Recommendation Demo</h2>
      <form onSubmit={handleSubmit} style={{ marginBottom: 20 }}>
        <label>
          <b>Enter your needs or question:</b>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            style={{ width: "100%", margin: "8px 0", padding: "6px", fontSize: 16 }}
            placeholder="e.g. energy, protein bar, immunity..."
            autoFocus
            required
          />
        </label>
        <button
          type="submit"
          style={{
            padding: "8px 20px",
            background: "#3477eb",
            color: "white",
            border: "none",
            borderRadius: 4,
            marginTop: 8,
            fontSize: 16,
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.7 : 1,
          }}
          disabled={loading}
        >
          {loading ? "Recommending..." : "Recommend"}
        </button>
      </form>

      {error && <div style={{ color: "red", margin: "10px 0" }}>{error}</div>}

      <div>
        {recommendations.length > 0 && (
          <>
            <h3>Recommended Products</h3>
            <ul style={{ paddingLeft: 0 }}>
              {recommendations.map((item) => (
                <li
                  key={item.id}
                  style={{
                    marginBottom: "18px",
                    borderBottom: "1px solid #eee",
                    paddingBottom: 8,
                    listStyle: "none",
                  }}
                >
                  <strong>{item.name}</strong>
                  <div style={{ fontSize: 15, margin: "2px 0" }}>{item.description}</div>
                  <div style={{ fontSize: 14, color: "#2c5877" }}>
                    <em>RAG Augmented:</em> {item.augmented}
                  </div>
                  {item.tags && (
                    <div style={{ fontSize: 13, color: "#888", marginTop: 2 }}>
                      <b>Tags:</b> {item.tags.join(", ")}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}

export default RecommendApp;
