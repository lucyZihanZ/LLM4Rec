import React, { useState } from "react";

// 注意：请确保 React 项目是用 create-react-app 或 vite 等新建的
//      后端 FastAPI 默认端口为 8000，如有不同请修改 baseUrl

const baseUrl = "http://localhost:8000"; // 后端API地址，生产环境换成你服务器地址

function RecommendApp() {
  const [query, setQuery] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // 处理提交
  const handleSubmit = async (e) => {
    e.preventDefault();
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
    } catch (err) {
      setError("Recommendation failed, please check your backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 500, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h2>Product Recommendation Demo</h2>
      <form onSubmit={handleSubmit}>
        <label>
          Enter your needs or question:
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            style={{ width: "100%", margin: "8px 0", padding: "6px" }}
            placeholder="e.g. energy, protein bar, immunity..."
            required
          />
        </label>
        <button type="submit" style={{ padding: "8px 20px" }} disabled={loading}>
          {loading ? "Recommending..." : "Recommend"}
        </button>
      </form>

      {error && <div style={{ color: "red", margin: "10px 0" }}>{error}</div>}

      <div>
        {recommendations.length > 0 && (
          <>
            <h3>Recommended Products:</h3>
            <ul>
              {recommendations.map((item) => (
                <li key={item.id} style={{ marginBottom: "12px", borderBottom: "1px solid #eee", paddingBottom: 6 }}>
                  <strong>{item.name}</strong><br />
                  <span>{item.description}</span><br />
                  <em style={{ color: "#666" }}>RAG Augmented: {item.augmented}</em>
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
