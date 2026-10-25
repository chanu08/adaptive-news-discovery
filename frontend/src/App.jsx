import { useState } from "react";
import { Home } from "./pages/Home";
import { Dataset } from "./pages/Dataset";
import UserProfile from "./pages/UserProfile";
import ArticleRepresentation from "./pages/ArticleRepresentation";
import "./App.css";

const PAGES = {
  home: Home,
  dataset: Dataset,
  profile: UserProfile,
  article_rep: ArticleRepresentation,
};

function App() {
  const [page, setPage] = useState("home");
  const PageComponent = PAGES[page] || Home;

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header__inner">
          <div className="app-header__brand">
            <span className="app-header__logo">🔍</span>
            <span className="app-header__name">NewsIR</span>
          </div>
          <nav className="app-header__nav">
            <button
              id="nav-home-btn"
              className={`app-header__link ${page === "home" ? "app-header__link--active" : ""}`}
              onClick={() => setPage("home")}
            >
              Home
            </button>
            <button
              id="nav-dataset-btn"
              className={`app-header__link ${page === "dataset" ? "app-header__link--active" : ""}`}
              onClick={() => setPage("dataset")}
            >
              Dataset
            </button>
            <button
              id="nav-profile-btn"
              className={`app-header__link ${page === "profile" ? "app-header__link--active" : ""}`}
              onClick={() => setPage("profile")}
            >
              User Profiles
            </button>
            <button
              className={`app-header__link ${page === "article_rep" ? "app-header__link--active" : ""}`}
              onClick={() => setPage("article_rep")}
            >
              Article Rep
            </button>
            <span className="app-header__phase">Phase 5 · IR</span>
          </nav>
        </div>
      </header>

      <PageComponent />

      <footer className="app-footer">
        <p>
          Adaptive News Discovery · Information Retrieval Mini-Project ·{" "}
          <span className="app-footer__phase">Phase 3</span>
        </p>
      </footer>
    </div>
  );
}

export default App;
