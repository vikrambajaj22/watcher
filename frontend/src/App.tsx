import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RequireAuth } from "./components/RequireAuth";
import { AdminPage } from "./pages/AdminPage";
import { HistoryPage } from "./pages/HistoryPage";
import { HomePage } from "./pages/HomePage";
import { RecommendPage } from "./pages/RecommendPage";
import { SimilarPage } from "./pages/SimilarPage";
import { VisualPage } from "./pages/VisualPage";
import { WillLikePage } from "./pages/WillLikePage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route element={<RequireAuth />}>
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/visual" element={<VisualPage />} />
          <Route path="/recommend" element={<RecommendPage />} />
          <Route path="/will-like" element={<WillLikePage />} />
          <Route path="/similar" element={<SimilarPage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Route>
      </Route>
    </Routes>
  );
}
