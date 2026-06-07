import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { RequireAuth } from "./components/RequireAuth";
import { ActorPage } from "./pages/ActorPage";
import { AdminPage } from "./pages/AdminPage";
import { ChatPage } from "./pages/ChatPage";
import { DiscoverPage } from "./pages/DiscoverPage";
import { HistoryPage } from "./pages/HistoryPage";
import { HomePage } from "./pages/HomePage";
import { NotFoundPage } from "./pages/NotFoundPage";
import { RecommendPage } from "./pages/RecommendPage";
import { SimilarPage } from "./pages/SimilarPage";
import { TasteProfilePage } from "./pages/TasteProfilePage";
import { WillLikePage } from "./pages/WillLikePage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<HomePage />} />
        <Route element={<RequireAuth />}>
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/recommend" element={<RecommendPage />} />
          <Route path="/will-like" element={<WillLikePage />} />
          <Route path="/similar" element={<SimilarPage />} />
          <Route path="/discover" element={<DiscoverPage />} />
          <Route path="/actor" element={<ActorPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/taste" element={<TasteProfilePage />} />
          <Route path="/admin" element={<AdminPage />} />
        </Route>
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
