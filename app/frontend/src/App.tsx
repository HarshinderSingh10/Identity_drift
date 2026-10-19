import { Route, Routes } from 'react-router-dom'
import { PageLayout } from './components/layout/PageLayout.tsx'
import Demo from './pages/Demo.tsx'
import Home from './pages/Home.tsx'
import Methodology from './pages/Methodology.tsx'
import NotFound from './pages/NotFound.tsx'
import Research from './pages/Research.tsx'
import Results from './pages/Results.tsx'
import Team from './pages/Team.tsx'

export default function App() {
  return (
    <Routes>
      {/* Every page renders inside the shared layout (navbar, main, footer). */}
      <Route element={<PageLayout />}>
        <Route index element={<Home />} />
        <Route path="research" element={<Research />} />
        <Route path="methodology" element={<Methodology />} />
        <Route path="results" element={<Results />} />
        <Route path="demo" element={<Demo />} />
        <Route path="team" element={<Team />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  )
}
