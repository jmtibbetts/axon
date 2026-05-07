import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import Dashboard from './Dashboard.tsx'

// Simple path-based routing — no react-router needed
const path = window.location.pathname;
const isDashboard = path === '/dashboard' || path.startsWith('/dashboard/');

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    {isDashboard ? <Dashboard /> : <App />}
  </StrictMode>,
)
