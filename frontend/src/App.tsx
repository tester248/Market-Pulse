import { Dashboard } from './components/Dashboard';
import { ErrorBoundary } from './components/ErrorBoundary';
// Import API test functions (available in browser console as window.testMarketPulseAPI)
import './utils/testNewAPI';

function App() {
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-slate-50 dark:bg-slate-900 p-4">
        <Dashboard />
      </div>
    </ErrorBoundary>
  );
}

export default App;