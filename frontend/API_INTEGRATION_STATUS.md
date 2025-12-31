# Financial Seismograph Dashboard - API Integration

## 🎯 Integration Status

### ✅ Completed Components
- **API Service Layer** (`src/services/api.ts`)
  - Full integration with Financial Seismograph backend
  - Type-safe API client with error handling
  - Data transformation utilities

- **Custom Hook** (`src/hooks/useFinancialAPI.ts`)
  - Real-time data management
  - Health monitoring
  - Automatic reconnection

- **Dashboard Updates** (`src/components/Dashboard.tsx`)
  - Connected to live API data
  - Fallback to mock data when API offline
  - Real-time status indicators

- **New Components**:
  - **NewsTicker**: Live RSS feed articles from `/api/articles`
  - **SystemStatus**: Real-time system health from `/api/system/stats`

### 🔗 API Endpoints Integrated

| Endpoint | Component | Status | Fallback |
|----------|-----------|--------|----------|
| `/api/seismograph/data` | SeismographChart | ✅ Connected | Mock data |
| `/api/tremors` | Dashboard stats | ✅ Connected | Mock data |
| `/api/epicenter/{id}` | EpicenterPanel | ✅ Connected | Mock analysis |
| `/api/query` | CommandQuery | ✅ Connected | Offline message |
| `/api/articles` | NewsTicker | ✅ Connected | Hidden when offline |
| `/api/system/health` | Header indicator | ✅ Connected | Shows offline |
| `/api/system/stats` | SystemStatus | ✅ Connected | Shows disconnected |

## 🚀 How to Test

### 1. Start Your Backend
```bash
# In your Financial Seismograph project directory
python production_startup.py
```

### 2. Start This Dashboard
```bash
# In this project directory
npm run dev
```

### 3. Test Scenarios

**API Connected (localhost:8000 running):**
- ✅ Green "API Connected" indicator in header
- ✅ Real seismograph data from backend
- ✅ Live news articles in sidebar
- ✅ System status shows all components
- ✅ AI queries work with actual responses
- ✅ Tremor analysis shows real pipeline traces

**API Offline:**
- 🔴 Red "API Offline" indicator with retry button
- 📊 Falls back to mock seismograph data
- 🎯 Mock tremor analysis for demonstration
- 📰 News ticker hidden
- ⚙️ System status shows disconnected
- 🤖 AI queries show offline message

## 🎨 UI/UX Features

### Real-time Updates
- **30-second intervals**: Seismograph and tremor data
- **5-minute intervals**: News articles
- **1-minute intervals**: System statistics

### Visual Indicators
- 🟢 Green WiFi icon: API healthy
- 🔴 Red WiFi off icon: API offline
- ⚠️ Warning icon: API errors
- 🔄 Spinner: Loading states

### Responsive Design
- Desktop: Full layout with sidebar
- Tablet: Stacked components
- Mobile: Single column layout

## 🔧 Configuration

### API Base URL
Default: `http://localhost:8000`
Change in: `src/services/api.ts`

### Update Intervals
Configure in: `src/hooks/useFinancialAPI.ts`
- Seismograph: 30s
- Health checks: On connection
- News: 5min
- Stats: 1min

### Mock Data Fallback
When API offline, dashboard uses:
- Generated seismograph patterns
- Demo tremor analysis
- Simulated real-time updates

## 📊 Data Flow

```
Backend API (localhost:8000)
    ↓
API Service Layer (api.ts)
    ↓
Financial Hook (useFinancialAPI.ts)
    ↓
Dashboard Components
    ↓
Real-time UI Updates
```

## 🎯 Next Steps

1. **Start your backend**: Run `python production_startup.py`
2. **Test live data**: Watch seismograph update with real market data
3. **Test AI queries**: Use command bar for market analysis
4. **Monitor status**: Check system health in sidebar
5. **Review news**: See live financial articles

## 🐛 Troubleshooting

**"API Offline" showing?**
- Check backend is running on localhost:8000
- Click "Retry" button to reconnect
- Check browser console for CORS issues

**No data showing?**
- Ensure RSS feeds are active (6+ feeds expected)
- Check backend logs for processing status
- Verify Ollama models are running

**Charts not updating?**
- Check `/api/seismograph/data` returns data
- Verify tremor detection is working
- Monitor browser network tab for API calls

---

**🌊 Your Financial Seismograph Dashboard is ready for real-time market analysis!**