import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  Shield, 
  Settings, 
  Camera, 
  Zap,
  Navigation,
  Clock
} from 'lucide-react';

// --- Types ---

interface Infraction {
  id: string;
  lane: number;
  timestamp: string;
  plate: string;
  speed: string;
  type: string;
}

interface Lane {
  id: number;
  pcu: number;
  light: string;
  infractions: Infraction[];
  ambulance_detected: boolean;
  accident_detected: boolean;
}

interface AppState {
  police_override: boolean;
  global_speed_limit: number;
  police_detected: boolean;
  active_lane_idx: number;
  mode: "general" | "testing";
  phase: string;
  lanes: Lane[];
}

// --- Sub-components ---

const LaneCard: React.FC<{ lane: Lane; isActive: boolean }> = ({ lane, isActive }) => {
  const pcuPercent = Math.min((lane.pcu / 20) * 100, 100);
  
  return (
    <div className={`lane-card ${isActive ? 'active' : ''}`}>
      <div className="lane-top">
        <span className="lane-name">LANE 0{lane.id + 1}</span>
        <div className="traffic-light">
          <div className={`signal-dot red ${lane.light === 'RED' ? 'on' : ''}`}></div>
          <div className={`signal-dot yellow ${lane.light === 'YELLOW' ? 'on' : ''}`}></div>
          <div className={`signal-dot green ${lane.light === 'GREEN' ? 'on' : ''}`}></div>
        </div>
      </div>
      <div className="metric">
        <span className="label">PCU Density</span>
        <span className="value">{lane.pcu.toFixed(1)}</span>
      </div>
      <div className="pcu-meter">
        <div className="pcu-fill" style={{ width: `${pcuPercent}%` }}></div>
      </div>
    </div>
  );
};

const InfractionItem: React.FC<{ infraction: Infraction }> = ({ infraction }) => (
  <div className="infraction-item">
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      <span className="plate-badge">{infraction.plate}</span>
      <span className="lane-indicator">LANE 0{infraction.lane} • {infraction.timestamp.split(' ')[1]}</span>
    </div>
    <div style={{ textAlign: 'right' }}>
      <span className="speed-badge">{infraction.speed} KM/H</span>
      <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{infraction.type}</div>
    </div>
  </div>
);

// --- Main App ---

function App() {
  const [state, setState] = useState<AppState | null>(null);
  const [speedLimit, setSpeedLimit] = useState(60);
  const [policeOverride, setPoliceOverride] = useState(false);
  const [appMode, setAppMode] = useState<'general' | 'testing'>('general');
  const [calibP1, setCalibP1] = useState(250);
  const [calibP2, setCalibP2] = useState(400);

  const handleCalibrate = async () => {
    try {
      await fetch('/api/calibrate_lane', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          lane_idx: 0, 
          p1: calibP1, 
          p2: calibP2,
          real_m: 6.0 
        })
      });
      alert("Calibration Updated! 6m reference set.");
    } catch (e) {
      console.error("Failed to calibrate", e);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) throw new Error("API Offline");
        const data = await res.json();
        setState(data);
        setSpeedLimit(data.global_speed_limit);
        setPoliceOverride(data.police_override);
        setAppMode(data.mode || 'general');
      } catch (e) {
        console.error("Failed to fetch state", e);
      }
    };

    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleSpeedLimitChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVal = parseInt(e.target.value);
    setSpeedLimit(newVal);
    try {
      await fetch('/api/set_speed_limit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit: newVal })
      });
    } catch (e) {
      console.error("Failed to update speed limit", e);
    }
  };

  const handleTogglePolice = async () => {
    const newVal = !policeOverride;
    setPoliceOverride(newVal);
    try {
      await fetch('/api/toggle_police', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: newVal })
      });
    } catch (e) {
      console.error("Failed to toggle police override", e);
    }
  };

  const handleModeChange = async (newMode: 'general' | 'testing') => {
    setAppMode(newMode);
    try {
      await fetch('/api/set_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: newMode })
      });
    } catch (e) {
      console.error('Failed to update app mode', e);
    }
  };

  if (!state) return <div className="loading" style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#020617', color: 'white' }}>Initializing Neural Engine...</div>;

  const allInfractions = state.lanes.flatMap(l => l.infractions).slice(0, 10);
  const activeEmergencies = state.lanes.filter(l => l.ambulance_detected || l.accident_detected);
  const modeLabel = appMode === 'testing' ? 'TESTING MODE' : 'GENERAL MODE';

  return (
    <div className="dashboard-container">
      <header>
        <div className="logo-section">
          <h1>ANTIGRAVITY TRAFFIC</h1>
          <p>AI-Powered Urban Logistics & Safety Dashboard</p>
        </div>
        
        <div style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Clock size={18} color="var(--text-secondary)" />
            <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>{new Date().toLocaleTimeString()}</span>
          </div>
          <div className="mode-selector">
            <button className={`mode-pill ${appMode === 'general' ? 'active' : ''}`} onClick={() => handleModeChange('general')}>GENERAL</button>
            <button className={`mode-pill ${appMode === 'testing' ? 'active testing' : ''}`} onClick={() => handleModeChange('testing')}>TESTING</button>
          </div>
          <span className={`mode-badge ${appMode === 'testing' ? 'testing' : ''}`}>{modeLabel}</span>
          <div className="toggle-container">
            <Shield size={18} color={policeOverride ? "var(--accent-blue)" : "var(--text-secondary)"} />
            <span className="toggle-label">POLICE OVERRIDE</span>
            <label className="switch">
              <input type="checkbox" checked={policeOverride} onChange={handleTogglePolice} />
              <span className="slider"></span>
            </label>
          </div>
        </div>
      </header>

      <main className="main-layout">
        <section className="video-grid">
          <div className="video-slot" style={{ position: 'relative' }}>
            <div className="video-label" style={{ display: 'flex', alignItems: 'center' }}><Navigation size={12} style={{marginRight: '6px'}} /> QUAD-STREAM LIVE FEED</div>
            <img 
              src="/video_feed" 
              alt="Traffic Feed" 
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const scaleY = 480 / rect.height;
                const y = (e.clientY - rect.top) * scaleY;
                
                // Toggle between setting P1 and P2
                if (Math.abs(y - calibP1) < Math.abs(y - calibP2)) {
                  setCalibP1(Math.round(y));
                } else {
                  setCalibP2(Math.round(y));
                }
              }}
              style={{ cursor: 'crosshair' }}
            />
            
            {/* Calibration Visual Overlays */}
            <div style={{ position: 'absolute', top: `${(calibP1/480)*100}%`, left: 0, width: '100%', height: '1px', background: 'rgba(34, 197, 94, 0.5)', borderTop: '1px dashed #22c55e', pointerEvents: 'none' }}></div>
            <div style={{ position: 'absolute', top: `${(calibP2/480)*100}%`, left: 0, width: '100%', height: '1px', background: 'rgba(239, 68, 68, 0.5)', borderTop: '1px dashed #ef4444', pointerEvents: 'none' }}></div>
            
            {state.police_detected && (
              <div className="emergency-alert" style={{ position: 'absolute', bottom: '20px', right: '20px' }}>
                <Shield size={18} /> POLICE IN FRAME
              </div>
            )}
          </div>
          
          <div className="video-slot" style={{ background: 'var(--panel-bg)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ padding: '40px', textAlign: 'center' }}>
               <Activity size={48} color="var(--accent-cyan)" style={{ marginBottom: '20px' }} />
               <h3>SYSTEM STATUS: OPTIMAL</h3>
               <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                 Neural Inference active on 4 lanes.<br/>
                 Overspeed Detection active at {state.global_speed_limit} KM/H.<br/>
                 CPU Usage Restricted.
               </p>
            </div>
          </div>
          
          <div className="video-slot" style={{ gridColumn: 'span 2', background: 'var(--panel-bg)', overflow: 'hidden' }}>
             <div style={{ padding: '20px' }}>
                <div className="panel-header" style={{ marginBottom: '20px' }}>
                  <Zap size={16} /> Real-time Performance Metrics
                </div>
                <div className="lanes-display">
                  {state.lanes.map(lane => (
                    <LaneCard key={lane.id} lane={lane} isActive={lane.id === state.active_lane_idx} />
                  ))}
                </div>
             </div>
          </div>
        </section>

        <aside className="sidebar">
          {activeEmergencies.length > 0 && (
            <div className="panel">
              <div className="panel-header" style={{ color: 'var(--accent-red)' }}>
                <AlertTriangle size={16} /> Critical Alerts
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {activeEmergencies.map((lane, i) => (
                  <div key={i} className="emergency-alert">
                    {lane.ambulance_detected ? '🚑 AMBULANCE IN LANE ' : '💥 ACCIDENT IN LANE '} {lane.id + 1}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="panel">
            <div className="panel-header">
              <Settings size={16} /> Detection Settings
            </div>
            <div className="settings-card">
              <div className="control-group">
                <div className="control-header">
                  <span>Speed Limit Boundary</span>
                  <span className="speed-value">{speedLimit} KM/H</span>
                </div>
                <input 
                  type="range" 
                  min="20" 
                  max="120" 
                  value={speedLimit} 
                  onChange={handleSpeedLimitChange} 
                />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
                  <span>20km/h</span>
                  <span>120km/h</span>
                </div>
              </div>
            </div>
          </div>

          <div className="panel">
            <div className="panel-header">
              <Settings size={16} /> Calibration Workshop
            </div>
            <div className="settings-card">
              <div className="control-group">
                <div className="control-header">
                  <span>6m Reference Gap (Lane 1)</span>
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '10px' }}>
                  <input 
                    type="range" min="0" max="640" value={calibP1} 
                    onChange={(e) => setCalibP1(parseInt(e.target.value))} 
                  />
                </div>
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px' }}>
                  <input 
                    type="range" min="0" max="640" value={calibP2} 
                    onChange={(e) => setCalibP2(parseInt(e.target.value))} 
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                   <span>P1: {calibP1}px</span>
                   <span>P2: {calibP2}px</span>
                </div>
                <button 
                  className="action-btn" 
                  style={{ width: '100%', marginTop: '12px', padding: '8px', borderRadius: '4px', background: 'var(--accent-blue)', color: 'white', border: 'none', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600 }}
                  onClick={handleCalibrate}
                >
                  SAVE CALIBRATION (6m)
                </button>
              </div>
            </div>
          </div>

          <div className="panel" style={{ flex: 1, minHeight: 0 }}>
            <div className="panel-header">
              <Camera size={16} /> OCR Capture Log
            </div>
            <div className="infraction-list" style={{ overflowY: 'auto' }}>
              {allInfractions.length > 0 ? (
                allInfractions.map((inf, i) => (
                  <InfractionItem key={i} infraction={inf} />
                ))
              ) : (
                <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-secondary)' }}>
                   <Zap size={32} style={{ opacity: 0.2, marginBottom: '10px' }} />
                   <p>Monitoring for speed infractions...</p>
                </div>
              )}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;
