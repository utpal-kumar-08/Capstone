import React, { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';

// ============================================================
// TYPES
// ============================================================

interface Infraction {
  id: string | number;
  lane?: number;
  timestamp: string;
  plate?: string;
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

interface TrafficState {
  global_speed_limit: number;
  police_detected: boolean;
  active_lane_idx: number;
  mode: 'general' | 'testing';
  phase: string;
  lanes: Lane[];
}

interface TestState {
  speed_limit: number;
  display_mode: string;
  test_type: 'overspeed' | 'accident';
  line_a: number;
  line_b: number;
  running: boolean;
  has_video: boolean;
  infractions: Infraction[];
  tracked_count: number;
  speed_count: number;
  accident_detected: boolean;
}

interface CalibState {
  laneIdx: number;
  point1: { x: number; y: number } | null;
  point2: { x: number; y: number } | null;
  imageData: string | null;
}

// ============================================================
// UTILITIES
// ============================================================

function useClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return time.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

// ============================================================
// LANE CARD
// ============================================================

const LaneCard: React.FC<{ lane: Lane; isActive: boolean }> = ({ lane, isActive }) => {
  const pct = Math.min((lane.pcu / 20) * 100, 100);
  const light = lane.light.toUpperCase();
  return (
    <div className={`lane-card ${isActive ? 'active-lane' : ''}`}>
      <div className="lane-card-top">
        <span className="lane-name">LANE {String(lane.id + 1).padStart(2, '0')}</span>
        <div className="traffic-light">
          <div className={`sig-dot sig-red   ${light === 'RED'    ? 'on' : ''}`} />
          <div className={`sig-dot sig-yellow ${light === 'YELLOW' ? 'on' : ''}`} />
          <div className={`sig-dot sig-green  ${light === 'GREEN'  ? 'on' : ''}`} />
        </div>
      </div>
      <div className="lane-pcu">
        <div className="lane-pcu-row">
          <span className="pcu-label">PCU Density</span>
          <span className="pcu-val">{lane.pcu.toFixed(1)}</span>
        </div>
        <div className="pcu-bar">
          <div className="pcu-fill" style={{ width: `${pct}%` }} />
        </div>
      </div>
      {(lane.ambulance_detected || lane.accident_detected) && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {lane.ambulance_detected && <span style={{ fontSize: '0.62rem', background: 'rgba(16,185,129,0.15)', color: '#10b981', padding: '2px 7px', borderRadius: 4, fontWeight: 700 }}>🚑 AMB</span>}
          {lane.accident_detected  && <span style={{ fontSize: '0.62rem', background: 'rgba(239,68,68,0.15)',   color: '#ef4444', padding: '2px 7px', borderRadius: 4, fontWeight: 700 }}>💥 ACC</span>}
        </div>
      )}
    </div>
  );
};

// ============================================================
// INFRACTION ITEM
// ============================================================

const InfractionItem: React.FC<{ inf: Infraction; isTest?: boolean }> = ({ inf, isTest }) => (
  <div className="infraction-item">
    <div className="inf-left">
      <span className="plate-tag">{isTest ? `VEHICLE #${inf.id}` : (inf.plate || 'UNKNOWN')}</span>
      <span className="inf-meta">
        {inf.type}{inf.lane !== undefined ? ` · LANE ${inf.lane}` : ''} · {isTest ? inf.timestamp : (inf.timestamp?.split(' ')[1] || inf.timestamp)}
      </span>
    </div>
    <span className="speed-tag">{inf.speed} KM/H</span>
  </div>
);

// ============================================================
// CALIBRATION MODAL (TRAFFIC)
// ============================================================

interface TrafficCalibModalProps {
  onClose: () => void;
}

const TrafficCalibModal: React.FC<TrafficCalibModalProps> = ({ onClose }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [state, setState] = useState<CalibState>({ laneIdx: -1, point1: null, point2: null, imageData: null });
  const [realDist, setRealDist] = useState(3);
  const [step, setStep] = useState(0); // 0=lane, 1=pick, 2=done
  const [hint, setHint] = useState('Select a lane to start calibration.');
  const [applied, setApplied] = useState(false);

  const drawImageOnCanvas = useCallback((b64: string) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const img = new Image();
    img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    img.src = 'data:image/jpeg;base64,' + b64;
  }, []);

  const captureFrame = async (laneIdx: number) => {
    setHint(`⏳ Capturing frame from Lane ${laneIdx + 1}...`);
    try {
      const r = await fetch(`/api/capture_frame/${laneIdx}`);
      const d = await r.json();
      setState(s => ({ ...s, laneIdx, imageData: d.image, point1: null, point2: null }));
      setStep(1);
      setHint('Click on the frame to set Point 1 (Line A).');
      setTimeout(() => drawImageOnCanvas(d.image), 50);
    } catch {
      setHint('❌ Failed to capture frame.');
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;

    if (!state.point1) {
      ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fillStyle = '#10b981'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 13px Inter'; ctx.fillText('LINE A', x + 12, y + 5);
      ctx.beginPath(); ctx.setLineDash([5, 4]); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
      ctx.strokeStyle = 'rgba(16,185,129,0.6)'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
      setState(s => ({ ...s, point1: { x: Math.round(x), y: Math.round(y) } }));
      setHint('Now click to set Point 2 (Line B).');
    } else if (!state.point2) {
      ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fillStyle = '#ef4444'; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 13px Inter'; ctx.fillText('LINE B', x + 12, y + 5);
      ctx.beginPath(); ctx.setLineDash([5, 4]); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
      ctx.strokeStyle = 'rgba(239,68,68,0.6)'; ctx.lineWidth = 1.5; ctx.stroke(); ctx.setLineDash([]);
      const newP2 = { x: Math.round(x), y: Math.round(y) };
      setState(s => ({ ...s, point2: newP2 }));
      setStep(2);
      setHint('✅ Both points set. Adjust distance and click Apply.');
    }
  };

  const apply = async () => {
    if (!state.point1 || !state.point2) return;
    try {
      const r = await fetch('/api/calibrate_lane', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lane_idx: state.laneIdx, p1: state.point1.y, p2: state.point2.y, real_m: realDist })
      });
      const d = await r.json();
      if (d.status === 'success') {
        setHint(`🎉 Calibration applied for Lane ${state.laneIdx + 1}!`);
        setApplied(true);
        setTimeout(onClose, 1800);
      }
    } catch {
      setHint('❌ Calibration failed.');
    }
  };

  const reset = () => {
    setState({ laneIdx: -1, point1: null, point2: null, imageData: null });
    setStep(0); setHint('Select a lane to start calibration.'); setApplied(false);
  };

  const pixDist = state.point1 && state.point2
    ? Math.round(Math.abs(state.point2.y - state.point1.y))
    : null;

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-box">
        <div className="modal-head">
          <h2>📐 Speed Trap Calibration</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="step-row">
            {['Select Lane', 'Pick Points', 'Apply'].map((s, i) => (
              <React.Fragment key={i}>
                {i > 0 && <span className="step-arrow">→</span>}
                <span className={`step-pill ${step === i ? 'active' : step > i ? 'done' : ''}`}>{i + 1}. {s}</span>
              </React.Fragment>
            ))}
          </div>
          <div className="calib-hint">{hint}</div>

          {step === 0 && (
            <div className="calib-lane-btns">
              {[0,1,2,3].map(i => (
                <button key={i} className="btn-lane" onClick={() => captureFrame(i)}>Lane {i + 1}</button>
              ))}
            </div>
          )}

          {step >= 1 && (
            <>
              <div className="canvas-wrapper">
                <canvas ref={canvasRef} width={960} height={540} onClick={step === 1 ? handleCanvasClick : undefined} style={{ cursor: step === 1 ? 'crosshair' : 'default' }} />
              </div>
              <div className="info-chips">
                <span className="info-chip">Line A: {state.point1 ? `Y=${state.point1.y}px` : 'Not set'}</span>
                <span className="info-chip">Line B: {state.point2 ? `Y=${state.point2.y}px` : 'Not set'}</span>
                {pixDist !== null && <span className="info-chip highlight">Pixel Gap: {pixDist}px</span>}
              </div>
              <div className="distance-row">
                <span>🛣️ Real-world distance between points:</span>
                <input type="number" value={realDist} min={1} max={50} step={0.5} onChange={e => setRealDist(parseFloat(e.target.value))} />
                <span>meters</span>
                {pixDist !== null && <span style={{ color: 'var(--cyan)', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem' }}>≈ {(pixDist / realDist).toFixed(2)} px/m</span>}
              </div>
            </>
          )}

          {step === 2 && state.point1 && state.point2 && pixDist !== null && (
            <div className="summary-grid">
              {[
                ['Lane', `Lane ${state.laneIdx + 1}`],
                ['Line A (Y)', `${state.point1.y}px`],
                ['Line B (Y)', `${state.point2.y}px`],
                ['Pixel Gap', `${pixDist}px`],
                ['Real Distance', `${realDist}m`],
                ['PPM', `${(pixDist / realDist).toFixed(2)} px/m`],
              ].map(([l, v]) => (
                <div key={l} className="sum-row">
                  <span className="sum-lbl">{l}</span>
                  <span className="sum-val">{v}</span>
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="modal-foot">
          <button className="btn btn-ghost" onClick={reset}>🔄 Reset</button>
          <button className="btn btn-primary" disabled={step < 2 || applied} onClick={apply}>
            {applied ? '✓ Applied!' : '✓ Apply Calibration'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================
// TEST CALIBRATION MODAL
// ============================================================

interface TestCalibModalProps {
  onClose: () => void;
}

const TestCalibModal: React.FC<TestCalibModalProps> = ({ onClose }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [p1, setP1] = useState<{ x: number; y: number } | null>(null);
  const [p2, setP2] = useState<{ x: number; y: number } | null>(null);
  const [realDist, setRealDist] = useState(3);
  const [hint, setHint] = useState('Click on the frame to set Line A, then Line B.');
  const [applied, setApplied] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch('/api/test/capture_frame');
        const d = await r.json();
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d')!;
        const img = new Image();
        img.onload = () => ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        img.src = 'data:image/jpeg;base64,' + d.image;
      } catch {
        setHint('❌ No test video loaded. Upload a video first.');
      }
    })();
  }, []);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || (p1 && p2)) return;
    const ctx = canvas.getContext('2d')!;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * sx;
    const y = (e.clientY - rect.top) * sy;

    if (!p1) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
      ctx.strokeStyle = 'rgba(16,185,129,0.85)'; ctx.lineWidth = 2; ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#10b981'; ctx.font = 'bold 13px Inter'; ctx.fillText('LINE A', 10, y - 8);
      setP1({ x: Math.round(x), y: Math.round(y) });
      setHint('Now click to set Line B.');
    } else {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y);
      ctx.strokeStyle = 'rgba(239,68,68,0.85)'; ctx.lineWidth = 2; ctx.setLineDash([5,4]); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#ef4444'; ctx.font = 'bold 13px Inter'; ctx.fillText('LINE B', 10, y - 8);
      setP2({ x: Math.round(x), y: Math.round(y) });
      setHint('✅ Both lines set. Adjust meters and click Apply.');
    }
  };

  const apply = async () => {
    if (!p1 || !p2) return;
    try {
      await fetch('/api/test/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ p1: p1.y, p2: p2.y, real_m: realDist })
      });
      setHint('🎉 Calibration applied!');
      setApplied(true);
      setTimeout(onClose, 1500);
    } catch {
      setHint('❌ Calibration failed.');
    }
  };

  const reset = () => {
    setP1(null); setP2(null); setApplied(false);
    setHint('Click on the frame to set Line A, then Line B.');
    const canvas = canvasRef.current;
    if (!canvas) return;
    // redraw image
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const gap = p1 && p2 ? Math.abs(p2.y - p1.y) : null;

  return (
    <div className="modal-overlay" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal-box">
        <div className="modal-head">
          <h2>📐 Test Speed Trap Calibration</h2>
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="modal-body">
          <div className="calib-hint">{hint}</div>
          <div className="canvas-wrapper">
            <canvas ref={canvasRef} width={960} height={540} onClick={handleClick} style={{ cursor: p1 && p2 ? 'default' : 'crosshair' }} />
          </div>
          <div className="info-chips">
            <span className="info-chip">Line A: {p1 ? `Y=${p1.y}px` : 'Not set'}</span>
            <span className="info-chip">Line B: {p2 ? `Y=${p2.y}px` : 'Not set'}</span>
            {gap !== null && <span className="info-chip highlight">Pixel Gap: {gap}px</span>}
          </div>
          <div className="distance-row">
            <span>🛣️ Real-world distance:</span>
            <input type="number" value={realDist} min={1} max={50} step={0.5} onChange={e => setRealDist(parseFloat(e.target.value))} />
            <span>meters</span>
            {gap !== null && <span style={{ color: 'var(--cyan)', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.8rem' }}>≈ {(gap / realDist).toFixed(2)} px/m</span>}
          </div>
        </div>
        <div className="modal-foot">
          <button className="btn btn-ghost" onClick={reset}>🔄 Reset</button>
          <button className="btn btn-amber" disabled={!p1 || !p2 || applied} onClick={apply}>
            {applied ? '✓ Applied!' : '✓ Apply Calibration'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================
// TRAFFIC MODE
// ============================================================

interface TrafficModeProps {
  state: TrafficState;
  speedLimit: number;
  onSpeedChange: (v: number) => void;
}

const TrafficMode: React.FC<TrafficModeProps> = ({ state, speedLimit, onSpeedChange }) => {
  const [laneView, setLaneView] = useState('all');
  const [displayMode, setDisplayMode] = useState('both');
  const [showCalib, setShowCalib] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const [frameZoom, setFrameZoom] = useState(100);

  const switchView = (val: string) => {
    setLaneView(val);
    setStreamKey(Date.now());
  };

  const applyDisplayMode = async (m: string) => {
    setDisplayMode(m);
    await fetch('/api/set_display_mode', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: m })
    }).catch(() => {});
  };

  const feedSrc = laneView === 'all'
    ? `/video_feed?t=${streamKey}`
    : `/video_feed/${laneView}?t=${streamKey}`;

  const allInfractions = state.lanes.flatMap(l => l.infractions).slice(0, 20);
  const emergencies = state.lanes.filter(l => l.ambulance_detected || l.accident_detected);

  return (
    <>
      <div className="content-area">
        {/* --- Control bar --- */}
        <div className="control-bar">
          <div className="ctrl-group">
            <span className="ctrl-label">View</span>
            <select value={laneView} onChange={e => switchView(e.target.value)}>
              <option value="all">All Lanes (Grid)</option>
              <option value="0">Lane 1 Only</option>
              <option value="1">Lane 2 Only</option>
              <option value="2">Lane 3 Only</option>
              <option value="3">Lane 4 Only</option>
            </select>
          </div>
          <div className="ctrl-divider" />
          <div className="ctrl-group speed-ctrl">
            <span className="ctrl-label">Speed Limit</span>
            <input
              type="range" min={20} max={120} step={5} value={speedLimit}
              onChange={e => onSpeedChange(parseInt(e.target.value))}
            />
            <span className="speed-display">{speedLimit} km/h</span>
          </div>
          <div className="ctrl-divider" />
          <div className="ctrl-group">
            <span className="ctrl-label">Display</span>
            <select value={displayMode} onChange={e => applyDisplayMode(e.target.value)}>
              <option value="both">Speed + Boxes</option>
              <option value="speed">Speed Only</option>
              <option value="bbox">Boxes Only</option>
            </select>
          </div>
          <div className="ctrl-divider" />
          <div className="ctrl-group speed-ctrl">
            <span className="ctrl-label">Zoom</span>
            <input
              type="range" min={50} max={250} step={10} value={frameZoom}
              onChange={e => setFrameZoom(parseInt(e.target.value))}
            />
            <span className="speed-display">{frameZoom}%</span>
          </div>
          <div className="ctrl-divider" />
          <button className="btn btn-ghost btn-sm" onClick={() => setShowCalib(true)}>
            📐 Calibrate
          </button>
        </div>

        {/* --- Video --- */}
        <div className="video-container" style={{ position: 'relative', flex: 1, overflow: 'hidden' }}>
          <div style={{ position: 'absolute', inset: 0, overflow: 'auto' }}>
            <div style={{ 
              width: `${frameZoom}%`, 
              height: `${frameZoom}%`,
              minWidth: '100%', 
              minHeight: '100%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center', 
              transition: 'width 0.2s, height 0.2s'
            }}>
              <img src={feedSrc} alt="Live Traffic Feed" key={streamKey} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            </div>
          </div>
          <div className="vid-overlay-badge live-badge" style={{ position: 'absolute', top: 14, left: 14, zIndex: 10 }}>🔴 LIVE</div>
        </div>
      </div>

      {/* --- Sidebar --- */}
      <div className="sidebar">
        <div className="sidebar-scroll">
          {/* Emergencies */}
          {emergencies.length > 0 && (
            <div className="panel alert-panel">
              <div className="panel-header">🚨 Active Emergencies</div>
              <div className="alert-items">
                {emergencies.map(l => (
                  <React.Fragment key={l.id}>
                    {l.ambulance_detected && (
                      <div className="alert-item ambulance">🚑 Ambulance — Lane {l.id + 1}</div>
                    )}
                    {l.accident_detected && (
                      <div className="alert-item accident">💥 Accident — Lane {l.id + 1}</div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}

          {/* Lane status */}
          <div className="panel">
            <div className="panel-header">📊 Lane Status & PCU Density</div>
            <div className="lane-grid">
              {state.lanes.map(lane => (
                <LaneCard key={lane.id} lane={lane} isActive={lane.id === state.active_lane_idx} />
              ))}
            </div>
          </div>

          {/* Infractions */}
          <div className="panel" style={{ flex: 1, minHeight: 0 }}>
            <div className="panel-header">📸 Overspeed Infractions</div>
            <div className="infraction-list">
              {allInfractions.length === 0 ? (
                <div className="infraction-empty">
                  <div className="infraction-empty-icon">🛡️</div>
                  <p>No infractions recorded yet.<br />Monitoring all lanes...</p>
                </div>
              ) : (
                allInfractions.map((inf, i) => <InfractionItem key={i} inf={inf} />)
              )}
            </div>
          </div>
        </div>
      </div>

      {showCalib && <TrafficCalibModal onClose={() => setShowCalib(false)} />}
    </>
  );
};

// ============================================================
// TESTING MODE
// ============================================================

const TestingMode: React.FC = () => {
  const [testState, setTestState] = useState<TestState | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [hasVideo, setHasVideo] = useState(false);
  const [showCalib, setShowCalib] = useState(false);
  const [streamKey, setStreamKey] = useState(Date.now());
  const fileInputRef = useRef<HTMLInputElement>(null);
  const changeInputRef = useRef<HTMLInputElement>(null);
  const [speedLimit, setSpeedLimit] = useState(60);
  const [displayMode, setDisplayMode] = useState('both');
  const [frameZoom, setFrameZoom] = useState(100);
  const [testType, setTestType] = useState<'overspeed' | 'accident'>('overspeed');

  // Poll test state
  useEffect(() => {
    const id = setInterval(async () => {
      try {
        const r = await fetch('/api/test/state');
        const d: TestState = await r.json();
        setTestState(d);
        if (d.has_video) setHasVideo(true);
        setSpeedLimit(d.speed_limit);
        setDisplayMode(d.display_mode);
        setTestType(d.test_type || 'overspeed');
      } catch {}
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const uploadVideo = async (file: File | null | undefined) => {
    if (!file) return;
    setUploading(true);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/api/test/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.status === 'success') {
        setHasVideo(true);
        setStreamKey(Date.now());
      }
    } catch {
      alert('Upload failed. Please try again.');
    }
    setUploading(false);
  };

  const applySettings = async (sl?: number, dm?: string, tt?: 'overspeed' | 'accident') => {
    const body: Record<string, unknown> = {};
    if (sl !== undefined) { body.speed_limit = sl; setSpeedLimit(sl); }
    if (dm !== undefined) { body.display_mode = dm; setDisplayMode(dm); }
    if (tt !== undefined) { body.test_type = tt; setTestType(tt); }
    await fetch('/api/test/settings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).catch(() => {});
  };

  const infractions = testState?.infractions ?? [];

  return (
    <>
      <div className="content-area">
        {/* Control bar */}
        <div className="control-bar">
          <div className="ctrl-group">
            <span className="ctrl-label">Goal</span>
            <select value={testType} onChange={e => applySettings(undefined, undefined, e.target.value as 'overspeed' | 'accident')} style={{ fontWeight: 600, color: 'var(--amber)' }}>
              <option value="overspeed">Overspeed (Model 1)</option>
              <option value="accident">Collisions (Model 2)</option>
            </select>
          </div>
          
          {hasVideo && (
            <>
              <div className="ctrl-divider" />
              {testType === 'overspeed' && (
                <>
                  <div className="ctrl-group speed-ctrl">
                    <span className="ctrl-label">Speed Limit</span>
                    <input
                      type="range" min={20} max={120} step={5} value={speedLimit}
                      onChange={e => applySettings(parseInt(e.target.value))}
                    />
                    <span className="speed-display">{speedLimit} km/h</span>
                  </div>
                  <div className="ctrl-divider" />
                </>
              )}

              {testType === 'overspeed' ? (
                <>
                  <div className="ctrl-group">
                    <span className="ctrl-label">Display</span>
                    <select value={displayMode} onChange={e => applySettings(undefined, e.target.value)}>
                      <option value="both">Speed + Boxes</option>
                      <option value="speed">Speed Only</option>
                      <option value="bbox">Boxes Only</option>
                    </select>
                  </div>
                  <div className="ctrl-divider" />
                </>
              ) : (
                <div className="ctrl-group" style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Collision mode ignores speed and overlay markings.
                </div>
              )}
              <div className="ctrl-group speed-ctrl">
                <span className="ctrl-label">Zoom</span>
                <input
                  type="range" min={50} max={250} step={10} value={frameZoom}
                  onChange={e => setFrameZoom(parseInt(e.target.value))}
                />
                <span className="speed-display">{frameZoom}%</span>
              </div>
              <div className="ctrl-divider" />
              {testType === 'overspeed' && (
                <>
                  <button className="btn btn-ghost btn-sm" onClick={() => setShowCalib(true)}>
                    📐 Calibrate Lines
                  </button>
                  <div className="ctrl-divider" />
                </>
              )}
              <button className="btn btn-ghost btn-sm" onClick={() => changeInputRef.current?.click()}>
                📂 Change Video
              </button>
              <input ref={changeInputRef} type="file" accept="video/*" style={{ display: 'none' }} onChange={e => uploadVideo(e.target.files?.[0])} />
            </>
          )}
        </div>

        {/* Video area */}
        <div className="video-container">
          {!hasVideo ? (
            <div
              className={`upload-zone ${dragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={e => {
                e.preventDefault(); setDragging(false);
                uploadVideo(e.dataTransfer.files?.[0]);
              }}
            >
              {uploading ? (
                <div className="upload-progress">
                  <div className="spinner" />
                  <span>Uploading video...</span>
                </div>
              ) : (
                <>
                  <div className="upload-icon">📹</div>
                  <div className="upload-text">
                    <h3><span className="highlight">Drop a video here</span> or click to browse</h3>
                    <p>Supports MP4, MOV, AVI · Any resolution</p>
                  </div>
                  <button className="btn btn-amber" onClick={e => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                    Choose Video File
                  </button>
                </>
              )}
              <input ref={fileInputRef} type="file" accept="video/*" style={{ display: 'none' }} onChange={e => uploadVideo(e.target.files?.[0])} />
            </div>
          ) : (
            <>
              {uploading && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 10, background: 'rgba(0,0,0,0.7)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 16 }}>
                  <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
                  <span style={{ color: 'var(--amber)', fontWeight: 600 }}>Uploading new video...</span>
                </div>
              )}
              <div style={{ position: 'absolute', inset: 0, overflow: 'auto' }}>
                <div style={{ 
                  width: `${frameZoom}%`, 
                  height: `${frameZoom}%`,
                  minWidth: '100%', 
                  minHeight: '100%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center', 
                  transition: 'width 0.2s, height 0.2s'
                }}>
                  <img key={streamKey} src={`/test_feed?t=${streamKey}`} alt="Test Feed" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                </div>
              </div>
              <div className="vid-overlay-badge test-badge" style={{ position: 'absolute', top: 14, left: 14, zIndex: 10 }}>🧪 TESTING MODE</div>
            </>
          )}
        </div>
      </div>

      {/* Sidebar */}
      <div className="sidebar">
        <div className="sidebar-scroll">
          {/* Accident Alert */}
          {testState?.accident_detected && (
            <div className="panel alert-panel" style={{ animation: 'pulse 1.5s infinite' }}>
              <div className="panel-header" style={{ color: 'var(--red)', borderColor: 'rgba(239, 68, 68, 0.4)' }}>
                🚨 EXTREME ALERT
              </div>
              <div className="alert-items">
                <div className="alert-item accident" style={{ fontSize: '1.2rem', padding: '16px', textAlign: 'center' }}>
                  💥 COLLISION DETECTED
                </div>
              </div>
            </div>
          )}

          {/* Stats */}
          <div className="panel">
            <div className="panel-header">📊 Live Detection Stats</div>
            <div className="stats-grid">
              <div className="stat-card">
                <div className="stat-val">{testState?.tracked_count ?? 0}</div>
                <div className="stat-lbl">Tracked Vehicles</div>
              </div>
              <div className="stat-card">
                <div className="stat-val">{testState?.speed_count ?? 0}</div>
                <div className="stat-lbl">Speeds Measured</div>
              </div>
              <div className="stat-card">
                <div className="stat-val" style={{ color: infractions.length > 0 ? 'var(--red)' : 'var(--amber)' }}>
                  {infractions.length}
                </div>
                <div className="stat-lbl">Infractions</div>
              </div>
            </div>
          </div>

          {/* Speed limit info */}
          {testType === 'overspeed' && (
            <div className="panel">
              <div className="panel-header">⚡ Test Configuration</div>
              <div className="settings-body">
                <div className="setting-row">
                  <div className="setting-row-header">
                    <span className="setting-label">Speed Limit</span>
                    <span className="setting-value">{speedLimit} KM/H</span>
                  </div>
                  <input type="range" min={20} max={120} step={5} value={speedLimit}
                    onChange={e => applySettings(parseInt(e.target.value))} />
                </div>
                <div className="setting-row">
                  <div className="setting-row-header">
                    <span className="setting-label">Status</span>
                    <span style={{ fontSize: '0.72rem', color: testState?.running ? 'var(--green)' : 'var(--text-muted)', fontWeight: 700 }}>
                      {testState?.running ? '● RUNNING' : '○ IDLE'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Infractions */}
          <div className="panel" style={{ flex: 1, minHeight: 0 }}>
            <div className="panel-header">{testType === 'overspeed' ? '🚨 Overspeed Detections' : '📹 Accident Check Log'}</div>
            <div className="infraction-list">
              {testType === 'accident' && !testState?.accident_detected && (
                 <div className="infraction-empty">
                   <div className="infraction-empty-icon">🛡️</div>
                   <p>{hasVideo ? 'Scanning for collisions...\nNo accidents detected.' : 'Upload a video to start testing.'}</p>
                 </div>
              )}
              {testType === 'overspeed' && infractions.length === 0 && (
                <div className="infraction-empty">
                  <div className="infraction-empty-icon">🔍</div>
                  <p>{hasVideo ? 'Monitoring for overspeed events...' : 'Upload a video to start testing.'}</p>
                </div>
              )}
              {testType === 'overspeed' && infractions.slice(0, 25).map((inf, i) => (
                <InfractionItem key={i} inf={inf} isTest />
              ))}
            </div>
          </div>
        </div>
      </div>

      {showCalib && <TestCalibModal onClose={() => setShowCalib(false)} />}
    </>
  );
};

// ============================================================
// HEADER
// ============================================================

interface HeaderProps {
  mode: 'general' | 'testing';
  onModeChange: (m: 'general' | 'testing') => void;
}

const Header: React.FC<HeaderProps> = ({ mode, onModeChange }) => {
  const clock = useClock();

  return (
    <header className="header">
      <div className="header-brand">
        <h1>🚦 Smart City AI Traffic Manager</h1>
        <p>Real-time PCU mapping · Ambulance Preemption · Overspeeding OCR</p>
      </div>

      <div className="header-center">
        <div className="mode-switcher">
          <button
            className={`mode-btn ${mode === 'general' ? 'active' : ''}`}
            onClick={() => onModeChange('general')}
          >
            <span className="btn-icon">🚦</span> Traffic Management
          </button>
          <button
            className={`mode-btn ${mode === 'testing' ? 'active testing' : ''}`}
            onClick={() => onModeChange('testing')}
          >
            <span className="btn-icon">🧪</span> Testing Mode
          </button>
        </div>
      </div>

      <div className="header-right">
        <div
          className={`status-badge ${mode === 'testing' ? 'testing-badge' : 'live'}`}
        >
          {mode === 'testing' ? '🧪 TESTING' : '🔴 LIVE'}
        </div>
        <div className="live-clock">
          <div className="live-dot" />
          {clock}
        </div>
      </div>
    </header>
  );
};

// ============================================================
// ROOT APP
// ============================================================

function App() {
  const [trafficState, setTrafficState] = useState<TrafficState | null>(null);
  const [mode, setMode] = useState<'general' | 'testing'>('general');
  const [speedLimit, setSpeedLimit] = useState(60);
  const [loading, setLoading] = useState(true);

  // Initial + periodic traffic state fetch
  useEffect(() => {
    const fetchTraffic = async () => {
      try {
        const r = await fetch('/api/state');
        if (!r.ok) throw new Error();
        const d: TrafficState = await r.json();
        setTrafficState(d);
        setSpeedLimit(Math.round(d.global_speed_limit));
        if (loading) { setMode(d.mode === 'testing' ? 'testing' : 'general'); setLoading(false); }
      } catch {
        if (loading) setLoading(false);
      }
    };

    fetchTraffic();
    const id = setInterval(fetchTraffic, 1000);
    return () => clearInterval(id);
  }, [loading]);

  const handleModeChange = async (m: 'general' | 'testing') => {
    setMode(m);
    await fetch('/api/set_mode', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: m })
    }).catch(() => {});
  };

  const handleSpeedChange = async (v: number) => {
    setSpeedLimit(v);
    await fetch('/api/set_speed_limit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ limit: v })
    }).catch(() => {});
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner" />
        <div className="loading-text">Connecting to Neural Engine...</div>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <Header mode={mode} onModeChange={handleModeChange} />
      <div className="main-layout">
        {mode === 'general' ? (
          <TrafficMode
            state={trafficState ?? {
              global_speed_limit: speedLimit,
              police_detected: false,
              active_lane_idx: 0,
              mode: 'general',
              phase: '',
              lanes: []
            }}
            speedLimit={speedLimit}
            onSpeedChange={handleSpeedChange}
          />
        ) : (
          <TestingMode />
        )}
      </div>
    </div>
  );
}

export default App;
