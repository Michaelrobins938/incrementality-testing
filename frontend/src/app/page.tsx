"use client";

import React, { useState, useEffect } from 'react';
import {
  TrendingUp, Target,
  CheckCircle2,
  Activity,
  Globe, Shield, Terminal, Download, Layers
} from 'lucide-react';
import {
  Line, XAxis, YAxis, CartesianGrid,
  Tooltip as RechartsTooltip, ResponsiveContainer,
  AreaChart, Area, ReferenceLine
} from 'recharts';
import { motion } from 'framer-motion';

import { Tooltip } from '@/components/shared/Tooltip';
import { InfoPanel } from '@/components/shared/InfoPanel';
import { StatCard } from '@/components/shared/StatCard';

// --- Mock Data ---

const TIME_SERIES = Array.from({ length: 40 }, (_, i) => ({
  date: `Point_${i}`,
  Actual: 100 + Math.sin(i * 0.3) * 20 + Math.random() * 10,
  Counterfactual: i < 20 ? 100 + Math.sin(i * 0.3) * 20 + Math.random() * 5 : 90 + Math.sin(i * 0.3) * 15 + Math.random() * 10
}));

const TERMINAL_LOGS = [
  { time: '10:41:02', event: 'GEO_SYNC', msg: 'Synchronizing 25 treatment nodes with synthetic baseline', status: 'LOCKED' },
  { time: '10:41:45', event: 'CALIBRATION', msg: 'Synthetic control matching score: 0.94 R-Squared', status: 'OPTIMAL' },
  { time: '10:42:15', event: 'INFERENCE', msg: 'Incremental Lift identified at +12.3% (p=0.002)', status: 'STABLE' }
];

export default function IncrementalityDashboard() {
  const [mounted, setMounted] = useState(false);
  const [activeLogIndex, setActiveLogIndex] = useState(0);

  useEffect(() => {
    setMounted(true);
    const interval = setInterval(() => {
      setActiveLogIndex(prev => (prev + 1) % TERMINAL_LOGS.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!mounted) return <div className="min-h-screen bg-[#020203]" />;

  return (
    <div className="min-h-screen bg-[#020203] text-zinc-100 font-mono selection:bg-indigo-500/30 overflow-x-hidden overflow-y-auto p-10">
      {/* Ambient Background Grid */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.03] z-0"
        style={{ backgroundImage: 'radial-gradient(#ffffff 1px, transparent 1px)', backgroundSize: '40px 40px' }}
      />
      <div className="scan-line" />

      <header className="max-w-[1600px] mx-auto flex flex-col lg:flex-row justify-between items-start lg:items-end gap-10 mb-20">
        <div>
          <div className="flex items-center gap-4 mb-4">
            <div className="px-4 py-1.5 glass-surface border border-indigo-500/30 text-indigo-500 text-[10px] font-black tracking-widest uppercase flex items-center gap-2 clip-tactical animate-in slide-in-from-left duration-700">
              <Shield size={14} className="animate-pulse" />
              <span className="text-zinc-500">ENGINE_PROTOCOL::</span>SYNTHETIC_CONTROL_V1.4
            </div>
            <div className="h-px w-24 bg-gradient-to-r from-indigo-600/50 to-transparent" />
            <span className="text-[10px] text-zinc-700 font-bold uppercase tracking-[0.4em] flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-ping" />
              CALC_ENGINE_READY
            </span>
          </div>
          <h1 className="text-9xl font-black italic tracking-tighter uppercase leading-[0.75] mb-8">
            CAUSAL <br />
            <span className="text-indigo-500">INCREMENTALITY</span>
          </h1>
          <p className="text-zinc-500 text-sm max-w-2xl leading-relaxed uppercase tracking-widest font-extrabold border-l-2 border-indigo-600/20 pl-8">
            Deterministic measurement of <span className="text-zinc-300 italic">incremental lift</span> via counterfactual modeling and synthetic control geo-calibration.
          </p>
        </div>

        <div className="flex items-center gap-6">
          <button className="bg-white text-black px-12 py-6 font-black uppercase text-sm tracking-[0.2em] hover:bg-indigo-500 transition-all shadow-[0_0_50px_rgba(255,255,255,0.05)] active:scale-95 flex items-center gap-4">
            <Activity size={20} />
            EXECUTE_NEW_TEST
          </button>
          <div className="h-16 w-px bg-white/5 hidden lg:block" />
          <div className="flex flex-col items-end">
            <span className="text-[10px] font-black text-zinc-600 uppercase tracking-widest mb-1 italic">Environment::Production</span>
            <div className="text-3xl font-black italic text-zinc-400 tracking-tighter">US-EAST-GRID-04</div>
          </div>
        </div>
      </header>

      <main className="max-w-[1700px] mx-auto space-y-16">
        {/* Method Overview */}
        <section>
          <InfoPanel
            title="Incremental Discovery Protocol"
            description="Determining the true impact of marketing spend by constructing a mathematical 'Counterfactual'—what would have happened without the spend."
            details="The system selects a weight-optimized group of 'Control' geographies that perfectly mirror the 'Treatment' group during a pre-test calibration period."
            useCase="Used to prove the true efficiency of Growth spend to the CFO. It bypasses the confusion of multi-touch systems by focusing on aggregate revenue-lift at the region level."
            technical="Difference-in-Differences (DiD) estimator enhanced with Synthetic Control Method (SCM) weights. Standard errors recovered via block-bootstrap in 80ms."
            detailsTitle="Control Strategy"
            useCaseTitle="Causal Context"
            technicalTitle="Synthetic Engine"
          />
        </section>

        {/* Primary Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <StatCard label="Incremental Lift" value="12.3%" trend="+2.1% Alpha" trendType="up" subValue="ABOVE_BASELINE" color="#6366f1" icon={Target} />
          <StatCard label="P-Value Integrity" value="0.002" trend="HIGHLY_SIGNIFICANT" trendType="up" subValue="STATISTICAL_PROB" color="#10b981" icon={Activity} />
          <StatCard label="Causal iROAS" value="3.45x" trend="OPTIMAL_YIELD" trendType="up" subValue="PER_AD_DOLLAR" color="#f59e0b" icon={TrendingUp} />
          <StatCard label="Match Quality" value="0.92" trend="MAP_VALIDATED" trendType="up" subValue="R_SQUARED_FIT" color="#3b82f6" icon={Globe} />
        </div>

        <div className="grid grid-cols-12 gap-10">
          {/* Main Counterfactual Chart */}
          <div className="col-span-12 lg:col-span-8 tactical-panel p-12 rounded-[3.5rem] border border-white/5 bg-zinc-900/10 relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-12 opacity-[0.02] text-white transition-opacity group-hover:opacity-10 pointer-events-none">
              <Layers size={320} />
            </div>

            <div className="flex justify-between items-start mb-16 relative z-10">
              <div>
                <h3 className="text-4xl font-black italic uppercase tracking-tighter leading-none mb-2">Causal Impact Analysis</h3>
                <p className="text-zinc-500 text-[10px] font-black uppercase tracking-[0.4em]">TimeSeries_Decomposition::Active</p>
              </div>
              <div className="flex gap-6">
                <div className="px-5 py-2.5 bg-black/40 border border-white/5 rounded-xl flex items-center gap-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 shadow-[0_0_15px_#6366f1]" />
                  <span className="text-[10px] font-black text-zinc-400 uppercase tracking-widest">ACTUAL_TREATED</span>
                </div>
                <div className="px-5 py-2.5 bg-black/40 border border-white/5 rounded-xl flex items-center gap-3">
                  <div className="w-2.5 h-2.5 rounded-full bg-zinc-700" />
                  <span className="text-[10px] font-black text-zinc-400 uppercase tracking-widest">COUNTERFACTUAL</span>
                </div>
              </div>
            </div>

            <div className="h-[450px] relative z-10">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={TIME_SERIES}>
                  <defs>
                    <linearGradient id="actualGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.03)" />
                  <XAxis dataKey="date" hide />
                  <YAxis hide domain={[60, 150]} />
                  <RechartsTooltip
                    cursor={{ stroke: '#6366f1', strokeWidth: 2 }}
                    contentStyle={{ backgroundColor: '#050505', border: '1px solid #1e293b', borderRadius: '12px' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="Actual"
                    stroke="#6366f1"
                    strokeWidth={4}
                    fill="url(#actualGrad)"
                    animationDuration={1500}
                  />
                  <Line
                    type="monotone"
                    dataKey="Counterfactual"
                    stroke="#27272a"
                    strokeWidth={2}
                    strokeDasharray="8 4"
                    dot={false}
                  />
                  <ReferenceLine x="Point_20" stroke="#ef4444" strokeDasharray="5 5" label={{ value: 'TEST_LAUNCH', position: 'top', fill: '#ef4444', fontSize: 10, fontWeight: 900, dy: 30 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Sidebar: Diagnostics & Log */}
          <div className="col-span-12 lg:col-span-4 space-y-8">
            {/* Match Quality Gauge */}
            <div className="tactical-panel p-10 rounded-[2.5rem] border-t-2 border-indigo-500 group relative overflow-hidden bg-zinc-900/10">
              <h3 className="text-xl font-black italic uppercase tracking-widest text-zinc-400 mb-12 text-center relative z-10">MATCH_PRECISION_V4</h3>

              <div className="flex flex-col items-center justify-center relative z-10">
                <div className="relative w-40 h-40 flex items-center justify-center">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle cx="80" cy="80" r="70" fill="transparent" stroke="#18181b" strokeWidth="12" />
                    <motion.circle
                      cx="80" cy="80" r="70" fill="transparent" stroke="#6366f1" strokeWidth="12"
                      strokeDasharray={440}
                      initial={{ strokeDashoffset: 440 }}
                      animate={{ strokeDashoffset: 440 * (1 - 0.92) }}
                      transition={{ duration: 2, ease: "easeOut" }}
                      strokeLinecap="round"
                    />
                  </svg>
                  <div className="absolute flex flex-col items-center">
                    <span className="text-5xl font-black italic text-white tracking-widest">0.92</span>
                    <span className="text-[10px] font-black text-zinc-600 uppercase tracking-widest mt-1 italic">R-Squared Fit</span>
                  </div>
                </div>
                <div className="mt-10 px-6 py-2 bg-indigo-500/10 border border-indigo-500/20 rounded-full text-[10px] font-black text-indigo-400 uppercase tracking-widest">
                  EXCELLENT_FIT_STABLE
                </div>
              </div>
            </div>

            {/* Analysis Hub Log */}
            <div className="tactical-panel p-10 rounded-[2.5rem] border border-white/5 bg-zinc-900/10">
              <div className="flex items-center justify-between mb-10">
                <div className="flex items-center gap-4">
                  <Terminal size={22} className="text-indigo-400" />
                  <h3 className="text-xl font-black italic uppercase tracking-widest text-zinc-400">Analysis_Terminal</h3>
                </div>
                <span className="text-[10px] font-black text-zinc-700 uppercase tracking-widest">Live_Inference</span>
              </div>

              <div className="space-y-6">
                {TERMINAL_LOGS.map((log, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className={`flex gap-6 p-6 rounded-2xl border transition-all cursor-default ${i === activeLogIndex ? 'bg-indigo-500/10 border-indigo-500/30 shadow-xl shadow-indigo-900/10' : 'bg-black/40 border-white/5 opacity-50'}`}
                  >
                    <div className="text-[10px] font-black text-zinc-700 py-1">[{log.time}]</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-1">
                        <span className={`text-[8px] font-black px-1.5 py-0.5 rounded tracking-tighter uppercase ${i === activeLogIndex ? 'bg-indigo-500 text-black' : 'bg-zinc-800 text-zinc-600'}`}>{log.event}</span>
                        <span className="text-xs font-black text-zinc-300 italic uppercase tracking-tight">{log.msg}</span>
                      </div>
                      <div className="text-[9px] font-black text-zinc-700 uppercase tracking-widest mt-2">{log.status}</div>
                    </div>
                  </motion.div>
                ))}
              </div>

              <button className="w-full mt-12 py-5 bg-white text-black rounded-xl font-black italic uppercase tracking-widest text-xs hover:bg-indigo-500 transition-all flex items-center justify-center gap-3 group">
                <Download size={16} className="group-hover:translate-y-1 transition-transform" />
                GENERATE_EXECUTIVE_DOSSIER
              </button>
            </div>
          </div>
        </div>

        {/* Robustness Checks Grid */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {[
            { label: 'Parallel Trends', desc: 'Pre-period assumption verification.', status: 'PASSED' },
            { label: 'Placebo Tests', desc: 'Robustness check against random noise.', status: 'PASSED' },
            { label: 'Spatial Decay', desc: 'Cross-geo contamination filtering.', status: 'LOCKED' },
            { label: 'Bootstrap CI', desc: '95% Confidence Interval convergence.', status: 'NOMINAL' }
          ].map((item, i) => (
            <div key={i} className="bg-black/40 border border-white/5 p-10 rounded-[2.5rem] hover:border-indigo-500/30 transition-all group/rob">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-emerald-500/10 rounded-lg">
                  <CheckCircle2 size={16} className="text-emerald-500" />
                </div>
                <span className="text-[10px] font-black text-zinc-500 uppercase tracking-widest italic group-hover:text-white transition-colors">{item.label}</span>
              </div>
              <h4 className="text-xl font-black text-white italic tracking-tighter mb-4">{item.status}</h4>
              <p className="text-[11px] text-zinc-600 leading-relaxed uppercase tracking-tight font-black">{item.desc}</p>
            </div>
          ))}
        </div>
      </main>

      <footer className="mt-40 p-16 border-t border-zinc-900 bg-zinc-900/20 flex flex-col md:flex-row justify-between items-center gap-10">
        <div className="flex items-center gap-10">
          <div className="flex items-center gap-4">
            <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 shadow-[0_0_15px_#6366f1] animate-pulse" />
            <span className="text-[11px] font-black text-zinc-600 uppercase tracking-[0.6em]">CAUSAL_CORE_ACTIVE</span>
          </div>
          <div className="h-6 w-px bg-zinc-800" />
          <span className="text-[11px] font-black uppercase tracking-[0.3em] text-zinc-800 italic">SYSTEM_SYNC::LOCKED</span>
        </div>
        <div className="flex gap-16 text-[11px] font-black uppercase tracking-[0.4em] text-zinc-800">
          <span className="cursor-pointer hover:text-white transition-all tracking-widest">CALIBRATION_PROTOCOLS</span>
          <span className="cursor-pointer hover:text-white transition-all tracking-widest">DATA_GOVERNANCE_V4</span>
          <span className="cursor-pointer hover:text-white transition-all tracking-widest">NODE_TERMINAL</span>
        </div>
        <div className="text-[11px] font-black text-zinc-950 uppercase tracking-[1em] italic">MAR_SCI_ENGINEERING_PRM</div>
      </footer>
    </div>
  );
}
