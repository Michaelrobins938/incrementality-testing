
import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, LucideIcon } from 'lucide-react';

interface StatCardProps {
    label: string;
    value: string;
    subValue?: string;
    trend: string;
    trendType?: 'up' | 'down' | 'neutral';
    color: string;
    icon?: LucideIcon;
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, subValue, trend, trendType = 'neutral', color, icon: Icon }) => (
    <motion.div
        whileHover={{ y: -5, scale: 1.02 }}
        className="tactical-panel p-6 rounded-2xl border-l-4 group cursor-default relative overflow-hidden"
        style={{ borderLeftColor: color }}
    >
        <div className="absolute inset-0 bg-gradient-to-br from-white/[0.02] to-transparent" />
        <div className="relative z-10">
            <div className="flex justify-between items-start mb-4">
                <div className="flex flex-col">
                    <span className="text-[9px] font-black text-zinc-500 uppercase tracking-widest leading-none mb-1">{label}</span>
                    <span className="text-[8px] font-bold text-zinc-700 uppercase tracking-tighter">METRIC_STABLE</span>
                </div>
                {Icon && <Icon size={14} className="text-zinc-600 group-hover:text-white transition-colors" />}
            </div>
            <div className="text-3xl font-black italic tracking-tighter text-white mb-1 group-hover:scale-110 transition-transform origin-left">{value}</div>
            {subValue && <div className="text-[8px] font-black text-zinc-600 uppercase tracking-widest mb-2">{subValue}</div>}
            <div className={`text-[10px] font-black uppercase flex items-center gap-2 ${trendType === 'up' ? 'text-emerald-500' : 'text-zinc-600'}`}>
                {(trendType === 'up' || trendType === 'down') && <TrendingUp size={10} className={trendType === 'down' ? 'rotate-180' : ''} />}
                {trend}
            </div>
        </div>
    </motion.div>
);
