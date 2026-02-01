
import React from 'react';
import { Info } from 'lucide-react';

interface InfoPanelProps {
    title: string;
    description: string;
    details: string;
    useCase: string;
    technical: string;
    detailsTitle?: string;
    useCaseTitle?: string;
    technicalTitle?: string;
}

export const InfoPanel: React.FC<InfoPanelProps> = ({
    title, description, details, useCase, technical,
    detailsTitle = "Operational Logic",
    useCaseTitle = "Use Case Context",
    technicalTitle = "Technical Engine"
}) => (
    <div className="bg-black/40 border border-white/5 p-6 rounded-2xl backdrop-blur-xl">
        <div className="flex items-center gap-3 mb-4">
            <Info size={16} className="text-blue-400" />
            <h4 className="text-[10px] font-black uppercase tracking-[0.3em] text-white italic">{title}</h4>
        </div>
        <p className="text-xs text-zinc-400 font-medium leading-relaxed mb-4 uppercase tracking-tighter">{description}</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-white/5">
            <div>
                <span className="block text-[8px] font-black text-zinc-600 uppercase mb-2">{detailsTitle}</span>
                <p className="text-[10px] text-zinc-500 leading-normal">{details}</p>
            </div>
            <div>
                <span className="block text-[8px] font-black text-zinc-600 uppercase mb-2">{useCaseTitle}</span>
                <p className="text-[10px] text-zinc-500 leading-normal">{useCase}</p>
            </div>
            <div>
                <span className="block text-[8px] font-black text-zinc-600 uppercase mb-2">{technicalTitle}</span>
                <p className="text-[10px] text-zinc-500 leading-normal italic font-mono">{technical}</p>
            </div>
        </div>
    </div>
);
