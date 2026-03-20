"use client";

import React, { useState, useMemo } from 'react';

type RawLeaderboardEntry = {
  model: string;
  variant: string;
  study_id: string;
  title?: string;
  average_bas: number;
  ecs?: number | string; // Effect Consistency Score
  total_cost: number;
  total_output_tokens: number;
  findings_breakdown?: Record<string, number>;
};

type AggregatedEntry = {
  id: string; // model + variant
  model: string;
  variant: string;
  overall_alignment: number;
  overall_ecs: string; // Display string for ECS
  total_cost: number;
  total_tokens: number;
  study_scores: Record<string, number>;
  study_ecs: Record<string, number | string>;
  raw_data: RawLeaderboardEntry[];
};

export default function LeaderboardList({ rawData }: { rawData: RawLeaderboardEntry[] }) {
  const [selectedVariant, setSelectedVariant] = useState<string>('All');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);

  // 1. Get unique variants for filter
  const variants = useMemo(() => {
    const v = new Set(rawData.map(d => d.variant));
    return ['All', ...Array.from(v)];
  }, [rawData]);

  // 2. Aggregate Data
  const aggregatedData = useMemo(() => {
    const grouped: Record<string, AggregatedEntry> = {};

    rawData.forEach(entry => {
      const key = `${entry.model}-${entry.variant}`;
      if (!grouped[key]) {
        grouped[key] = {
          id: key,
          model: entry.model,
          variant: entry.variant,
          overall_alignment: 0,
          overall_ecs: "N/A",
          total_cost: 0,
          total_tokens: 0,
          study_scores: {},
          study_ecs: {},
          raw_data: []
        };
      }
      
      grouped[key].study_scores[entry.study_id] = entry.average_bas;
      if (entry.ecs !== undefined) {
          grouped[key].study_ecs[entry.study_id] = entry.ecs;
      }

      grouped[key].total_cost += entry.total_cost;
      grouped[key].total_tokens += entry.total_output_tokens;
      grouped[key].raw_data.push(entry);
    });

    // Calculate averages and finalize
    return Object.values(grouped).map(group => {
      const studies = Object.values(group.study_scores);
      const avg = studies.reduce((a, b) => a + b, 0) / studies.length;
      
      // Calculate ECS average
      const ecsValues = Object.values(group.study_ecs).filter(v => typeof v === 'number') as number[];
      let avgEcsStr = "N/A";
      if (ecsValues.length > 0) {
          const avgEcs = ecsValues.reduce((a, b) => a + b, 0) / ecsValues.length;
          avgEcsStr = avgEcs.toFixed(4);
      }

      return {
        ...group,
        overall_alignment: avg * 100, // Convert to percentage
        overall_ecs: avgEcsStr
      };
    }).sort((a, b) => b.overall_alignment - a.overall_alignment); // Sort by score desc

  }, [rawData]);

  // 3. Filter
  const filteredData = useMemo(() => {
    if (selectedVariant === 'All') return aggregatedData;
    return aggregatedData.filter(d => d.variant === selectedVariant);
  }, [aggregatedData, selectedVariant]);

  // Helpers
  const formatScore = (val: number) => val.toFixed(1) + '%';
  const formatCost = (val: number) => '$' + val.toFixed(4);
  const formatInteger = (val: number) => new Intl.NumberFormat().format(val);
  
  const getScoreColor = (score: number) => {
      // Input score is percentage 0-100
      if (score >= 80) return "bg-green-500";
      if (score >= 60) return "bg-yellow-400";
      if (score >= 40) return "bg-orange-400";
      return "bg-red-400";
  };

  const modelShortName = (name: string) => name.split('/')[1] || name;

  const toggleExpand = (id: string) => {
    if (expandedRow === id) setExpandedRow(null);
    else setExpandedRow(id);
  };

  return (
    <div className="bg-white">
      {/* Filters */}
      <div className="flex justify-end mb-4 gap-2">
        <div className="flex items-center">
            <span className="mr-2 text-sm text-gray-600 font-medium">Filter by Variant:</span>
            <select 
                value={selectedVariant}
                onChange={(e) => setSelectedVariant(e.target.value)}
                className="block w-40 rounded-md border-0 py-1.5 pl-3 pr-10 text-gray-900 ring-1 ring-inset ring-gray-300 focus:ring-2 focus:ring-blue-600 sm:text-sm sm:leading-6"
            >
                {variants.map(v => <option key={v} value={v}>{v}</option>)}
            </select>
        </div>
      </div>

      <div className="overflow-x-auto shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
        <table className="min-w-full divide-y divide-gray-300">
          <thead className="bg-gray-50">
            <tr>
              <th scope="col" className="py-3.5 pl-4 pr-3 text-center text-sm font-semibold text-gray-900 sm:pl-6 w-12">Rank</th>
              <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Model</th>
              <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">Variant</th>
              <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900">PAS (Alignment)</th>
              <th scope="col" className="px-3 py-3.5 text-center text-sm font-semibold text-gray-900">ECS</th>
              <th scope="col" className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 hidden md:table-cell">Study Scores</th>
              <th scope="col" className="px-3 py-3.5 text-right text-sm font-semibold text-gray-900">Cost</th>
              <th scope="col" className="px-3 py-3.5 text-right text-sm font-semibold text-gray-900">Tokens</th>
              <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6"><span className="sr-only">Details</span></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 bg-white">
            {filteredData.map((row, index) => (
              <React.Fragment key={row.id}>
              <tr className="hover:bg-gray-50 cursor-pointer" onClick={() => toggleExpand(row.id)}>
                <td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm text-center font-medium text-gray-900 sm:pl-6">
                  {index + 1}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm font-medium text-gray-900">
                  {modelShortName(row.model)}
                </td>
                 <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500 font-mono text-xs">
                  {row.variant}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-center">
                    <span className={`inline-flex items-center rounded-md px-2 py-1 text-sm font-medium ring-1 ring-inset ring-gray-500/10 ${
                        row.overall_alignment > 75 ? 'text-green-700 bg-green-50' : 
                        row.overall_alignment > 50 ? 'text-yellow-700 bg-yellow-50' : 'text-red-700 bg-red-50'
                    }`}>
                        {formatScore(row.overall_alignment)}
                    </span>
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-center font-mono text-gray-600">
                    {row.overall_ecs}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm hidden md:table-cell">
                    <div className="flex gap-1">
                        {/* Only showing first 12 studies to save space if dataset grows */}
                        {Array.from({length: 12}).map((_, i) => {
                            const studyId = `study_${String(i+1).padStart(3, '0')}`;
                            const score = row.study_scores[studyId] ? row.study_scores[studyId] * 100 : 0;
                            const hasData = !!row.study_scores[studyId];
                            
                            return (
                                <div 
                                    key={studyId} 
                                    className={`w-3 h-6 rounded-sm ${hasData ? getScoreColor(score) : 'bg-gray-100'}`}
                                    title={`${studyId}: ${hasData ? score.toFixed(1) + '%' : 'N/A'}`}
                                ></div>
                            )
                        })}
                    </div>
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-right text-gray-500 tabular-nums">
                  {formatCost(row.total_cost)}
                </td>
                <td className="whitespace-nowrap px-3 py-4 text-sm text-right text-gray-500 tabular-nums">
                  {formatInteger(row.total_tokens)}
                </td>
                 <td className="whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                  <span className="text-blue-600 hover:text-blue-900">
                    {expandedRow === row.id ? 'Hide' : 'Show'}
                  </span>
                </td>
              </tr>
              {expandedRow === row.id && (
                  <tr className="bg-gray-50">
                      <td colSpan={8} className="p-4 sm:px-6">
                          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {row.raw_data.sort((a,b) => a.study_id.localeCompare(b.study_id)).map(studyData => {
                                const findingsStr = studyData.findings_breakdown 
                                    ? Object.entries(studyData.findings_breakdown)
                                        .map(([k, v]) => `${k} (${v})`)
                                        .join(', ')
                                    : '';
                                return (
                                <div 
                                    key={studyData.study_id} 
                                    className="bg-white p-3 rounded shadow-sm border border-gray-200 group relative"
                                >
                                    <div className="absolute inset-0 z-10 opacity-0 group-hover:opacity-100 bg-black/80 text-white p-3 text-xs rounded transition-opacity pointer-events-none flex flex-col justify-center">
                                        <div className="font-bold mb-1">{studyData.study_id}</div>
                                        <div className="mb-2 line-clamp-2">{studyData.title}</div>
                                        {findingsStr && (
                                            <div className="border-t border-white/20 pt-1">
                                                <span className="opacity-70">Findings:</span> {findingsStr}
                                            </div>
                                        )}
                                    </div>
                                    <h4 className="border-b pb-2 mb-2">
                                        <div className="flex justify-between items-baseline">
                                            <span className="font-bold text-xs uppercase text-gray-500">{studyData.study_id}</span>
                                        </div>
                                        <div className="text-sm font-medium text-gray-900 leading-tight pt-1" title={studyData.title}>
                                            {studyData.title || 'Untitled Study'}
                                        </div>
                                    </h4>
                                    
                                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                                        <div className="text-gray-500">ECS</div>
                                        <div className="text-right font-mono text-gray-900">
                                            {typeof studyData.ecs === 'number' ? studyData.ecs.toPrecision(4) : studyData.ecs || 'N/A'}
                                        </div>
                                        
                                        <div className="text-gray-500">PAS</div>
                                        <div className={`text-right font-mono ${studyData.average_bas > 0.8 ? "text-green-600" : "text-gray-900"}`}>
                                            {studyData.average_bas.toPrecision(4)}
                                        </div>

                                        <div className="text-gray-500">Cost</div>
                                        <div className="text-right font-mono text-gray-500">
                                            ${studyData.total_cost.toPrecision(4)}
                                        </div>

                                        <div className="text-gray-500">Tokens</div>
                                        <div className="text-right font-mono text-gray-500">
                                            {formatInteger(studyData.total_output_tokens)}
                                        </div>
                                    </div>
                                </div>
                                );
                            })}
                          </div>
                      </td>
                  </tr>
              )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
