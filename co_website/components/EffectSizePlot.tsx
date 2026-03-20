"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const DATA_URL = "/data/effects/gemini_flash_v4_effect_data.json";
const COLOR_SIG = "#2BA8AF";
const COLOR_NONSIG = "#C5EEF0";
const AXIS_HUMAN = "Human Effect Size";
const AXIS_AGENT = "Agent Effect Size";

export type EffectDatum = {
  Model: string;
  Method: string;
  ES_Original: number;
  ES_Replication: number;
  Significant_Replication: boolean;
  Power: number;
  Study_ID: string;
  Test_Name: string;
  Sub_Study_ID: string;
};

function weightedLinreg(
  x: number[],
  y: number[],
  w: number[]
): [number, number] | [null, null] {
  if (x.length !== y.length || x.length !== w.length || x.length < 2)
    return [null, null];
  const valid: number[] = [];
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    const yi = y[i];
    const wi = w[i];
    if (
      xi != null &&
      yi != null &&
      wi != null &&
      Number.isFinite(xi) &&
      Number.isFinite(yi) &&
      Number.isFinite(wi) &&
      wi > 0
    )
      valid.push(i);
  }
  if (valid.length < 2) return [null, null];
  const xv = valid.map((i) => x[i]);
  const yv = valid.map((i) => y[i]);
  const wv = valid.map((i) => w[i]);
  const tw = wv.reduce((a, b) => a + b, 0);
  if (tw <= 0) return [null, null];
  const muX = xv.reduce((s, v, i) => s + v * wv[i], 0) / tw;
  const muY = yv.reduce((s, v, i) => s + v * wv[i], 0) / tw;
  const cov = xv.reduce((s, v, i) => s + wv[i] * (v - muX) * (yv[i] - muY), 0);
  const vx = xv.reduce((s, v, i) => s + wv[i] * (v - muX) ** 2, 0);
  if (vx <= 0) return [null, null];
  const a = cov / vx;
  const b = muY - a * muX;
  return [a, b];
}

function markerSize(power: number): number {
  return 4 + power * power * 10;
}

/** From Study_ID e.g. "study_012" → "Study 012" */
function studyHoverLabel(studyId: string): string {
  const m = String(studyId ?? "").match(/study_(\d+)/i);
  return m ? `Study ${m[1]}` : `Study ${studyId || "—"}`;
}

export default function EffectSizePlot() {
  const [rawData, setRawData] = useState<EffectDatum[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(DATA_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((arr: EffectDatum[]) => {
        if (!cancelled) {
          setRawData(Array.isArray(arr) ? arr : []);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(e?.message ?? "Failed to load data");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const { slope, intercept, regressionX, regressionY, xMin, xMax, yMin, yMax } =
    useMemo(() => {
      if (rawData.length < 2) {
        const xMin = -1.5;
        const xMax = 3.5;
        return {
          slope: null as number | null,
          intercept: null as number | null,
          regressionX: [] as number[],
          regressionY: [] as number[],
          xMin: xMin,
          xMax: xMax,
          yMin: xMin,
          yMax: xMax,
        };
      }
      const studyCounts: Record<string, number> = {};
      rawData.forEach((d) => {
        const s = d.Study_ID ?? "";
        studyCounts[s] = (studyCounts[s] ?? 0) + 1;
      });
      const weights = rawData.map(
        (d) => 1 / (studyCounts[d.Study_ID ?? ""] || 1)
      );
      const x = rawData.map((d) => d.ES_Original);
      const y = rawData.map((d) => d.ES_Replication);
      const [a, b] = weightedLinreg(x, y, weights);
      const allX = rawData.map((d) => d.ES_Original);
      const allY = rawData.map((d) => d.ES_Replication);
      const xP2 = Math.min(...allX);
      const xP98 = Math.max(...allX);
      const yP2 = Math.min(...allY);
      const yP98 = Math.max(...allY);
      const padX = Math.max(0.1, (xP98 - xP2) * 0.05);
      const padY = Math.max(0.1, (yP98 - yP2) * 0.05);
      const xMin = xP2 - padX;
      const xMax = xP98 + padX;
      const yMin = yP2 - padY;
      const yMax = yP98 + padY;
      const rangeMin = Math.min(xMin, yMin);
      const rangeMax = Math.max(xMax, yMax);
      const lo = rangeMin;
      const hi = rangeMax;
      const regX = [lo, hi];
      const regY = a != null && b != null ? [a * lo + b, a * hi + b] : [];
      return {
        slope: a ?? null,
        intercept: b ?? null,
        regressionX: regX,
        regressionY: regY,
        xMin: lo,
        xMax: hi,
        yMin: lo,
        yMax: hi,
      };
    }, [rawData]);

  const sigData = useMemo(
    () => rawData.filter((d) => d.Significant_Replication),
    [rawData]
  );
  const nonsigData = useMemo(
    () => rawData.filter((d) => !d.Significant_Replication),
    [rawData]
  );

  const scatterTraces: Plotly.Data[] = [
    ...(sigData.length > 0
      ? [
          {
            x: sigData.map((d) => d.ES_Original),
            y: sigData.map((d) => d.ES_Replication),
            mode: "markers" as const,
            type: "scatter" as const,
            marker: {
              size: sigData.map((d) => markerSize(d.Power)),
              color: COLOR_SIG,
              line: { color: "white", width: 1 },
            },
            name: "Significant",
            text: sigData.map((d) => studyHoverLabel(d.Study_ID)),
            hoverinfo: "text" as const,
            hovertemplate: "%{text}<extra></extra>",
          },
        ]
      : []),
    ...(nonsigData.length > 0
      ? [
          {
            x: nonsigData.map((d) => d.ES_Original),
            y: nonsigData.map((d) => d.ES_Replication),
            mode: "markers" as const,
            type: "scatter" as const,
            marker: {
              size: nonsigData.map((d) => markerSize(d.Power)),
              color: COLOR_NONSIG,
              line: { color: "white", width: 1 },
            },
            name: "Not Significant",
            text: nonsigData.map((d) => studyHoverLabel(d.Study_ID)),
            hoverinfo: "text" as const,
            hovertemplate: "%{text}<extra></extra>",
          },
        ]
      : []),
  ];

  const shapes: Partial<Plotly.Shape>[] = [
    { type: "line", x0: xMin, x1: xMax, y0: xMin, y1: xMax, line: { color: "#999", width: 1.5, dash: "solid" } },
    { type: "line", x0: 0, x1: 0, y0: yMin, y1: yMax, line: { color: "#999", width: 1, dash: "dot" } },
    { type: "line", x0: xMin, x1: xMax, y0: 0, y1: 0, line: { color: "#999", width: 1, dash: "dot" } },
  ];

  const layout: Partial<Plotly.Layout> = {
    title: { text: "Gemini 3 Flash: Human vs Agent Effect Size", font: { size: 14 }, xref: "paper", x: 0.5 },
    xaxis: { 
      title: { text: AXIS_HUMAN, font: { size: 12 } }, 
      range: [xMin, xMax], 
      constrain: "domain", 
      fixedrange: true, 
      tickfont: { size: 10 },
      automargin: true,
      gridcolor: '#f3f4f6',
      zerolinecolor: '#9ca3af'
    },
    yaxis: { 
      title: { text: AXIS_AGENT, font: { size: 12 } }, 
      range: [yMin, yMax], 
      scaleanchor: "x", 
      scaleratio: 1, 
      fixedrange: true, 
      tickfont: { size: 10 },
      automargin: true,
      gridcolor: '#f3f4f6',
      zerolinecolor: '#9ca3af'
    },
    showlegend: true,
    legend: { x: 1, y: 1, xanchor: "right", font: { size: 11 } },
    shapes,
    margin: { t: 45, r: 10, b: 45, l: 50 },
    hovermode: "closest" as const,
    dragmode: false,
    autosize: true,
    plot_bgcolor: 'white',
    paper_bgcolor: 'white'
  };

  const regressionTrace: Plotly.Data | null =
    slope != null && intercept != null && regressionX.length === 2
      ? {
          x: regressionX,
          y: regressionY,
          mode: "lines",
          type: "scatter",
          line: { color: "#333", width: 2, dash: "dash" },
          name: "Regression",
          hoverinfo: "skip",
        }
      : null;

  const allTraces: Plotly.Data[] = regressionTrace
    ? [...scatterTraces, regressionTrace]
    : scatterTraces;

  if (loading) {
    return (
      <div className="w-full flex justify-center items-center min-h-[350px] bg-gray-50 rounded-xl">
        <p className="text-gray-500">Loading effect size data…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full rounded-xl bg-red-50 border border-red-200 p-4 text-red-700">
        {error}
      </div>
    );
  }

  return (
    <div id="effects" className="w-full h-full flex flex-col">
      <div className="bg-white overflow-hidden flex-grow">
        <Plot
          data={allTraces}
          layout={layout}
          config={{ responsive: true, displayModeBar: false }}
          style={{ width: "100%", height: "100%", minHeight: 520 }}
          useResizeHandler
        />
      </div>
    </div>
  );
}