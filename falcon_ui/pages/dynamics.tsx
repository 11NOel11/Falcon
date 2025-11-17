import React, { useState, useMemo } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import Toggle from '@/components/Toggle';
import dynamicsData from '@/data/dynamics.json';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type MetricType = 'loss' | 'accuracy' | 'schedule';

export default function DynamicsPage() {
  const [selectedMetric, setSelectedMetric] = useState<MetricType>('loss');
  const [visibleOptimizers, setVisibleOptimizers] = useState({
    AdamW: true,
    Muon: true,
    Scion: true,
    Falcon: true,
  });

  const optimizerColors = {
    AdamW: '#4FACF7',
    Muon: '#E87BF8',
    Scion: '#9D4EDD',
    Falcon: '#00F5FF',
  };

  const plotData = useMemo(() => {
    const data: any[] = [];

    if (selectedMetric === 'loss') {
      Object.entries(dynamicsData.trainingLoss).forEach(([key, values]) => {
        if (key !== 'epochs' && visibleOptimizers[key as keyof typeof visibleOptimizers]) {
          data.push({
            type: 'scatter',
            mode: 'lines+markers',
            x: dynamicsData.trainingLoss.epochs,
            y: values,
            name: key,
            line: {
              color: optimizerColors[key as keyof typeof optimizerColors],
              width: 3,
            },
            marker: { size: 6 },
          });
        }
      });
    } else if (selectedMetric === 'accuracy') {
      Object.entries(dynamicsData.validationAccuracy).forEach(([key, values]) => {
        if (key !== 'epochs' && visibleOptimizers[key as keyof typeof visibleOptimizers]) {
          data.push({
            type: 'scatter',
            mode: 'lines+markers',
            x: dynamicsData.validationAccuracy.epochs,
            y: values,
            name: key,
            line: {
              color: optimizerColors[key as keyof typeof optimizerColors],
              width: 3,
            },
            marker: { size: 6 },
          });
        }
      });
    } else if (selectedMetric === 'schedule') {
      data.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: dynamicsData.retainFraction.epochs,
        y: dynamicsData.retainFraction.values,
        name: 'Retain Fraction (ρ)',
        line: { color: '#4FACF7', width: 3 },
        marker: { size: 6 },
        yaxis: 'y',
      });
      data.push({
        type: 'scatter',
        mode: 'lines+markers',
        x: dynamicsData.interleavingPeriod.epochs,
        y: dynamicsData.interleavingPeriod.values,
        name: 'Interleaving Period (K)',
        line: { color: '#E87BF8', width: 3 },
        marker: { size: 6 },
        yaxis: 'y2',
      });
    }

    return data;
  }, [selectedMetric, visibleOptimizers]);

  const getLayout = () => {
    const baseLayout: any = {
      paper_bgcolor: '#0A0F24',
      plot_bgcolor: '#1C2240',
      margin: { l: 60, r: 60, t: 40, b: 60 },
      height: 500,
      showlegend: true,
      legend: {
        font: { color: '#9CA3AF' },
        bgcolor: '#1C2240',
        bordercolor: '#4FACF7',
        borderwidth: 1,
      },
      xaxis: {
        title: 'Epoch',
        gridcolor: '#2D3748',
        color: '#9CA3AF',
      },
    };

    if (selectedMetric === 'loss') {
      baseLayout.yaxis = {
        title: 'Training Loss',
        gridcolor: '#2D3748',
        color: '#9CA3AF',
      };
    } else if (selectedMetric === 'accuracy') {
      baseLayout.yaxis = {
        title: 'Validation Accuracy',
        gridcolor: '#2D3748',
        color: '#9CA3AF',
        range: [0, 1],
      };
    } else if (selectedMetric === 'schedule') {
      baseLayout.yaxis = {
        title: 'Retain Fraction (ρ)',
        gridcolor: '#2D3748',
        color: '#4FACF7',
      };
      baseLayout.yaxis2 = {
        title: 'Interleaving Period (K)',
        overlaying: 'y',
        side: 'right',
        color: '#E87BF8',
      };
    }

    return baseLayout;
  };

  const metricButtons: { type: MetricType; label: string; color: string }[] = [
    { type: 'loss', label: 'Training Loss', color: 'falcon-blue' },
    { type: 'accuracy', label: 'Validation Accuracy', color: 'falcon-pink' },
    { type: 'schedule', label: 'Falcon Schedule', color: 'falcon-cyan' },
  ];

  return (
    <>
      <Head>
        <title>Training Dynamics - Falcon UI</title>
      </Head>

      <main className="min-h-screen pt-24 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <h1 className="text-5xl font-display font-bold mb-4 text-white">
            Training Dynamics
          </h1>
          <p className="text-xl text-gray-400 mb-12">
            Explore how different optimizers evolve across training epochs
          </p>

          <div className="grid lg:grid-cols-4 gap-8">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-blue">
                  Metric Selection
                </h3>
                <div className="space-y-2">
                  {metricButtons.map((btn) => (
                    <button
                      key={btn.type}
                      onClick={() => setSelectedMetric(btn.type)}
                      className={`w-full px-4 py-3 rounded-lg font-semibold transition-all text-left ${
                        selectedMetric === btn.type
                          ? `bg-${btn.color} text-white shadow-lg`
                          : 'bg-falcon-bg text-gray-400 hover:bg-falcon-card'
                      }`}
                    >
                      {btn.label}
                    </button>
                  ))}
                </div>
              </Card>

              {selectedMetric !== 'schedule' && (
                <Card>
                  <h3 className="text-xl font-bold mb-4 text-falcon-pink">
                    Visible Optimizers
                  </h3>
                  <div className="space-y-3">
                    {Object.keys(visibleOptimizers).map((key) => (
                      <Toggle
                        key={key}
                        label={key}
                        checked={visibleOptimizers[key as keyof typeof visibleOptimizers]}
                        onChange={(checked) =>
                          setVisibleOptimizers((prev) => ({
                            ...prev,
                            [key]: checked,
                          }))
                        }
                        color={optimizerColors[key as keyof typeof optimizerColors]}
                      />
                    ))}
                  </div>
                </Card>
              )}

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">Insights</h3>
                <div className="space-y-3 text-sm text-gray-400">
                  {selectedMetric === 'loss' && (
                    <p>
                      Falcon achieves lower training loss faster by adaptively
                      filtering high-frequency gradient noise.
                    </p>
                  )}
                  {selectedMetric === 'accuracy' && (
                    <p>
                      Superior generalization comes from low-rank updates that avoid
                      overfitting to training data.
                    </p>
                  )}
                  {selectedMetric === 'schedule' && (
                    <p>
                      The retain fraction ρ decreases over time, while interleaving
                      period K reduces to balance exploration and exploitation.
                    </p>
                  )}
                </div>
              </Card>
            </div>

            {/* Visualization */}
            <div className="lg:col-span-3">
              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-white">
                  {metricButtons.find((b) => b.type === selectedMetric)?.label}
                </h3>
                <div className="bg-falcon-bg rounded-lg overflow-hidden">
                  <Plot
                    data={plotData}
                    layout={getLayout()}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </div>

                <div className="mt-6 grid md:grid-cols-3 gap-4">
                  <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-blue/30">
                    <h4 className="text-sm font-bold text-falcon-blue mb-1">
                      Energy Distribution
                    </h4>
                    <p className="text-xs text-gray-500">
                      Most gradient energy concentrates in low frequencies
                    </p>
                  </div>

                  <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-pink/30">
                    <h4 className="text-sm font-bold text-falcon-pink mb-1">
                      Rank-1 Focus
                    </h4>
                    <p className="text-xs text-gray-500">
                      Principal direction captures essential update information
                    </p>
                  </div>

                  <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-cyan/30">
                    <h4 className="text-sm font-bold text-falcon-cyan mb-1">
                      Adaptive Schedule
                    </h4>
                    <p className="text-xs text-gray-500">
                      Dynamic masking balances noise reduction and signal preservation
                    </p>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-falcon-bg/50 rounded-lg border border-falcon-purple/30">
                  <p className="text-sm text-gray-400 italic font-display text-center">
                    "Training unfolds as a symphony—frequencies harmonize,
                    <br />
                    structure emerges, and convergence sings its final note."
                  </p>
                </div>
              </Card>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
