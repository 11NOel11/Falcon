import React, { useState, useMemo } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import Slider from '@/components/Slider';
import Toggle from '@/components/Toggle';
import trajectoryData from '@/data/trajectories.json';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type OptimizerKey = 'AdamW' | 'Muon' | 'Scion' | 'Falcon';

export default function TrajectoryPage() {
  const [visibleOptimizers, setVisibleOptimizers] = useState<Record<OptimizerKey, boolean>>({
    AdamW: true,
    Muon: true,
    Scion: true,
    Falcon: true,
  });

  const [learningRate, setLearningRate] = useState(0.01);
  const [iterations, setIterations] = useState(10);

  const plotData = useMemo(() => {
    const data: any[] = [];

    // Add loss surface
    data.push({
      type: 'surface',
      x: trajectoryData.lossSurface.x,
      y: trajectoryData.lossSurface.y,
      z: trajectoryData.lossSurface.z,
      colorscale: 'Viridis',
      opacity: 0.7,
      showscale: false,
      name: 'Loss Surface',
    });

    // Add optimizer trajectories
    Object.entries(trajectoryData.optimizers).forEach(([key, optimizer]) => {
      if (visibleOptimizers[key as OptimizerKey]) {
        const trajectory = optimizer.trajectory.slice(0, iterations);
        data.push({
          type: 'scatter3d',
          mode: 'lines+markers',
          x: trajectory.map((p) => p.x),
          y: trajectory.map((p) => p.y),
          z: trajectory.map((p) => p.z),
          line: {
            color: optimizer.color,
            width: 4,
          },
          marker: {
            size: 5,
            color: optimizer.color,
          },
          name: optimizer.name,
        });
      }
    });

    return data;
  }, [visibleOptimizers, iterations]);

  const layout = {
    paper_bgcolor: '#0A0F24',
    plot_bgcolor: '#0A0F24',
    scene: {
      xaxis: { title: 'θ₁', gridcolor: '#1C2240', color: '#9CA3AF' },
      yaxis: { title: 'θ₂', gridcolor: '#1C2240', color: '#9CA3AF' },
      zaxis: { title: 'Loss', gridcolor: '#1C2240', color: '#9CA3AF' },
      bgcolor: '#0A0F24',
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    height: 600,
    showlegend: true,
    legend: {
      font: { color: '#9CA3AF' },
      bgcolor: '#1C2240',
    },
  };

  return (
    <>
      <Head>
        <title>Trajectory Visualizer - Falcon UI</title>
      </Head>

      <main className="min-h-screen pt-24 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <h1 className="text-5xl font-display font-bold mb-4 text-white">
            Trajectory Visualizer
          </h1>
          <p className="text-xl text-gray-400 mb-12">
            Watch optimizers navigate the loss landscape, each with its unique path
          </p>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-blue">
                  Optimizer Selection
                </h3>
                <div className="space-y-3">
                  {Object.entries(trajectoryData.optimizers).map(([key, optimizer]) => (
                    <Toggle
                      key={key}
                      label={optimizer.name}
                      checked={visibleOptimizers[key as OptimizerKey]}
                      onChange={(checked) =>
                        setVisibleOptimizers((prev) => ({
                          ...prev,
                          [key]: checked,
                        }))
                      }
                      color={optimizer.color}
                    />
                  ))}
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-pink">
                  Parameters
                </h3>
                <div className="space-y-4">
                  <Slider
                    label="Learning Rate"
                    value={learningRate}
                    min={0.001}
                    max={0.1}
                    step={0.001}
                    onChange={setLearningRate}
                  />
                  <Slider
                    label="Iterations"
                    value={iterations}
                    min={1}
                    max={10}
                    step={1}
                    onChange={setIterations}
                  />
                </div>
              </Card>

              {/* Optimizer Info */}
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">
                  About Optimizers
                </h3>
                <div className="space-y-4 text-sm">
                  {Object.values(trajectoryData.optimizers).map((optimizer) => (
                    <div key={optimizer.name} className="space-y-1">
                      <h4
                        className="font-semibold"
                        style={{ color: optimizer.color }}
                      >
                        {optimizer.name}
                      </h4>
                      <p className="text-gray-400 italic text-xs">
                        {optimizer.description}
                      </p>
                      <code className="text-xs text-gray-500 block font-mono">
                        {optimizer.equation}
                      </code>
                    </div>
                  ))}
                </div>
              </Card>
            </div>

            {/* Visualization */}
            <div className="lg:col-span-2">
              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-white">
                  3D Loss Landscape
                </h3>
                <div className="bg-falcon-bg rounded-lg overflow-hidden">
                  <Plot
                    data={plotData}
                    layout={layout}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </div>

                <div className="mt-6 p-4 bg-falcon-bg/50 rounded-lg border border-falcon-blue/30">
                  <p className="text-sm text-gray-400 italic font-display">
                    "Each optimizer carves its own path through the manifold of loss—
                    some swift and direct, others spiraling with geometric grace."
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
