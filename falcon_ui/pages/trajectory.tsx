import React, { useState, useMemo, useEffect } from 'react';
import Head from 'next/head';
import dynamic from 'next/dynamic';
import Card from '@/components/Card';
import Slider from '@/components/Slider';
import Toggle from '@/components/Toggle';
import trajectoryDataRaw from '@/data/trajectories.json';
import {
  getLossFunction,
  generateLossSurface,
  generateAdamWTrajectory,
  generateMuonTrajectory,
  generateFalconTrajectory,
  LOSS_FUNCTIONS,
} from '@/utils/lossLandscapes';

// Dynamically import Plot to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

type OptimizerKey = 'AdamW' | 'Muon' | 'Falcon';
type LandscapeType = 'rosenbrock' | 'rastrigin' | 'beale' | 'himmelblau' | 'ackley';

const trajectoryData: any = trajectoryDataRaw;

export default function TrajectoryPage() {
  const [visibleOptimizers, setVisibleOptimizers] = useState<Record<OptimizerKey, boolean>>({
    AdamW: true,
    Muon: true,
    Falcon: true,
  });

  const [learningRate, setLearningRate] = useState(0.01);
  const [momentum, setMomentum] = useState(0.9);
  const [batchSize, setBatchSize] = useState(256);
  const [maxIterations, setMaxIterations] = useState(10);
  const [currentIteration, setCurrentIteration] = useState(10);
  const [isAnimating, setIsAnimating] = useState(false);
  const [landscape, setLandscape] = useState<LandscapeType>('rosenbrock');
  const [showMetrics, setShowMetrics] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(300);

  const currentLandscape = trajectoryData.landscapes[landscape];

  // Animation loop
  useEffect(() => {
    if (!isAnimating) return;

    const interval = setInterval(() => {
      setCurrentIteration((prev) => {
        if (prev >= maxIterations) {
          setIsAnimating(false);
          return maxIterations;
        }
        return prev + 1;
      });
    }, animationSpeed);

    return () => clearInterval(interval);
  }, [isAnimating, maxIterations, animationSpeed]);

  const startAnimation = () => {
    setCurrentIteration(1);
    setIsAnimating(true);
  };

  const togglePause = () => {
    setIsAnimating(!isAnimating);
  };

  const resetAnimation = () => {
    setIsAnimating(false);
    setCurrentIteration(maxIterations);
  };

  // Calculate optimizer metrics
  const optimizerMetrics = useMemo(() => {
    const metrics: Record<string, { finalLoss: number; convergenceSpeed: number; pathLength: number }> = {};

    Object.entries(currentLandscape.optimizers).forEach(([key, optimizerData]: [string, any]) => {
      const trajectory = optimizerData.trajectory.slice(0, currentIteration);
      const finalLoss = trajectory[trajectory.length - 1]?.z || 100;

      // Find iteration where loss dropped below threshold
      const convergenceIter = trajectory.findIndex((p: any) => p.z < 1.5);
      const convergenceSpeed = convergenceIter > 0 ? convergenceIter : currentIteration;

      // Calculate path length
      let pathLength = 0;
      for (let i = 1; i < trajectory.length; i++) {
        const dx = trajectory[i].x - trajectory[i-1].x;
        const dy = trajectory[i].y - trajectory[i-1].y;
        const dz = trajectory[i].z - trajectory[i-1].z;
        pathLength += Math.sqrt(dx*dx + dy*dy + dz*dz);
      }

      metrics[key] = { finalLoss, convergenceSpeed, pathLength };
    });

    return metrics;
  }, [currentIteration, landscape]);

  // Find winner based on current data
  const winner = useMemo(() => {
    const losses = Object.entries(optimizerMetrics)
      .filter(([key]) => visibleOptimizers[key as OptimizerKey])
      .map(([key, m]) => ({ key, loss: m.finalLoss }));
    losses.sort((a, b) => a.loss - b.loss);
    return losses[0]?.key || currentLandscape.winner;
  }, [optimizerMetrics, visibleOptimizers, landscape]);

  const plotData = useMemo(() => {
    const data: any[] = [];

    // Add loss surface
    data.push({
      type: 'surface',
      x: currentLandscape.lossSurface.x,
      y: currentLandscape.lossSurface.y,
      z: currentLandscape.lossSurface.z,
      colorscale: 'Viridis',
      opacity: 0.65,
      showscale: false,
      name: 'Loss Surface',
      hoverinfo: 'skip',
    });

    // Add optimizer trajectories
    Object.entries(currentLandscape.optimizers).forEach(([key, optimizerData]: [string, any]) => {
      if (visibleOptimizers[key as OptimizerKey]) {
        const trajectory = optimizerData.trajectory.slice(0, currentIteration);
        const optimizer = trajectoryData.optimizers[key];

        // Add trajectory line
        data.push({
          type: 'scatter3d',
          mode: 'lines+markers',
          x: trajectory.map((p: any) => p.x),
          y: trajectory.map((p: any) => p.y),
          z: trajectory.map((p: any) => p.z),
          line: {
            color: optimizer.color,
            width: 6,
          },
          marker: {
            size: 6,
            color: optimizer.color,
            symbol: 'circle',
          },
          name: optimizer.name,
          hovertemplate: `${optimizer.name}<br>Loss: %{z:.2f}<extra></extra>`,
        });

        // Add current position marker (larger)
        if (trajectory.length > 0) {
          const current = trajectory[trajectory.length - 1];
          data.push({
            type: 'scatter3d',
            mode: 'markers',
            x: [current.x],
            y: [current.y],
            z: [current.z],
            marker: {
              size: 15,
              color: optimizer.color,
              symbol: 'diamond',
              line: {
                color: '#ffffff',
                width: 3,
              },
            },
            name: `${optimizer.name} (current)`,
            showlegend: false,
            hoverinfo: 'skip',
          });
        }
      }
    });

    return data;
  }, [visibleOptimizers, currentIteration, landscape]);

  const layout = {
    paper_bgcolor: '#0A0F24',
    plot_bgcolor: '#0A0F24',
    scene: {
      xaxis: { title: 'θ₁', gridcolor: '#1C2240', color: '#9CA3AF' },
      yaxis: { title: 'θ₂', gridcolor: '#1C2240', color: '#9CA3AF' },
      zaxis: { title: 'Loss', gridcolor: '#1C2240', color: '#9CA3AF' },
      bgcolor: '#0A0F24',
      camera: {
        eye: { x: 1.5, y: 1.5, z: 1.3 }
      },
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    height: 600,
    showlegend: true,
    legend: {
      font: { color: '#9CA3AF', size: 12 },
      bgcolor: '#1C224099',
      x: 0.02,
      y: 0.98,
    },
  };

  return (
    <>
      <Head>
        <title>Trajectory Visualizer - Falcon UI</title>
      </Head>

      <main className="min-h-screen pt-24 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <h1 className="text-5xl font-display font-bold mb-4 text-white animate-fade-in-up">
            Trajectory Visualizer
          </h1>
          <p className="text-xl text-gray-400 mb-8 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            Watch optimizers navigate the loss landscape in real-time
          </p>

          {/* Winner Banner */}
          {currentIteration === maxIterations && (
            <div className="mb-8 p-6 bg-gradient-to-r from-falcon-cyan/20 to-falcon-blue/20 border border-falcon-cyan/50 rounded-lg animate-pulse-glow">
              <div className="flex items-center justify-between flex-wrap gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-falcon-cyan mb-2">
                    🏆 Winner: {trajectoryData.optimizers[winner]?.name || 'Unknown'}
                  </h2>
                  <p className="text-gray-300">
                    Achieved lowest loss of <strong className="text-falcon-cyan">
                      {optimizerMetrics[winner]?.finalLoss.toFixed(3)}
                    </strong> in {currentIteration} iterations
                  </p>
                  <p className="text-sm text-gray-400 mt-2 italic">
                    {currentLandscape.explanation}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-400">Landscape: <span className="text-white font-semibold">{currentLandscape.name}</span></p>
                  <p className="text-sm text-gray-400 mt-1">Expected Winner: <span className="text-falcon-pink font-semibold">{currentLandscape.winner}</span></p>
                </div>
              </div>
            </div>
          )}

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              {/* Animation Controls */}
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">
                  ⏯️ Animation Controls
                </h3>
                <div className="space-y-3">
                  <button
                    onClick={startAnimation}
                    disabled={isAnimating && currentIteration === 1}
                    className="w-full px-4 py-3 bg-gradient-to-r from-falcon-blue to-falcon-cyan text-white font-semibold rounded-lg hover:shadow-lg hover:shadow-falcon-blue/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    ▶️ Start Animation
                  </button>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={togglePause}
                      className="px-4 py-2 bg-falcon-card text-gray-300 font-semibold rounded-lg hover:bg-falcon-bg transition-all"
                    >
                      {isAnimating ? '⏸️ Pause' : '▶️ Resume'}
                    </button>
                    <button
                      onClick={resetAnimation}
                      className="px-4 py-2 bg-falcon-card text-gray-300 font-semibold rounded-lg hover:bg-falcon-bg transition-all"
                    >
                      ⏹️ Reset
                    </button>
                  </div>
                  <div className="mt-4 p-3 bg-falcon-bg rounded-lg">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">Progress</span>
                      <span className="text-falcon-cyan font-mono">{currentIteration}/{maxIterations}</span>
                    </div>
                    <div className="w-full bg-falcon-card rounded-full h-2 overflow-hidden">
                      <div
                        className="bg-gradient-to-r from-falcon-blue to-falcon-cyan h-full transition-all duration-300"
                        style={{ width: `${(currentIteration / maxIterations) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              </Card>

              {/* Landscape Selection */}
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">
                  🗺️ Loss Landscape
                </h3>
                <div className="space-y-2">
                  {(Object.keys(trajectoryData.landscapes) as LandscapeType[]).map((key) => {
                    const landData = trajectoryData.landscapes[key];
                    return (
                      <button
                        key={key}
                        onClick={() => {
                          setLandscape(key);
                          resetAnimation();
                        }}
                        className={`w-full text-left px-4 py-3 rounded-lg font-semibold transition-all ${
                          landscape === key
                            ? 'bg-gradient-to-r from-falcon-blue/30 to-falcon-cyan/30 border border-falcon-cyan text-white'
                            : 'bg-falcon-bg text-gray-400 hover:bg-falcon-card'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <div className="font-bold text-sm">{landData.name}</div>
                          {landData.winner === 'AdamW' && <span className="text-xs px-2 py-0.5 bg-blue-500/30 rounded">AdamW wins</span>}
                          {landData.winner === 'Muon' && <span className="text-xs px-2 py-0.5 bg-pink-500/30 rounded">Muon wins</span>}
                          {landData.winner === 'Falcon' && <span className="text-xs px-2 py-0.5 bg-cyan-500/30 rounded">Falcon wins</span>}
                        </div>
                        <div className="text-xs text-gray-500 mt-1">{landData.explanation}</div>
                      </button>
                    );
                  })}
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-blue">
                  🎯 Optimizer Selection
                </h3>
                <div className="space-y-3">
                  {Object.entries(trajectoryData.optimizers).map(([key, optimizer]: [string, any]) => (
                    <div key={key} className="flex items-center justify-between p-2 bg-falcon-bg rounded-lg">
                      <Toggle
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
                      {winner === key && currentIteration === maxIterations && (
                        <span className="text-2xl animate-pulse">🏆</span>
                      )}
                    </div>
                  ))}
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-pink">
                  ⚙️ Hyperparameters
                </h3>
                <div className="space-y-4">
                  <Slider
                    label="Learning Rate (α)"
                    value={learningRate}
                    min={0.001}
                    max={0.1}
                    step={0.001}
                    onChange={setLearningRate}
                  />
                  <Slider
                    label="Momentum (β)"
                    value={momentum}
                    min={0.5}
                    max={0.99}
                    step={0.01}
                    onChange={setMomentum}
                  />
                  <Slider
                    label="Batch Size"
                    value={batchSize}
                    min={32}
                    max={512}
                    step={32}
                    onChange={setBatchSize}
                  />
                  <Slider
                    label="Max Iterations"
                    value={maxIterations}
                    min={5}
                    max={10}
                    step={1}
                    onChange={(val) => {
                      setMaxIterations(val);
                      if (currentIteration > val) setCurrentIteration(val);
                    }}
                  />
                  <Slider
                    label="Animation Speed (ms)"
                    value={animationSpeed}
                    min={100}
                    max={800}
                    step={50}
                    onChange={setAnimationSpeed}
                  />
                  <Toggle
                    label="Show Live Metrics"
                    checked={showMetrics}
                    onChange={setShowMetrics}
                  />
                </div>
              </Card>
            </div>

            {/* Visualization */}
            <div className="lg:col-span-2 space-y-6">
              <Card hover={false}>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-bold text-white">
                    3D Loss Landscape - {currentLandscape.name}
                  </h3>
                  <div className="text-sm text-gray-400 font-mono">
                    Iteration: <span className="text-falcon-cyan font-bold">{currentIteration}</span>
                  </div>
                </div>
                <div className="bg-falcon-bg rounded-lg overflow-hidden border border-falcon-blue/30">
                  <Plot
                    data={plotData}
                    layout={layout}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </div>
              </Card>

              {/* Live Metrics */}
              {showMetrics && (
                <Card>
                  <h3 className="text-xl font-bold mb-6 text-white">
                    📊 Live Performance Metrics
                  </h3>
                  <div className="grid md:grid-cols-3 gap-4">
                    {Object.entries(optimizerMetrics)
                      .filter(([key]) => visibleOptimizers[key as OptimizerKey])
                      .map(([key, metrics]) => {
                        const optimizer = trajectoryData.optimizers[key];
                        const isWinner = winner === key && currentIteration === maxIterations;
                        return (
                          <div
                            key={key}
                            className={`p-4 rounded-lg border-2 transition-all ${
                              isWinner
                                ? 'border-falcon-cyan bg-falcon-cyan/10 animate-pulse-glow'
                                : 'border-falcon-card bg-falcon-card/50'
                            }`}
                          >
                            <div className="flex items-center gap-2 mb-3">
                              <div
                                className="w-4 h-4 rounded-full"
                                style={{ backgroundColor: optimizer.color }}
                              />
                              <h4 className="font-bold text-white">{optimizer.name}</h4>
                              {isWinner && <span className="text-xl">🏆</span>}
                            </div>
                            <div className="space-y-2 text-sm">
                              <div className="flex justify-between">
                                <span className="text-gray-400">Current Loss:</span>
                                <span className="font-mono text-falcon-cyan font-bold">
                                  {metrics.finalLoss.toFixed(3)}
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Convergence:</span>
                                <span className="font-mono text-gray-300">
                                  {metrics.convergenceSpeed} iter
                                </span>
                              </div>
                              <div className="flex justify-between">
                                <span className="text-gray-400">Path Length:</span>
                                <span className="font-mono text-gray-300">
                                  {metrics.pathLength.toFixed(2)}
                                </span>
                              </div>
                            </div>
                            {isWinner && (
                              <div className="mt-3 pt-3 border-t border-falcon-cyan/30">
                                <p className="text-xs text-gray-400 italic">
                                  {currentLandscape.explanation}
                                </p>
                              </div>
                            )}
                          </div>
                        );
                      })}
                  </div>
                </Card>
              )}

              {/* Loss Function Mathematics */}
              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-falcon-pink">
                  🧮 Loss Function Mathematics
                </h3>
                <div className="p-4 bg-falcon-bg rounded-lg border border-falcon-pink/20">
                  <h4 className="font-semibold text-lg text-white mb-2">
                    {LOSS_FUNCTIONS[landscape]?.name || currentLandscape.name}
                  </h4>
                  <div className="mb-3 p-3 bg-falcon-card rounded font-mono text-sm text-falcon-cyan">
                    {LOSS_FUNCTIONS[landscape]?.equation}
                  </div>
                  <p className="text-gray-400 text-sm mb-2">
                    {LOSS_FUNCTIONS[landscape]?.description}
                  </p>
                  <div className="grid grid-cols-2 gap-3 mt-3 text-xs">
                    <div className="p-2 bg-falcon-card rounded">
                      <div className="text-gray-500">Global Minimum</div>
                      <div className="text-falcon-cyan font-mono">
                        ({LOSS_FUNCTIONS[landscape]?.globalMinimum.x}, {LOSS_FUNCTIONS[landscape]?.globalMinimum.y})
                      </div>
                    </div>
                    <div className="p-2 bg-falcon-card rounded">
                      <div className="text-gray-500">Local Minima</div>
                      <div className="text-falcon-pink font-mono">
                        {LOSS_FUNCTIONS[landscape]?.hasLocalMinima ? 'Yes (Multiple)' : 'No'}
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Optimizer Info */}
              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">
                  📖 About Optimizers
                </h3>
                <div className="grid md:grid-cols-3 gap-4">
                  {Object.values(trajectoryData.optimizers).map((optimizer: any) => (
                    <div key={optimizer.name} className="p-4 bg-falcon-bg rounded-lg border border-falcon-blue/20">
                      <h4
                        className="font-semibold text-lg mb-2"
                        style={{ color: optimizer.color }}
                      >
                        {optimizer.name}
                      </h4>
                      <p className="text-gray-400 italic text-sm mb-2">
                        {optimizer.description}
                      </p>
                      <code className="text-xs text-gray-500 block font-mono bg-falcon-card p-2 rounded">
                        {optimizer.equation}
                      </code>
                    </div>
                  ))}
                </div>
              </Card>

              <Card hover={false}>
                <div className="p-4 bg-gradient-to-br from-falcon-card to-falcon-bg border border-falcon-purple/30 rounded-lg">
                  <p className="text-sm text-gray-400 italic font-display text-center">
                    "Each optimizer carves its own path through the manifold of loss—
                    <br />
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
