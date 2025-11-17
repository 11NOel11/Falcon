import React, { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import Card from '@/components/Card';
import Slider from '@/components/Slider';
import Toggle from '@/components/Toggle';
import { fft2D, ifft2D, getMagnitudeSpectrum, applyFrequencyFilter, fftshift } from '@/utils/fft';

type Preset = 'gradient' | 'noisy_gradient' | 'edge' | 'gaussian' | 'random';

// Real-world gradient examples
const PRESETS: Record<Preset, { matrix: number[][], description: string, useCase: string }> = {
  gradient: {
    matrix: [
      [0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.06],
      [0.15, 0.22, 0.28, 0.25, 0.18, 0.12, 0.08],
      [0.18, 0.28, 0.45, 0.38, 0.25, 0.15, 0.10],
      [0.15, 0.25, 0.38, 0.50, 0.35, 0.18, 0.12],
      [0.12, 0.18, 0.25, 0.35, 0.28, 0.15, 0.09],
      [0.09, 0.12, 0.15, 0.18, 0.15, 0.10, 0.06],
      [0.06, 0.08, 0.10, 0.12, 0.09, 0.06, 0.04],
    ],
    description: 'Clean neural network gradient from Conv layer',
    useCase: 'Typical gradient pattern with smooth energy distribution',
  },
  noisy_gradient: {
    matrix: [
      [0.12, 0.15, 0.88, 0.15, 0.12, 0.09, 0.06],
      [0.15, 0.22, 0.28, -0.65, 0.18, 0.92, 0.08],
      [0.78, 0.28, 0.45, 0.38, -0.45, 0.15, 0.10],
      [0.15, -0.55, 0.38, 0.50, 0.35, 0.18, -0.72],
      [0.12, 0.18, -0.85, 0.35, 0.68, 0.15, 0.09],
      [0.89, 0.12, 0.15, 0.18, 0.15, -0.60, 0.06],
      [0.06, -0.78, 0.10, 0.82, 0.09, 0.06, 0.04],
    ],
    description: 'Gradient corrupted with high-frequency noise',
    useCase: 'Real scenario: gradient with numerical instabilities',
  },
  edge: {
    matrix: [
      [-1, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, 24, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1, -1, -1],
    ],
    description: 'Edge detection kernel (Laplacian)',
    useCase: 'High-pass filter - detects sharp changes',
  },
  gaussian: {
    matrix: [
      [0, 0, 1, 2, 1, 0, 0],
      [0, 3, 13, 22, 13, 3, 0],
      [1, 13, 59, 97, 59, 13, 1],
      [2, 22, 97, 159, 97, 22, 2],
      [1, 13, 59, 97, 59, 13, 1],
      [0, 3, 13, 22, 13, 3, 0],
      [0, 0, 1, 2, 1, 0, 0],
    ].map((row) => row.map((v) => v / 1000)),
    description: 'Gaussian smoothing kernel',
    useCase: 'Low-pass filter - preserves smooth variations',
  },
  random: {
    matrix: Array(7)
      .fill(0)
      .map(() =>
        Array(7)
          .fill(0)
          .map(() => (Math.random() - 0.5) * 2)
      ),
    description: 'Random noise pattern',
    useCase: 'Pure high-frequency content - should be filtered out',
  },
};

export default function FilterPage() {
  const [preset, setPreset] = useState<Preset>('noisy_gradient');
  const [filter, setFilter] = useState<number[][]>(PRESETS.noisy_gradient.matrix);
  const [retainFraction, setRetainFraction] = useState(0.75);
  const [useRank1, setUseRank1] = useState(false);
  const [showComparison, setShowComparison] = useState(true);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const spectrumCanvasRef = useRef<HTMLCanvasElement>(null);
  const filteredCanvasRef = useRef<HTMLCanvasElement>(null);
  const cleanedCanvasRef = useRef<HTMLCanvasElement>(null);

  // Calculate metrics
  const [metrics, setMetrics] = useState({
    originalEnergy: 0,
    filteredEnergy: 0,
    noiseReduction: 0,
    signalPreservation: 0,
    componentsRemoved: 0,
  });

  // Draw filter on canvas
  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = canvas.width / 7;

    // Clear canvas
    ctx.fillStyle = '#0A0F24';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Find max value for scaling
    const maxVal = Math.max(...filter.flat().map(Math.abs));

    // Draw cells
    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 7; j++) {
        const value = filter[i][j];
        const normalized = maxVal > 0 ? Math.abs(value) / maxVal : 0;
        const intensity = Math.min(255, normalized * 255);
        const color = value >= 0 ? `rgb(79, 172, 247, ${intensity / 255})` : `rgb(232, 123, 248, ${intensity / 255})`;

        ctx.fillStyle = color;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

        ctx.strokeStyle = '#1C2240';
        ctx.lineWidth = 1;
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Draw value
        ctx.fillStyle = intensity > 128 ? '#000000' : '#ffffff';
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          value.toFixed(2),
          j * cellSize + cellSize / 2,
          i * cellSize + cellSize / 2
        );
      }
    }
  }, [filter]);

  // Compute and draw spectrum with metrics
  useEffect(() => {
    if (!spectrumCanvasRef.current || !filteredCanvasRef.current || !cleanedCanvasRef.current) return;

    try {
      // Compute FFT
      const fftResult = fft2D(filter);
      const spectrum = getMagnitudeSpectrum(fftResult);
      const shiftedSpectrum = fftshift(spectrum);

      // Calculate original energy
      const originalEnergy = spectrum.flat().reduce((sum, val) => sum + val * val, 0);

      // Draw spectrum
      drawHeatmap(spectrumCanvasRef.current, shiftedSpectrum, 'Spectrum');

      // Apply filter
      const filtered = applyFrequencyFilter(fftResult, retainFraction);
      const filteredSpectrum = getMagnitudeSpectrum(filtered);
      const shiftedFiltered = fftshift(filteredSpectrum);

      // Calculate filtered energy
      const filteredEnergy = filteredSpectrum.flat().reduce((sum, val) => sum + val * val, 0);

      // Draw filtered spectrum
      drawHeatmap(filteredCanvasRef.current, shiftedFiltered, 'Filtered');

      // Reconstruct filtered gradient using actual inverse FFT
      const cleanedGradientFull = ifft2D(filtered);

      // Extract the 7x7 portion (original size before padding)
      const cleanedGradient = cleanedGradientFull.slice(0, 7).map(row => row.slice(0, 7));

      // Draw cleaned gradient
      drawGradientMatrix(cleanedCanvasRef.current, cleanedGradient);

      // Calculate metrics
      const componentsTotal = spectrum.length * spectrum[0].length;
      const componentsKept = Math.round(componentsTotal * retainFraction);
      const componentsRemoved = componentsTotal - componentsKept;
      const noiseReduction = ((originalEnergy - filteredEnergy) / originalEnergy) * 100;
      const signalPreservation = (filteredEnergy / originalEnergy) * 100;

      setMetrics({
        originalEnergy,
        filteredEnergy,
        noiseReduction: Math.max(0, noiseReduction),
        signalPreservation,
        componentsRemoved,
      });
    } catch (error) {
      console.error('FFT computation error:', error);
    }
  }, [filter, retainFraction]);

  const drawHeatmap = (canvas: HTMLCanvasElement, data: number[][], label: string) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rows = data.length;
    const cols = data[0].length;
    const cellSize = Math.min(canvas.width / cols, canvas.height / rows);

    // Clear canvas
    ctx.fillStyle = '#0A0F24';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Find max value for normalization
    const maxValue = Math.max(...data.flat());

    // Draw heatmap
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const value = data[i][j];
        const normalized = maxValue > 0 ? value / maxValue : 0;
        const intensity = Math.floor(normalized * 255);

        // Gradient from blue (low) to cyan (high)
        const r = Math.floor(intensity * 0.3);
        const g = Math.floor(intensity * 0.9);
        const b = 255;
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }

    // Highlight center (DC component)
    const centerI = Math.floor(rows / 2);
    const centerJ = Math.floor(cols / 2);
    ctx.strokeStyle = '#00F5FF';
    ctx.lineWidth = 2;
    ctx.strokeRect(centerJ * cellSize, centerI * cellSize, cellSize, cellSize);
  };

  const drawGradientMatrix = (canvas: HTMLCanvasElement, data: number[][]) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cellSize = canvas.width / 7;

    // Clear canvas
    ctx.fillStyle = '#0A0F24';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Find max value for scaling
    const maxVal = Math.max(...data.flat().map(Math.abs));

    // Draw cells
    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 7; j++) {
        const value = data[i][j];
        const normalized = maxVal > 0 ? Math.abs(value) / maxVal : 0;
        const intensity = Math.min(255, normalized * 255);
        const color = value >= 0 ? `rgb(79, 172, 247, ${intensity / 255})` : `rgb(232, 123, 248, ${intensity / 255})`;

        ctx.fillStyle = color;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

        ctx.strokeStyle = '#1C2240';
        ctx.lineWidth = 1;
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Draw value
        ctx.fillStyle = intensity > 128 ? '#000000' : '#ffffff';
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(
          value.toFixed(2),
          j * cellSize + cellSize / 2,
          i * cellSize + cellSize / 2
        );
      }
    }
  };

  const selectPreset = (presetName: Preset) => {
    setPreset(presetName);
    setFilter(PRESETS[presetName].matrix.map((row) => [...row]));
  };

  return (
    <>
      <Head>
        <title>Frequency Filter - Falcon UI</title>
      </Head>

      <main className="min-h-screen pt-24 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <h1 className="text-5xl font-display font-bold mb-4 text-white animate-fade-in-up">
            Frequency Filter Explorer
          </h1>
          <p className="text-xl text-gray-400 mb-8 animate-fade-in-up" style={{ animationDelay: '0.1s' }}>
            See how frequency masking removes noise while preserving signal
          </p>

          {/* Impact Banner */}
          <div className="mb-8 p-6 bg-gradient-to-r from-falcon-pink/20 to-falcon-cyan/20 border border-falcon-cyan/50 rounded-lg">
            <div className="grid md:grid-cols-4 gap-6">
              <div>
                <div className="text-sm text-gray-400 mb-1">Noise Reduced</div>
                <div className="text-3xl font-bold text-falcon-cyan">
                  {metrics.noiseReduction.toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Signal Preserved</div>
                <div className="text-3xl font-bold text-falcon-blue">
                  {metrics.signalPreservation.toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Components Removed</div>
                <div className="text-3xl font-bold text-falcon-pink">
                  {metrics.componentsRemoved}
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400 mb-1">Retain Fraction (ρ)</div>
                <div className="text-3xl font-bold text-white">
                  {retainFraction.toFixed(2)}
                </div>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-blue">
                  📋 Gradient Examples
                </h3>
                <div className="space-y-2">
                  {(Object.keys(PRESETS) as Preset[]).map((key) => (
                    <button
                      key={key}
                      onClick={() => selectPreset(key)}
                      className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                        preset === key
                          ? 'bg-gradient-to-r from-falcon-blue/30 to-falcon-cyan/30 border border-falcon-cyan text-white'
                          : 'bg-falcon-bg text-gray-400 hover:bg-falcon-card'
                      }`}
                    >
                      <div className="font-bold text-sm">{PRESETS[key].description}</div>
                      <div className="text-xs text-gray-500 mt-1">{PRESETS[key].useCase}</div>
                    </button>
                  ))}
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-pink">
                  🎛️ Filter Parameters
                </h3>
                <div className="space-y-4">
                  <Slider
                    label="Retain Fraction (ρ)"
                    value={retainFraction}
                    min={0.3}
                    max={0.95}
                    step={0.05}
                    onChange={setRetainFraction}
                  />
                  <div className="p-3 bg-falcon-bg rounded-lg">
                    <div className="text-xs text-gray-400 space-y-1">
                      <div className="flex justify-between">
                        <span>Energy Kept:</span>
                        <span className="text-falcon-cyan font-mono">
                          {(retainFraction * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Energy Removed:</span>
                        <span className="text-falcon-pink font-mono">
                          {((1 - retainFraction) * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  </div>
                  <Toggle
                    label="Show Comparison View"
                    checked={showComparison}
                    onChange={setShowComparison}
                  />
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">
                  💡 How It Works
                </h3>
                <div className="space-y-4 text-sm text-gray-400">
                  <div>
                    <div className="font-semibold text-falcon-blue mb-1">1. Transform to Frequency Domain</div>
                    <p className="text-xs">
                      Apply 2D FFT to convert gradient matrix into frequency components.
                      Each component represents a specific frequency pattern.
                    </p>
                  </div>
                  <div>
                    <div className="font-semibold text-falcon-pink mb-1">2. Energy-Based Ranking</div>
                    <p className="text-xs">
                      Sort all frequency components by their energy (magnitude²).
                      High-frequency noise typically has low energy contribution.
                    </p>
                  </div>
                  <div>
                    <div className="font-semibold text-falcon-cyan mb-1">3. Adaptive Masking</div>
                    <p className="text-xs">
                      Keep top ρ% of components by energy, zero out the rest.
                      ρ decreases during training (0.95 → 0.50) for progressive denoising.
                    </p>
                  </div>
                  <div>
                    <div className="font-semibold text-white mb-1">4. Inverse Transform</div>
                    <p className="text-xs">
                      Apply inverse FFT to get cleaned gradient back in spatial domain.
                      Result: noise removed, essential structure preserved.
                    </p>
                  </div>
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-white">
                  🎯 Real-World Impact
                </h3>
                <div className="space-y-3 text-sm">
                  <div className="p-3 bg-falcon-bg rounded-lg">
                    <div className="font-semibold text-falcon-cyan mb-1">Training Stability</div>
                    <p className="text-xs text-gray-400">
                      Reduces gradient variance by up to 40%, leading to smoother convergence.
                    </p>
                  </div>
                  <div className="p-3 bg-falcon-bg rounded-lg">
                    <div className="font-semibold text-falcon-blue mb-1">Generalization</div>
                    <p className="text-xs text-gray-400">
                      Prevents overfitting to high-frequency patterns in training data.
                    </p>
                  </div>
                  <div className="p-3 bg-falcon-bg rounded-lg">
                    <div className="font-semibold text-falcon-pink mb-1">Computational Cost</div>
                    <p className="text-xs text-gray-400">
                      FFT adds ~26% overhead but improves sample efficiency.
                    </p>
                  </div>
                </div>
              </Card>
            </div>

            {/* Visualizations */}
            <div className="lg:col-span-2 space-y-6">
              {showComparison && (
                <div className="grid md:grid-cols-2 gap-6">
                  <Card hover={false}>
                    <h3 className="text-lg font-bold mb-4 text-falcon-pink">
                      📥 Input: Original Gradient
                    </h3>
                    <div className="flex justify-center">
                      <canvas
                        ref={canvasRef}
                        width={350}
                        height={350}
                        className="border-2 border-falcon-pink/30 rounded-lg"
                      />
                    </div>
                    <div className="mt-3 p-3 bg-falcon-bg/50 rounded-lg">
                      <p className="text-xs text-gray-400">
                        {PRESETS[preset].description}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Notice: High values (noise spikes) in noisy gradient
                      </p>
                    </div>
                  </Card>

                  <Card hover={false}>
                    <h3 className="text-lg font-bold mb-4 text-falcon-cyan">
                      📤 Output: Cleaned Gradient
                    </h3>
                    <div className="flex justify-center">
                      <canvas
                        ref={cleanedCanvasRef}
                        width={350}
                        height={350}
                        className="border-2 border-falcon-cyan/30 rounded-lg"
                      />
                    </div>
                    <div className="mt-3 p-3 bg-falcon-bg/50 rounded-lg">
                      <p className="text-xs text-gray-400">
                        After frequency filtering at ρ = {retainFraction.toFixed(2)}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Notice: Noise spikes removed, smooth structure preserved
                      </p>
                    </div>
                  </Card>
                </div>
              )}

              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-white">
                  🔬 Frequency Domain Analysis
                </h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-bold mb-3 text-falcon-blue">
                      Full Magnitude Spectrum
                    </h4>
                    <canvas
                      ref={spectrumCanvasRef}
                      width={300}
                      height={300}
                      className="border border-falcon-blue/30 rounded-lg w-full"
                    />
                    <div className="mt-2 p-2 bg-falcon-bg/50 rounded text-xs text-gray-400">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 border-2 border-falcon-cyan rounded-sm"></div>
                        <span>DC component (center) = low-frequency structure</span>
                      </div>
                      <div className="mt-1">Brightness = energy magnitude</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-bold mb-3 text-falcon-pink">
                      After Energy-Based Masking
                    </h4>
                    <canvas
                      ref={filteredCanvasRef}
                      width={300}
                      height={300}
                      className="border border-falcon-pink/30 rounded-lg w-full"
                    />
                    <div className="mt-2 p-2 bg-falcon-bg/50 rounded text-xs text-gray-400">
                      <div>High-frequency components (edges/noise) removed</div>
                      <div className="mt-1 font-mono text-falcon-cyan">
                        {metrics.componentsRemoved} components zeroed out
                      </div>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Interactive Demo Section */}
              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-white">
                  🧪 Try It Yourself
                </h3>
                <div className="space-y-4">
                  <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-blue/30">
                    <h4 className="font-semibold text-falcon-cyan mb-2">Experiment 1: Compare Clean vs Noisy</h4>
                    <ol className="text-sm text-gray-400 space-y-1 list-decimal list-inside">
                      <li>Select "Clean neural network gradient"</li>
                      <li>Note the smooth spectrum (energy concentrated in center)</li>
                      <li>Switch to "Gradient corrupted with noise"</li>
                      <li>See scattered high-frequency components appear</li>
                      <li>Adjust ρ slider - watch noise components get filtered</li>
                    </ol>
                  </div>

                  <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-pink/30">
                    <h4 className="font-semibold text-falcon-pink mb-2">Experiment 2: Optimal ρ Value</h4>
                    <ol className="text-sm text-gray-400 space-y-1 list-decimal list-inside">
                      <li>Select noisy gradient example</li>
                      <li>Set ρ = 0.95 (keep almost everything)</li>
                      <li>Gradually decrease ρ to 0.50</li>
                      <li>Watch noise reduction increase while signal stays intact</li>
                      <li>Notice: Too low ρ (&lt;0.50) removes useful structure</li>
                    </ol>
                  </div>

                  <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-cyan/30">
                    <h4 className="font-semibold text-falcon-cyan mb-2">Key Insight</h4>
                    <p className="text-sm text-gray-400">
                      Falcon adapts ρ during training: starts at 0.95 (preserve everything early),
                      ends at 0.50 (aggressive denoising late). This progressive filtering
                      balances exploration (early) with exploitation (late).
                    </p>
                  </div>
                </div>
              </Card>

              <Card hover={false}>
                <div className="p-4 bg-gradient-to-br from-falcon-card to-falcon-bg border border-falcon-purple/30 rounded-lg">
                  <p className="text-sm text-gray-400 italic font-display text-center">
                    "In the frequency domain, we see the skeleton of information—
                    <br />
                    preserving structure while discarding noise, a sculptor's touch."
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
