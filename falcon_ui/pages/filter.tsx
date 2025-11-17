import React, { useState, useEffect, useRef } from 'react';
import Head from 'next/head';
import Card from '@/components/Card';
import Slider from '@/components/Slider';
import Toggle from '@/components/Toggle';
import { fft2D, getMagnitudeSpectrum, applyFrequencyFilter, fftshift } from '@/utils/fft';

type Preset = 'edge' | 'gaussian' | 'random' | 'custom';

const PRESETS: Record<Preset, number[][]> = {
  edge: [
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, 24, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1],
  ],
  gaussian: [
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0],
  ].map((row) => row.map((v) => v / 1000)),
  random: Array(7)
    .fill(0)
    .map(() =>
      Array(7)
        .fill(0)
        .map(() => Math.random())
    ),
  custom: Array(7)
    .fill(0)
    .map(() => Array(7).fill(0)),
};

export default function FilterPage() {
  const [filter, setFilter] = useState<number[][]>(PRESETS.gaussian);
  const [retainFraction, setRetainFraction] = useState(0.75);
  const [useRank1, setUseRank1] = useState(false);
  const [preset, setPreset] = useState<Preset>('gaussian');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const spectrumCanvasRef = useRef<HTMLCanvasElement>(null);
  const filteredCanvasRef = useRef<HTMLCanvasElement>(null);

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

    // Draw cells
    for (let i = 0; i < 7; i++) {
      for (let j = 0; j < 7; j++) {
        const value = filter[i][j];
        const intensity = Math.min(255, Math.max(0, Math.abs(value) * 255));
        const color = value >= 0 ? `rgb(79, 172, 247, ${intensity / 255})` : `rgb(232, 123, 248, ${intensity / 255})`;

        ctx.fillStyle = color;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);

        ctx.strokeStyle = '#1C2240';
        ctx.strokeRect(j * cellSize, i * cellSize, cellSize, cellSize);

        // Draw value
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px Inter';
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

  // Compute and draw spectrum
  useEffect(() => {
    if (!spectrumCanvasRef.current || !filteredCanvasRef.current) return;

    try {
      // Compute FFT
      const fftResult = fft2D(filter);
      const spectrum = getMagnitudeSpectrum(fftResult);
      const shiftedSpectrum = fftshift(spectrum);

      // Draw spectrum
      drawHeatmap(spectrumCanvasRef.current, shiftedSpectrum, 'Spectrum');

      // Apply filter
      const filtered = applyFrequencyFilter(fftResult, retainFraction);
      const filteredSpectrum = getMagnitudeSpectrum(filtered);
      const shiftedFiltered = fftshift(filteredSpectrum);

      // Draw filtered spectrum
      drawHeatmap(filteredCanvasRef.current, shiftedFiltered, 'Filtered');
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

        ctx.fillStyle = `rgb(${intensity}, ${Math.floor(intensity * 0.6)}, ${255 - intensity})`;
        ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
      }
    }
  };

  const selectPreset = (presetName: Preset) => {
    setPreset(presetName);
    setFilter(PRESETS[presetName].map((row) => [...row]));
  };

  return (
    <>
      <Head>
        <title>Frequency Filter - Falcon UI</title>
      </Head>

      <main className="min-h-screen pt-24 pb-12 px-6">
        <div className="container mx-auto max-w-7xl">
          <h1 className="text-5xl font-display font-bold mb-4 text-white">
            Frequency Filter Explorer
          </h1>
          <p className="text-xl text-gray-400 mb-12">
            Visualize how frequency masking shapes gradient information
          </p>

          <div className="grid lg:grid-cols-3 gap-8">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-blue">Presets</h3>
                <div className="grid grid-cols-2 gap-2">
                  {Object.keys(PRESETS).map((key) => (
                    <button
                      key={key}
                      onClick={() => selectPreset(key as Preset)}
                      className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                        preset === key
                          ? 'bg-falcon-blue text-white'
                          : 'bg-falcon-bg text-gray-400 hover:bg-falcon-card'
                      }`}
                    >
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </button>
                  ))}
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-pink">
                  Filter Parameters
                </h3>
                <div className="space-y-4">
                  <Slider
                    label="Retain Fraction (ρ)"
                    value={retainFraction}
                    min={0.5}
                    max={0.95}
                    step={0.05}
                    onChange={setRetainFraction}
                  />
                  <Toggle
                    label="Rank-1 Approximation"
                    checked={useRank1}
                    onChange={setUseRank1}
                  />
                </div>
              </Card>

              <Card>
                <h3 className="text-xl font-bold mb-4 text-falcon-cyan">
                  Explanation
                </h3>
                <div className="space-y-3 text-sm text-gray-400">
                  <p>
                    <strong className="text-falcon-blue">Energy-based Masking:</strong>{' '}
                    Frequency components are ranked by their energy contribution and
                    filtered to retain the specified fraction.
                  </p>
                  <p>
                    <strong className="text-falcon-pink">Adaptive Schedule:</strong>{' '}
                    The retain fraction ρ decreases over training (0.95 → 0.50) to
                    progressively remove high-frequency noise.
                  </p>
                  <p>
                    <strong className="text-falcon-cyan">Rank-1 Updates:</strong>{' '}
                    Power iteration finds the dominant direction, focusing updates
                    along the principal gradient axis.
                  </p>
                </div>
              </Card>
            </div>

            {/* Visualizations */}
            <div className="lg:col-span-2 space-y-6">
              <Card hover={false}>
                <h3 className="text-xl font-bold mb-4 text-white">
                  Original Filter (7×7)
                </h3>
                <div className="flex justify-center">
                  <canvas
                    ref={canvasRef}
                    width={350}
                    height={350}
                    className="border border-falcon-blue/30 rounded-lg"
                  />
                </div>
              </Card>

              <div className="grid md:grid-cols-2 gap-6">
                <Card hover={false}>
                  <h3 className="text-lg font-bold mb-4 text-falcon-blue">
                    Magnitude Spectrum
                  </h3>
                  <canvas
                    ref={spectrumCanvasRef}
                    width={300}
                    height={300}
                    className="border border-falcon-blue/30 rounded-lg w-full"
                  />
                  <p className="text-xs text-gray-500 mt-2 italic">
                    DC component centered; brightness indicates energy
                  </p>
                </Card>

                <Card hover={false}>
                  <h3 className="text-lg font-bold mb-4 text-falcon-pink">
                    Filtered Spectrum
                  </h3>
                  <canvas
                    ref={filteredCanvasRef}
                    width={300}
                    height={300}
                    className="border border-falcon-pink/30 rounded-lg w-full"
                  />
                  <p className="text-xs text-gray-500 mt-2 italic">
                    After energy-based masking at ρ = {retainFraction.toFixed(2)}
                  </p>
                </Card>
              </div>

              <Card hover={false}>
                <div className="p-4 bg-falcon-bg/50 rounded-lg border border-falcon-purple/30">
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
