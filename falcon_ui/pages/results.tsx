import React, { useState } from 'react';
import Head from 'next/head';
import summaryData from '@/data/summary.json';
import ImageComparison from '@/components/ImageComparison';

interface ExperimentData {
  experiment: string;
  optimizer: string;
  data_fraction: number;
  best_val_acc: number;
  best_epoch: number;
  total_time_min: number;
  median_epoch_time_s: number;
  images_per_sec: number;
  time_to_85_percent: number | null;
}

export default function Results() {
  const [selectedCategory, setSelectedCategory] = useState<'full_training' | 'fixed_time' | 'data_efficiency'>('full_training');

  // Base path for GitHub Pages deployment
  const basePath = process.env.NODE_ENV === 'production' ? '/Falcon' : '';

  const figureCategories = [
    {
      title: 'Core Optimization Results',
      figures: [
        { src: `${basePath}/fig_top1_vs_time.png`, caption: 'Top-1 Accuracy vs Training Time', description: 'Comparison of convergence speed across optimizers' },
        { src: `${basePath}/fig_time_to_85.png`, caption: 'Time to 85% Accuracy', description: 'How quickly each optimizer reaches key accuracy milestones' },
        { src: `${basePath}/fig_data_efficiency.png`, caption: 'Data Efficiency Comparison', description: 'Performance with reduced training data (10%, 20%, 100%)' },
        { src: `${basePath}/fig_fixed_time_10min.png`, caption: 'Fixed Time Budget (10 min)', description: 'Best accuracy achievable within a fixed time constraint' },
      ],
    },
    {
      title: 'Frequency Filtering & Adaptive Schedules',
      figures: [
        { src: `${basePath}/fig_frequency_filtering_demo.png`, caption: 'Frequency Filtering Demonstration', description: 'How FFT-based filtering removes high-frequency noise' },
        { src: `${basePath}/fig_frequency_masks.png`, caption: 'Frequency Domain Masks', description: 'Energy-based masking in the frequency domain' },
        { src: `${basePath}/fig_real_image_filtering.png`, caption: 'Real Image Filtering Example', description: 'Visual demonstration of frequency filtering on actual images' },
        { src: `${basePath}/fig_progressive_filtering.png`, caption: 'Progressive Filtering Schedule', description: 'How retain fraction decreases over training epochs' },
        { src: `${basePath}/fig_adaptive_schedules.png`, caption: 'Adaptive Scheduling Dynamics', description: 'Evolution of retain fraction and interleaving period' },
      ],
    },
    {
      title: 'Architecture & Performance Analysis',
      figures: [
        { src: `${basePath}/fig_architecture_comparison.png`, caption: 'Architecture Comparison', description: 'Optimizer performance across different network architectures' },
        { src: `${basePath}/fig_computational_breakdown.png`, caption: 'Computational Cost Breakdown', description: 'Time analysis showing 40% overhead for Falcon' },
        { src: `${basePath}/fig_mask_sharing.png`, caption: 'Mask Sharing Strategy', description: 'How frequency masks are shared across parameter groups' },
        { src: `${basePath}/fig_ema_averaging.png`, caption: 'EMA Averaging Effects', description: 'Impact of exponential moving average on convergence' },
        { src: `${basePath}/fig_robustness_noise.png`, caption: 'Robustness to Noise', description: 'Performance under different noise conditions' },
      ],
    },
  ];

  const categoryTitles = {
    full_training: 'Full Training Results',
    fixed_time: 'Fixed Time Budget',
    data_efficiency: 'Data Efficiency',
  };

  const categoryDescriptions = {
    full_training: 'Complete training on 100% of CIFAR-10 data until convergence',
    fixed_time: 'Best accuracy achieved within a 10-minute time constraint',
    data_efficiency: 'Performance with reduced training data (10% and 20%)',
  };

  const getOptimizerColor = (optimizer: string) => {
    if (optimizer === 'AdamW') return '#4FACF7';
    if (optimizer === 'Muon') return '#E87BF8';
    if (optimizer === 'FALCON' || optimizer === 'Falcon' || optimizer === 'FALCON v5') return '#00F5FF';
    return '#888';
  };

  const selectedData = summaryData[selectedCategory] as ExperimentData[];

  return (
    <>
      <Head>
        <title>Falcon - Experimental Results</title>
        <meta name="description" content="Complete experimental results and figures from Falcon research" />
      </Head>

      <main className="min-h-screen py-24 px-6">
        <div className="container mx-auto max-w-7xl">
          {/* Header */}
          <div className="mb-12 text-center">
            <h1 className="text-5xl font-display font-bold text-white mb-4">
              FALCON Experimental Results
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering
            </p>
            <p className="text-sm text-gray-400 mt-3">
              Research by Noel Thomas, MBZUAI • 2025
            </p>
          </div>

          {/* Stats Cards */}
          <div className="grid md:grid-cols-4 gap-6 mb-12">
            <div className="p-6 bg-gradient-to-br from-falcon-blue/20 to-falcon-cyan/20 border border-falcon-blue/40 rounded-xl hover:border-falcon-blue transition-all">
              <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Best Accuracy</div>
              <div className="text-4xl font-bold text-white mb-1">90.49%</div>
              <div className="text-xs text-falcon-pink">Muon Optimizer</div>
            </div>
            <div className="p-6 bg-gradient-to-br from-falcon-cyan/20 to-falcon-blue/20 border border-falcon-cyan/40 rounded-xl hover:border-falcon-cyan transition-all">
              <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Falcon Accuracy</div>
              <div className="text-4xl font-bold text-white mb-1">90.33%</div>
              <div className="text-xs text-falcon-cyan">Competitive Performance</div>
            </div>
            <div className="p-6 bg-gradient-to-br from-falcon-pink/20 to-falcon-purple/20 border border-falcon-pink/40 rounded-xl hover:border-falcon-pink transition-all">
              <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Epoch Time</div>
              <div className="text-4xl font-bold text-white mb-1">6.7s</div>
              <div className="text-xs text-falcon-pink">+40% vs AdamW</div>
            </div>
            <div className="p-6 bg-gradient-to-br from-falcon-purple/20 to-falcon-pink/20 border border-falcon-purple/40 rounded-xl hover:border-falcon-purple transition-all">
              <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Convergence</div>
              <div className="text-4xl font-bold text-white mb-1">1.56m</div>
              <div className="text-xs text-falcon-purple">Time to 85%</div>
            </div>
          </div>

          {/* Key Findings Summary */}
          <div className="mb-12 p-8 bg-gradient-to-br from-falcon-card to-falcon-bg border border-falcon-blue/30 rounded-lg">
            <h2 className="text-2xl font-bold text-falcon-cyan mb-6">Key Research Findings</h2>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="p-4 bg-falcon-bg/50 rounded-lg border-l-4 border-falcon-blue">
                <div className="flex items-start gap-3">
                  <div className="text-2xl">✓</div>
                  <div>
                    <h3 className="text-lg font-bold text-falcon-blue mb-2">Competitive Accuracy</h3>
                    <p className="text-gray-300 text-sm">
                      Falcon achieves <strong className="text-falcon-cyan">90.33%</strong> on CIFAR-10 with VGG11,
                      comparable to AdamW (90.28%) and Muon (90.49%)
                    </p>
                  </div>
                </div>
              </div>
              <div className="p-4 bg-falcon-bg/50 rounded-lg border-l-4 border-falcon-pink">
                <div className="flex items-start gap-3">
                  <div className="text-2xl">⚡</div>
                  <div>
                    <h3 className="text-lg font-bold text-falcon-pink mb-2">Computational Trade-off</h3>
                    <p className="text-gray-300 text-sm">
                      40% slower than AdamW due to FFT operations and rank-1 approximations
                      (<strong className="text-falcon-pink">6.7s</strong> vs 4.8s per epoch)
                    </p>
                  </div>
                </div>
              </div>
              <div className="p-4 bg-falcon-bg/50 rounded-lg border-l-4 border-falcon-cyan">
                <div className="flex items-start gap-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h3 className="text-lg font-bold text-falcon-cyan mb-2">Data Efficiency</h3>
                    <p className="text-gray-300 text-sm">
                      Frequency filtering shows strongest benefits with full training data,
                      suggesting optimization for large-scale scenarios
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Experimental Results Table */}
          <section className="mb-16">
            <h2 className="text-3xl font-display font-bold text-white mb-6">Experimental Results</h2>

            {/* Category Tabs */}
            <div className="flex gap-4 mb-6 border-b border-gray-700">
              {(Object.keys(categoryTitles) as Array<keyof typeof categoryTitles>).map((category) => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`pb-3 px-4 font-medium transition-all ${
                    selectedCategory === category
                      ? 'text-falcon-cyan border-b-2 border-falcon-cyan'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {categoryTitles[category]}
                </button>
              ))}
            </div>

            <p className="text-gray-400 mb-6">{categoryDescriptions[selectedCategory]}</p>

            {/* Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full bg-falcon-card border border-falcon-blue/20 rounded-lg overflow-hidden">
                <thead className="bg-falcon-bg border-b border-falcon-blue/20">
                  <tr>
                    <th className="px-4 py-3 text-left text-sm font-semibold text-gray-300">Optimizer</th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-gray-300">Accuracy</th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-gray-300">Best Epoch</th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-gray-300">Total Time</th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-gray-300">Epoch Time</th>
                    <th className="px-4 py-3 text-right text-sm font-semibold text-gray-300">Throughput</th>
                    {selectedCategory === 'full_training' && (
                      <th className="px-4 py-3 text-right text-sm font-semibold text-gray-300">Time to 85%</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {selectedData.map((exp, idx) => (
                    <tr key={exp.experiment} className={idx % 2 === 0 ? 'bg-falcon-card/40' : 'bg-falcon-card/20'}>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: getOptimizerColor(exp.optimizer) }}
                          />
                          <span className="font-medium text-gray-200">{exp.optimizer}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-gray-200">{exp.best_val_acc.toFixed(2)}%</td>
                      <td className="px-4 py-3 text-right text-gray-300">{exp.best_epoch}</td>
                      <td className="px-4 py-3 text-right text-gray-300">{exp.total_time_min.toFixed(2)} min</td>
                      <td className="px-4 py-3 text-right text-gray-300">{exp.median_epoch_time_s.toFixed(1)}s</td>
                      <td className="px-4 py-3 text-right font-mono text-gray-300">
                        {exp.images_per_sec.toLocaleString()} img/s
                      </td>
                      {selectedCategory === 'full_training' && (
                        <td className="px-4 py-3 text-right text-gray-300">
                          {exp.time_to_85_percent ? `${exp.time_to_85_percent.toFixed(2)} min` : '—'}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {/* Real Image Training Comparison */}
          <section className="mb-16">
            <h2 className="text-3xl font-display font-bold text-white mb-4">
              Real Image Training Comparison
            </h2>
            <p className="text-gray-400 mb-6">
              See how each optimizer learns to classify actual CIFAR-10 images across training epochs
            </p>
            <ImageComparison />
          </section>

          {/* Figures Gallery */}
          {figureCategories.map((category) => (
            <section key={category.title} className="mb-16">
              <h2 className="text-3xl font-display font-bold text-white mb-6">{category.title}</h2>
              <div className="grid md:grid-cols-2 gap-6">
                {category.figures.map((fig) => (
                  <div
                    key={fig.src}
                    className="bg-falcon-card border border-falcon-blue/20 rounded-lg overflow-hidden hover:border-falcon-blue/40 transition-all"
                  >
                    <div className="bg-black p-4">
                      <img
                        src={fig.src}
                        alt={fig.caption}
                        className="w-full h-auto"
                      />
                    </div>
                    <div className="p-4">
                      <h3 className="text-lg font-semibold text-white mb-1">{fig.caption}</h3>
                      <p className="text-sm text-gray-400">{fig.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}

          {/* Repository Link */}
          <section className="mt-16 p-8 bg-gradient-to-br from-falcon-card to-falcon-bg border border-falcon-blue/30 rounded-lg text-center">
            <h2 className="text-2xl font-bold text-falcon-cyan mb-3">Full Research Repository</h2>
            <p className="text-gray-300 mb-4">
              Access the complete codebase, datasets, papers, and reproducible experiments
            </p>
            <a
              href="https://github.com/11NOel11/Falcon"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block px-6 py-3 bg-falcon-blue hover:bg-falcon-cyan text-white font-semibold rounded-lg transition-colors"
            >
              View on GitHub
            </a>
          </section>
        </div>
      </main>
    </>
  );
}
