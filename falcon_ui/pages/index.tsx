import React from 'react';
import Head from 'next/head';
import Hero from '@/components/Hero';
import SVDExplorer from '@/components/SVDExplorer';
import NetworkDiagram from '@/components/NetworkDiagram';
import StatsCounter from '@/components/StatsCounter';
import AnimatedCard from '@/components/AnimatedCard';
import Link from 'next/link';

export default function Home() {
  return (
    <>
      <Head>
        <title>Falcon UI - Optimizer Visualization Suite</title>
        <meta
          name="description"
          content="Experience learning as art through trajectories, spectra and structure"
        />
      </Head>

      <main className="min-h-screen">
        <Hero />

        {/* Additional content section */}
        <section className="relative py-20 px-6">
          <div className="container mx-auto max-w-6xl">
            <h2 className="text-4xl font-display font-bold text-center mb-12 text-white">
              The Art of Optimization
            </h2>

            <div className="grid md:grid-cols-2 gap-12">
              <div className="space-y-6">
                <h3 className="text-2xl font-bold text-falcon-blue">
                  What is Falcon?
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  Falcon (Frequency-Aware Low-rank Conditioning Optimizer) is a novel
                  optimization algorithm that combines frequency domain analysis with
                  low-rank matrix approximations for deep neural network training.
                </p>
                <p className="text-gray-300 leading-relaxed">
                  Through comprehensive experiments on CIFAR-10 with VGG11, Falcon achieves
                  <strong className="text-falcon-cyan"> 90.33% accuracy</strong>, demonstrating
                  competitive performance with AdamW (90.28%) and Muon (90.49%).
                </p>
                <p className="text-sm text-gray-400 mt-2">
                  Paper: "FALCON: Frequency-Adaptive Learning with Conserved Orthogonality and Noise Filtering"
                  (<a href="https://github.com/11NOel11/Falcon" className="text-falcon-blue hover:underline" target="_blank" rel="noopener noreferrer">
                    GitHub
                  </a>)
                </p>
              </div>

              <div className="space-y-6">
                <h3 className="text-2xl font-bold text-falcon-pink">
                  Why Visualize?
                </h3>
                <p className="text-gray-300 leading-relaxed">
                  Understanding optimization algorithms requires more than equations.
                  This interactive suite lets you explore how different optimizers
                  traverse loss landscapes, how frequency filtering shapes gradients,
                  and how training dynamics evolve over time.
                </p>
                <p className="text-gray-300 leading-relaxed">
                  Each visualization reveals a unique perspective on the geometry of
                  learning, transforming abstract mathematics into tangible insight.
                </p>
              </div>
            </div>

            {/* Key Statistics */}
            <div className="mt-20 mb-16">
              <h2 className="text-4xl font-display font-bold text-center mb-12 text-white">
                Performance Highlights
              </h2>
              <div className="grid md:grid-cols-4 gap-6">
                <StatsCounter value={90.33} label="CIFAR-10 Accuracy" suffix="%" decimals={2} color="text-falcon-cyan" />
                <StatsCounter value={7486} label="Images per Second" suffix="" decimals={0} color="text-falcon-pink" />
                <StatsCounter value={6.7} label="Seconds per Epoch" suffix="s" decimals={1} color="text-falcon-blue" />
                <StatsCounter value={1.56} label="Convergence Time" suffix=" min" decimals={2} color="text-falcon-cyan" />
              </div>
            </div>

            {/* Feature Showcase */}
            <div className="mt-16 mb-12">
              <h2 className="text-4xl font-display font-bold text-center mb-12 text-white">
                Interactive Visualizations
              </h2>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                <Link href="/trajectory">
                  <AnimatedCard
                    title="3D Trajectory Viewer"
                    description="Explore optimizer paths across loss landscapes with interactive 3D visualization"
                    gradient="from-falcon-blue/20 to-falcon-cyan/20"
                    icon={
                      <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                      </svg>
                    }
                  />
                </Link>
                <Link href="/filter">
                  <AnimatedCard
                    title="Frequency Filter Explorer"
                    description="Draw custom filters and visualize 2D FFT transformations in real-time"
                    gradient="from-falcon-pink/20 to-falcon-blue/20"
                    icon={
                      <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    }
                  />
                </Link>
                <Link href="/dynamics">
                  <AnimatedCard
                    title="Training Dynamics"
                    description="Track loss curves, accuracy evolution, and adaptive scheduling across epochs"
                    gradient="from-falcon-cyan/20 to-falcon-pink/20"
                    icon={
                      <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                      </svg>
                    }
                  />
                </Link>
                <Link href="/results">
                  <AnimatedCard
                    title="Experimental Results"
                    description="Complete analysis of CIFAR-10 experiments with all figures and metrics"
                    gradient="from-falcon-blue/20 to-falcon-pink/20"
                    icon={
                      <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    }
                  />
                </Link>
                <AnimatedCard
                  title="SVD Explorer"
                  description="Interactive rank-k matrix approximation demonstration"
                  gradient="from-falcon-pink/20 to-falcon-cyan/20"
                  icon={
                    <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v7a1 1 0 01-1 1h-4a1 1 0 01-1-1V5z" />
                    </svg>
                  }
                />
                <AnimatedCard
                  title="Network Diagram"
                  description="Layer-wise optimization strategy visualization"
                  gradient="from-falcon-cyan/20 to-falcon-blue/20"
                  icon={
                    <svg className="w-12 h-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                    </svg>
                  }
                />
              </div>
            </div>

            <div className="mt-16 p-8 bg-gradient-to-br from-falcon-card to-falcon-bg border border-falcon-blue/30 rounded-lg animate-pulse-glow">
              <h3 className="text-xl font-display italic text-center text-gray-300 mb-4">
                "In the dance of gradients and frequencies, patterns emerge—
                <br />
                each optimizer a unique choreography across the loss landscape."
              </h3>
              <p className="text-center text-sm text-gray-500">
                — The Mathematics of Learning
              </p>
            </div>

            {/* Interactive Components */}
            <div className="mt-16 grid md:grid-cols-2 gap-8">
              <SVDExplorer />
              <NetworkDiagram />
            </div>
          </div>
        </section>
      </main>
    </>
  );
}
