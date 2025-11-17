import React from 'react';
import Link from 'next/link';
import WaveBackground from './WaveBackground';
import ParticleField from './ParticleField';
import RippleButton from './RippleButton';

const Hero: React.FC = () => {
  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <WaveBackground />
      <ParticleField />

      <div className="relative z-10 text-center px-6 max-w-5xl">
        <h1 className="text-6xl md:text-7xl font-display font-bold mb-6 text-white">
          Falcon Optimizer
        </h1>
        <h2 className="text-3xl md:text-4xl font-display text-falcon-blue mb-4">
          Where Frequency Meets Geometry
        </h2>
        <p className="text-xl md:text-2xl text-gray-300 mb-12 font-light italic animate-fade-in-up" style={{ animationDelay: '0.2s' }}>
          Experience learning as art, through trajectories, spectra and structure
        </p>

        <div className="flex flex-wrap gap-4 justify-center mb-16">
          <RippleButton href="/trajectory" variant="gradient">
            Explore Visualizations
          </RippleButton>
          <RippleButton href="/falcon_v5" variant="secondary">
            View Results
          </RippleButton>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
          <div className="bg-falcon-card/50 backdrop-blur-sm p-6 rounded-lg border border-falcon-blue/30 hover:border-falcon-blue transition-all duration-300 hover:transform hover:scale-105 hover:shadow-xl hover:shadow-falcon-blue/20 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
            <div className="text-4xl mb-4 animate-float">🌊</div>
            <h3 className="text-xl font-bold text-falcon-blue mb-2">
              Frequency Masking
            </h3>
            <p className="text-gray-400 text-sm">
              Energy-aware spectral filtering adapts gradient updates through 2D FFT analysis
            </p>
          </div>

          <div className="bg-falcon-card/50 backdrop-blur-sm p-6 rounded-lg border border-falcon-pink/30 hover:border-falcon-pink transition-all duration-300 hover:transform hover:scale-105 hover:shadow-xl hover:shadow-falcon-pink/20 animate-fade-in-up" style={{ animationDelay: '0.4s' }}>
            <div className="text-4xl mb-4 animate-float" style={{ animationDelay: '0.5s' }}>⚡</div>
            <h3 className="text-xl font-bold text-falcon-pink mb-2">
              Rank-1 Updates
            </h3>
            <p className="text-gray-400 text-sm">
              Low-rank approximations preserve essential gradient directions via power iteration
            </p>
          </div>

          <div className="bg-falcon-card/50 backdrop-blur-sm p-6 rounded-lg border border-falcon-cyan/30 hover:border-falcon-cyan transition-all duration-300 hover:transform hover:scale-105 hover:shadow-xl hover:shadow-falcon-cyan/20 animate-fade-in-up" style={{ animationDelay: '0.5s' }}>
            <div className="text-4xl mb-4 animate-float" style={{ animationDelay: '1s' }}>🎯</div>
            <h3 className="text-xl font-bold text-falcon-cyan mb-2">
              Orthogonal Projection
            </h3>
            <p className="text-gray-400 text-sm">
              Gram-Schmidt orthogonalization ensures decorrelated parameter updates
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hero;
