import React, { useEffect, useState } from 'react';

const ScrollProgress: React.FC = () => {
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <>
      {/* Top progress bar */}
      <div className="fixed top-0 left-0 w-full h-1 bg-falcon-bg/50 backdrop-blur-sm z-50">
        <div
          className="h-full bg-gradient-to-r from-falcon-blue via-falcon-cyan to-falcon-pink transition-all duration-150 shadow-lg shadow-falcon-cyan/50"
          style={{ width: `${scrollProgress}%` }}
        />
      </div>

      {/* Circular progress indicator */}
      <div className="fixed bottom-8 right-8 z-50 group">
        <div className="relative w-12 h-12">
          <svg className="transform -rotate-90 w-12 h-12">
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="rgba(0, 245, 255, 0.2)"
              strokeWidth="3"
              fill="none"
            />
            <circle
              cx="24"
              cy="24"
              r="20"
              stroke="url(#progressGradient)"
              strokeWidth="3"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 20}`}
              strokeDashoffset={`${2 * Math.PI * 20 * (1 - scrollProgress / 100)}`}
              className="transition-all duration-150"
            />
            <defs>
              <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#00F5FF" />
                <stop offset="50%" stopColor="#4FACF7" />
                <stop offset="100%" stopColor="#E87BF8" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs font-bold text-falcon-cyan">
              {Math.round(scrollProgress)}%
            </span>
          </div>
        </div>

        {/* Scroll to top button */}
        <button
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
          className={`absolute inset-0 rounded-full bg-falcon-card/80 backdrop-blur-sm border border-falcon-cyan/50 opacity-0 group-hover:opacity-100 transition-all duration-300 hover:scale-110 hover:shadow-lg hover:shadow-falcon-cyan/50 ${
            scrollProgress > 10 ? 'pointer-events-auto' : 'pointer-events-none'
          }`}
          aria-label="Scroll to top"
        >
          <svg
            className="w-12 h-12 text-falcon-cyan"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M5 10l7-7m0 0l7 7m-7-7v18"
            />
          </svg>
        </button>
      </div>
    </>
  );
};

export default ScrollProgress;
