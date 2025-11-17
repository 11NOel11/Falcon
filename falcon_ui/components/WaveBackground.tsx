import React from 'react';

const WaveBackground: React.FC = () => {
  return (
    <div className="absolute inset-0 overflow-hidden">
      <svg
        className="absolute w-full h-full"
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 1200 600"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style={{ stopColor: '#4FACF7', stopOpacity: 0.3 }} />
            <stop offset="50%" style={{ stopColor: '#E87BF8', stopOpacity: 0.3 }} />
            <stop offset="100%" style={{ stopColor: '#00F5FF', stopOpacity: 0.3 }} />
          </linearGradient>
          <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style={{ stopColor: '#E87BF8', stopOpacity: 0.2 }} />
            <stop offset="50%" style={{ stopColor: '#9D4EDD', stopOpacity: 0.2 }} />
            <stop offset="100%" style={{ stopColor: '#4FACF7', stopOpacity: 0.2 }} />
          </linearGradient>
        </defs>

        {/* First wave */}
        <path d="M0,300 Q300,200 600,300 T1200,300 L1200,600 L0,600 Z" fill="url(#grad1)">
          <animate
            attributeName="d"
            dur="10s"
            repeatCount="indefinite"
            values="
              M0,300 Q300,200 600,300 T1200,300 L1200,600 L0,600 Z;
              M0,300 Q300,400 600,300 T1200,300 L1200,600 L0,600 Z;
              M0,300 Q300,200 600,300 T1200,300 L1200,600 L0,600 Z
            "
          />
        </path>

        {/* Second wave */}
        <path d="M0,350 Q300,250 600,350 T1200,350 L1200,600 L0,600 Z" fill="url(#grad2)">
          <animate
            attributeName="d"
            dur="15s"
            repeatCount="indefinite"
            values="
              M0,350 Q300,250 600,350 T1200,350 L1200,600 L0,600 Z;
              M0,350 Q300,450 600,350 T1200,350 L1200,600 L0,600 Z;
              M0,350 Q300,250 600,350 T1200,350 L1200,600 L0,600 Z
            "
          />
        </path>

        {/* Third wave */}
        <path
          d="M0,400 Q300,300 600,400 T1200,400 L1200,600 L0,600 Z"
          fill="rgba(79, 172, 247, 0.1)"
        >
          <animate
            attributeName="d"
            dur="20s"
            repeatCount="indefinite"
            values="
              M0,400 Q300,300 600,400 T1200,400 L1200,600 L0,600 Z;
              M0,400 Q300,500 600,400 T1200,400 L1200,600 L0,600 Z;
              M0,400 Q300,300 600,400 T1200,400 L1200,600 L0,600 Z
            "
          />
        </path>
      </svg>

      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-falcon-bg/50 to-falcon-bg" />
    </div>
  );
};

export default WaveBackground;
