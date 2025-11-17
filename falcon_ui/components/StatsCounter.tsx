import React, { useEffect, useState, useRef } from 'react';

interface StatsCounterProps {
  value: number;
  label: string;
  suffix?: string;
  prefix?: string;
  decimals?: number;
  duration?: number;
  color?: string;
}

const StatsCounter: React.FC<StatsCounterProps> = ({
  value,
  label,
  suffix = '',
  prefix = '',
  decimals = 0,
  duration = 2000,
  color = 'text-falcon-cyan',
}) => {
  const [count, setCount] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [isVisible]);

  useEffect(() => {
    if (!isVisible) return;

    let startTime: number | null = null;
    const startValue = 0;
    const endValue = value;

    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);

      // Easing function (ease-out-cubic)
      const easeOut = 1 - Math.pow(1 - progress, 3);
      const currentCount = startValue + (endValue - startValue) * easeOut;

      setCount(currentCount);

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setCount(endValue);
      }
    };

    requestAnimationFrame(animate);
  }, [isVisible, value, duration]);

  return (
    <div ref={ref} className="text-center p-6 bg-falcon-card/50 rounded-lg border border-falcon-blue/20 hover:border-falcon-blue/50 transition-all duration-300 group">
      <div className={`text-4xl md:text-5xl font-bold ${color} mb-2 group-hover:scale-110 transition-transform duration-300`}>
        {prefix}
        {count.toFixed(decimals)}
        {suffix}
      </div>
      <div className="text-gray-400 text-sm uppercase tracking-wide">{label}</div>
      <div className="mt-2 h-1 bg-gradient-to-r from-falcon-blue via-falcon-cyan to-falcon-pink rounded-full transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500" />
    </div>
  );
};

export default StatsCounter;
