import React, { useState, useRef } from 'react';

interface AnimatedCardProps {
  title: string;
  description: string;
  icon?: React.ReactNode;
  gradient?: string;
  href?: string;
  onClick?: () => void;
}

const AnimatedCard: React.FC<AnimatedCardProps> = ({
  title,
  description,
  icon,
  gradient = 'from-falcon-blue/20 to-falcon-pink/20',
  href,
  onClick,
}) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    setMousePosition({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  const CardContent = (
    <div
      ref={cardRef}
      className="relative group overflow-hidden rounded-xl bg-falcon-card border border-falcon-blue/30 p-6 transition-all duration-500 hover:border-falcon-blue hover:shadow-2xl hover:shadow-falcon-blue/20 cursor-pointer"
      onMouseMove={handleMouseMove}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
      style={{
        transform: isHovered ? 'translateY(-8px) scale(1.02)' : 'translateY(0) scale(1)',
      }}
    >
      {/* Animated gradient background */}
      <div
        className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}
      />

      {/* Spotlight effect */}
      {isHovered && (
        <div
          className="absolute rounded-full bg-white/10 blur-2xl transition-all duration-300"
          style={{
            width: '200px',
            height: '200px',
            left: mousePosition.x - 100,
            top: mousePosition.y - 100,
            pointerEvents: 'none',
          }}
        />
      )}

      {/* Content */}
      <div className="relative z-10">
        {icon && (
          <div className="mb-4 text-falcon-cyan transform group-hover:scale-110 transition-transform duration-300">
            {icon}
          </div>
        )}
        <h3 className="text-xl font-bold text-white mb-2 group-hover:text-falcon-cyan transition-colors">
          {title}
        </h3>
        <p className="text-gray-400 text-sm group-hover:text-gray-300 transition-colors">
          {description}
        </p>
      </div>

      {/* Shine effect */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700"
        style={{
          background:
            'linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%)',
          transform: isHovered ? 'translateX(100%)' : 'translateX(-100%)',
          transition: 'transform 0.7s',
        }}
      />
    </div>
  );

  if (href) {
    return <a href={href}>{CardContent}</a>;
  }

  return CardContent;
};

export default AnimatedCard;
