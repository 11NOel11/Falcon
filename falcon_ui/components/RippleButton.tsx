import React, { useState } from 'react';
import Link from 'next/link';

interface RippleButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  href?: string;
  variant?: 'primary' | 'secondary' | 'gradient';
  className?: string;
}

const RippleButton: React.FC<RippleButtonProps> = ({
  children,
  onClick,
  href,
  variant = 'primary',
  className = '',
}) => {
  const [ripples, setRipples] = useState<{ x: number; y: number; id: number }[]>([]);

  const handleClick = (e: React.MouseEvent<HTMLButtonElement | HTMLAnchorElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const newRipple = { x, y, id: Date.now() };

    setRipples((prev) => [...prev, newRipple]);

    setTimeout(() => {
      setRipples((prev) => prev.filter((ripple) => ripple.id !== newRipple.id));
    }, 600);

    if (onClick) onClick();
  };

  const variantClasses = {
    primary: 'bg-falcon-blue hover:bg-falcon-cyan text-white',
    secondary: 'bg-falcon-card border border-falcon-blue hover:border-falcon-cyan text-falcon-cyan',
    gradient: 'bg-gradient-to-r from-falcon-blue via-falcon-cyan to-falcon-pink text-white',
  };

  const baseClasses = `
    relative overflow-hidden px-8 py-4 rounded-lg font-semibold text-lg
    transition-all duration-300 transform hover:scale-105 hover:shadow-xl
    ${variantClasses[variant]}
    ${className}
  `;

  const content = (
    <>
      <span className="relative z-10">{children}</span>
      {ripples.map((ripple) => (
        <span
          key={ripple.id}
          className="absolute rounded-full bg-white/30 animate-ripple"
          style={{
            left: ripple.x,
            top: ripple.y,
            width: 0,
            height: 0,
          }}
        />
      ))}
    </>
  );

  if (href) {
    return (
      <Link href={href} className={baseClasses} onClick={handleClick}>
        {content}
      </Link>
    );
  }

  return (
    <button className={baseClasses} onClick={handleClick}>
      {content}
    </button>
  );
};

export default RippleButton;
