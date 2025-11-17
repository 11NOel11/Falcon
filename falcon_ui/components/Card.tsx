import React, { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}

const Card: React.FC<CardProps> = ({ children, className = '', hover = true }) => {
  return (
    <div
      className={`bg-falcon-card rounded-lg p-6 border border-gray-700 ${
        hover ? 'hover:border-falcon-blue hover:shadow-lg hover:shadow-falcon-blue/20 transition-all duration-300' : ''
      } ${className}`}
    >
      {children}
    </div>
  );
};

export default Card;
