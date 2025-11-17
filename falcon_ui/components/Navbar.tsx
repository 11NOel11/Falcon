import React, { useState, useEffect } from 'react';
import Link from 'next/link';

const Navbar: React.FC = () => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { href: '/', label: 'Home' },
    { href: '/trajectory', label: 'Trajectory' },
    { href: '/filter', label: 'Filter' },
    { href: '/dynamics', label: 'Dynamics' },
    { href: '/results', label: 'Results' },
  ];

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-falcon-bg/95 backdrop-blur-sm py-3'
          : 'bg-transparent py-5'
      }`}
    >
      <div className="container mx-auto px-6 flex justify-between items-center">
        <Link href="/" className="text-2xl font-display font-bold text-falcon-cyan">
          Falcon
        </Link>

        <div className="flex gap-8">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="text-gray-300 hover:text-falcon-blue transition-colors duration-200 font-medium"
            >
              {link.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
