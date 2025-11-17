/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'falcon-bg': '#0A0F24',
        'falcon-card': '#1C2240',
        'falcon-blue': '#4FACF7',
        'falcon-pink': '#E87BF8',
        'falcon-purple': '#9D4EDD',
        'falcon-cyan': '#00F5FF',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Playfair Display', 'serif'],
      },
      animation: {
        'wave': 'wave 10s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        wave: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(79, 172, 247, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(79, 172, 247, 1)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
          '33%': { transform: 'translateY(-10px) rotate(2deg)' },
          '66%': { transform: 'translateY(5px) rotate(-2deg)' },
        },
      },
    },
  },
  plugins: [],
}
