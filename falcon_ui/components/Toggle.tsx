import React from 'react';

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  color?: string;
}

const Toggle: React.FC<ToggleProps> = ({ label, checked, onChange, color = '#4FACF7' }) => {
  return (
    <label className="flex items-center space-x-3 cursor-pointer group">
      <div className="relative">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only"
        />
        <div
          className={`w-11 h-6 rounded-full transition-colors duration-300 ${
            checked ? 'bg-falcon-blue' : 'bg-gray-600'
          }`}
          style={checked ? { backgroundColor: color } : {}}
        />
        <div
          className={`absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform duration-300 ${
            checked ? 'transform translate-x-5' : ''
          }`}
        />
      </div>
      <span className="text-sm font-medium text-gray-300 group-hover:text-white transition-colors">
        {label}
      </span>
    </label>
  );
};

export default Toggle;
