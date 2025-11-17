import React from 'react';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  unit?: string;
}

const Slider: React.FC<SliderProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  unit = '',
}) => {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium text-gray-300">{label}</label>
        <span className="text-sm text-falcon-blue font-semibold">
          {value.toFixed(2)}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
        style={{
          background: `linear-gradient(to right, #4FACF7 0%, #4FACF7 ${
            ((value - min) / (max - min)) * 100
          }%, #374151 ${((value - min) / (max - min)) * 100}%, #374151 100%)`,
        }}
      />
    </div>
  );
};

export default Slider;
