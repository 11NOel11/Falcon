import React, { useState } from 'react';
import Card from './Card';

type Layer = {
  name: string;
  optimizer: 'spectral' | 'orthogonal' | 'adamw';
};

const NetworkDiagram: React.FC = () => {
  const [hoveredLayer, setHoveredLayer] = useState<string | null>(null);

  const layers: Layer[] = [
    { name: 'Input', optimizer: 'adamw' },
    { name: 'Conv1', optimizer: 'spectral' },
    { name: 'Conv2', optimizer: 'spectral' },
    { name: 'FC1', optimizer: 'orthogonal' },
    { name: 'FC2', optimizer: 'orthogonal' },
    { name: 'Output', optimizer: 'adamw' },
  ];

  const getLayerColor = (optimizer: Layer['optimizer']) => {
    switch (optimizer) {
      case 'spectral':
        return '#4FACF7';
      case 'orthogonal':
        return '#E87BF8';
      case 'adamw':
        return '#9D4EDD';
    }
  };

  const getLayerDescription = (optimizer: Layer['optimizer']) => {
    switch (optimizer) {
      case 'spectral':
        return 'Frequency-domain filtering via FFT masking';
      case 'orthogonal':
        return 'Gram-Schmidt orthogonal projection';
      case 'adamw':
        return 'Standard adaptive moment estimation';
    }
  };

  return (
    <Card>
      <h3 className="text-xl font-bold mb-6 text-falcon-cyan">
        Network Architecture
      </h3>

      <div className="space-y-4">
        {layers.map((layer, idx) => (
          <div
            key={layer.name}
            className="relative"
            onMouseEnter={() => setHoveredLayer(layer.name)}
            onMouseLeave={() => setHoveredLayer(null)}
          >
            <div
              className="p-4 rounded-lg border-2 transition-all cursor-pointer"
              style={{
                borderColor: getLayerColor(layer.optimizer),
                backgroundColor:
                  hoveredLayer === layer.name
                    ? `${getLayerColor(layer.optimizer)}20`
                    : '#1C2240',
              }}
            >
              <div className="flex justify-between items-center">
                <div>
                  <h4 className="font-bold text-white">{layer.name}</h4>
                  <p className="text-xs text-gray-400">
                    {layer.optimizer === 'spectral'
                      ? 'Spectral Filtering'
                      : layer.optimizer === 'orthogonal'
                      ? 'Orthogonal Projection'
                      : 'AdamW'}
                  </p>
                </div>
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getLayerColor(layer.optimizer) }}
                />
              </div>

              {hoveredLayer === layer.name && (
                <div className="mt-3 pt-3 border-t border-gray-600">
                  <p className="text-xs text-gray-300">
                    {getLayerDescription(layer.optimizer)}
                  </p>
                </div>
              )}
            </div>

            {idx < layers.length - 1 && (
              <div
                className="h-6 w-0.5 mx-auto"
                style={{ backgroundColor: getLayerColor(layer.optimizer) }}
              />
            )}
          </div>
        ))}
      </div>

      <div className="mt-6 grid grid-cols-3 gap-3 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-falcon-blue" />
          <span className="text-gray-400">FFT Masking</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-falcon-pink" />
          <span className="text-gray-400">Gram-Schmidt</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-falcon-purple" />
          <span className="text-gray-400">AdamW</span>
        </div>
      </div>

      <div className="mt-4 p-3 bg-falcon-bg/50 rounded-lg border border-falcon-cyan/30">
        <p className="text-xs text-gray-400 italic">
          Falcon adaptively applies different update strategies to different layers
          based on their characteristics and training phase.
        </p>
      </div>
    </Card>
  );
};

export default NetworkDiagram;
