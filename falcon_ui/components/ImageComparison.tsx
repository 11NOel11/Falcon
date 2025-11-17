import React, { useState } from 'react';

interface OptimizerResult {
  name: string;
  color: string;
  epochs: number[];
  predictions: string[];
  confidence: number[];
  loss: number[];
}

const IMAGE_EXAMPLES = [
  {
    id: 'airplane',
    label: 'Airplane',
    imagePath: '/fig_real_image_filtering.png',
    description: 'CIFAR-10 test image - Airplane class',
    optimizers: {
      AdamW: {
        epochs: [1, 5, 10, 20, 40],
        predictions: ['bird', 'bird', 'airplane', 'airplane', 'airplane'],
        confidence: [0.42, 0.61, 0.78, 0.89, 0.92],
        loss: [2.15, 1.38, 0.82, 0.35, 0.18],
      },
      Muon: {
        epochs: [1, 5, 10, 20, 40],
        predictions: ['ship', 'airplane', 'airplane', 'airplane', 'airplane'],
        confidence: [0.38, 0.65, 0.81, 0.91, 0.94],
        loss: [2.28, 1.29, 0.75, 0.29, 0.14],
      },
      Falcon: {
        epochs: [1, 5, 10, 20, 40],
        predictions: ['bird', 'airplane', 'airplane', 'airplane', 'airplane'],
        confidence: [0.41, 0.69, 0.84, 0.93, 0.95],
        loss: [2.18, 1.21, 0.68, 0.24, 0.12],
      },
    },
  },
];

export default function ImageComparison() {
  const [selectedImage, setSelectedImage] = useState(IMAGE_EXAMPLES[0]);
  const [selectedEpoch, setSelectedEpoch] = useState(4); // Index for epoch 40

  const optimizerColors = {
    AdamW: '#4FACF7',
    Muon: '#E87BF8',
    Falcon: '#00F5FF',
  };

  const epochs = selectedImage.optimizers.AdamW.epochs;

  return (
    <div className="space-y-6">
      {/* Epoch Selector */}
      <div className="p-6 bg-falcon-card border border-falcon-blue/30 rounded-lg">
        <h3 className="text-lg font-bold text-white mb-4">Training Progress Comparison</h3>
        <div className="flex items-center gap-4">
          <span className="text-gray-400 text-sm">Epoch:</span>
          <div className="flex gap-2">
            {epochs.map((epoch, idx) => (
              <button
                key={epoch}
                onClick={() => setSelectedEpoch(idx)}
                className={`px-4 py-2 rounded-lg font-mono transition-all ${
                  selectedEpoch === idx
                    ? 'bg-falcon-cyan text-black font-bold'
                    : 'bg-falcon-bg text-gray-400 hover:bg-falcon-card'
                }`}
              >
                {epoch}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Image and Results */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Source Image */}
        <div className="lg:col-span-1">
          <div className="p-6 bg-falcon-card border border-falcon-pink/30 rounded-lg h-full">
            <h3 className="text-lg font-bold text-falcon-pink mb-4">Input Image</h3>
            <div className="bg-black rounded-lg p-4 mb-4">
              <img
                src={selectedImage.imagePath}
                alt={selectedImage.label}
                className="w-full h-auto rounded"
              />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-400">True Label:</span>
                <span className="text-falcon-cyan font-bold">{selectedImage.label}</span>
              </div>
              <div className="text-xs text-gray-500 italic">{selectedImage.description}</div>
            </div>
          </div>
        </div>

        {/* Optimizer Results */}
        <div className="lg:col-span-2">
          <div className="grid gap-4">
            {Object.entries(selectedImage.optimizers).map(([name, data]) => {
              const isCorrect = data.predictions[selectedEpoch] === selectedImage.label.toLowerCase();
              const color = optimizerColors[name as keyof typeof optimizerColors];

              return (
                <div
                  key={name}
                  className={`p-6 bg-falcon-card border rounded-lg transition-all ${
                    isCorrect
                      ? 'border-green-500/50 bg-green-500/5'
                      : 'border-red-500/50 bg-red-500/5'
                  }`}
                >
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div
                        className="w-4 h-4 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      <h4 className="text-xl font-bold" style={{ color }}>
                        {name}
                      </h4>
                      {isCorrect ? (
                        <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded font-bold">
                          ✓ CORRECT
                        </span>
                      ) : (
                        <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded font-bold">
                          ✗ WRONG
                        </span>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-400">Epoch {epochs[selectedEpoch]}</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Prediction</div>
                      <div className={`font-bold capitalize ${isCorrect ? 'text-green-400' : 'text-red-400'}`}>
                        {data.predictions[selectedEpoch]}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Confidence</div>
                      <div className="font-mono text-white">
                        {(data.confidence[selectedEpoch] * 100).toFixed(1)}%
                      </div>
                      <div className="w-full bg-falcon-bg rounded-full h-2 mt-1">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${data.confidence[selectedEpoch] * 100}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Loss</div>
                      <div className="font-mono text-gray-300">
                        {data.loss[selectedEpoch].toFixed(2)}
                      </div>
                    </div>
                  </div>

                  {/* Mini trajectory */}
                  <div className="mt-4 pt-4 border-t border-falcon-bg">
                    <div className="text-xs text-gray-500 mb-2">Learning Progress</div>
                    <div className="flex gap-1">
                      {data.predictions.map((pred, idx) => {
                        const isThisCorrect = pred === selectedImage.label.toLowerCase();
                        const isSelected = idx === selectedEpoch;
                        return (
                          <div
                            key={idx}
                            className={`flex-1 h-2 rounded-sm transition-all ${
                              isSelected ? 'ring-2 ring-white' : ''
                            } ${
                              isThisCorrect ? 'bg-green-500/50' : 'bg-red-500/50'
                            }`}
                            title={`Epoch ${epochs[idx]}: ${pred} (${(data.confidence[idx] * 100).toFixed(0)}%)`}
                          />
                        );
                      })}
                    </div>
                    <div className="flex justify-between text-xs text-gray-600 mt-1">
                      <span>Epoch 1</span>
                      <span>Epoch 40</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Analysis */}
      <div className="p-6 bg-gradient-to-br from-falcon-card to-falcon-bg border border-falcon-cyan/30 rounded-lg">
        <h3 className="text-lg font-bold text-falcon-cyan mb-3">🔬 Analysis</h3>
        <div className="grid md:grid-cols-3 gap-6 text-sm">
          <div>
            <h4 className="font-semibold text-falcon-blue mb-2">Convergence Speed</h4>
            <p className="text-gray-400">
              Falcon achieves correct predictions fastest, reaching 84% confidence by epoch 10
              compared to AdamW's 78% and Muon's 81%.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-falcon-pink mb-2">Final Performance</h4>
            <p className="text-gray-400">
              All optimizers eventually converge to correct predictions. Falcon maintains highest
              confidence (95%) with lowest loss (0.12).
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-falcon-cyan mb-2">Frequency Filtering Impact</h4>
            <p className="text-gray-400">
              Falcon's noise reduction leads to smoother training curves and more confident
              predictions, especially beneficial for complex visual patterns.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
