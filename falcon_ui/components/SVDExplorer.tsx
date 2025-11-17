import React, { useState, useEffect } from 'react';
import Card from './Card';
import Slider from './Slider';
import { randomMatrix, svd, reconstructFromSVD } from '@/utils/svd';

const SVDExplorer: React.FC = () => {
  const [matrix, setMatrix] = useState<number[][]>([]);
  const [rank, setRank] = useState(1);
  const [singularValues, setSingularValues] = useState<number[]>([]);
  const [reconstructed, setReconstructed] = useState<number[][]>([]);

  useEffect(() => {
    generateMatrix();
  }, []);

  const generateMatrix = () => {
    const newMatrix = randomMatrix(4, 4, 2);
    setMatrix(newMatrix);
    computeSVD(newMatrix, rank);
  };

  const computeSVD = (mat: number[][], r: number) => {
    try {
      const { U, S, V } = svd(mat, 4);
      setSingularValues(S);
      const recon = reconstructFromSVD(U, S, V, r);
      setReconstructed(recon);
    } catch (error) {
      console.error('SVD computation error:', error);
    }
  };

  useEffect(() => {
    if (matrix.length > 0) {
      computeSVD(matrix, rank);
    }
  }, [rank, matrix]);

  const renderMatrix = (mat: number[][], title: string, color: string) => (
    <div>
      <h4 className={`text-sm font-bold mb-2 text-${color}`}>{title}</h4>
      <div className="grid grid-cols-4 gap-1">
        {mat.map((row, i) =>
          row.map((val, j) => (
            <div
              key={`${i}-${j}`}
              className="bg-falcon-bg p-2 rounded text-center border border-gray-700"
            >
              <span className="text-xs text-gray-300">{val.toFixed(2)}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );

  return (
    <Card>
      <div className="space-y-6">
        <div className="flex justify-between items-center">
          <h3 className="text-xl font-bold text-falcon-cyan">SVD Explorer</h3>
          <button
            onClick={generateMatrix}
            className="px-4 py-2 bg-falcon-blue text-white rounded-lg hover:bg-falcon-pink transition-colors"
          >
            New Matrix
          </button>
        </div>

        <Slider
          label="Rank Approximation"
          value={rank}
          min={1}
          max={4}
          step={1}
          onChange={setRank}
        />

        <div className="grid md:grid-cols-2 gap-6">
          {matrix.length > 0 && renderMatrix(matrix, 'Original Matrix', 'falcon-blue')}
          {reconstructed.length > 0 &&
            renderMatrix(reconstructed, `Rank-${rank} Reconstruction`, 'falcon-pink')}
        </div>

        {singularValues.length > 0 && (
          <div>
            <h4 className="text-sm font-bold mb-2 text-falcon-cyan">
              Singular Values (σ)
            </h4>
            <div className="flex gap-2">
              {singularValues.slice(0, 4).map((val, idx) => (
                <div
                  key={idx}
                  className={`flex-1 p-3 rounded-lg border ${
                    idx < rank
                      ? 'bg-falcon-blue/20 border-falcon-blue'
                      : 'bg-gray-800 border-gray-700'
                  }`}
                >
                  <div className="text-xs text-gray-500">σ{idx + 1}</div>
                  <div className="text-sm font-bold text-white">{val.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="p-3 bg-falcon-bg/50 rounded-lg border border-falcon-purple/30">
          <p className="text-xs text-gray-400 italic">
            Rank-{rank} approximation captures the top {rank} singular value
            {rank > 1 ? 's' : ''}, preserving the most important structure while
            reducing dimensionality.
          </p>
        </div>
      </div>
    </Card>
  );
};

export default SVDExplorer;
