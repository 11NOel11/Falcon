/**
 * Complex number representation
 */
export interface Complex {
  real: number;
  imag: number;
}

/**
 * Add two complex numbers
 */
export function complexAdd(a: Complex, b: Complex): Complex {
  return {
    real: a.real + b.real,
    imag: a.imag + b.imag,
  };
}

/**
 * Subtract two complex numbers
 */
export function complexSubtract(a: Complex, b: Complex): Complex {
  return {
    real: a.real - b.real,
    imag: a.imag - b.imag,
  };
}

/**
 * Multiply two complex numbers
 */
export function complexMultiply(a: Complex, b: Complex): Complex {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real,
  };
}

/**
 * Compute magnitude of complex number
 */
export function complexMagnitude(c: Complex): number {
  return Math.sqrt(c.real * c.real + c.imag * c.imag);
}

/**
 * Compute 1D FFT using Cooley-Tukey algorithm
 */
export function fft(input: Complex[]): Complex[] {
  const n = input.length;
  if (n <= 1) return input;

  // Ensure length is power of 2
  if ((n & (n - 1)) !== 0) {
    throw new Error('FFT input length must be a power of 2');
  }

  // Divide
  const even: Complex[] = [];
  const odd: Complex[] = [];
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) even.push(input[i]);
    else odd.push(input[i]);
  }

  // Conquer
  const fftEven = fft(even);
  const fftOdd = fft(odd);

  // Combine
  const result: Complex[] = new Array(n);
  for (let k = 0; k < n / 2; k++) {
    const angle = (-2 * Math.PI * k) / n;
    const w: Complex = {
      real: Math.cos(angle),
      imag: Math.sin(angle),
    };
    const t = complexMultiply(w, fftOdd[k]);
    result[k] = complexAdd(fftEven[k], t);
    result[k + n / 2] = complexSubtract(fftEven[k], t);
  }

  return result;
}

/**
 * Compute 1D inverse FFT
 */
export function ifft(input: Complex[]): Complex[] {
  const n = input.length;

  // Conjugate input
  const conjugated = input.map(c => ({ real: c.real, imag: -c.imag }));

  // Apply FFT
  const result = fft(conjugated);

  // Conjugate and scale
  return result.map(c => ({
    real: c.real / n,
    imag: -c.imag / n,
  }));
}

/**
 * Compute 2D FFT
 */
export function fft2D(input: number[][]): Complex[][] {
  const rows = input.length;
  const cols = input[0].length;

  // Pad to power of 2 if necessary
  const paddedRows = Math.pow(2, Math.ceil(Math.log2(rows)));
  const paddedCols = Math.pow(2, Math.ceil(Math.log2(cols)));

  // Convert to complex and pad
  const complexInput: Complex[][] = [];
  for (let i = 0; i < paddedRows; i++) {
    complexInput[i] = [];
    for (let j = 0; j < paddedCols; j++) {
      complexInput[i][j] = {
        real: i < rows && j < cols ? input[i][j] : 0,
        imag: 0,
      };
    }
  }

  // FFT along rows
  const rowFFT: Complex[][] = complexInput.map(row => fft(row));

  // Transpose
  const transposed: Complex[][] = [];
  for (let j = 0; j < paddedCols; j++) {
    transposed[j] = [];
    for (let i = 0; i < paddedRows; i++) {
      transposed[j][i] = rowFFT[i][j];
    }
  }

  // FFT along columns (which are now rows after transpose)
  const colFFT = transposed.map(row => fft(row));

  // Transpose back
  const result: Complex[][] = [];
  for (let i = 0; i < paddedRows; i++) {
    result[i] = [];
    for (let j = 0; j < paddedCols; j++) {
      result[i][j] = colFFT[j][i];
    }
  }

  return result;
}

/**
 * Compute 2D inverse FFT
 */
export function ifft2D(input: Complex[][]): number[][] {
  const rows = input.length;
  const cols = input[0].length;

  // Conjugate input
  const conjugated = input.map(row =>
    row.map(c => ({ real: c.real, imag: -c.imag }))
  );

  // Apply 2D FFT
  const fftResult = fft2D(conjugated.map(row => row.map(c => c.real)));

  // Conjugate and scale
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result[i] = [];
    for (let j = 0; j < cols; j++) {
      result[i][j] = fftResult[i][j].real / (rows * cols);
    }
  }

  return result;
}

/**
 * Get magnitude spectrum from FFT result
 */
export function getMagnitudeSpectrum(fftResult: Complex[][]): number[][] {
  return fftResult.map(row => row.map(c => complexMagnitude(c)));
}

/**
 * Shift zero-frequency component to center
 */
export function fftshift(input: number[][]): number[][] {
  const rows = input.length;
  const cols = input[0].length;
  const result: number[][] = [];

  for (let i = 0; i < rows; i++) {
    result[i] = [];
    for (let j = 0; j < cols; j++) {
      const newI = (i + Math.floor(rows / 2)) % rows;
      const newJ = (j + Math.floor(cols / 2)) % cols;
      result[i][j] = input[newI][newJ];
    }
  }

  return result;
}

/**
 * Apply frequency domain filtering based on energy retention
 */
export function applyFrequencyFilter(
  fftResult: Complex[][],
  retainFraction: number
): Complex[][] {
  const magnitudes = getMagnitudeSpectrum(fftResult);
  const flatMagnitudes = magnitudes.flat();
  const energies = flatMagnitudes.map(m => m * m);

  // Calculate total energy
  const totalEnergy = energies.reduce((sum, e) => sum + e, 0);
  const targetEnergy = totalEnergy * retainFraction;

  // Sort energies in descending order with indices
  const sortedIndices = energies
    .map((e, idx) => ({ energy: e, index: idx }))
    .sort((a, b) => b.energy - a.energy);

  // Find threshold
  let cumulativeEnergy = 0;
  let threshold = 0;
  for (const { energy } of sortedIndices) {
    cumulativeEnergy += energy;
    if (cumulativeEnergy >= targetEnergy) {
      threshold = Math.sqrt(energy);
      break;
    }
  }

  // Apply mask
  const result: Complex[][] = [];
  for (let i = 0; i < fftResult.length; i++) {
    result[i] = [];
    for (let j = 0; j < fftResult[i].length; j++) {
      const mag = magnitudes[i][j];
      if (mag >= threshold) {
        result[i][j] = fftResult[i][j];
      } else {
        result[i][j] = { real: 0, imag: 0 };
      }
    }
  }

  return result;
}
