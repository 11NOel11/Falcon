/**
 * SVD and matrix utilities for FALCON UI
 */

/**
 * Matrix type
 */
export type Matrix = number[][];

/**
 * Multiply two matrices
 */
export function matrixMultiply(A: Matrix, B: Matrix): Matrix {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;

  const result: Matrix = [];
  for (let i = 0; i < rowsA; i++) {
    result[i] = [];
    for (let j = 0; j < colsB; j++) {
      let sum = 0;
      for (let k = 0; k < colsA; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * Transpose a matrix
 */
export function transpose(A: Matrix): Matrix {
  const rows = A.length;
  const cols = A[0].length;
  const result: Matrix = [];

  for (let j = 0; j < cols; j++) {
    result[j] = [];
    for (let i = 0; i < rows; i++) {
      result[j][i] = A[i][j];
    }
  }

  return result;
}

/**
 * Compute Frobenius norm
 */
export function frobeniusNorm(A: Matrix): number {
  let sum = 0;
  for (const row of A) {
    for (const val of row) {
      sum += val * val;
    }
  }
  return Math.sqrt(sum);
}

/**
 * Normalize a vector
 */
export function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
  return v.map(val => val / norm);
}

/**
 * Power iteration to find dominant singular value and vectors
 */
export function powerIteration(
  A: Matrix,
  maxIterations: number = 100,
  tolerance: number = 1e-6
): { u: number[]; sigma: number; v: number[] } {
  const m = A.length;
  const n = A[0].length;

  // Initialize random vector
  let v = new Array(n).fill(0).map(() => Math.random());
  v = normalize(v);

  let sigma = 0;
  let u: number[] = [];

  for (let iter = 0; iter < maxIterations; iter++) {
    // Compute A^T * A * v
    const Av = new Array(m).fill(0);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        Av[i] += A[i][j] * v[j];
      }
    }

    const ATAv = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < m; i++) {
        ATAv[j] += A[i][j] * Av[i];
      }
    }

    // Normalize
    const newSigmaSquared = ATAv.reduce((sum, val, idx) => sum + val * v[idx], 0);
    const newV = normalize(ATAv);

    // Check convergence
    const diff = newV.reduce((sum, val, idx) => sum + Math.abs(val - v[idx]), 0);
    if (diff < tolerance) {
      v = newV;
      sigma = Math.sqrt(newSigmaSquared);
      break;
    }

    v = newV;
    sigma = Math.sqrt(newSigmaSquared);
  }

  // Compute u = A * v / sigma
  u = new Array(m).fill(0);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      u[i] += A[i][j] * v[j];
    }
  }
  u = normalize(u);

  return { u, sigma, v };
}

/**
 * Simplified SVD using power iteration (computes top k singular values/vectors)
 */
export function svd(A: Matrix, k: number = 1): {
  U: Matrix;
  S: number[];
  V: Matrix;
} {
  const m = A.length;
  const n = A[0].length;

  const U: Matrix = [];
  const S: number[] = [];
  const V: Matrix = [];

  let residual = A.map(row => [...row]); // Copy matrix

  for (let i = 0; i < Math.min(k, m, n); i++) {
    const { u, sigma, v } = powerIteration(residual);

    U.push(u);
    S.push(sigma);
    V.push(v);

    // Deflate: residual = residual - sigma * u * v^T
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        residual[row][col] -= sigma * u[row] * v[col];
      }
    }
  }

  return { U, S, V };
}

/**
 * Reconstruct matrix from SVD with rank k
 */
export function reconstructFromSVD(
  U: Matrix,
  S: number[],
  V: Matrix,
  rank: number
): Matrix {
  const m = U.length;
  const n = V.length > 0 ? V[0].length : 0;

  const result: Matrix = [];
  for (let i = 0; i < m; i++) {
    result[i] = new Array(n).fill(0);
  }

  for (let k = 0; k < Math.min(rank, S.length); k++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        result[i][j] += S[k] * U[k][i] * V[k][j];
      }
    }
  }

  return result;
}

/**
 * Compute rank-1 approximation (outer product of two vectors)
 */
export function rank1Approximation(A: Matrix): Matrix {
  const { U, S, V } = svd(A, 1);
  return reconstructFromSVD(U, S, V, 1);
}

/**
 * Generate random matrix
 */
export function randomMatrix(rows: number, cols: number, scale: number = 1): Matrix {
  const result: Matrix = [];
  for (let i = 0; i < rows; i++) {
    result[i] = [];
    for (let j = 0; j < cols; j++) {
      result[i][j] = (Math.random() - 0.5) * 2 * scale;
    }
  }
  return result;
}

/**
 * Compute spectral norm (largest singular value)
 */
export function spectralNorm(A: Matrix): number {
  const { S } = svd(A, 1);
  return S[0];
}

/**
 * Create identity matrix
 */
export function identity(n: number): Matrix {
  const result: Matrix = [];
  for (let i = 0; i < n; i++) {
    result[i] = [];
    for (let j = 0; j < n; j++) {
      result[i][j] = i === j ? 1 : 0;
    }
  }
  return result;
}

/**
 * Apply orthogonal projection using Gram-Schmidt
 */
export function gramSchmidt(vectors: Matrix): Matrix {
  const result: Matrix = [];

  for (let i = 0; i < vectors.length; i++) {
    let v = [...vectors[i]];

    // Subtract projections onto previous orthonormal vectors
    for (let j = 0; j < i; j++) {
      const dot = v.reduce((sum, val, idx) => sum + val * result[j][idx], 0);
      v = v.map((val, idx) => val - dot * result[j][idx]);
    }

    // Normalize
    result.push(normalize(v));
  }

  return result;
}
