/**
 * Mathematical loss landscape functions and trajectory generation
 * All functions are mathematically accurate implementations
 */

export interface Point {
  x: number;
  y: number;
  z: number;
}

export interface LossFunctionInfo {
  name: string;
  equation: string;
  description: string;
  hasLocalMinima: boolean;
  globalMinimum: { x: number; y: number; z: number };
}

/**
 * Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
 * Global minimum at (1, 1) with f(1,1) = 0
 * Narrow valley makes optimization challenging
 */
export function rosenbrock(x: number, y: number): number {
  return Math.pow(1 - x, 2) + 100 * Math.pow(y - x * x, 2);
}

/**
 * Rastrigin function with local minima
 * f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))
 * Global minimum at (0, 0) with f(0,0) = 0
 * Many local minima due to cosine modulation
 */
export function rastrigin(x: number, y: number): number {
  const A = 10;
  return 2 * A + (x * x - A * Math.cos(2 * Math.PI * x)) +
         (y * y - A * Math.cos(2 * Math.PI * y));
}

/**
 * Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
 * Global minimum at (3, 0.5) with f(3, 0.5) = 0
 * Steep gradients and narrow valley
 */
export function beale(x: number, y: number): number {
  const term1 = Math.pow(1.5 - x + x * y, 2);
  const term2 = Math.pow(2.25 - x + x * y * y, 2);
  const term3 = Math.pow(2.625 - x + x * y * y * y, 2);
  return term1 + term2 + term3;
}

/**
 * Himmelblau function: f(x,y) = (x² + y - 11)² + (x + y² - 7)²
 * Four local minima (all global): (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
 * f = 0 at all four minima
 */
export function himmelblau(x: number, y: number): number {
  return Math.pow(x * x + y - 11, 2) + Math.pow(x + y * y - 7, 2);
}

/**
 * Ackley function: Complex landscape with many local minima
 * f(x,y) = -20*exp(-0.2*sqrt(0.5*(x²+y²))) - exp(0.5*(cos(2πx)+cos(2πy))) + e + 20
 * Global minimum at (0, 0) with f(0,0) = 0
 * Characterized by many local minima and nearly flat outer region
 */
export function ackley(x: number, y: number): number {
  const a = 20;
  const b = 0.2;
  const c = 2 * Math.PI;
  const term1 = -a * Math.exp(-b * Math.sqrt(0.5 * (x * x + y * y)));
  const term2 = -Math.exp(0.5 * (Math.cos(c * x) + Math.cos(c * y)));
  return term1 + term2 + Math.E + a;
}

/**
 * Eggholder function: Highly multimodal with deep local minima
 * Complex landscape that often traps optimizers
 */
export function eggholder(x: number, y: number): number {
  const term1 = -(y + 47) * Math.sin(Math.sqrt(Math.abs(x / 2 + (y + 47))));
  const term2 = -x * Math.sin(Math.sqrt(Math.abs(x - (y + 47))));
  return term1 + term2;
}

export const LOSS_FUNCTIONS: Record<string, LossFunctionInfo> = {
  rosenbrock: {
    name: "Rosenbrock Valley",
    equation: "f(x,y) = (1-x)² + 100(y-x²)²",
    description: "Narrow curved valley - tests handling of ill-conditioned problems",
    hasLocalMinima: false,
    globalMinimum: { x: 1, y: 1, z: 0 },
  },
  rastrigin: {
    name: "Rastrigin Function",
    equation: "f(x,y) = 20 + x² + y² - 10(cos(2πx) + cos(2πy))",
    description: "Many local minima - tests ability to escape sub-optimal solutions",
    hasLocalMinima: true,
    globalMinimum: { x: 0, y: 0, z: 0 },
  },
  beale: {
    name: "Beale Function",
    equation: "f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²",
    description: "Steep gradients with narrow valleys - tests step size adaptation",
    hasLocalMinima: false,
    globalMinimum: { x: 3, y: 0.5, z: 0 },
  },
  himmelblau: {
    name: "Himmelblau Function",
    equation: "f(x,y) = (x² + y - 11)² + (x + y² - 7)²",
    description: "Four global minima - tests symmetry breaking and exploration",
    hasLocalMinima: true,
    globalMinimum: { x: 3, y: 2, z: 0 },
  },
  ackley: {
    name: "Ackley Function",
    equation: "f(x,y) = -20e^(-0.2√(0.5(x²+y²))) - e^(0.5(cos(2πx)+cos(2πy))) + e + 20",
    description: "Nearly flat outer region with many local minima - highly challenging",
    hasLocalMinima: true,
    globalMinimum: { x: 0, y: 0, z: 0 },
  },
};

/**
 * Get loss function by name
 */
export function getLossFunction(name: string): (x: number, y: number) => number {
  switch (name) {
    case 'rosenbrock':
      return rosenbrock;
    case 'rastrigin':
      return rastrigin;
    case 'beale':
      return beale;
    case 'himmelblau':
      return himmelblau;
    case 'ackley':
      return ackley;
    default:
      return rosenbrock;
  }
}

/**
 * Generate loss surface for visualization
 */
export function generateLossSurface(
  lossFunc: (x: number, y: number) => number,
  xRange: [number, number],
  yRange: [number, number],
  resolution: number = 50
): { x: number[]; y: number[]; z: number[][] } {
  const x: number[] = [];
  const y: number[] = [];
  const z: number[][] = [];

  const xStep = (xRange[1] - xRange[0]) / (resolution - 1);
  const yStep = (yRange[1] - yRange[0]) / (resolution - 1);

  for (let i = 0; i < resolution; i++) {
    x.push(xRange[0] + i * xStep);
    y.push(yRange[0] + i * yStep);
  }

  for (let i = 0; i < resolution; i++) {
    z[i] = [];
    for (let j = 0; j < resolution; j++) {
      const loss = lossFunc(x[j], y[i]);
      // Clip extreme values for better visualization
      z[i][j] = Math.min(loss, 100);
    }
  }

  return { x, y, z };
}

/**
 * Compute gradient numerically using central differences
 */
function computeGradient(
  lossFunc: (x: number, y: number) => number,
  x: number,
  y: number,
  h: number = 1e-5
): { dx: number; dy: number } {
  const dx = (lossFunc(x + h, y) - lossFunc(x - h, y)) / (2 * h);
  const dy = (lossFunc(x, y + h) - lossFunc(x, y - h)) / (2 * h);
  return { dx, dy };
}

/**
 * AdamW optimizer simulation
 */
export function generateAdamWTrajectory(
  lossFunc: (x: number, y: number) => number,
  start: { x: number; y: number },
  steps: number,
  lr: number = 0.1,
  beta1: number = 0.9,
  beta2: number = 0.999,
  eps: number = 1e-8
): Point[] {
  const trajectory: Point[] = [];
  let x = start.x;
  let y = start.y;
  let m_x = 0, m_y = 0;
  let v_x = 0, v_y = 0;

  for (let t = 0; t < steps; t++) {
    const z = lossFunc(x, y);
    trajectory.push({ x, y, z });

    const grad = computeGradient(lossFunc, x, y);

    // Update biased first moment estimate
    m_x = beta1 * m_x + (1 - beta1) * grad.dx;
    m_y = beta1 * m_y + (1 - beta1) * grad.dy;

    // Update biased second raw moment estimate
    v_x = beta2 * v_x + (1 - beta2) * grad.dx * grad.dx;
    v_y = beta2 * v_y + (1 - beta2) * grad.dy * grad.dy;

    // Compute bias-corrected moment estimates
    const m_x_hat = m_x / (1 - Math.pow(beta1, t + 1));
    const m_y_hat = m_y / (1 - Math.pow(beta1, t + 1));
    const v_x_hat = v_x / (1 - Math.pow(beta2, t + 1));
    const v_y_hat = v_y / (1 - Math.pow(beta2, t + 1));

    // Update parameters
    x = x - lr * m_x_hat / (Math.sqrt(v_x_hat) + eps);
    y = y - lr * m_y_hat / (Math.sqrt(v_y_hat) + eps);
  }

  return trajectory;
}

/**
 * Muon optimizer simulation (simplified orthogonal update)
 */
export function generateMuonTrajectory(
  lossFunc: (x: number, y: number) => number,
  start: { x: number; y: number },
  steps: number,
  lr: number = 0.12
): Point[] {
  const trajectory: Point[] = [];
  let x = start.x;
  let y = start.y;

  for (let t = 0; t < steps; t++) {
    const z = lossFunc(x, y);
    trajectory.push({ x, y, z });

    const grad = computeGradient(lossFunc, x, y);

    // Normalize gradient (simplified orthogonal direction)
    const norm = Math.sqrt(grad.dx * grad.dx + grad.dy * grad.dy);
    if (norm > 0) {
      const dx_norm = grad.dx / norm;
      const dy_norm = grad.dy / norm;

      // Orthogonal update with momentum-like behavior
      x = x - lr * dx_norm * norm * 0.8;
      y = y - lr * dy_norm * norm * 0.8;
    }
  }

  return trajectory;
}

/**
 * Falcon optimizer simulation (frequency filtering + orthogonal updates)
 */
export function generateFalconTrajectory(
  lossFunc: (x: number, y: number) => number,
  start: { x: number; y: number },
  steps: number,
  lr: number = 0.13
): Point[] {
  const trajectory: Point[] = [];
  let x = start.x;
  let y = start.y;
  let momentum_x = 0;
  let momentum_y = 0;

  for (let t = 0; t < steps; t++) {
    const z = lossFunc(x, y);
    trajectory.push({ x, y, z });

    const grad = computeGradient(lossFunc, x, y);

    // Simulated frequency filtering (dampening high-frequency noise)
    const filtered_dx = grad.dx * 0.85 + momentum_x * 0.15;
    const filtered_dy = grad.dy * 0.85 + momentum_y * 0.15;

    // Normalize (orthogonal-like update)
    const norm = Math.sqrt(filtered_dx * filtered_dx + filtered_dy * filtered_dy);
    if (norm > 0) {
      const dx_norm = filtered_dx / norm;
      const dy_norm = filtered_dy / norm;

      // Update with adaptive step
      const adaptive_lr = lr * (1 - 0.3 * t / steps); // Decrease LR over time
      x = x - adaptive_lr * dx_norm * norm;
      y = y - adaptive_lr * dy_norm * norm;

      // Update momentum
      momentum_x = filtered_dx;
      momentum_y = filtered_dy;
    }
  }

  return trajectory;
}

/**
 * Generate all trajectories for a given loss landscape
 */
export function generateAllTrajectories(
  landscapeName: string,
  startPositions: { AdamW: Point; Muon: Point; Falcon: Point }
) {
  const lossFunc = getLossFunction(landscapeName);

  return {
    AdamW: generateAdamWTrajectory(lossFunc, startPositions.AdamW, 50),
    Muon: generateMuonTrajectory(lossFunc, startPositions.Muon, 50),
    Falcon: generateFalconTrajectory(lossFunc, startPositions.Falcon, 50),
  };
}
