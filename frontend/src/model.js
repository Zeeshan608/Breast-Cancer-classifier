import modelData from "./model_data.json" with { type: "json" };

function relu(arr) {
  return arr.map((x) => Math.max(0, x));
}

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

export function predictCancer(features) {
  const { scaler, layers } = modelData;
  const { mean, scale } = scaler;

  // 1. Scale input features: (x - mean) / scale
  const scaled = features.map((x, i) => (x - mean[i]) / scale[i]);

  // 2. Layer 1: Dense (30 -> 64)
  const z1 = new Array(64).fill(0);
  for (let j = 0; j < 64; j++) {
    let sum = layers.dense1.b[j];
    for (let i = 0; i < 30; i++) {
      sum += scaled[i] * layers.dense1.w[i][j];
    }
    z1[j] = sum;
  }
  const h1 = relu(z1);

  // 3. Layer 2: BatchNormalization
  const h1_bn = new Array(64);
  for (let j = 0; j < 64; j++) {
    const std = Math.sqrt(layers.bn1.var[j] + layers.bn1.eps);
    h1_bn[j] = layers.bn1.gamma[j] * ((h1[j] - layers.bn1.mean[j]) / std) + layers.bn1.beta[j];
  }

  // 4. Layer 3: Dense (64 -> 32)
  const z2 = new Array(32).fill(0);
  for (let j = 0; j < 32; j++) {
    let sum = layers.dense2.b[j];
    for (let i = 0; i < 64; i++) {
      sum += h1_bn[i] * layers.dense2.w[i][j];
    }
    z2[j] = sum;
  }
  const h2 = relu(z2);

  // 5. Layer 4: BatchNormalization
  const h2_bn = new Array(32);
  for (let j = 0; j < 32; j++) {
    const std = Math.sqrt(layers.bn2.var[j] + layers.bn2.eps);
    h2_bn[j] = layers.bn2.gamma[j] * ((h2[j] - layers.bn2.mean[j]) / std) + layers.bn2.beta[j];
  }

  // 6. Layer 5: Output Dense (32 -> 1) with Sigmoid
  let z3 = layers.dense3.b[0];
  for (let i = 0; i < 32; i++) {
    z3 += h2_bn[i] * layers.dense3.w[i][0];
  }

  const probability = sigmoid(z3);
  return {
    probability,
    label: probability >= 0.5 ? "Malignant" : "Benign",
  };
}
