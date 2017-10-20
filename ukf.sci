funcprot(0);                                 // suppress warning about redefining functions

// SR-UKF   Square Root Unscented Kalman Filter for nonlinear dynamic systems
// [x, S] = ukf(f, x, S, h, z, Qs, Rs) returns state estimate, x and state covariance, P
// for nonlinear dynamic system (for simplicity, noise is assumed as additive):
//           x_k+1 = f(x_k) + w_k
//           z_k   = h(x_k) + v_k
// where w ~ N(0, Q) meaning w is gaussian noise with covariance Q
//       v ~ N(0, R) meaning v is gaussian noise with covariance R
// Inputs:   f: function handle for f(x)
//           x: "a priori" state estimate
//           S: "a priori" estimated square root of state covariance
//           h: function handle for h(x)
//           z: current measurement
//          Qs: process noise standard deviation
//          Rs: measurement noise standard deviation
// Output:   x: "a posteriori" state estimate
//           S: "a posteriori" square root of state covariance
//
// Example:  refer to ukf_main.sce
//
// Reference: R. van der Merwe and E. Wan. 
// The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
//
// By Zhe Hu at City University of Hong Kong, 05/01/2017
// Original Matlab code converted to Scilab 6.0.0 by Arthur de Beun 20171018
function [x, S] = ukf(fstate, x, S, hmeas, z, Qs, Rs)
  L = prod(size(x));                         // number of states (state dimension))
  m = prod(size(z));                         // number of measurements
  /* from section 2 of paper */
  alpha = 1e-3;                              // usually 1e-4 <= alpha <= 1e-1
  beta = 2;                                  // 2 is optimal for Gaussian distributions
  lambda = L * (alpha^2 - 1);
  c = L + lambda;
  eta = sqrt(c);                             // eta = sqrt(L + lambda)
                                             // sample mean weights
  Wm = [lambda/c 0.5/c+zeros(1, 2*L)];       // W(m)0 = lambda / (L + lambda)
                                             // W(m)i = 1 / (2 * (L + lambda)), i = 1..2L
                                             // covariance weights
  Wc = Wm;                                   // W(c)i = W(m)i for i = 1..2L
  Wc(1) = Wc(1) + (1 - alpha^2 + beta);      // W(c)0 = lambda / (L + lambda) + (1 - alpha^2 + beta)
  /* equation (17) in Algorithm 3.1 from paper */
  X = sigmas(x, S, eta);                     // sigma points around x
  /* equations (18) through (21) in Algorithm 3.1 from paper */
  [x1, X1, S1, X2] = ut(fstate, X, Wm, Wc, L, Qs ); // unscented transformation of process
  /* equations (22) through (25) in Algorithm 3.1 from paper */
  [z1, Z1, S2, Z2] = ut(hmeas, X1, Wm, Wc, m, Rs); // unscented transformation of measurements
  P12 = X2 * diag(Wc) * Z2';                 // equation (26) transformed cross-covariance
  K = P12 / S2 / S2';                        // equation (27) Kalman gain
  x = x1 + K * (z - z1);                     // state update
  /* posterior measurement update */
  U = K * S2';                               // equation (28)
  for i = 1:m
    S1 = choldowndate(S1, U(:, i));          // equation (29), for each measurement dimension
  end
  S = S1;
endfunction

function [y, Y, S, Y1] = ut(f, X, Wm, Wc, n, Rs)
// Unscented Transformation
// First call: Equations (18) through (21) in Algorithm 3.1 from paper
// Second call: Equations (22) through (25) in Algorithm 3.1 from paper
// Inputs:
//        f: nonlinear map
//        X: sigma points
//       Wm: weights for mean
//       Wc: weights for covariance
//        n: number of outputs of f
//       Rs: additive std
// Output:
//        y: transformed mean
//        Y: transformed sampling points
//        S: transformed square root of covariance
//       Y1: transformed deviations
  L = size(X, 2);
  y = zeros(n, 1);                           // initialise mean
  Y = zeros(n, L);
  for k = 1:L                                // L is actually 2L
    Y(:, k) = f(X(:, k));                    // equation (18) or (22)
    y = y + Wm(k) * Y(:, k);                 // equation (19) or (23) 
  end
  // time-update of Cholesky factor
  Y1 = Y - y(:, ones(1, L));                 // inner brackets of equation (20) or (24)
  residual = Y1 * diag(sqrt(abs(Wc)));       // left part of matrix in equation (20) or (24)
  [dummy, S] = qr([residual(:,2:L) Rs]', "e"); // equation (20) or (24)
                                             // equation (21) or (25)
  if Wc(1) < 0
    S = choldowndate(S, residual(:, 1));     // Matlab: cholupdate(S, residual(:, 1),'-')
  else
    S = cholupdate(S, residual(:, 1));       // Matlab: cholupdate(S, residual(:, 1),'+')
  end
endfunction

function X = sigmas(x, S, eta)
// Sigma points around reference point
// Equation (17) in Algorithm 3.1 from paper
// Inputs:
//       x: reference point
//       S: square root of covariance
//     eta: coefficient
// Output:
//       X: Sigma points
  A = eta * S';
  Y = x(:, ones(1, prod(size(x))));
  X = [x Y+A Y-A]; 
endfunction

/* Scilab 6.0.0 does not have a built-in function for cholupdate,
   this implementation is from
   https://en.wikipedia.org/wiki/Cholesky_decomposition */
function L = cholupdate(L, x)                // Matlab: cholupdate(L, x, '+')
  n = length(x);
  for k = 1:n
    r = sqrt(L(k, k)^2 + x(k)^2);
    c = r / L(k, k);
    s = x(k) / L(k, k);
    L(k, k) = r;
    L(k+1:n, k) = (L(k+1:n, k) + s * x(k+1:n)) / c;
    x(k+1:n) = c * x(k+1:n) - s * L(k+1:n, k);
  end
endfunction

function L = choldowndate(L, x)              // Matlab: cholupdate(L, x, '-')
  n = length(x);
  for k = 1:n
    r = sqrt(L(k, k)^2 - x(k)^2);
    c = r / L(k, k);
    s = x(k) / L(k, k);
    L(k, k) = r;
    L(k+1:n, k) = (L(k+1:n, k) - s * x(k+1:n)) / c;
    x(k+1:n) = c * x(k+1:n) - s * L(k+1:n, k);
  end
endfunction
