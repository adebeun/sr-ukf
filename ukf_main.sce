// Reference: R. van der Merwe and E. Wan. 
// The Square-Root Unscented Kalman Filter for State and Parameter-Estimation, 2001
//
// By Zhe Hu at City University of Hong Kong, 05/01/2017
// Original Matlab code converted to Scilab 6.0.0 by Arthur de Beun 20171018

exec("ukf.sci");

n = 3;                                       // state dimension
m = 2;                                       // measurement dimension
q = 0.1;                                     // process noise
r = 0.1;                                     // measurement noise
Qs = q * eye(n, n);                          // process noise covariance
Rs = r * eye(m, m);                          // measurement noise covariance
deff('x = f(x)', ['temp = 0.05*x(1)*(x(2)+x(3))';
                  'x(1) = x(2)';
                  'x(2) = x(3)';
                  'x(3) = temp']);           // nonlinear state equations
deff('x = h(x)', 'x = x(1:m)');              // measurement equation
s = [0; 0; 1];                               // initial state
x = s + q * rand(n, 1, "normal");            // initial state with noise
S = eye(n, n);                               // initial square root of state covariance
N = 20;                                      // total dynamic steps
                                             // allocate memory
xV = zeros(n, N);                            // estimate
sV = zeros(n, N);                            // actual
zV = zeros(m, N);
for k = 1:N
  z = h(s) + r * rand(m, 1, "normal");       // measurements
  sV(:, k) = s;                              // save actual state
  zV(:, k) = z;                              // save measurement
  [x, S] = ukf(f, x, S, h, z, Qs, Rs);       // ukf 
  xV(:, k) = x;                              // save estimate
  s = f(s) + q * rand(n, 1, "normal");       // update process 
end
for k = 1:n                                  // plot results
  subplot(3, 1, k)
  plot(1:N, sV(k, :), '-', 1:N, xV(k, :), '--')
end
xV
