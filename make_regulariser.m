function reg = make_regulariser(X, R, N, v0)
%MAKE_REGULARISER  Coefficients of R(theta; r) with mu = 1.01 K0, gamma = K2.

[K0, K2] = reg_constants(X, R, N);
rn  = sqrt(sum(R.^2, 2));
Er  = mean(rn.^2);                             % residual energy
reg = struct('mu', 1.01*K0, 'gamma', K2, 'tilt', Er*v0, 'K0', K0, 'J0', Er/2);
end
