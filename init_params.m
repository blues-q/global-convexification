function theta = init_params(N, d, seed)
%INIT_PARAMS  Theta_0 ~ N(0, I) in the layout of OBJECTIVE.

rng(seed);
W2 = randn(N, d);  W1 = randn(N, 2);  b1 = randn(N, 1);
theta = [W2(:); W1(:); b1];
end
