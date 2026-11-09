function v0 = v0_direction(X, R, N)
%V0_DIRECTION  Direction v0 of the linear term (Proposition 2).
%   Neuron k gets output weights c_k*wbar and input weights and bias
%   c_k*ubar, where (ubar, wbar) is the leading singular pair of
%   C = mean(r_n a_n'), a_n = (x_n, 1), and c_k/sqrt(2) is equally
%   spaced in [0.5, 1.5]. Layout as in OBJECTIVE.

C = (R.'*[X, ones(size(X, 1), 1)])/size(X, 1);
[U, ~, V] = svd(C);
wbar = U(:, 1);  ubar = V(:, 1);
[~, i] = max(abs(wbar));                       % sign convention
if wbar(i) < 0, wbar = -wbar;  ubar = -ubar; end
c  = sqrt(2)*linspace(0.5, 1.5, N)';
v0 = [reshape(c*wbar.', [], 1); c*ubar(1); c*ubar(2); c*ubar(3)];
end
