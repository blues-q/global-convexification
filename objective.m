function [J, grad, L0] = objective(theta, X, R, N, reg)
%OBJECTIVE  Loss L0, regularised objective J and its gradient.
%   theta = [W2(:); W1(:); b1] with output weights W2 (N x d), input
%   weights W1 (N x 2) and biases b1 (N x 1); the network is
%   g(x) = W2' tanh(W1 x + b1).
%   L0 = 1/(2M) sum_n ||g(x_n) - r_n||^2.
%   reg = struct('mu', mu, 'gamma', gamma, 'tilt', Er*v0) gives
%   J = L0 + mu/2 ||theta||^2 + gamma/4 ||theta||^4 - Er v0'theta;
%   reg = [] gives J = L0.

d  = size(R, 2);
W2 = reshape(theta(1:d*N), N, d);
W1 = reshape(theta(d*N+1:(d+2)*N), N, 2);
b1 = theta((d+2)*N+1:end);
H  = tanh(X*W1' + b1.');
E  = H*W2 - R;
L0 = sum(mean(E.^2, 1))/2;
th2 = sum(theta.^2);
if isempty(reg)
    J = L0;
else
    J = L0 + (reg.mu/2*th2 + reg.gamma/4*th2^2 - reg.tilt.'*theta);
end
if nargout > 1
    M  = size(X, 1);
    S  = (E*W2.') .* (1 - H.^2);
    grad = [reshape((1/M)*(H.'*E), [], 1); reshape((1/M)*(S.'*X), [], 1); ...
            (1/M)*sum(S, 1).'];
    if ~isempty(reg)
        grad = grad + ((reg.mu + reg.gamma*th2)*theta - reg.tilt);
    end
end
end
