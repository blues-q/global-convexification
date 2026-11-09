function [X, R] = henon_data(M, nBurn, a, b, aNom, bNom)
%HENON_DATA  States of the Henon map and residuals of a nominal model.
%   [X, R] = HENON_DATA(M, NBURN, A, B, ANOM, BNOM) iterates
%       x+ = [1 - a x(1)^2 + x(2); b x(1)]
%   from x = (0.1, 0.1), discards NBURN iterates and returns M states
%   X (M x 2) and the residuals R = x+ - fhat(x) (M x 2), where fhat is
%   the Henon map with coefficients ANOM and BNOM.

x = 0.1;  y = 0.1;
for n = 1:nBurn
    xn = 1 - a*x^2 + y;  y = b*x;  x = xn;
end
X = zeros(M, 2);
for n = 1:M
    X(n, :) = [x, y];
    xn = 1 - a*x^2 + y;  y = b*x;  x = xn;
end
R = [(aNom - a)*X(:, 1).^2, (b - bNom)*X(:, 1)];
end
