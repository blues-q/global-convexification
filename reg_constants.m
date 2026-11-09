function [K0, K2] = reg_constants(X, R, N)
%REG_CONSTANTS  Constants K0 and K2 of Proposition 1.
%   [K0, K2] = REG_CONSTANTS(X, R, N) for states X (M x d), residuals
%   R (M x d) and a tanh network with N neurons. J = L0 + R is strongly
%   convex for mu > K0 and gamma >= K2.

beta = 4/(3*sqrt(3));                          % max |tanh''|
A    = sqrt(max(sum(X.^2, 2) + 1));            % max ||(x_n, 1)||
Rmax = max(sqrt(sum(R.^2, 2)));                % max ||r_n||
D    = sqrt(N)*A + beta*A^2*Rmax;
K0   = A*Rmax + D/2;
K2   = sqrt(N)*beta*A^2 + D/2;
end
