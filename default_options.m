function opt = default_options()
%DEFAULT_OPTIONS  Adam and step-size settings of Section 4.

opt.lr    = 1e-3;                  % initial step size alpha_0
opt.beta1 = 0.9;   opt.beta2 = 0.999;
opt.epsA  = 1e-10;                 % Adam denominator constant
opt.nb    = 1000;                  % block length of the plateau test
opt.np    = 3;                     % blocks without progress
opt.delta = 0.02;                  % progress per block (decades)
opt.rho   = 0.5;                   % step-size decay factor
opt.lrMin = 1e-12;                 % minimum step size
opt.epsR  = 1e-3;                  % release tolerance epsilon_r
opt.Tg    = 2e4;                   % grace period after release
opt.maxIter = 2e5;
end
