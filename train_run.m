function out = train_run(theta, X, R, N, mode, reg, opt)
%TRAIN_RUN  One training run with full-batch Adam.
%   mode = 'L0'       : minimise L0
%          'J'        : minimise J throughout
%          'two-phase': minimise J until release (Algorithm 1), then L0
%   reg  : regulariser (see OBJECTIVE) with extra fields K0 and J0 = J(0)
%   opt  : maxIter, lr, epsA, beta1, beta2, nb, np, delta, rho, lrMin,
%          epsR (release tolerance), Tg (grace period)
%   out  : L0, J, nrm, step (one value per iteration), tRel, theta.
%
%   Step-size control: the minimum of the objective is recorded for blocks
%   of nb iterations; if log10 of it decreases by less than delta for np
%   blocks in a row (a plateau), the step size is multiplied by rho, down
%   to lrMin. In Phase 1 of 'two-phase', a plateau at lrMin also releases.

twoPhase = strcmp(mode, 'two-phase');
regOn    = ~strcmp(mode, 'L0');
released = ~twoPhase;                          % 'L0', 'J': no release
tRel = NaN;
m = zeros(size(theta));  v = m;  b1p = 1;  b2p = 1;
step = opt.lr;  graceEnd = 0;
blkMin = inf;  prevMin = inf;  nBad = 0;
[out.L0, out.J, out.nrm, out.step] = deal(zeros(opt.maxIter, 1));

for t = 1:opt.maxIter
    if regOn && ~(twoPhase && released), r = reg; else, r = []; end
    [J, g, L0] = objective(theta, X, R, N, r);
    out.L0(t) = L0;  out.J(t) = J;  out.nrm(t) = norm(theta);

    % plateau test on block minima of the current objective
    plateau = false;
    blkMin = min(blkMin, J);
    if mod(t, opt.nb) == 0
        if isfinite(prevMin) && log10(prevMin/max(blkMin, realmin)) < opt.delta
            nBad = nBad + 1;
        else
            nBad = 0;
        end
        prevMin = blkMin;  blkMin = inf;
        plateau = nBad >= opt.np;
    end

    % release condition (cert)
    release = ~released && norm(g)^2/(2*(reg.mu - reg.K0)) <= opt.epsR*reg.J0;
    if plateau
        if ~released && ~release && step > opt.lrMin
            step = max(step*opt.rho, opt.lrMin);
        elseif ~released
            release = true;                    % plateau at lrMin
        elseif t > graceEnd
            step = max(step*opt.rho, opt.lrMin);
        end
        nBad = 0;  prevMin = inf;  blkMin = inf;
    end
    if release
        released = true;  tRel = t;
        step = opt.lr;  graceEnd = t + opt.Tg;
        m(:) = 0;  v(:) = 0;  b1p = 1;  b2p = 1;
        [~, g] = objective(theta, X, R, N, []);
        nBad = 0;  prevMin = inf;  blkMin = inf;
    end

    % Adam step
    out.step(t) = step;
    b1p = b1p*opt.beta1;  b2p = b2p*opt.beta2;
    m = opt.beta1*m + (1 - opt.beta1)*g;
    v = opt.beta2*v + (1 - opt.beta2)*g.^2;
    theta = theta - step*(m/(1 - b1p)) ./ (sqrt(v/(1 - b2p)) + opt.epsA);
end
out.tRel = tRel;  out.theta = theta;
end
