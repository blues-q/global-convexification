%RUN_PERFECT  Exact nominal model (Section 4.1): Figures 1 and 3.
%   Unregularised (L0) and regularised (J) training of a network with
%   N = 5 neurons from five random initialisations.

clear; close all;
N = 5;  nInit = 5;
[X, R] = henon_data(1000, 500, 1.4, 0.3, 1.4, 0.3);      % R = 0
opt = default_options();
opt.maxIter = 1e5;  opt.epsA = 1e-4;
reg = make_regulariser(X, R, N, zeros(5*N, 1));          % E_r = 0: no linear term
[K0, K2] = reg_constants(X, R, N);
fprintf('K0 = %.4f, K2 = %.4f\n', K0, K2);

modes = {'L0', 'J'};
res = cell(numel(modes), nInit);
for s = 1:numel(modes)
    parfor i = 1:nInit
        res{s, i} = train_run(init_params(N, 2, i), X, R, N, modes{s}, reg, opt);
    end
end
if ~isfolder('results'), mkdir('results'); end
save(fullfile('results', 'perfect.mat'), 'res', 'modes', 'opt', 'N', 'K0', 'K2');

for s = 1:numel(modes)
    L0 = cellfun(@(o) o.L0(end), res(s, :));
    nr = cellfun(@(o) o.nrm(end), res(s, :));
    fprintf('%-3s final L0: %s\n    final ||theta||: %s\n', modes{s}, ...
            sprintf('%.2e ', L0), sprintf('%.2e ', nr));
end
plot_perfect(res);
