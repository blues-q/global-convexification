%RUN_IMPERFECT  Imperfect nominal model (Section 4.2): Figure 4 and Table 2.
%   Nominal coefficients a = 1.3, b = 0.25 (true 1.4, 0.3); unregularised
%   and two-phase training for N = 50, 150, 400 from five random
%   initialisations. N = 400 takes about an hour.

clear; close all;
Ns = [50 150 400];  nInit = 5;
[X, R] = henon_data(1000, 500, 1.4, 0.3, 1.3, 0.25);
opt = default_options();
modes = {'L0', 'two-phase'};
if ~isfolder('results'), mkdir('results'); end

for N = Ns
    reg = make_regulariser(X, R, N, v0_direction(X, R, N));
    fprintf('N = %d: K0 = %.4f, K2 = %.4f, Er = %.4e\n', N, reg.K0, reg.gamma, 2*reg.J0);
    res = cell(numel(modes), nInit);
    for s = 1:numel(modes)
        parfor i = 1:nInit
            res{s, i} = train_run(init_params(N, 2, i), X, R, N, modes{s}, reg, opt);
        end
    end
    save(fullfile('results', sprintf('imperfect_N%d.mat', N)), 'res', 'modes', 'opt', 'N', 'reg');
    plot_imperfect(res, N);
end
table_imperfect(Ns);
