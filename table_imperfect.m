function table_imperfect(Ns)
%TABLE_IMPERFECT  Table 2: geometric means of the final L0 and ||theta||
%   and the median release iteration, from results/imperfect_N*.mat.

gm = @(v) exp(mean(log(v)));
fprintf('%5s %-10s %12s %10s %12s\n', 'N', 'training', 'final L0', '||theta||', 't_r [1e4]');
for N = Ns
    S = load(fullfile('results', sprintf('imperfect_N%d.mat', N)));
    for s = 1:numel(S.modes)
        L0 = cellfun(@(o) o.L0(end), S.res(s, :));
        nr = cellfun(@(o) o.nrm(end), S.res(s, :));
        tr = cellfun(@(o) o.tRel, S.res(s, :));
        if all(isnan(tr)), trs = '-'; else, trs = sprintf('%.2f', median(tr)/1e4); end
        fprintf('%5d %-10s %12.2e %10.2f %12s\n', N, S.modes{s}, gm(L0), gm(nr), trs);
    end
end
end
