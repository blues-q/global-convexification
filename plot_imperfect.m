function plot_imperfect(res, N)
%PLOT_IMPERFECT  One panel of Figure 4: L0 (top) and ||theta|| (bottom),
%   unregularised and two-phase training; dashed: median release iteration.

blue = [0.25 0.43 0.88];  red = [0.89 0.23 0.17];
runs = @(s, f) cell2mat(cellfun(@(o) o.(f), res(s, :), 'UniformOutput', false));
tr  = median(cellfun(@(o) o.tRel, res(2, :)));

f = figure('Units', 'inches', 'Position', [1 1 3.4 4.6]);
q = {'L0', 'nrm'};  lab = {'L_0', '||\Theta||'};
for p = 1:2
    ax = subplot(2, 1, p);  hold(ax, 'on');  box(ax, 'on');
    plot_band(ax, runs(1, q{p}), blue, p == 1);
    plot_band(ax, runs(2, q{p}), red, p == 1);
    plot(ax, [tr tr]/1e4, [1e-300 1e300], '--', 'Color', red, 'HandleVisibility', 'off');
    yl = [min(min(runs(2, q{p}))), max(max(runs(1, q{p})))];
    ylim(ax, [yl(1)/3, yl(2)*3]);
    ylabel(ax, lab{p});
    if p == 1
        title(ax, sprintf('N = %d', N), 'FontWeight', 'normal');
        legend(ax, {'Unregularised', 'Two-phase'}, 'Location', 'northeast', 'Box', 'off');
    else
        xlabel(ax, 'iteration (\times10^4)');
    end
end
save_fig(f, sprintf('fig_imperfect_N%d', N));
end
