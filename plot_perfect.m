function plot_perfect(res)
%PLOT_PERFECT  Figure 1 (unregularised L0) and Figure 3 (L0 and ||theta||,
%   unregularised and regularised, insets: unregularised on own scale).

blue = [0.25 0.43 0.88];  red = [0.89 0.23 0.17];
runs = @(s, f) cell2mat(cellfun(@(o) o.(f), res(s, :), 'UniformOutput', false));

f1 = figure('Units', 'inches', 'Position', [1 1 3.3 2.5]);
ax = axes(f1);  hold(ax, 'on');  box(ax, 'on');
plot_band(ax, runs(1, 'L0'), blue, true);
xlabel(ax, 'iteration (\times10^4)');  ylabel(ax, 'L_0');
save_fig(f1, 'fig_unreg');

f3 = figure('Units', 'inches', 'Position', [1 1 6.2 2.5]);
q = {'L0', 'nrm'};  lab = {'L_0', '||\Theta||'};
for p = 1:2
    ax = subplot(1, 2, p);  hold(ax, 'on');  box(ax, 'on');
    plot_band(ax, runs(1, q{p}), blue, p == 1);
    plot_band(ax, runs(2, q{p}), red, p == 1);
    ylim(ax, [1e-310 1e20]);  set(ax, 'YTick', 10.^(-300:100:0));
    xlabel(ax, 'iteration (\times10^4)');  ylabel(ax, lab{p});
    if p == 1, legend(ax, {'unregularised (L_0)', 'regularised (J)'}, 'Location', 'northoutside', ...
                      'Orientation', 'horizontal', 'Box', 'off'); end
    pos = get(ax, 'Position');
    in = axes(f3, 'Position', [pos(1)+0.42*pos(3), pos(2)+0.22*pos(4), 0.52*pos(3), 0.5*pos(4)]);
    hold(in, 'on');  box(in, 'on');
    plot_band(in, runs(1, q{p}), blue, p == 1);
    if p == 2, set(in, 'YScale', 'linear'); end
    set(in, 'FontSize', 7);
end
save_fig(f3, 'fig_perfect');
end
