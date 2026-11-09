function plot_band(ax, Y, col, showBand)
%PLOT_BAND  Geometric mean over runs (columns of Y) and, if showBand,
%   a band of one standard deviation of log10(Y), on a log axis.
%   Values <= 0 are left out (e.g. a loss of exactly zero).

it = (1:size(Y, 1))';
L  = log10(Y);  L(~isfinite(L)) = NaN;
n  = sum(~isnan(L), 2);
mu = sum(L, 2, 'omitnan')./n;
if showBand && size(Y, 2) > 1
    sd = sqrt(sum((L - mu).^2, 2, 'omitnan')./max(n - 1, 1));
    nb = 2000;                                  % band drawn on nb blocks
    e  = round(linspace(0, numel(it), nb + 1));
    xb = zeros(nb, 1);  lo = xb;  hi = xb;
    for k = 1:nb
        j = e(k)+1:e(k+1);
        xb(k) = mean(it(j));
        lo(k) = mean(mu(j) - sd(j), 'omitnan');  hi(k) = mean(mu(j) + sd(j), 'omitnan');
    end
    ok = isfinite(lo) & isfinite(hi);
    fill(ax, [xb(ok); flipud(xb(ok))]/1e4, 10.^[lo(ok); flipud(hi(ok))], col, ...
         'FaceAlpha', 0.2, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
plot(ax, it/1e4, 10.^mu, '-', 'Color', col, 'LineWidth', 1);
set(ax, 'YScale', 'log');
end
