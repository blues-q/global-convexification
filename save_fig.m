function save_fig(fig, name)
%SAVE_FIG  Save a figure as results/<name>.pdf and .png.

if ~isfolder('results'), mkdir('results'); end
set(findall(fig, 'Type', 'axes'), 'FontName', 'Times New Roman', 'TickDir', 'in');
print(fig, fullfile('results', [name '.pdf']), '-dpdf', '-bestfit');
print(fig, fullfile('results', [name '.png']), '-dpng', '-r200');
end
