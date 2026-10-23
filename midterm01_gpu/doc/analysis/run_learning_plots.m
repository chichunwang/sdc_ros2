function figures = run_learning_plots(matFile, outputDirectory)
%RUN_LEARNING_PLOTS 讀教學資料、開啟五張圖，存 PNG/FIG 到本目錄 figures。
% MATLAB Command Window: cd('.../midterm01_gpu/doc/analysis'); run_learning_plots
% Optional: run_learning_plots('/path/to/bundle.mat', '/path/to/new/figure/folder')
% Requires base MATLAB R2020b or later; no specialized toolbox is used.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(matFile)
        matFile = fullfile(here, 'data', 'track02_learning.mat');
    end
    assert(isfile(matFile), 'Teaching MAT missing: run export_learning_data.py first (see README).');
    loaded = load(matFile, 'sdc');
    assert(isfield(loaded, 'sdc') && strcmp(loaded.sdc.schema, 'sdc_learning_v1'), ...
        'Unsupported teaching bundle.');
    if nargin < 2 || isempty(outputDirectory)
        base = fullfile(here, 'figures', ['learning_' datestr(now, 'yymmddHHMMSS')]);
        outputDirectory = base;
        sequence = 2;
        while isfolder(outputDirectory) || isfile(outputDirectory)
            outputDirectory = sprintf('%s_%04d', base, sequence);
            sequence = sequence+1;
        end
    end
    assert(~isfolder(outputDirectory) && ~isfile(outputDirectory), ...
        'Preserve existing figures: select a new output directory.');
    mkdir(outputDirectory);
    figures = [plot_saved_run(loaded.sdc); demo_icp_observability(); demo_kalman_rts()];
    names = {'01_trajectory_changes', '02_map_reliability', '03_bias_and_timing', ...
        '04_icp_observability_synthetic', '05_kalman_bias_synthetic'};
    for k = 1:numel(figures)
        savefig(figures(k), fullfile(outputDirectory, [names{k} '.fig']));
        print(figures(k), fullfile(outputDirectory, [names{k} '.png']), '-dpng', '-r150');
    end
    fprintf('Saved %d teaching figures to: %s\n', numel(figures), outputDirectory);
    fprintf('Real-data plots contain no ground truth; synthetic truth is only for the demos.\n');
end
