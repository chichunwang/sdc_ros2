function figures = plot_saved_run(d)
%PLOT_SAVED_RUN 已保存真實結果的圖解；所有曲線都不是 ground truth。
% d 由 export_learning_data.py 產生，時間已先減 int64 origin 再轉秒。
% map_xy 包括被拒絕的候選，不能將整條候選當成接受後的定位輸出。

    t = d.time_s(:);
    good = logical(d.accepted(:));
    temporal = logical(d.temporal_ok(:));
    origin = d.gnss_xy(1, :);
    figures = gobjects(3, 1);
    blue = [0.15 0.35 0.75];
    orange = [0.85 0.35 0.10];
    green = [0.10 0.55 0.35];
    gray = [0.65 0.65 0.65];

    figures(1) = figure('Name', 'Saved trajectories: version differences', ...
        'Color', 'w', 'Position', [60 60 1200 760]);
    subplot(2, 2, 1);
    plot(d.gnss_xy(:, 1)-origin(1), d.gnss_xy(:, 2)-origin(2), '.', 'Color', gray);
    hold on;
    plot(d.prior_xy(:, 1)-origin(1), d.prior_xy(:, 2)-origin(2), 'Color', blue);
    plot(d.reference_xy(:, 1)-origin(1), d.reference_xy(:, 2)-origin(2), 'Color', green);
    plot(d.latest_xy(:, 1)-origin(1), d.latest_xy(:, 2)-origin(2), 'Color', orange);
    axis equal; grid on;
    xlabel('Map X relative to first GNSS (m)'); ylabel('Map Y relative to first GNSS (m)');
    title('Same map coordinates; pan/zoom to inspect');
    legend('Raw GNSS', 'Sensor prior', 'Previous best', 'Slow-map graph', 'Location', 'best');
    delta = d.latest_xy-d.prior_xy;
    oldDelta = d.reference_xy-d.prior_xy;
    for axisIndex = 1:2
        subplot(2, 2, axisIndex+1);
        plot(t, oldDelta(:, axisIndex), 'Color', green); hold on;
        plot(t, delta(:, axisIndex), 'Color', orange); grid on;
        xlabel('Time from GNSS origin (s)'); ylabel('Position change from prior (m)');
        title(sprintf('Map %s correction; not truth error', char('X'+axisIndex-1)));
        legend('Previous best - prior', 'Graph - prior', 'Location', 'best');
    end
    subplot(2, 2, 4);
    plot(t, wrapped(d.reference_yaw_rad-d.prior_yaw_rad)*180/pi, 'Color', green);
    hold on;
    plot(t, wrapped(d.latest_yaw_rad-d.prior_yaw_rad)*180/pi, 'Color', orange);
    grid on; xlabel('Time (s)'); ylabel('Wrapped heading change (deg)');
    title('Yaw changes from prior; not heading RMSE');
    legend('Previous best - prior', 'Graph - prior', 'Location', 'best');
    sgtitle(sprintf('Track %d: input, prior and two outputs (no ground truth)', d.track));

    figures(2) = figure('Name', 'Map observation reliability', ...
        'Color', 'w', 'Position', [85 85 1200 760]);
    subplot(2, 2, 1);
    scatter(t(~good), d.scale(~good), 8, gray, 'filled'); hold on;
    scatter(t(good), d.scale(good), 8, green, 'filled');
    yline(d.scale_bounds(1), '--'); yline(d.scale_bounds(2), '--');
    grid on; xlabel('Time (s)'); ylabel('Candidate scale (dimensionless)');
    title('Scale is a fitted candidate, including rejected fits');
    legend('Rejected', 'Geometrically accepted', 'Search lower bound', 'Search upper bound', 'Location', 'best');
    subplot(2, 2, 2);
    stairs(t, double(good), 'Color', blue); hold on;
    stairs(t, double(temporal)+0.04, 'Color', orange);
    ylim([-0.1 1.2]); yticks([0 1]); grid on;
    xlabel('Time (s)'); ylabel('Pass flag (0 / 1)');
    title('Temporal flag offset +0.04 for visibility');
    legend('Geometry', 'Adjacent displacement gate', 'Location', 'best');
    subplot(2, 2, 3);
    scatter(t, d.heldout_median_m, 8, gray, 'filled'); hold on;
    yline(d.heldout_limit_m, '--', 'Saved acceptance threshold');
    grid on; xlabel('Time (s)'); ylabel('Held-out median point-to-plane residual (m)');
    title('Residual measures local fit, not pose accuracy');
    subplot(2, 2, 4);
    plot(t(2:end), d.adjacent_discrepancy_m, 'Color', blue); hold on;
    plot(t(2:end), d.adjacent_gate_m, '--', 'Color', orange);
    grid on; xlabel('Second frame time of each adjacent pair (s)'); ylabel('Displacement discrepancy (m)');
    title('Gate also requires both frames geometrically accepted');
    legend('Map displacement - GNSS displacement norm', 'Saved gate', 'Location', 'best');
    sgtitle('Three stages: geometry -> adjacent consistency -> slow anchors');

    figures(3) = figure('Name', 'Slow GNSS bias and raw gyro model', ...
        'Color', 'w', 'Position', [110 110 1200 760]);
    mapDelta = d.map_xy-d.gnss_xy_at_frames;
    for axisIndex = 1:2
        subplot(2, 2, axisIndex);
        scatter(t(~good), mapDelta(~good, axisIndex), 6, gray, 'filled'); hold on;
        scatter(t(good), mapDelta(good, axisIndex), 6, blue, 'filled');
        plot(d.anchor_time_s, -d.anchor_bias_xy(:, axisIndex), 'kd', 'MarkerFaceColor', 'k');
        plot(d.node_time_s, -d.states(:, 6+axisIndex), 'Color', orange, 'LineWidth', 1.4);
        grid on; xlabel('Time (s)'); ylabel('Map minus raw GNSS (m)');
        title(sprintf('%s: z_{GNSS}=p+b; correction is -b', char('X'+axisIndex-1)));
        legend('Rejected candidate', 'Geometry accepted candidate', 'Slow anchor (-bias)', ...
            'Graph (-bias state)', 'Location', 'best');
    end
    subplot(2, 2, 3);
    plot(d.node_time_s, d.states(:, 6)*180/pi, 'Color', blue);
    grid on; xlabel('Time (s)'); ylabel('Effective yaw-rate bias (deg/s)');
    title('Estimated state; not calibrated 3-axis IMU bias');
    subplot(2, 2, 4);
    plot(t, d.load_ms, 'Color', gray); hold on;
    plot(t, d.registration_ms, 'Color', blue);
    plot(t, d.total_ms, 'Color', orange);
    grid on; xlabel('Time (s)'); ylabel('Per-frame wall time (ms)');
    title('Preparation and batch graph excluded');
    legend('PLY load + voxel', 'Registration', 'Frame', 'Location', 'best');
    sgtitle('Bias sign, sparse anchors and computation boundaries');
end

function result = wrapped(value)
    result = atan2(sin(value), cos(value));
end
