function fig = demo_kalman_rts()
%DEMO_KALMAN_RTS 合成的一維 [position,velocity] KF/RTS 與 bias 可辨識性。
% 教學參數只控制合成資料；不讀寫 production config，不是完整八狀態圖。
% 真值只存在於本合成例；真實 GNSS、map、最佳 CSV 都不能當真值。

    previousRng = rng;
    cleanup = onCleanup(@() rng(previousRng)); %#ok<NASGU>
    rng(21, 'twister');
    t = (0:0.2:40)';
    truth = 2*t+0.8*sin(t/5);
    bias = 0.8*sin(t/12);
    measurementStd = 0.3;
    measurement = truth+bias+measurementStd*randn(size(t));
    n = numel(t);
    h = [1 0];
    r = measurementStd^2;
    accelerationPsd = 0.2; % Illustrative model setting, not sensor calibration.
    means = zeros(2, n); covariances = zeros(2, 2, n);
    predicted = zeros(2, n); predictedCov = zeros(2, 2, n);
    state = [measurement(1); (measurement(2)-measurement(1))/(t(2)-t(1))];
    p = diag([10 10]);
    for k = 1:n
        if k == 1
            dt = 0;
        else
            dt = t(k)-t(k-1);
        end
        f = [1 dt; 0 1];
        q = accelerationPsd*[dt^3/3 dt^2/2; dt^2/2 dt];
        state = f*state; p = f*p*f'+q;
        predicted(:, k) = state; predictedCov(:, :, k) = p;
        innovation = measurement(k)-h*state;
        gain = p*h'/(h*p*h'+r);
        state = state+gain*innovation;
        a = eye(2)-gain*h;
        p = a*p*a'+gain*r*gain'; % Joseph covariance update.
        means(:, k) = state; covariances(:, :, k) = (p+p')/2;
    end
    smooth = means;
    for k = n-1:-1:1
        f = [1 t(k+1)-t(k); 0 1];
        gain = (covariances(:, :, k)*f')/predictedCov(:, :, k+1);
        smooth(:, k) = means(:, k)+gain*(smooth(:, k+1)-predicted(:, k+1));
    end

    fig = figure('Name', 'Synthetic Kalman, RTS and GNSS bias', 'Color', 'w', ...
        'Position', [160 160 1200 760]);
    subplot(2, 2, 1);
    plot(t, measurement, '.', 'Color', [0.7 0.7 0.7]); hold on;
    plot(t, truth, 'k', 'LineWidth', 1.4);
    plot(t, means(1, :), 'Color', [0.15 0.35 0.75]);
    plot(t, smooth(1, :), 'Color', [0.85 0.35 0.1]);
    grid on; xlabel('Synthetic time (s)'); ylabel('Position (m)');
    title('KF uses past/current data; RTS also uses later data');
    legend('Synthetic GNSS', 'Known synthetic truth', 'KF', 'RTS', 'Location', 'best');
    subplot(2, 2, 2);
    plot(t, means(1, :)'-truth); hold on;
    plot(t, smooth(1, :)'-truth); plot(t, bias, 'k--', 'LineWidth', 1.3);
    grid on; xlabel('Synthetic time (s)'); ylabel('Error against known synthetic truth (m)');
    title('Smoothing alone cannot separate slow GNSS bias');
    legend('KF error', 'RTS error', 'Injected slow bias', 'Location', 'best');

    % One-sample identifiability example, independent of the time-series demo.
    % With only z=p+b, all p+b=z pairs have the same likelihood.
    [position, offset] = meshgrid(linspace(8.5, 11.5, 101), linspace(-1, 2.5, 101));
    gnssValue = 10.8;
    mapValue = 10.0; % Known synthetic absolute observation; not a real map pose.
    costGnss = ((position+offset-gnssValue)/0.3).^2;
    costBoth = costGnss+((position-mapValue)/0.2).^2;
    levels = [0.25 1 4 9 16];
    subplot(2, 2, 3);
    contour(position, offset, costGnss, levels); hold on;
    plot(10, 0.8, 'kp', 'MarkerFaceColor', 'k');
    xlabel('Unknown position p (m)'); ylabel('Unknown GNSS bias b (m)'); grid on;
    title('GNSS only: a valley of equally valid (p,b)');
    subplot(2, 2, 4);
    contour(position, offset, costBoth, levels); hold on;
    plot(10, 0.8, 'kp', 'MarkerFaceColor', 'k');
    xlabel('Unknown position p (m)'); ylabel('Unknown GNSS bias b (m)'); grid on;
    title('An independent absolute anchor constrains the valley');
    sgtitle('Synthetic lesson: temporal smoothing and absolute bias identification differ');
end
