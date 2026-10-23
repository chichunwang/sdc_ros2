function fig = demo_icp_observability()
%DEMO_ICP_OBSERVABILITY 合成幾何教學，不重新執行定位或修改 production 設定。
% 已知正確對應，用相同 point-to-plane Jacobian 觀察局部可觀測性。
% 此處省略 ICP 的最近鄰更新、robust weights、ground gate 與 holdout。
% r = n'*(s*R*p+t-q), J = [n, cross(s*R*p,n), n'*(s*R*p)]。

    [x, y] = meshgrid(linspace(-6, 6, 13), linspace(-5, 5, 11));
    ground = [x(:), y(:), zeros(numel(x), 1)];
    groundN = repmat([0 0 1], size(ground, 1), 1);
    [a, z] = meshgrid(linspace(-5, 5, 11), linspace(0, 4, 9));
    wallX = [6*ones(numel(a), 1), a(:), z(:)];
    [a, z] = meshgrid(linspace(-6, 6, 13), linspace(0, 4, 9));
    wallY = [a(:), 5*ones(numel(a), 1), z(:)];
    room = [ground; wallX; wallX.*[-1 1 1]; wallY; wallY.*[1 -1 1]];
    normal = [groundN; repmat([1 0 0], 2*size(wallX, 1), 1); ...
        repmat([0 1 0], 2*size(wallY, 1), 1)];
    sensor = [0 0 1.5];
    % Illustrative variable increments, same dimensional roles as production;
    % singular values depend on these units and are NOT a covariance/accuracy.
    units = [1 1 1 0.05 0.05 0.05 0.1];
    [jGround, sGround, hGround] = geometry(ground-sensor, groundN, units);
    [jRoom, sRoom, hRoom] = geometry(room-sensor, normal, units);
    assert(norm(jGround(:, [1 2 6]), 'fro') < 1e-12);
    assert(sRoom(end) > 1e-6);

    fig = figure('Name', 'Synthetic ICP observability', 'Color', 'w', ...
        'Position', [135 135 1200 760]);
    subplot(2, 2, 1);
    scatter3(room(:, 1), room(:, 2), room(:, 3), 8, normal, 'filled');
    hold on; plot3(sensor(1), sensor(2), sensor(3), 'kp', 'MarkerSize', 12, 'MarkerFaceColor', 'k');
    axis equal; grid on; view(35, 25);
    xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
    title('Synthetic ground + opposing walls; known correspondences');
    subplot(2, 2, 2);
    semilogy(1:7, max(sGround, 1e-14), 'o-'); hold on;
    semilogy(1:7, max(sRoom, 1e-14), 's-'); grid on;
    xlabel('Singular-value index'); ylabel('Scaled Jacobian singular value');
    title('7 unknowns: translation, rotation, log-scale');
    legend('Ground only', 'Ground + walls', 'Location', 'best');
    subplot(2, 2, 3);
    semilogy(1:3, max(hGround, 1e-14), 'o-'); hold on;
    semilogy(1:3, max(hRoom, 1e-14), 's-'); grid on;
    xlabel('Horizontal singular-value index'); ylabel('After nuisance projection');
    title('XY/yaw support after removing Z/roll/pitch/scale');
    legend('Ground only', 'Ground + walls', 'Location', 'best');
    subplot(2, 2, 4);
    shift = linspace(-2, 2, 101);
    groundCost = mean(groundN(:, 1).^2)*shift.^2;
    roomCost = mean(normal(:, 1).^2)*shift.^2;
    plot(shift, groundCost, 'LineWidth', 1.5); hold on;
    plot(shift, roomCost, 'LineWidth', 1.5); grid on;
    xlabel('Pure X translation (m)'); ylabel('Mean squared point-to-plane residual (m^2)');
    title('Sliding along a plane can leave residual unchanged');
    legend('Ground only', 'Ground + walls', 'Location', 'best');
    sgtitle('Synthetic local linearization: a small residual need not locate the vehicle');
end

function [jacobian, singular, horizontal] = geometry(rotated, normal, units)
    jacobian = [normal, cross(rotated, normal, 2), sum(normal.*rotated, 2)];
    scaled = jacobian.*units;
    singular = svd(scaled, 'econ');
    nuisance = scaled(:, [3 4 5 7]);
    [u, s, ~] = svd(nuisance, 'econ');
    s = diag(s);
    u = u(:, s > max(size(nuisance))*eps(max(s)));
    target = scaled(:, [1 2 6]);
    target = target-u*(u'*target);
    horizontal = svd(target, 'econ');
end
