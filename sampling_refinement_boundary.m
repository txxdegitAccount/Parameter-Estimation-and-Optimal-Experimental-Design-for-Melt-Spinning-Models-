clc;
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

base_file = fullfile(script_dir, 'sigma_results_M10000_consistent.mat');
assert(isfile(base_file), 'Missing baseline MAT file: %s', base_file);

S = load(base_file, 'theta_samples', 'sigma_min_all', ...
    'rankJ_all', 'fullrank_all', 'shoot_ok', 'rp_ok_all');

theta_all = S.theta_samples;                 % [A_op, A_c]
sigma_min_all = S.sigma_min_all(:);
rank_all = S.rankJ_all(:);
ok_all = logical(S.fullrank_all(:)) & ...
         logical(S.shoot_ok(:)) & logical(S.rp_ok_all(:)) & ...
         isfinite(sigma_min_all) & (rank_all == 2);

assert(size(theta_all,1) == 10076, ...
    'Expected 10076 saved sampling points; found %d.', size(theta_all,1));

W = 0.009;
T_0 = 298;
A_0 = pi*(0.22*0.1)^2 / 4;
T_a_abs = 15;
u_a_abs = 30;
rho_0 = 1.356 - 5e-4 * T_0;
u_0 = W / (A_0 * rho_0);
T_a = T_a_abs / T_0;
u_a = u_a_abs / u_0;
numPoints = 1001;
numeric_param = [W, T_0, A_0, T_a, u_a, numPoints-1, u_0];
u_target_single = 3730;
F_init = 270;
rank_tol = 1e-8;
sigma_tol = 1e-8;

levels = [20, 39, 77];     
n_levels = numel(levels);
count_all = zeros(n_levels,1);
rank2_all = zeros(n_levels,1);
min_sv_all = nan(n_levels,1);
failure_all = zeros(n_levels,1);
new_point_count = zeros(n_levels,1);

count_all(1) = size(theta_all, 1);
rank2_all(1) = sum(ok_all);
failure_all(1) = count_all(1) - rank2_all(1);
if any(ok_all)
    min_sv_all(1) = min(sigma_min_all(ok_all));
end

fprintf('\nBaseline: N=%d, full-rank=%d, min singular value=%.9e\n', ...
    count_all(1), rank2_all(1), min_sv_all(1));

theta_added = zeros(0,2);
sigma_added = zeros(0,1);
rank_added = zeros(0,1);
ok_added = false(0,1);

for i_level = 2:n_levels
    n_edge = levels(i_level);
    boundary_samples = generate_boundary_samples(n_edge);

    is_existing = ismembertol(boundary_samples, theta_all, ...
        1e-10, 'ByRows', true);
    new_points = boundary_samples(~is_existing,:);
    N_new = size(new_points, 1);
    new_point_count(i_level) = N_new;

    fprintf('\nn_edge=%d: %d genuinely new boundary points.\n', ...
        n_edge, N_new);

    sigma_new = nan(N_new,1);
    rank_new = nan(N_new,1);
    ok_new = false(N_new,1);

    parfor k = 1:N_new
        try
            [~, ~, shoot_ok, ~, rp_ok, J] = compute_local_jacobian( ...
                new_points(k,:), numeric_param, u_target_single, F_init);
            if shoot_ok && rp_ok && all(isfinite(J), 'all')
                singular_vals = svd(J);
                sigma_new(k) = singular_vals(end);
                rank_new(k) = rank(J, rank_tol);
                ok_new(k) = (rank_new(k) == 2) && ...
                            (sigma_new(k) > sigma_tol);
            end
        catch
        end
    end

    theta_all = [theta_all; new_points];
    sigma_min_all = [sigma_min_all; sigma_new];
    rank_all = [rank_all; rank_new];
    ok_all = [ok_all; ok_new];

    theta_added = [theta_added; new_points];
    sigma_added = [sigma_added; sigma_new];
    rank_added = [rank_added; rank_new];
    ok_added = [ok_added; ok_new];

    count_all(i_level) = size(theta_all,1);
    rank2_all(i_level) = sum(ok_all);
    failure_all(i_level) = count_all(i_level) - rank2_all(i_level);

    if any(ok_all)
        min_sv_all(i_level) = min(sigma_min_all(ok_all));
    end

    expected = 10000 + 4*n_edge - 4;
    assert(count_all(i_level) == expected, ...
        'Unexpected N=%d; expected %d. Check point matching.', ...
        count_all(i_level), expected);

    fprintf('N=%d, verified full rank=%d, failures=%d, min sigma=%.9e\n', ...
        count_all(i_level), rank2_all(i_level), ...
        failure_all(i_level), min_sv_all(i_level));
end

summary = table(levels(:), count_all, new_point_count, ...
    rank2_all, failure_all, min_sv_all, ...
    'VariableNames', {'PointsPerEdge', 'TotalPoints', ...
    'NewlyCalculated', 'FullRankCount', 'FailedCount', ...
    'MinimumSingularValue'});
disp(summary);

csv_file = fullfile(script_dir, 'sampling_refinement_summary.csv');
mat_file = fullfile(script_dir, 'sampling_refinement_results.mat');
writetable(summary, csv_file);
save(mat_file, 'summary', 'theta_added', 'sigma_added', ...
    'rank_added', 'ok_added', 'theta_all', 'sigma_min_all', ...
    'rank_all', 'ok_all', '-v7.3');

fprintf('\nSaved summary: %s\nSaved values: %s\n', csv_file, mat_file);

function boundary_samples = generate_boundary_samples(n_edge)
    x = linspace(1, 2, n_edge)';     % A_op
    y = linspace(90, 120, n_edge)';  % A_c
    edge1 = [ones(n_edge,1), y];
    edge2 = [2*ones(n_edge,1), y];
    edge3 = [x, 90*ones(n_edge,1)];
    edge4 = [x, 120*ones(n_edge,1)];
    boundary_samples = unique([edge1; edge2; edge3; edge4], ...
        'rows', 'stable');
end

function [F_0, r_star, has_F0] = get_force_new(numeric_param, truth_value, u_target, F_init)

    try
        F_0 = get_force_consistent(numeric_param, truth_value, u_target, F_init);
        F_0 = F_0(1);
        r_star = boundary_residual(F_0, numeric_param, truth_value, u_target);
        res_tol = 1e-3;
        has_F0 = isfinite(F_0) && isfinite(r_star) && abs(r_star) <= res_tol;

    catch
        F_0 = nan;
        r_star = nan;
        has_F0 = false;
    end
end

%% =========================================================
% 子函数1.1：保留原始 fmincon，只将内部 ode45 精度统一为 1e-8/1e-9
%% =========================================================
function F_0 = get_force_consistent(numeric_param, truth_value, u_target, F_init)

    options = optimoptions('fmincon', ...
        'MaxIterations', 6000, ...
        'Display', 'off', ...
        'UseParallel', false);

    objective = @(F) obj_function_consistent( ...
        u_target, numeric_param, truth_value, F);

    num_targets = length(u_target);
    F_init_vector = repmat(F_init, num_targets, 1);

    [F_0, ~] = fmincon(objective, F_init_vector, ...
        [], [], [], [], [], [], ...
        @(x) nonlcon_consistent(x, u_target, 1e-6, ...
        numeric_param, truth_value), options);
end

%% =========================================================
% 子函数1.2：目标函数
%% =========================================================
function err = obj_function_consistent(u_target, numeric_param, param_value, F)

    m   = numeric_param(6);
    u_0 = numeric_param(7);
    z   = linspace(0, 1, m+1);

    u_list = zeros(size(u_target));

    % 与后续 boundary_residual / sensitivity ODE 使用完全相同的精度
    odeOptions = odeset('RelTol',1e-8,'AbsTol',1e-9);

    for index = 1:length(u_target)
        [~, model_datas] = ode45(@steady_solution, z, ...
            [1 1 1 F(index) 0 0], odeOptions, ...
            numeric_param, [param_value(1), param_value(2)]);

        u_list(index) = model_datas(end,2) * u_0 * 60 / 100;
    end

    err = sum((u_list - u_target).^2);
end

%% =========================================================
% 子函数1.3：非线性约束
%% =========================================================
function [c, ceq] = nonlcon_consistent(x, u_target, tolerance, numeric_param, param_value)

    m   = numeric_param(6);
    u_0 = numeric_param(7);
    z   = linspace(0, 1, m+1);

    u_list = zeros(size(u_target));

    % 与目标函数、边界残差、灵敏度方程使用相同精度
    odeOptions = odeset('RelTol',1e-8,'AbsTol',1e-9);

    for index = 1:length(u_target)
        [~, model_datas] = ode45(@steady_solution, z, ...
            [1 1 1 x(index) 0 0], odeOptions, ...
            numeric_param, [param_value(1), param_value(2)]);

        u_list(index) = model_datas(end,2) * u_0 * 60 / 100;
    end

    err = sum((u_list - u_target).^2);
    c   = err - tolerance;
    ceq = [];
end

%% =========================================================
% 子函数2：边界残差
%% =========================================================
function r = boundary_residual(F, numeric_param, truth_value, u_target)

    m   = numeric_param(6);
    u_0 = numeric_param(7);

    z = linspace(0, 1, m+1);
    odeOptions = odeset('RelTol',1e-8,'AbsTol',1e-9);

    [~, model_datas] = ode45(@steady_solution, z, [1 1 1 F 0 0], ...
        odeOptions, numeric_param, [truth_value(1), truth_value(2)]);

    u_end = model_datas(end,2) * u_0 * 60 / 100;
    r = u_end - u_target;
end

%% =========================================================
% 子函数3：计算局部灵敏度矩阵 J
%% =========================================================
function [F_0, r_star, has_F0, rp, rp_ok, J] = compute_local_jacobian( ...
    param_value, numeric_param, u_target_single, F_init)

    [F_0, r_star, has_F0] = get_force_new(numeric_param, param_value, u_target_single, F_init);

    if ~has_F0 || ~isfinite(F_0)
        rp = NaN;
        rp_ok = false;
        J = NaN(3,2);
        return;
    end

    numPoints = numeric_param(6) + 1;
    z_span = linspace(0, 1, numPoints);
    ode_opts = odeset('AbsTol', 1e-9, 'RelTol', 1e-8);

    function dydz = augmented_ode(~, y)
        x        = y(1:6);
        s_theta1 = y(7:12);
        s_theta2 = y(13:18);
        s_F_vec  = y(19:24);

        f = steady_solution([], x, numeric_param, param_value);

        eps_x = 1e-8;
        A = zeros(6,6);
        for i = 1:6
            x_plus  = x; x_plus(i)  = x_plus(i)  + eps_x;
            x_minus = x; x_minus(i) = x_minus(i) - eps_x;

            f_plus  = steady_solution([], x_plus,  numeric_param, param_value);
            f_minus = steady_solution([], x_minus, numeric_param, param_value);

            A(:,i) = (f_plus - f_minus) / (2*eps_x);
        end

        eps_theta = 1e-7 * (abs(param_value) + 1e-10);
        b = zeros(6,2);

        for j = 1:2
            theta_plus  = param_value;
            theta_minus = param_value;

            epsj = eps_theta(j);
            if epsj == 0
                epsj = 1e-7;
            end

            theta_plus(j)  = theta_plus(j)  + epsj;
            theta_minus(j) = theta_minus(j) - epsj;

            f_plus  = steady_solution([], x, numeric_param, theta_plus);
            f_minus = steady_solution([], x, numeric_param, theta_minus);

            b(:,j) = (f_plus - f_minus) / (2*epsj);
        end

        dydz = [f; ...
                A * s_theta1 + b(:,1); ...
                A * s_theta2 + b(:,2); ...
                A * s_F_vec];
    end

    init_x = [1; 1; 1; F_0; 0; 0];

    aug_init = [init_x; ...
                zeros(6,1); ...
                zeros(6,1); ...
                [0;0;0;1;0;0]];

    [~, aug_datas] = ode45(@augmented_ode, z_span, aug_init, ode_opts);
    aug_end = aug_datas(end, :)';

    s_theta1_end = aug_end(7:12);
    s_theta2_end = aug_end(13:18);
    s_F_end      = aug_end(19:24);

    rp = s_F_end(2);

    tol_rp = 1e-8;
    rp_ok = abs(rp) > tol_rp;

    if ~rp_ok
        J = NaN(3,2);
        return;
    end

    dF_dtheta1 = - s_theta1_end(2) / rp;
    dF_dtheta2 = - s_theta2_end(2) / rp;

    total_s1 = s_theta1_end + s_F_end * dF_dtheta1;
    total_s2 = s_theta2_end + s_F_end * dF_dtheta2;

    J = [total_s1(1), total_s2(1); ...
         total_s1(5), total_s2(5); ...
         total_s1(6), total_s2(6)];
end
