%% Interior LHS sampling-refinement test for structural identifiability
% Requires:
%   1. sampling_refinement_results.mat  (created by boundary refinement)
%   2. steady_solution.m
% Place this script in the same folder.
%
% Existing 10304 points (10000 interior + 304 boundary) are reused.
% We append two independent random Latin-hypercube batches of 5000 points
% each to check 10000 -> 15000 -> 20000 interior points.
% No previously evaluated points are recomputed.
%
% IMPORTANT: a sampling refinement is numerical evidence, not a rigorous
% mathematical error bound over the full continuous parameter rectangle.

clc;
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

input_file = fullfile(script_dir, 'sampling_refinement_results.mat');
assert(isfile(input_file), ...
    'Missing sampling_refinement_results.mat. Run the boundary test first.');

S = load(input_file, 'theta_all', 'sigma_min_all', 'rank_all', 'ok_all');

theta_all = S.theta_all;
sigma_all = S.sigma_min_all(:);
rank_all = S.rank_all(:);
ok_all = logical(S.ok_all(:));

assert(size(theta_all, 2) == 2, 'Expected [A_op, A_c] sample pairs.');
assert(size(theta_all, 1) == 10304, ...
    'Expected 10304 precomputed points after edge refinement (got %d).', ...
    size(theta_all, 1));
assert(numel(sigma_all) == 10304 && numel(rank_all) == 10304 ...
       && numel(ok_all) == 10304, 'The saved arrays have inconsistent sizes.');

% Same model settings and rank tolerance as boundary refinement
W = 0.009;
T_0 = 298;
A_0 = pi*(0.22*0.1)^2 / 4;
T_a_abs = 15;
u_a_abs = 30;
rho_0 = 1.356 - 5e-4*T_0;
u_0 = W/(A_0*rho_0);
T_a = T_a_abs/T_0;
u_a = u_a_abs/u_0;
numPoints = 1001;
numeric_param = [W, T_0, A_0, T_a, u_a, numPoints-1, u_0];

u_target_single = 3730;
F_init = 270;
rank_tol = 1e-8;
sigma_tol = 1e-8;

Aop_lower = 1;
Aop_upper = 2;
Ac_lower = 90;
Ac_upper = 120;

boundary_mask = ...
    abs(theta_all(:,1)-Aop_lower)<1e-10 | ...
    abs(theta_all(:,1)-Aop_upper)<1e-10 | ...
    abs(theta_all(:,2)-Ac_lower)<1e-10 | ...
    abs(theta_all(:,2)-Ac_upper)<1e-10;

assert(sum(boundary_mask)==304, ...
    'Expected 304 boundary samples; found %d.', sum(boundary_mask));
assert(sum(~boundary_mask)==10000, ...
    'Expected 10000 interior samples; found %d.', sum(~boundary_mask));

interior_levels = [10000, 15000, 20000];
n_levels = numel(interior_levels);
edge_count = 304;
total_points = zeros(n_levels,1);
fullrank_count = zeros(n_levels,1);
failure_count = zeros(n_levels,1);
min_sigma_all = nan(n_levels,1);
min_sigma_interior = nan(n_levels,1);
min_sigma_boundary = nan(n_levels,1);
min_sigma_new = nan(n_levels,1);
new_point_count = zeros(n_levels,1);

% Report existing numerical results at 10000 interior + 304 boundary points
total_points(1) = numel(sigma_all);
fullrank_count(1) = sum(ok_all);
failure_count(1) = total_points(1)-fullrank_count(1);
min_sigma_all(1) = min(sigma_all(ok_all));
min_sigma_interior(1) = min(sigma_all(ok_all & ~boundary_mask));
min_sigma_boundary(1) = min(sigma_all(ok_all & boundary_mask));

fprintf('\nBaseline: %d interior + %d boundary = %d total\n', ...
    interior_levels(1), edge_count, total_points(1));
fprintf('  full rank: %d/%d; overall min singular value: %.9e\n', ...
    fullrank_count(1), total_points(1), min_sigma_all(1));
fprintf('  interior minimum: %.9e; boundary minimum: %.9e\n', ...
    min_sigma_interior(1), min_sigma_boundary(1));

% Use a fixed seed for reproducible INDEPENDENT supplementary LHS blocks.
% The UNION of blocks is not itself a single N-point Latin hypercube.
rng(20261023, 'twister');

for level = 2:n_levels
    N_new = interior_levels(level)-interior_levels(level-1);
    new_point_count(level) = N_new;

    % Elementary randomized Latin hypercube sampling (toolbox-free)
    U = zeros(N_new,2);
    for j = 1:2
        U(:,j) = (randperm(N_new)' - rand(N_new,1))/N_new;
    end

    new_points = [Aop_lower + U(:,1)*(Aop_upper-Aop_lower), ...
                  Ac_lower + U(:,2)*(Ac_upper-Ac_lower)];

    sigma_new = nan(N_new,1);
    rank_new = nan(N_new,1);
    ok_new = false(N_new,1);
    failure_messages = cell(N_new,1);

    fprintf('\nAdding %d new interior LHS points (interior total = %d)...\n', ...
        N_new, interior_levels(level));

    parfor k = 1:N_new
        try
            [~, ~, shoot_ok, ~, rp_ok, J] = compute_local_jacobian( ...
                new_points(k,:), numeric_param, u_target_single, F_init);
            if shoot_ok && rp_ok && all(isfinite(J), 'all')
                vals = svd(J);
                sigma_new(k) = vals(end);
                rank_new(k) = rank(J, rank_tol);
                ok_new(k) = (rank_new(k)==2) && (sigma_new(k)>sigma_tol);
                if ~ok_new(k)
                    failure_messages{k} = 'Rank or singular-value threshold failed.';
                end
            else
                failure_messages{k} = 'Shooting, boundary, or sensitivity check failed.';
            end
        catch ME
            failure_messages{k} = ME.message;
        end
    end

    % Each new point is inside the parameter rectangle
    theta_all = [theta_all; new_points];
    sigma_all = [sigma_all; sigma_new];
    rank_all = [rank_all; rank_new];
    ok_all = [ok_all; ok_new];
    boundary_mask = [boundary_mask; false(N_new,1)];

    total_points(level) = numel(sigma_all);
    fullrank_count(level) = sum(ok_all);
    failure_count(level) = total_points(level)-fullrank_count(level);

    if any(ok_all)
        min_sigma_all(level) = min(sigma_all(ok_all));
    end
    if any(ok_all & ~boundary_mask)
        min_sigma_interior(level) = min(sigma_all(ok_all & ~boundary_mask));
    end
    if any(ok_all & boundary_mask)
        min_sigma_boundary(level) = min(sigma_all(ok_all & boundary_mask));
    end
    if any(ok_new)
        min_sigma_new(level) = min(sigma_new(ok_new));
    end

    fprintf('N=%d, full rank=%d, failures=%d\n', ...
        total_points(level), fullrank_count(level), failure_count(level));
    fprintf('  min(new interior)=%.9e, min(all interior)=%.9e, min(all)=%.9e\n', ...
        min_sigma_new(level), min_sigma_interior(level), min_sigma_all(level));

    bad = find(~ok_new, 3);
    for i_bad=1:numel(bad)
        fprintf('  Failure at added point %d: %s\n', ...
            bad(i_bad), failure_messages{bad(i_bad)});
    end

    % Save a checkpoint after EACH batch, including all results already computed
    summary = table(interior_levels(:), repmat(edge_count,n_levels,1), ...
        total_points, new_point_count, fullrank_count, failure_count, ...
        min_sigma_new, min_sigma_interior, min_sigma_boundary, min_sigma_all, ...
        'VariableNames', {'InteriorPoints', 'BoundaryPoints', 'TotalPoints', ...
        'NewInteriorPoints', 'FullRankCount', 'FailedCount', ...
        'NewInteriorMinSV', 'InteriorMinSV', 'BoundaryMinSV', 'OverallMinSV'});

    summary_file = fullfile(script_dir, 'sampling_refinement_interior_summary.csv');
    results_file = fullfile(script_dir, 'sampling_refinement_interior_results.mat');
    writetable(summary, summary_file);
    save(results_file, 'summary', 'theta_all', 'sigma_all', 'rank_all', ...
        'ok_all', 'boundary_mask', '-v7.3');
end

fprintf('\n=== Sampling refinement summary ===\n');
disp(summary);
fprintf('Saved summary: %s\nSaved results: %s\n', ...
    summary_file, results_file);

%% The same shooting and reduced-Jacobian functions used in the previous test

function [F_0, r_star, has_F0] = get_force_new(numeric_param, truth_value, u_target, F_init)

    try
        F_0 = get_force_consistent(numeric_param, truth_value, u_target, F_init);
        F_0 = F_0(1);

        % 使用与灵敏度方程相同的高精度再次检查边界残差
        r_star = boundary_residual(F_0, numeric_param, truth_value, u_target);

        % 原非线性约束为 r^2 <= 1e-6，因此等价于 |r| <= 1e-3
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
