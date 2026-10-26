clc; clear; close all;
% rng(1);

%% =========================
% 0) 基本参数
%% =========================
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

%% =========================
% 1) 参数采样
% theta(:,1)=A_op, theta(:,2)=A_c
%% =========================
M = 1000;

theta_samples = lhsdesign(M, 2);
theta_samples(:,1) = 1.0 + theta_samples(:,1) * (2.0 - 1.0);   % A_op in [1,2]
theta_samples(:,2) = 90  + theta_samples(:,2) * (120 - 90);    % A_c  in [90,120]

disp('前5个采样点（A_op, A_c）：');
disp(theta_samples(1:5,:));

%% =========================
% 2) 结果存储
%% =========================
F0_all        = nan(M,1);
r_all         = nan(M,1);
shoot_ok      = false(M,1);

rp_all        = nan(M,1);
rp_ok_all     = false(M,1);

J_all         = cell(M,1);
rankJ_all     = nan(M,1);
fullrank_all  = false(M,1);

sigma_all        = nan(M,2);
sigma_min_all    = nan(M,1);
sigma_max_all    = nan(M,1);

rank_tol  = 1e-8;
sigma_tol = 1e-8;

%% =========================
% 3) 主循环
%% =========================
for m_idx = 1:M
    theta_m = theta_samples(m_idx,:);

    [F_0, r_star, has_F0, rp, rp_ok, J_m] = compute_local_jacobian( ...
        theta_m, numeric_param, u_target_single, F_init);

    F0_all(m_idx)    = F_0;
    r_all(m_idx)     = r_star;
    shoot_ok(m_idx)  = has_F0;

    rp_all(m_idx)    = rp;
    rp_ok_all(m_idx) = rp_ok;

    J_all{m_idx} = J_m;

    if all(isfinite(J_m), 'all')
        rankJ_all(m_idx) = rank(J_m, rank_tol);

        s = svd(J_m);
        sigma_all(m_idx,1:length(s)) = s(:)';
        sigma_max_all(m_idx) = s(1);
        sigma_min_all(m_idx) = s(end);

        fullrank_all(m_idx) = (sigma_min_all(m_idx) > sigma_tol);
    end
end

%% =========================
% 4) 统计信息
%% =========================
fprintf('总采样点数 M = %d\n', M);
fprintf('成功拿到 F0 的点数 = %d\n', sum(shoot_ok));
fprintf('r_p 检验通过点数 = %d\n', sum(rp_ok_all));
fprintf('J 满列秩(rank=2)点数 = %d\n', sum(fullrank_all));

valid_sigma = isfinite(sigma_min_all) & sigma_min_all > 0;
if any(valid_sigma)
    fprintf('\n==== 最小奇异值统计 ====\n');
    fprintf('sigma_min: min    = %.3e\n', min(sigma_min_all(valid_sigma)));
    fprintf('sigma_min: median = %.3e\n', median(sigma_min_all(valid_sigma)));
    fprintf('sigma_min: max    = %.3e\n', max(sigma_min_all(valid_sigma)));
end

%% =========================
% 5) 有效数据
%% =========================
valid_idx = isfinite(sigma_min_all) & (sigma_min_all > 0);

A_op_plot   = theta_samples(valid_idx, 1);
A_c_plot    = theta_samples(valid_idx, 2);
sigma_plot  = sigma_min_all(valid_idx);
log_sigma   = log10(sigma_plot);

%% =========================
% 6) 图1：sigma_min(J) 散点图
%% =========================
if any(valid_idx)
    fig1 = figure('Color','w','Position',[80,80,860,650]);

    scatter(A_c_plot, A_op_plot, 95, sigma_plot, 'filled', ...
        'MarkerEdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.4);
    box on;
    grid on;

    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20);
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20);
    title('\sigma_{min}(J)', 'Interpreter','tex', 'FontSize',18, 'FontWeight','normal');

    colormap(parula);
    cb1 = colorbar;
    cb1.Label.String = '\sigma_{min}(J)';
    cb1.Label.Interpreter = 'tex';
    cb1.FontSize = 13;
    cb1.LineWidth = 1.0;

    ax = gca;
    ax.FontSize = 15;
    ax.LineWidth = 1.2;
    ax.Box = 'on';
    ax.Layer = 'top';
    ax.TickDir = 'out';

    exportgraphics(fig1, 'sigma_min_scatter.png', 'Resolution', 600);
    exportgraphics(fig1, 'sigma_min_scatter.pdf', 'ContentType', 'vector');
end

%% =========================
% 7) 图2：log10(sigma_min(J)) 散点图
%% =========================
if any(valid_idx)
    fig2 = figure('Color','w','Position',[100,100,860,650]);

    scatter(A_c_plot, A_op_plot, 95, log_sigma, 'filled', ...
        'MarkerEdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.4);
    box on;
    grid on;

    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20);
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20);
    title('log_{10}(\sigma_{min}(J))', 'Interpreter','tex', 'FontSize',18, 'FontWeight','normal');

    colormap(parula);
    cb2 = colorbar;
    cb2.Label.String = 'log_{10}(\sigma_{min}(J))';
    cb2.Label.Interpreter = 'tex';
    cb2.FontSize = 13;
    cb2.LineWidth = 1.0;

    ax = gca;
    ax.FontSize = 15;
    ax.LineWidth = 1.2;
    ax.Box = 'on';
    ax.Layer = 'top';
    ax.TickDir = 'out';

    exportgraphics(fig2, 'log_sigma_min_scatter.png', 'Resolution', 600);
    exportgraphics(fig2, 'log_sigma_min_scatter.pdf', 'ContentType', 'vector');
end

%% =========================
% 8) 图3：sigma_min(J) 插值热图
%% =========================
if any(valid_idx) && nnz(valid_idx) >= 8
    Ac_grid  = linspace(min(theta_samples(:,2)), max(theta_samples(:,2)), 300);
    Aop_grid = linspace(min(theta_samples(:,1)), max(theta_samples(:,1)), 300);
    [ACG, AOPG] = meshgrid(Ac_grid, Aop_grid);

    SIGMAG = griddata(A_c_plot, A_op_plot, sigma_plot, ACG, AOPG, 'natural');

    fig3 = figure('Color','w','Position',[120,120,820,620]);

    contourf(ACG, AOPG, SIGMAG, 24, 'LineStyle', 'none');
    hold on;

    % [~,hc1] = contour(ACG, AOPG, SIGMAG, 8);
    % hc1.LineColor = [0.35 0.35 0.35];
    % hc1.LineWidth = 0.5;

    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20);
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20);
    title('\sigma_{min}(J)', 'Interpreter','tex', 'FontSize',18, 'FontWeight','normal');

    colormap(parula);
    cb3 = colorbar;
    cb3.Label.String = '\sigma_{min}(J)';
    cb3.Label.Interpreter = 'tex';
    cb3.FontSize = 13;
    cb3.LineWidth = 1.0;

    ax = gca;
    ax.FontSize = 15;
    ax.LineWidth = 1.2;
    ax.Box = 'on';
    ax.Layer = 'top';
    ax.TickDir = 'out';
    grid off;

    xlim([min(theta_samples(:,2)), max(theta_samples(:,2))]);
    ylim([min(theta_samples(:,1)), max(theta_samples(:,1))]);

    hold off;

    exportgraphics(fig3, 'sigma_min_heatmap.png', 'Resolution', 600);
    exportgraphics(fig3, 'sigma_min_heatmap.pdf', 'ContentType', 'vector');
end

%% =========================
% 9) 图4：log10(sigma_min(J)) 插值热图
%% =========================
if any(valid_idx) && nnz(valid_idx) >= 8
    Ac_grid  = linspace(min(theta_samples(:,2)), max(theta_samples(:,2)), 300);
    Aop_grid = linspace(min(theta_samples(:,1)), max(theta_samples(:,1)), 300);
    [ACG, AOPG] = meshgrid(Ac_grid, Aop_grid);

    LOGSIG = griddata(A_c_plot, A_op_plot, log_sigma, ACG, AOPG, 'natural');

    fig4 = figure('Color','w','Position',[140,140,820,620]);

    contourf(ACG, AOPG, LOGSIG, 24, 'LineStyle', 'none');
    hold on;

    % [~,hc2] = contour(ACG, AOPG, LOGSIG, 8);
    % hc2.LineColor = [0.35 0.35 0.35];
    % hc2.LineWidth = 0.5;

    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20);
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20);
    title('log_{10}(\sigma_{min}(J))', 'Interpreter','tex', 'FontSize',18, 'FontWeight','normal');

    colormap(parula);
    cb4 = colorbar;
    cb4.Label.String = 'log_{10}(\sigma_{min}(J))';
    cb4.Label.Interpreter = 'tex';
    cb4.FontSize = 13;
    cb4.LineWidth = 1.0;

    ax = gca;
    ax.FontSize = 15;
    ax.LineWidth = 1.2;
    ax.Box = 'on';
    ax.Layer = 'top';
    ax.TickDir = 'out';
    grid off;

    xlim([min(theta_samples(:,2)), max(theta_samples(:,2))]);
    ylim([min(theta_samples(:,1)), max(theta_samples(:,1))]);

    hold off;

    exportgraphics(fig4, 'log_sigma_min_heatmap.png', 'Resolution', 600);
    exportgraphics(fig4, 'log_sigma_min_heatmap.pdf', 'ContentType', 'vector');
end

fprintf('\n图片已保存：\n');
fprintf('  sigma_min_scatter.png\n');
fprintf('  sigma_min_scatter.pdf\n');
fprintf('  log_sigma_min_scatter.png\n');
fprintf('  log_sigma_min_scatter.pdf\n');
if any(valid_idx) && nnz(valid_idx) >= 8
    fprintf('  sigma_min_heatmap.png\n');
    fprintf('  sigma_min_heatmap.pdf\n');
    fprintf('  log_sigma_min_heatmap.png\n');
    fprintf('  log_sigma_min_heatmap.pdf\n');
end

%% =========================================================
% 子函数1：调用你现有的 get_force
%% =========================================================
function [F_0, r_star, has_F0] = get_force_new(numeric_param, truth_value, u_target, F_init)

    try
        F_0 = get_force(numeric_param, truth_value, u_target, F_init);
        F_0 = F_0(1);

        r_star = boundary_residual(F_0, numeric_param, truth_value, u_target);
        has_F0 = isfinite(F_0);

    catch
        F_0 = nan;
        r_star = nan;
        has_F0 = false;
    end
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