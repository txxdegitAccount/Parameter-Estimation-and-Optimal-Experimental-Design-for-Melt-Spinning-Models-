clc; clear; close all;
rng(1);   % 固定随机种子，保证不同版本使用完全相同的 LHS 样本

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
% 标准 LHS + 手动补角点/边界点
% theta(:,1)=A_op, theta(:,2)=A_c
%% =========================
M_lhs = 10000;   % LHS 主体样本数
n_edge = 20;     % 每条边补点数

% ---- 标准 LHS 主体采样 ----
theta_samples = lhsdesign(M_lhs, 2);
theta_samples(:,1) = 1.0 + theta_samples(:,1) * (2.0 - 1.0);   % A_op in [1,2]
theta_samples(:,2) = 90  + theta_samples(:,2) * (120 - 90);    % A_c  in [90,120]

% ---- 四个角点 [A_op, A_c] ----
corner_pts = [1.0,  90;
              1.0, 120;
              2.0,  90;
              2.0, 120];

% ---- 四条边界点 ----
edge1 = [1.0 * ones(n_edge,1), linspace(90,120,n_edge)'];   % A_op = 1
edge2 = [2.0 * ones(n_edge,1), linspace(90,120,n_edge)'];   % A_op = 2
edge3 = [linspace(1,2,n_edge)', 90  * ones(n_edge,1)];      % A_c  = 90
edge4 = [linspace(1,2,n_edge)', 120 * ones(n_edge,1)];      % A_c  = 120

% ---- 合并并去重 ----
theta_samples = [theta_samples;
                 corner_pts;
                 edge1;
                 edge2;
                 edge3;
                 edge4];

theta_samples = unique(theta_samples, 'rows', 'stable');

M = size(theta_samples, 1);

disp('前5个采样点（A_op, A_c）：');
disp(theta_samples(1:5,:));
fprintf('LHS主体点数 = %d\n', M_lhs);
fprintf('补边界后总点数 M = %d\n', M);

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
% 3) 启动并行池 + 主循环（52核服务器推荐）
%% =========================
N_workers = 48;   % 52个物理核，先用48个worker，给系统/MATLAB主进程留4核

poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool('local', N_workers);
elseif poolobj.NumWorkers ~= N_workers
    delete(poolobj);
    parpool('local', N_workers);
end

tic;
parfor m_idx = 1:M
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
        sigma_all(m_idx,:) = s(:)';
        sigma_max_all(m_idx) = s(1);
        sigma_min_all(m_idx) = s(end);

        fullrank_all(m_idx) = (sigma_min_all(m_idx) > sigma_tol);
    end
end

elapsed_time = toc;
fprintf('\n并行主循环耗时 = %.2f min\n', elapsed_time/60);

% 先保存数值结果，避免后续作图异常导致长时间计算结果丢失
result_file = fullfile(pwd, 'sigma_results_M10000_consistent.mat');
save(result_file, 'theta_samples', 'F0_all', 'r_all', 'shoot_ok', ...
    'rp_all', 'rp_ok_all', 'J_all', 'rankJ_all', 'fullrank_all', ...
    'sigma_all', 'sigma_min_all', 'sigma_max_all', 'elapsed_time', '-v7.3');
fprintf('数值结果已保存：%s\n', result_file);

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

valid_rstar = isfinite(r_all);
if any(valid_rstar)
    fprintf('\n==== 边界残差 |r(F_0,theta)| 统计 ====\n');
    fprintf('|r|: median = %.3e\n', median(abs(r_all(valid_rstar))));
    fprintf('|r|: max    = %.3e\n', max(abs(r_all(valid_rstar))));
    fprintf('|r| > 1e-3 的点数 = %d\n', sum(abs(r_all(valid_rstar)) > 1e-3));
end

valid_rpstat = isfinite(rp_all);
if any(valid_rpstat)
    fprintf('\n==== |r_p| 统计 ====\n');
    fprintf('|r_p|: min    = %.3e\n', min(abs(rp_all(valid_rpstat))));
    fprintf('|r_p|: median = %.3e\n', median(abs(rp_all(valid_rpstat))));
end

%% =========================
% 5) 有效数据
%% =========================
valid_idx = shoot_ok & rp_ok_all & isfinite(sigma_min_all) & (sigma_min_all > 0);

A_op_plot   = theta_samples(valid_idx, 1);
A_c_plot    = theta_samples(valid_idx, 2);
sigma_plot  = sigma_min_all(valid_idx);
log_sigma   = log10(sigma_plot);

%% =========================
% 图片保存文件夹
%% =========================
fig_folder = fullfile(pwd, 'figures_M10000_consistent');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end

%% ========================= 
% 6) 图1：d_min(J) 散点图 
%% ========================= 
if any(valid_idx) 
    fig1 = figure('Color','w','Position',[80,80,860,650]); 
 
    scatter(A_c_plot, A_op_plot, 36, sigma_plot, 'filled', ... 
        'MarkerEdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.2); 
    box on; 
    grid on; 
 
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20); 
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20); 

    % d_min(J) 全部斜体
    title('\it d_{min}(J)', ...
        'Interpreter','tex', 'FontSize',18, 'FontWeight','normal'); 
 
    colormap(parula); 
    cb1 = colorbar; 
    cb1.Label.String = '\it d_{min}(J)'; 
    cb1.Label.Interpreter = 'tex'; 
    cb1.FontSize = 13; 
    cb1.LineWidth = 1.0; 
 
    ax = gca; 
    ax.FontSize = 15; 
    ax.LineWidth = 1.2; 
    ax.Box = 'on'; 
    ax.Layer = 'top'; 
    ax.TickDir = 'out'; 
 
    exportgraphics(fig1, fullfile(fig_folder, 'd_min_scatter.png'), 'Resolution', 600); 
    exportgraphics(fig1, fullfile(fig_folder, 'd_min_scatter.pdf'), 'ContentType', 'vector'); 
end 
 
%% ========================= 
% 7) 图2：log10(d_min(J)) 散点图 
%% ========================= 
if any(valid_idx) 
    fig2 = figure('Color','w','Position',[100,100,860,650]); 
 
    scatter(A_c_plot, A_op_plot, 36, log_sigma, 'filled', ... 
        'MarkerEdgeColor', [0.25 0.25 0.25], 'LineWidth', 0.2); 
    box on; 
    grid on; 
 
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20); 
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20); 

    % log 保持正体，仅 d_min(J) 斜体
    title('log_{10}(\it d_{min}(J)\rm)', ...
        'Interpreter','tex', 'FontSize',18, 'FontWeight','normal'); 
 
    colormap(parula); 
    cb2 = colorbar; 
    cb2.Label.String = 'log_{10}(\it d_{min}(J)\rm)'; 
    cb2.Label.Interpreter = 'tex'; 
    cb2.FontSize = 13; 
    cb2.LineWidth = 1.0; 
 
    ax = gca; 
    ax.FontSize = 15; 
    ax.LineWidth = 1.2; 
    ax.Box = 'on'; 
    ax.Layer = 'top'; 
    ax.TickDir = 'out'; 
 
    exportgraphics(fig2, fullfile(fig_folder, 'log_d_min_scatter.png'), 'Resolution', 600); 
    exportgraphics(fig2, fullfile(fig_folder, 'log_d_min_scatter.pdf'), 'ContentType', 'vector'); 
end 
 
%% ========================= 
% 8) 图3：d_min(J) 热图 
%% ========================= 
if any(valid_idx) && nnz(valid_idx) >= 8 
    Ac_grid  = linspace(min(theta_samples(:,2)), max(theta_samples(:,2)), 300); 
    Aop_grid = linspace(min(theta_samples(:,1)), max(theta_samples(:,1)), 300); 
    [ACG, AOPG] = meshgrid(Ac_grid, Aop_grid); 
 
    F_sigma = scatteredInterpolant(A_c_plot, A_op_plot, sigma_plot, ... 
                                   'natural', 'nearest'); 
    SIGMAG = F_sigma(ACG, AOPG); 
 
    fig3 = figure('Color','w','Position',[120,120,820,620]); 
 
    contourf(ACG, AOPG, SIGMAG, 50, 'LineStyle', 'none'); 
 
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20); 
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20); 

    title('\it d_{min}(J)', ...
        'Interpreter','tex', 'FontSize',18, 'FontWeight','normal'); 
 
    colormap(parula); 
    cb3 = colorbar; 
    cb3.Label.String = '\it d_{min}(J)'; 
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
 
    exportgraphics(fig3, fullfile(fig_folder, 'd_min_heatmap.png'), 'Resolution', 600); 
    exportgraphics(fig3, fullfile(fig_folder, 'd_min_heatmap.pdf'), ...
        'ContentType', 'image', 'Resolution', 600); 
end 
 
%% ========================= 
% 9) 图4：log10(d_min(J)) 热图 
%% ========================= 
if any(valid_idx) && nnz(valid_idx) >= 8 
    Ac_grid  = linspace(min(theta_samples(:,2)), max(theta_samples(:,2)), 300); 
    Aop_grid = linspace(min(theta_samples(:,1)), max(theta_samples(:,1)), 300); 
    [ACG, AOPG] = meshgrid(Ac_grid, Aop_grid); 
 
    F_logsigma = scatteredInterpolant(A_c_plot, A_op_plot, log_sigma, ... 
                                      'natural', 'nearest'); 
    LOGSIG = F_logsigma(ACG, AOPG); 
 
    fig4 = figure('Color','w','Position',[140,140,820,620]); 
 
    contourf(ACG, AOPG, LOGSIG, 50, 'LineStyle', 'none'); 
 
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20); 
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20); 

    title('log_{10}(\it d_{min}(J)\rm)', ...
        'Interpreter','tex', 'FontSize',18, 'FontWeight','normal'); 
 
    colormap(parula); 
    cb4 = colorbar; 
    cb4.Label.String = 'log_{10}(\it d_{min}(J)\rm)'; 
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
 
    exportgraphics(fig4, fullfile(fig_folder, 'log_d_min_heatmap.png'), 'Resolution', 600); 
    exportgraphics(fig4, fullfile(fig_folder, 'log_d_min_heatmap.pdf'), ...
        'ContentType', 'image', 'Resolution', 600); 
end 
 
%% ========================= 
% 10) 图5：d_min(J) 的 3D 散点图 
%% ========================= 
if any(valid_idx) 
    fig5 = figure('Color','w','Position',[160,160,880,680]); 
 
    scatter3(A_c_plot, A_op_plot, sigma_plot, ... 
        24, sigma_plot, 'filled', ... 
        'MarkerEdgeColor', [0.25 0.25 0.25], ... 
        'LineWidth', 0.15); 
    hold on; 
 
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20); 
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20); 

    zlabel('\it d_{min}(J)', ...
        'Interpreter','tex', 'FontSize',18); 
 
    % 前面的文字正体，仅 d_min(J) 斜体
    title('3D view of \it d_{min}(J)\rm', ...
        'Interpreter','tex', ...
        'FontSize',18, 'FontWeight','normal'); 
 
    colormap(parula); 
    cb5 = colorbar; 
    cb5.Label.String = '\it d_{min}(J)'; 
    cb5.Label.Interpreter = 'tex'; 
    cb5.FontSize = 13; 
    cb5.LineWidth = 1.0; 
 
    ax = gca; 
    ax.FontSize = 15; 
    ax.LineWidth = 1.2; 
    ax.Box = 'on'; 
    ax.Layer = 'top'; 
    ax.TickDir = 'out'; 
    grid on; 
    view(45, 28); 
 
    hold off; 
 
    exportgraphics(fig5, fullfile(fig_folder, 'd_min_3D_scatter.png'), 'Resolution', 600); 
    exportgraphics(fig5, fullfile(fig_folder, 'd_min_3D_scatter.pdf'), 'ContentType', 'vector'); 
end 
 
%% ========================= 
% 11) 图6：log10(d_min(J)) 的 3D 曲面图 
%% ========================= 
if any(valid_idx) && nnz(valid_idx) >= 8 
    Ac_grid  = linspace(min(theta_samples(:,2)), max(theta_samples(:,2)), 220); 
    Aop_grid = linspace(min(theta_samples(:,1)), max(theta_samples(:,1)), 220); 
    [ACG, AOPG] = meshgrid(Ac_grid, Aop_grid); 
 
    F_logsigma = scatteredInterpolant(A_c_plot, A_op_plot, log_sigma, ... 
                                      'natural', 'nearest'); 
    LOGSIG_3D = F_logsigma(ACG, AOPG); 
 
    fig6 = figure('Color','w','Position',[180,180,900,700]); 
 
    surf(ACG, AOPG, LOGSIG_3D, LOGSIG_3D, ... 
        'EdgeColor', 'none', 'FaceAlpha', 1.0); 
    hold on; 
 
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20); 
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20); 

    zlabel('log_{10}(\it d_{min}(J)\rm)', ...
        'Interpreter','tex', 'FontSize',18); 
 
    title('3D surface of log_{10}(\it d_{min}(J)\rm)', ... 
        'Interpreter','tex', 'FontSize',18, 'FontWeight','normal'); 
 
    colormap(parula); 
    cb6 = colorbar; 
    cb6.Label.String = 'log_{10}(\it d_{min}(J)\rm)'; 
    cb6.Label.Interpreter = 'tex'; 
    cb6.FontSize = 13; 
    cb6.LineWidth = 1.0; 
 
    ax = gca; 
    ax.FontSize = 15; 
    ax.LineWidth = 1.2; 
    ax.Box = 'on'; 
    ax.Layer = 'top'; 
    ax.TickDir = 'out'; 
    grid on; 
    view(48, 30); 
 
    hold off; 
 
    exportgraphics(fig6, fullfile(fig_folder, 'log_d_min_3D_surface.png'), 'Resolution', 600); 
    exportgraphics(fig6, fullfile(fig_folder, 'log_d_min_3D_surface.pdf'), ... 
        'ContentType', 'image', 'Resolution', 600); 
end

%% =========================
% 12) 诊断图：log10|r(F0,theta)|
%% =========================
valid_r_plot = isfinite(r_all);
if any(valid_r_plot)
    fig7 = figure('Color','w','Position',[200,200,860,650]);
    scatter(theta_samples(valid_r_plot,2), theta_samples(valid_r_plot,1), ...
        28, log10(abs(r_all(valid_r_plot)) + eps), 'filled');
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20);
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20);
    title('$\log_{10}|r(F_0,\theta)|$', 'Interpreter','latex', ...
        'FontSize',18, 'FontWeight','normal');
    colormap(parula);
    cb7 = colorbar;
    cb7.Label.String = '$\log_{10}|r(F_0,\theta)|$';
    cb7.Label.Interpreter = 'latex';
    cb7.FontSize = 13;
    ax = gca;
    ax.FontSize = 15;
    ax.LineWidth = 1.2;
    ax.Box = 'on';
    ax.TickDir = 'out';
    exportgraphics(fig7, fullfile(fig_folder, 'diagnostic_log_boundary_residual.png'), 'Resolution', 600);
end

%% =========================
% 13) 诊断图：log10|r_p|
%% =========================
valid_rp_plot = isfinite(rp_all);
if any(valid_rp_plot)
    fig8 = figure('Color','w','Position',[220,220,860,650]);
    scatter(theta_samples(valid_rp_plot,2), theta_samples(valid_rp_plot,1), ...
        28, log10(abs(rp_all(valid_rp_plot)) + eps), 'filled');
    xlabel('$A_c$', 'Interpreter','latex', 'FontSize',20);
    ylabel('$A_{op}$', 'Interpreter','latex', 'FontSize',20);
    title('$\log_{10}|r_p|$', 'Interpreter','latex', ...
        'FontSize',18, 'FontWeight','normal');
    colormap(parula);
    cb8 = colorbar;
    cb8.Label.String = '$\log_{10}|r_p|$';
    cb8.Label.Interpreter = 'latex';
    cb8.FontSize = 13;
    ax = gca;
    ax.FontSize = 15;
    ax.LineWidth = 1.2;
    ax.Box = 'on';
    ax.TickDir = 'out';
    exportgraphics(fig8, fullfile(fig_folder, 'diagnostic_log_rp.png'), 'Resolution', 600);
end

fprintf('\n图片已保存到文件夹：\n%s\n', fig_folder);
fprintf('图片文件：\n');
fprintf('  d_min_scatter.png\n');
fprintf('  d_min_scatter.pdf\n');
fprintf('  log_d_min_scatter.png\n');
fprintf('  log_d_min_scatter.pdf\n');

if any(valid_idx) && nnz(valid_idx) >= 8
    fprintf('  d_min_heatmap.png\n');
    fprintf('  d_min_heatmap.pdf\n');
    fprintf('  log_d_min_heatmap.png\n');
    fprintf('  log_d_min_heatmap.pdf\n');
end

if any(valid_idx)
    fprintf('  d_min_3D_scatter.png\n');
    fprintf('  d_min_3D_scatter.pdf\n');
end

if any(valid_idx) && nnz(valid_idx) >= 8
    fprintf('  log_d_min_3D_surface.png\n');
    fprintf('  log_d_min_3D_surface.pdf\n');
end

fprintf('  diagnostic_log_boundary_residual.png\n');
fprintf('  diagnostic_log_rp.png\n');

%% =========================================================
% 子函数1：高精度一致的 fmincon shooting
% 保留原 fmincon 方法，只统一 ode45 的 RelTol/AbsTol
%% =========================================================
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