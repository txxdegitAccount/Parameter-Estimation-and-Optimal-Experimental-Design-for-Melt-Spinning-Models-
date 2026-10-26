clc; clear; close all;
rng(1);   % 固定 LHS，便于重复比较

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

% 保留该字段仅用于兼容原 numeric_param 的结构。
% 优化后的 ODE 只需要终点，因此不再强制输出 1001 个 z 点。
numPoints = 1001;
numeric_param = [W, T_0, A_0, T_a, u_a, numPoints-1, u_0];

u_target_single = 3730;
F_init = 270;

%% =========================
% 并行设置
%% =========================
% 当前电脑使用 8 个 workers
N_workers = 8;

poolobj = gcp('nocreate');
if isempty(poolobj)
    if isempty(N_workers)
        poolobj = parpool;
    else
        poolobj = parpool(N_workers);
    end
end
fprintf('Parallel pool workers = %d\n', poolobj.NumWorkers);

%% =========================
% 1) 参数采样
% 标准 LHS + 手动补角点/边界点
% theta(:,1)=A_op, theta(:,2)=A_c
%% =========================
M_lhs = 10000;   % 最终 LHS 主体样本数
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

rankJ_all     = nan(M,1);
fullrank_all  = false(M,1);

sigma_min_all = nan(M,1);
sigma_max_all = nan(M,1);

rank_tol  = 1e-8;
sigma_tol = 1e-8;

%% =========================
% 3) 主循环：不同 (A_op,A_c) 采样点并行
%% =========================
fprintf('\n开始并行计算 %d 个采样点 ...\n', M);
t_main = tic;

parfor m_idx = 1:M

    theta_m = theta_samples(m_idx,:);

    try
        [F_0, r_star, has_F0, rp, rp_ok, J_m] = compute_local_jacobian_fast( ...
            theta_m, numeric_param, u_target_single, F_init);

        F0_all(m_idx)    = F_0;
        r_all(m_idx)     = r_star;
        shoot_ok(m_idx)  = has_F0;

        rp_all(m_idx)    = rp;
        rp_ok_all(m_idx) = rp_ok;

        if all(isfinite(J_m), 'all')
            % 只做一次 SVD；原代码 rank() + svd() 相当于重复分解。
            s = svd(J_m, 'econ');

            sigma_max_all(m_idx) = s(1);
            sigma_min_all(m_idx) = s(end);

            rankJ_all(m_idx) = sum(s > rank_tol);
            fullrank_all(m_idx) = (sigma_min_all(m_idx) > sigma_tol);
        end

    catch
        % 单个采样点失败时保留 NaN，不让整个 10000 点任务中断。
        shoot_ok(m_idx)  = false;
        rp_ok_all(m_idx) = false;
    end
end

elapsed_main = toc(t_main);
fprintf('并行主循环完成，用时 %.2f min (%.2f h)\n', ...
    elapsed_main/60, elapsed_main/3600);

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

% 保存数值结果；绘图失败也不会丢掉长时间计算结果
save('sigma_results_M10000_fast.mat', ...
    'theta_samples', 'F0_all', 'r_all', 'shoot_ok', ...
    'rp_all', 'rp_ok_all', 'rankJ_all', 'fullrank_all', ...
    'sigma_min_all', 'sigma_max_all', ...
    'rank_tol', 'sigma_tol', 'M_lhs', 'n_edge', ...
    'numeric_param', 'u_target_single', 'F_init');

%% =========================
% 5) 有效数据
%% =========================
valid_idx = isfinite(sigma_min_all) & (sigma_min_all > 0);

A_op_plot   = theta_samples(valid_idx, 1);
A_c_plot    = theta_samples(valid_idx, 2);
sigma_plot  = sigma_min_all(valid_idx);
log_sigma   = log10(sigma_plot);


%% =========================
% 图片输出文件夹
%% =========================
fig_folder = fullfile(pwd, 'figures_M10000');
if ~exist(fig_folder, 'dir')
    mkdir(fig_folder);
end
fprintf('\n图片输出文件夹：%s\n', fig_folder);

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

fprintf('\n图片已保存到文件夹：%s\n', fig_folder);
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


%% =========================================================
% 子函数1：快速求解 F_0
% 当前问题中 F 为标量，u_target_single 也为标量。
% 原 get_force 用 fmincon 同时最小化残差平方并施加同一个残差约束，
% 会重复调用 ODE。这里直接把 shooting residual = 0 作为一维根问题。
%% =========================================================
function [F_0, r_star, has_F0] = solve_force_fast( ...
    numeric_param, truth_value, u_target, F_init)

    residual_tol = 1e-3;  % 对应原约束 (u_end-u_target)^2 <= 1e-6

    % 与后续灵敏度积分保持较高精度；Refine=1 减少无用输出。
    ode_opts_force = odeset('RelTol',1e-8, ...
                            'AbsTol',1e-9, ...
                            'Refine',1);

    residual_fun = @(F) terminal_residual_fast( ...
        F, numeric_param, truth_value, u_target, ode_opts_force);

    F_0 = NaN;
    r_star = NaN;
    has_F0 = false;

    % 先检查初值本身
    r0 = residual_fun(F_init);
    if isfinite(r0) && abs(r0) <= residual_tol
        F_0 = F_init;
        r_star = r0;
        has_F0 = true;
        return;
    end

    % 以 F_init 为中心逐步扩展正值区间，寻找符号变化。
    expand = 1.25;
    bracket_found = false;
    F_lo = NaN; F_hi = NaN;

    for k = 1:10
        Flo_try = max(F_init / expand, 1e-8);
        Fhi_try = F_init * expand;

        rlo = residual_fun(Flo_try);
        rhi = residual_fun(Fhi_try);

        if isfinite(rlo) && isfinite(r0) && rlo * r0 <= 0
            F_lo = Flo_try;
            F_hi = F_init;
            bracket_found = true;
            break;
        elseif isfinite(rhi) && isfinite(r0) && r0 * rhi <= 0
            F_lo = F_init;
            F_hi = Fhi_try;
            bracket_found = true;
            break;
        elseif isfinite(rlo) && isfinite(rhi) && rlo * rhi <= 0
            F_lo = Flo_try;
            F_hi = Fhi_try;
            bracket_found = true;
            break;
        end

        expand = expand * 1.6;
    end

    % 首选 fzero：一维 shooting 比 fmincon 更适合当前标量 F 问题。
    if bracket_found
        try
            fz_opts = optimset('Display','off', ...
                               'TolX',1e-8, ...
                               'MaxIter',100, ...
                               'MaxFunEvals',200);

            [F_0, r_star, exitflag] = fzero( ...
                residual_fun, [F_lo, F_hi], fz_opts);

            has_F0 = exitflag > 0 && ...
                     isfinite(F_0) && ...
                     isfinite(r_star) && ...
                     abs(r_star) <= residual_tol;

            if has_F0
                return;
            end
        catch
            % 进入后面的保底搜索
        end
    end

    % 少数没有找到符号变化的点：在 log(F) 上做无约束保底搜索，
    % 自动保证 F>0，且只在 fzero 失败时使用。
    try
        q0 = log(max(F_init, 1e-8));
        fm_opts = optimset('Display','off', ...
                           'TolX',1e-7, ...
                           'MaxIter',150, ...
                           'MaxFunEvals',300);

        obj_logF = @(q) safe_squared_residual_logF( ...
            q, numeric_param, truth_value, u_target, ode_opts_force);

        q_star = fminsearch(obj_logF, q0, fm_opts);
        F_0 = exp(q_star);
        r_star = residual_fun(F_0);

        has_F0 = isfinite(F_0) && ...
                 isfinite(r_star) && ...
                 abs(r_star) <= residual_tol;
    catch
        F_0 = NaN;
        r_star = NaN;
        has_F0 = false;
    end
end

%% =========================================================
% 子函数2：终端 shooting residual
% 只关心 z=1 的终点，因此 tspan 改为 [0 1]，
% 不再要求 ode45 输出 1001 个网格点。
%% =========================================================
function r = terminal_residual_fast( ...
    F, numeric_param, truth_value, u_target, ode_opts)

    try
        u_0 = numeric_param(7);

        [~, model_datas] = ode45( ...
            @steady_solution_fast, [0 1], ...
            [1 1 1 F 0 0], ...
            ode_opts, numeric_param, truth_value);

        u_end = model_datas(end,2) * u_0 * 60 / 100;
        r = u_end - u_target;

        if ~isfinite(r)
            r = NaN;
        end
    catch
        r = NaN;
    end
end

%% =========================================================
% 子函数3：fzero 失败时的保底目标函数
%% =========================================================
function val = safe_squared_residual_logF( ...
    q, numeric_param, truth_value, u_target, ode_opts)

    F = exp(q);
    r = terminal_residual_fast( ...
        F, numeric_param, truth_value, u_target, ode_opts);

    if isfinite(r)
        val = r.^2;
    else
        val = 1e100;
    end
end

%% =========================================================
% 子函数4：计算局部灵敏度矩阵 J
%% =========================================================
function [F_0, r_star, has_F0, rp, rp_ok, J] = ...
    compute_local_jacobian_fast( ...
    param_value, numeric_param, u_target_single, F_init)

    [F_0, r_star, has_F0] = solve_force_fast( ...
        numeric_param, param_value, u_target_single, F_init);

    if ~has_F0 || ~isfinite(F_0)
        rp = NaN;
        rp_ok = false;
        J = NaN(3,2);
        return;
    end

    % 只需要 z=1 的最终灵敏度，不再输出 1001 个位置。
    z_span = [0 1];

    ode_opts = odeset('AbsTol',1e-9, ...
                      'RelTol',1e-8, ...
                      'Refine',1);

    % 这些常数在整个一次 sensitivity ODE 中不变，提前算好。
    T0 = numeric_param(2);
    u0 = numeric_param(7);

    IV = 0.67;
    L  = 150;
    G  = 1e9;
    n0 = 0.275;
    T1 = 267;
    Tg = 67;

    mu0 = 3*(IV)^(5.15) * ...
        exp(2.303*(3280/(T0+273)-1.54));

    q_pref = mu0*u0/(L*G);

    function dydz = augmented_ode(~, y)

        x        = y(1:6);
        s_theta1 = y(7:12);
        s_theta2 = y(13:18);
        s_F_vec  = y(19:24);

        f = steady_solution_fast([], x, numeric_param, param_value);

        % -------------------------------------------------
        % 状态 Jacobian A = df/dx
        % 保留原来的中心差分，避免改变 J 的精度等级。
        % -------------------------------------------------
        eps_x = 1e-8;
        A = zeros(6,6);

        for i = 1:6
            x_plus  = x;
            x_minus = x;

            x_plus(i)  = x_plus(i)  + eps_x;
            x_minus(i) = x_minus(i) - eps_x;

            f_plus  = steady_solution_fast( ...
                [], x_plus, numeric_param, param_value);
            f_minus = steady_solution_fast( ...
                [], x_minus, numeric_param, param_value);

            A(:,i) = (f_plus - f_minus) / (2*eps_x);
        end

        % -------------------------------------------------
        % 参数 Jacobian b = df/dtheta
        % theta1 = A_op, theta2 = A_c
        %
        % 原代码这里又做 4 次 steady_solution 有限差分。
        % 由于模型中：
        %   A_op 只显式进入第 6 个方程；
        %   A_c   只显式进入结晶速率 K，从而进入第 5 个方程；
        % 可以直接写出精确偏导，减少 4 次 RHS 评价。
        % -------------------------------------------------
        b = zeros(6,2);

        active = double(x(3)*T0 < T1 && x(3)*T0 > Tg);

        % d f6 / d A_op
        b(6,1) = active * f(2) / (x(2)*n0);

        % d f5 / d A_c
        q_ac = q_pref * x(4) / x(1);
        b(5,2) = f(5) * q_ac;

        dydz = [f; ...
                A*s_theta1 + b(:,1); ...
                A*s_theta2 + b(:,2); ...
                A*s_F_vec];
    end

    init_x = [1; 1; 1; F_0; 0; 0];

    aug_init = [init_x; ...
                zeros(6,1); ...
                zeros(6,1); ...
                [0;0;0;1;0;0]];

    try
        [~, aug_datas] = ode45( ...
            @augmented_ode, z_span, aug_init, ode_opts);
    catch
        rp = NaN;
        rp_ok = false;
        J = NaN(3,2);
        return;
    end

    aug_end = aug_datas(end,:)';

    s_theta1_end = aug_end(7:12);
    s_theta2_end = aug_end(13:18);
    s_F_end      = aug_end(19:24);

    rp = s_F_end(2);

    tol_rp = 1e-8;
    rp_ok = isfinite(rp) && abs(rp) > tol_rp;

    if ~rp_ok
        J = NaN(3,2);
        return;
    end

    dF_dtheta1 = -s_theta1_end(2) / rp;
    dF_dtheta2 = -s_theta2_end(2) / rp;

    total_s1 = s_theta1_end + s_F_end*dF_dtheta1;
    total_s2 = s_theta2_end + s_F_end*dF_dtheta2;

    J = [total_s1(1), total_s2(1); ...
         total_s1(5), total_s2(5); ...
         total_s1(6), total_s2(6)];
end
