clear; 

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

F_init = 270;              

u_target_single = 3730;    
% 采样参数
M = 1000;                    
theta_samples = lhsdesign(M, 2);  
theta_samples(:,1) = 1.0 + theta_samples(:,1) * (2.0 - 1.0);   
theta_samples(:,2) = 90 + theta_samples(:,2) * (120 - 90);     

G = zeros(2, 2);
W_matrix = eye(3);          

% 开启并行池（如果有 Parallel Computing Toolbox）
if isempty(gcp('nocreate'))  
    parpool; 
end

parfor m = 1:M
    param_value_m = theta_samples(m, :);
    
    J_m = compute_local_jacobian(param_value_m, numeric_param, u_target_single, F_init);
    
    G_local{m} = J_m' * W_matrix * J_m;  
end

for m = 1:M
    G = G + G_local{m};
end

disp('全局信息矩阵 G:');
disp(G);

rank_G = rank(G, 1e-6);
disp(['Rank of G: ', num2str(rank_G)]);

cond_G = cond(G);
disp(['Condition number of G: ', num2str(cond_G)]);

lambda = eig(G);
lambda_min = min(lambda);
disp(['Minimum eigenvalue of G: ', num2str(lambda_min)]);

if rank_G == 2 && lambda_min > 1e-6
    disp('结论：全局可辨识且数值稳定');
else
    disp('结论：存在全局不可辨识或参数估计不稳定问题');
end

function J = compute_local_jacobian(param_value, numeric_param, u_target_single, F_init)
    F_nom = get_force(numeric_param, param_value, u_target_single, F_init);
   
    numPoints = numeric_param(6) + 1;
    z_span = linspace(0, 1, numPoints);
    ode_opts = odeset('AbsTol', 1e-9, 'RelTol', 1e-8);
    
    function dydz = augmented_ode(~, y, numeric_param, param_value)
        x = y(1:6);
        s_theta1 = y(7:12);   % ?x/?A_op
        s_theta2 = y(13:18);  % ?x/?A_c
        s_F_vec  = y(19:24);  % ?x/?F(0)
        
        f = steady_solution([], x, numeric_param, param_value);
        
        % 数值中心差分计算 A = df/dx
        eps_x = 1e-8;
        A = zeros(6,6);
        for i = 1:6
            x_plus  = x; x_plus(i)  = x_plus(i)  + eps_x;
            x_minus = x; x_minus(i) = x_minus(i) - eps_x;
            f_plus  = steady_solution([], x_plus,  numeric_param, param_value);
            f_minus = steady_solution([], x_minus, numeric_param, param_value);
            A(:,i) = (f_plus - f_minus) / (2*eps_x);
        end
        
        % 数值中心差分计算 b_j = df/dθ_j
        eps_theta = 1e-7 * (abs(param_value) + 1e-10);
        b = zeros(6,2);
        for j = 1:2
            theta_plus  = param_value; theta_plus(j)  = theta_plus(j)  + eps_theta(j);
            theta_minus = param_value; theta_minus(j) = theta_minus(j) - eps_theta(j);
            f_plus  = steady_solution([], x, numeric_param, theta_plus);
            f_minus = steady_solution([], x, numeric_param, theta_minus);
            b(:,j) = (f_plus - f_minus) / (2*eps_theta(j));
        end
        
        dydz = [f; ...
                A * s_theta1 + b(:,1); ...
                A * s_theta2 + b(:,2); ...
                A * s_F_vec];
    end
    
    init_x = [1 1 1 F_nom 0 0]';
    
    aug_init = [init_x; ...
                zeros(6,1); ...      % s_theta1(0) = 0
                zeros(6,1); ...      % s_theta2(0) = 0
                [0;0;0;1;0;0]];      % s_F(0) = e_4
    
    [~, aug_datas] = ode45(@(z,y) augmented_ode(z,y,numeric_param,param_value), ...
                           z_span, aug_init, ode_opts);
    aug_end = aug_datas(end, :)'; 
    
    s_theta1_end = aug_end(7:12);
    s_theta2_end = aug_end(13:18);
    s_F_end      = aug_end(19:24);
    
    if abs(s_F_end(2)) < 1e-8
        warning('采样点 %d: s_F_u(1) 接近0，打靶敏感度奇异，使用 NaN', m);
        J = NaN(3,2);
        return;
    end
    
    dF_dtheta1 = - s_theta1_end(2) / s_F_end(2);
    dF_dtheta2 = - s_theta2_end(2) / s_F_end(2);
    
    total_s1 = s_theta1_end + s_F_end * dF_dtheta1;
    total_s2 = s_theta2_end + s_F_end * dF_dtheta2;
    
    J = [total_s1(1), total_s2(1); ...
         total_s1(5), total_s2(5); ...
         total_s1(6), total_s2(6)];
end