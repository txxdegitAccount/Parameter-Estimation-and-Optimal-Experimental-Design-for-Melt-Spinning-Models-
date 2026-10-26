function J = compute_jacobian()
    % 根据 test.m 中的参数设置
    W = 0.009;                  % g/s
    T_0 = 298;                  % °C
    A_0 = pi*(0.22*0.1)^2 / 4;   % cm? ≈ 0.00038013
    T_a_abs = 15;               % °C
    u_a_abs = 30;               % cm/s
    
    rho_0 = 1.356 - 5e-4 * T_0;
    u_0 = W / (A_0 * rho_0);     % ≈19.6155 cm/s
    
    T_a = T_a_abs / T_0;         % normalized
    u_a = u_a_abs / u_0;         % normalized
    
    numPoints = 1001;
    u_target = [3480, 3730, 3980];  % 你的目标速度
    
    numeric_param = [W, T_0, A_0, T_a, u_a, numPoints-1, u_0];  % 注意：get_force 中 m = numeric_param(6)，z有 m+1 点，故这里 m = numPoints-1
    
    param_value = [1.41, 105.24];  % A_op, A_c
    F_init = 270;
    
    % Step 1: 使用打靶法求标称 F_nom (向量，长度=3)
    F_nom = get_force(numeric_param, param_value, u_target, F_init);
    
    if length(F_nom) ~= length(u_target)
        error('get_force 返回的 F_0 长度不匹配 u_target');
    end
    
    % 定义固定网格
    z_span = linspace(0, 1, numPoints);
    ode_opts = odeset('AbsTol', 1e-9, 'RelTol', 1e-8);
    
    % Step 2: 定义扩展 ODE 函数（24维状态）
    function dydz = augmented_ode(~, y)
        x = y(1:6);
        s_theta1 = y(7:12);   % ?x/?A_op
        s_theta2 = y(13:18);  % ?x/?A_c
        s_F_vec = y(19:24);   % 这里只用一个代表性的 s_F (对第一个F)，假设敏感度相似，或取平均；简化处理
        
        % 当前 f
        f = steady_solution([], x, numeric_param, param_value);
        
        % 数值中心差分计算 A = df/dx (6x6)
        eps_x = 1e-8;
        A = zeros(6,6);
        for i = 1:6
            x_plus = x;  x_plus(i) = x_plus(i) + eps_x;
            x_minus = x; x_minus(i) = x_minus(i) - eps_x;
            f_plus = steady_solution([], x_plus, numeric_param, param_value);
            f_minus = steady_solution([], x_minus, numeric_param, param_value);
            A(:,i) = (f_plus - f_minus) / (2*eps_x);
        end
        
        % 数值中心差分计算 b_j = df/dθ_j (6x2)
        eps_theta = 1e-7 * (abs(param_value) + 1e-10);  % 相对扰动
        b = zeros(6,2);
        for j = 1:2
            theta_plus = param_value; theta_plus(j) = theta_plus(j) + eps_theta(j);
            theta_minus = param_value; theta_minus(j) = theta_minus(j) - eps_theta(j);
            f_plus = steady_solution([], x, numeric_param, theta_plus);
            f_minus = steady_solution([], x, numeric_param, theta_minus);
            b(:,j) = (f_plus - f_minus) / (2*eps_theta(j));
        end
        
        % 扩展方程 (这里 s_F_vec 对应 ?x/?F，b_F=0)
        dydz = [f; ...
                A * s_theta1 + b(:,1); ...
                A * s_theta2 + b(:,2); ...
                A * s_F_vec];
    end
    
    % Step 3: 为每个 target 分别积分扩展系统（因为 F(0) 不同，轨迹不同）
    % 为简化，我们取中间 target (index=2) 的轨迹作为代表性轨迹计算敏感度
    % （实际多 target 时，Jacobian 对所有输出平均敏感度；这里输出是端点标量，Φ 是 3x1 向量独立于多个u）
    mid_idx = 2;
    init_x = [1 1 1 F_nom(mid_idx) 0 0]';
    
    aug_init = [init_x; ...
                zeros(6,1); ...  % s_theta1(0)
                zeros(6,1); ...  % s_theta2(0)
                [0;0;0;1;0;0]];  % s_F(0) = e4
    
    [~, aug_datas] = ode45(@augmented_ode, z_span, aug_init, ode_opts);
    aug_end = aug_datas(end, :)';  % z=1 处
    
    % 提取
    s_theta1_end = aug_end(7:12);
    s_theta2_end = aug_end(13:18);
    s_F_end = aug_end(19:24);
    
    % Step 4: 计算 ?F/?θ_j （针对中间 target）
    if abs(s_F_end(2)) < 1e-8
        warning('s_F_u(1) 接近0，打靶敏感度奇异');
        J = NaN(3,2);
        return;
    end
    
    dF_dtheta1 = - s_theta1_end(2) / s_F_end(2);
    dF_dtheta2 = - s_theta2_end(2) / s_F_end(2);
    
    % 总敏感度
    total_s1 = s_theta1_end + s_F_end * dF_dtheta1;
    total_s2 = s_theta2_end + s_F_end * dF_dtheta2;
    
    % Step 5: 组装 Jacobian (3x2): 行对应 A(1), Θ(1), n(1) → indices 1,5,6
    J = [total_s1(1), total_s2(1); ...
         total_s1(5), total_s2(5); ...
         total_s1(6), total_s2(6)];
    
    % 显示结果
    disp('Jacobian J (rows: A(1), Θ(1), n(1); columns: A_op, A_c):');
    disp(J);
    disp(['Rank of J: ', num2str(rank(J, 1e-6))]);
end