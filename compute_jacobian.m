function J = compute_jacobian()
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
    u_target = [3480, 3730, 3980]; 
    
    numeric_param = [W, T_0, A_0, T_a, u_a, numPoints-1, u_0];  
    param_value = [1.41, 105.24];  % A_op, A_c
    F_init = 270;
    
    F_nom = get_force(numeric_param, param_value, u_target, F_init);
    
    if length(F_nom) ~= length(u_target)
        error('get_force 返回的 F_0 长度不匹配 u_target');
    end
    
    
    z_span = linspace(0, 1, numPoints);
    ode_opts = odeset('AbsTol', 1e-9, 'RelTol', 1e-8);
    
    
    function dydz = augmented_ode(~, y)
        x = y(1:6);
        s_theta1 = y(7:12);   
        s_theta2 = y(13:18); 
        s_F_vec = y(19:24);   
        f = steady_solution([], x, numeric_param, param_value);
        
        eps_x = 1e-8;
        A = zeros(6,6);
        for i = 1:6
            x_plus = x;  x_plus(i) = x_plus(i) + eps_x;
            x_minus = x; x_minus(i) = x_minus(i) - eps_x;
            f_plus = steady_solution([], x_plus, numeric_param, param_value);
            f_minus = steady_solution([], x_minus, numeric_param, param_value);
            A(:,i) = (f_plus - f_minus) / (2*eps_x);
        end
        
        eps_theta = 1e-7 * (abs(param_value) + 1e-10);  
        b = zeros(6,2);
        for j = 1:2
            theta_plus = param_value; theta_plus(j) = theta_plus(j) + eps_theta(j);
            theta_minus = param_value; theta_minus(j) = theta_minus(j) - eps_theta(j);
            f_plus = steady_solution([], x, numeric_param, theta_plus);
            f_minus = steady_solution([], x, numeric_param, theta_minus);
            b(:,j) = (f_plus - f_minus) / (2*eps_theta(j));
        end
        
        dydz = [f; ...
                A * s_theta1 + b(:,1); ...
                A * s_theta2 + b(:,2); ...
                A * s_F_vec];
    end
    
    mid_idx = 2;
    init_x = [1 1 1 F_nom(mid_idx) 0 0]';
    
    aug_init = [init_x; ...
                zeros(6,1); ...  
                zeros(6,1); ...  
                [0;0;0;1;0;0]];  
    
    [~, aug_datas] = ode45(@augmented_ode, z_span, aug_init, ode_opts);
    aug_end = aug_datas(end, :)';  
    
    s_theta1_end = aug_end(7:12);
    s_theta2_end = aug_end(13:18);
    s_F_end = aug_end(19:24);
    
    if abs(s_F_end(2)) < 1e-8
        warning('s_F_u(1) 接近0，打靶敏感度奇异');
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

    disp('Jacobian J (rows: A(1), Θ(1), n(1); columns: A_op, A_c):');
    disp(J);
    disp(['Rank of J: ', num2str(rank(J, 1e-6))]);
end