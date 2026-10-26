function ff = obj_function(param_value,numeric_param,x,sigma, u_vars)

    % numeric_param=[W,T_0,A_0,T_a,u_a,numPoints,u_0];
    % u_target T_0 T_a u_a w;
    %  lb = [2000*ones(1,u_vars), 285, 15, 30, 6.75e-3]; 
    %  ub = [4000*ones(1,u_vars), 310, 30, 50, 9e-3]; 

    u_target=x(1:u_vars) .* 10;
    numeric_param(1) = x(end) * 1e-5;  % W
    numeric_param(2) = x(u_vars + 1); % T_0
    numeric_param(4) = x(u_vars + 2) / x(u_vars +  1); % T_a / T_0
    numeric_param(end) =  numeric_param(1) / ( numeric_param(3) * (1.356-5*10^(-4) * numeric_param(2) ) ); % u_0
    numeric_param(5) = x(u_vars + 3) / numeric_param(end); % u_a / u_0
  
    F_0=get_force(numeric_param,param_value,u_target,270);
    % disp(F_0)
    % odeOptions=odeset('AbsTol', 1e-9, 'RelTol', 1e-8);
    odeOptions = [];
    sigma  = zeros(length(F_0), 3);
    perturbance = 5e-3;

    for index  = 1:length(F_0)
        [~,datas]=ode45( @steady_solution,linspace(0,1,numeric_param(6)),[1 1 1 F_0(index) 0 0],odeOptions,numeric_param,param_value);
        sigma(index, :) = [datas(end,1) datas(end,5) datas(end,6)];
    end

     sigma_list = max(sigma, [], 1);

    J=zeros(length(F_0)*3, 2);
    for index=1:length(F_0)
       J(index*3 - 2 : index * 3, :)= hessian_differential(param_value,numeric_param,perturbance,F_0(index),sigma_list);
    end

    CV=J'*J;
  
    % D-optimal：最大化矩阵的行列式
    % ff=-det(CV);

    % % E-optimal：最大化矩阵的最小特征值
    if any( isnan(CV(:)) ) || any ( isinf(CV(:)) )
        ff = Inf;
    else
        eigCV=eig(CV);
        minEig=min(eigCV);
        if minEig < 1e-10
            minEig = 1e-10;
        end
        ff = -minEig;
    end

end
