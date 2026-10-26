function [J] = hessian_differential(param_value,numeric_param,perturbance,F_0, sigma)
   
    m=numeric_param(6);
    simulation_param=param_value;
    odeOptions = [];
    J =zeros(3, 2);
    
    for index = 1: 2

        simulation_param(index)=(1+perturbance)*param_value(index);
        [~,datas]=ode45( @steady_solution,linspace(0,1,m),[1 1 1 F_0 0 0],odeOptions,numeric_param,simulation_param);
        A_incr=datas(end,1);
        Theta_incr=datas(end,5);
        n_incr=datas(end,6);

        simulation_param(index)=(1-perturbance)*param_value(index);
        [~,datas]=ode45( @steady_solution,linspace(0,1,m),[1 1 1 F_0 0 0],odeOptions,numeric_param,simulation_param);
        A_decr=datas(end,1);
        Theta_decr=datas(end,5);
        n_decr=datas(end,6);

        J(:, index) = [(A_incr-A_decr)./(perturbance*2* sigma(1));
            (Theta_incr-Theta_decr)./(perturbance*2* sigma(2));
            (n_incr-n_decr)./(perturbance*2* sigma(3)); ]; 
    end

end
