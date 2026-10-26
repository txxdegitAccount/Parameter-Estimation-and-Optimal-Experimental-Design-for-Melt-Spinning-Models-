function F_0 = get_force(numeric_param,truth_value,u_target,F_init)

  options = optimoptions('fmincon','MaxIterations',6000,'Display','off');
  objective = @(F) obj_function(u_target,numeric_param,truth_value,F);

  num_targets = length(u_target);
  F_init_vector = repmat(F_init, num_targets, 1); 
  
  [F_0, ~] = fmincon(objective, F_init_vector, [], [], [], [], [], [],@(x) nonlcon(x,u_target,1e-6,numeric_param,truth_value), options);
end

%%
function error = obj_function(u_target, numeric_param, param_value, F)

    m = numeric_param(6);
    u_0 = numeric_param(7);
    z = linspace(0, 1, m+1);
    u_list = linspace(0,0, length(u_target));

   odeOptions=[];

    for index = 1:length(u_target)
        [~,model_datas]=ode45( @steady_solution,z,[1 1 1 F(index) 0 0],odeOptions,numeric_param,[param_value(1),param_value(2)]);
        u_list(index)=model_datas(end,2) * u_0 * 60 / 100;
    end

    error = sum((u_list - u_target).^2);

end

%%
function [c, ceq] = nonlcon(x, u_target, tolerance,numeric_param,param_value) 
    m = numeric_param(6);
    u_0 = numeric_param(7);
    z = linspace(0, 1, m+1);
    u_list = linspace(0,0, length(u_target));

   odeOptions=[];
    
    for index = 1:length(u_target)
        [~,model_datas]=ode45( @steady_solution,z,[1 1 1 x(index) 0 0],odeOptions,numeric_param,[param_value(1),param_value(2)]);
        u_list(index)=model_datas(end,2) * u_0 * 60 / 100;
    end

   error = sum((u_list - u_target).^2);
   c = error - tolerance;
   ceq =[];
    
end