function [minimizer,param_str,min_sq_err,grad,hessian,max_l,sigma] = optimize_likelihood_1(fixed,fixed_param_val,lb,ub,numeric_param,measurement_datas,u_setValue)
                                
    param_str='[';
    paramcount=1;

    for i=1:size(fixed_param_val,2)
        if ~fixed(i)
            param_str=strcat(param_str,'x(',num2str(paramcount),'),');
            paramcount = paramcount+1;
        else
            param_str=strcat(param_str,num2str(fixed_param_val(i)),',');
        end
    end

    param_str=strcat(param_str,']');
    % the anonymous function takes an argument x and calls a function named squared_error
    f_str=strcat('f=@(x) square_1(u_setValue,numeric_param,measurement_datas,',param_str,');');
    eval(f_str);

    % the optimoptions function is used to create an option object for the optimization problem
    options=optimoptions('fmincon','Algorithm','interior-point');

    % set to display iteration information, which shows the optimized iteration process in the command window
    % opens diagnostics, which displays detailed optimized diagnostics
    % sets the maximum number of function evaluations to prevent the optimization process from endless loops
    options.Display='final';
    options.Diagnostics='off';
    options.MaxFunctionEvaluations=6000;
    options.ConstraintTolerance=1.0000e-06;
    problem.objective=f;
    problem.x0=fixed_param_val(fixed==0);
    problem.solver='fmincon';
    problem.lb=lb(fixed==0);
    problem.ub=ub(fixed==0);
    problem.options=options;
    problem.nonlcon = @(x) nonlincon(x, fixed, fixed_param_val, numeric_param, u_setValue);
    [minimizer,min_sq_err,exitflag,fmincon_output,~,grad,hessian] = fmincon(problem);
    fprintf('fmincon exitflag: %d\n',exitflag);
    disp(fmincon_output); % full display

    nonzero_index=find(fixed ~= 0);
    if(nonzero_index > 1)
        minimizer_1=[minimizer(1:nonzero_index-1) fixed_param_val(nonzero_index) minimizer(nonzero_index:end)];
    else
        minimizer_1=[fixed_param_val(nonzero_index) minimizer];
    end

    [max_l,sigma]=  log_likelihood(min_sq_err, measurement_datas);
    fprintf(['optimization outcome: ',repmat('%.3f ',size(minimizer))],minimizer);

end

%%
function [c, ceq] = nonlincon(x, fixed, fixed_param_val, numeric_param, u_setValue)
    params = fixed_param_val;
    params(fixed == 0) = x;

    m = numeric_param(6);
    u_0 = numeric_param(7);
    z = linspace(0, 1, m+1);
    u_list = linspace(0,0, length(u_setValue));

   odeOptions= odeset('AbsTol', 1e-9, 'RelTol', 1e-8);
    for index = 1:length(u_setValue)
        [~,model_datas]=ode45( @steady_solution,z,[1 1 1 params(index+2) 0 0],odeOptions,numeric_param,[params(1),params(2)]);
        u_list(index)=model_datas(end,2) * u_0 * 60 / 100;
    end

    tolerance = 1e-3; 
    error = sum((u_list - u_setValue).^2);
    
    c = error - tolerance;
    ceq = [];
end
