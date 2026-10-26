function [error] = square_1(u_setValue,numeric_param,measurement_datas,param_value)

   
    data_diff=zeros(length(u_setValue), 3);

    % numeric_param=[W,T_0,A_0,T_a,u_a,m,u_0];
    m=numeric_param(6);
    z=linspace(0,1,m+1);
    
    normalization_factors = max(measurement_datas, [], 1);
    % options = odeset('AbsTol', 1e-9, 'RelTol', 1e-8);
    options = [];

    for index = 1:length(u_setValue)
        [~,model_datas]=ode45( @steady_solution,z,[1 1 1 param_value(index+2) 0 0],options,numeric_param, param_value(1:2));
    
        
        model_data=[model_datas(end,1)*numeric_param(3) model_datas(end,5)*100 model_datas(end,6)];
        data_diff(index, :) = (measurement_datas(index, :) - model_data) ./ normalization_factors;
    end
    
    data_diff2=data_diff.^2;
    error=sum(data_diff2,"all");

end




