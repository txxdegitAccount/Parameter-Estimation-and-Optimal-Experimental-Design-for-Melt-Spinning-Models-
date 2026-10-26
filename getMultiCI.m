function [CI_param1,CI_param2,min_sq_err] = getMultiCI(measurement_datas,numeric_param,u_setValue, type)
    
    CI_param1=zeros(3,1);
    CI_param2=zeros(3,1);

    if type == 1
       initial = [1.4,105,323];
    else
        initial=[1.4,105,linspace(280,286,length(u_setValue))];
    end

        fixed=[0,0,zeros(1,length(u_setValue))];
        lb=[1.0,90,ones(1,length(u_setValue))*100];
        ub=[2.0,120,ones(1,length(u_setValue))*400]; 

    [overall_minimizer,~,min_sq_err,~,~,max_l,~] = optimize_likelihood_1(fixed,initial,lb,ub,numeric_param,measurement_datas,u_setValue);

    CI_param1(2)=overall_minimizer(1);
    CI_param2(2)=overall_minimizer(2);


    %% profile likelihood

    base_param_names = {'A_o_p', 'A_c'};
    repeated_param_name = 'F_0';
    param_names = [base_param_names, repmat({repeated_param_name}, 1,length(u_setValue))];

    adjustments=[1,4, 5*ones(1,length(u_setValue))];
    lb = overall_minimizer-adjustments;
    ub = overall_minimizer+adjustments;

    numpts=50;
    num_params = size(fixed, 2);
    param_vals = zeros(num_params, numpts+1); 
    max_ls = zeros(num_params, numpts+1);
    minimizers = cell(num_params, numpts+1);
     
    parfor param=1:num_params   
    
        if fixed(param) || param >= 3
            continue;
        end
    
        local_param_vals = linspace(lb(param), ub(param), numpts);
        local_param_vals = [local_param_vals, overall_minimizer(param)];
        local_param_vals = sort(local_param_vals);
        [~,mle_idx] = min(abs(local_param_vals - overall_minimizer(param)));
    

        opt_fixed = fixed;
        opt_fixed(param) = 1;
        opt_initial = initial;

        local_minimizers = cell(1, numpts+1);
        local_max_ls = zeros(1, numpts+1);
        local_minimizers{mle_idx} = overall_minimizer(opt_fixed == 0);
        local_max_ls(mle_idx) = max_l;
        
        for i=mle_idx+1:numpts+1
            fprintf('Optimizing for %s=%.3f\n',param_names{param},local_param_vals(i));
            opt_initial(opt_fixed==0)=local_minimizers{i-1};
            opt_initial(param)=local_param_vals(i);

             [minimizer,~,~,~,~,local_maxls,~] = optimize_likelihood_1(opt_fixed,opt_initial,lb,ub,numeric_param,measurement_datas,u_setValue);

            local_max_ls(i) = local_maxls;
            local_minimizers{i}=minimizer;
        end
    
        for i=mle_idx-1:-1:1
            fprintf('Optimizing for %s=%.3f\n',param_names{param},local_param_vals(i));
            opt_initial(opt_fixed==0)=local_minimizers{i+1};
            opt_initial(param)=local_param_vals(i);
    
              [minimizer,~,~,~,~,local_maxls,~] = optimize_likelihood_1(opt_fixed,opt_initial,lb,ub,numeric_param,measurement_datas,u_setValue);
            local_max_ls(i) = local_maxls;
            local_minimizers{i}=minimizer;
        end
    
        param_vals(param, :) = local_param_vals;
        minimizers(param, :) = local_minimizers;
        max_ls(param, :) = local_max_ls;
    end
    
    %%
    % stores the zero of a Profile Likelihood curve
    zs = cell(num_params,1);
   
    for param=1:num_params

        if fixed(param) || param >= 3
            continue;
        end
    
        xx=param_vals(param,:);
        yy=max_ls(param,:)-max(max_ls(param,:));    
        zs{param}=interp_zero(xx,yy+1.92);

        if size(zs{param},2) >= 2

            if param==1
                CI_param1(1)=zs{param}(1);
                CI_param1(3)=zs{param}(end);
            elseif param==2
                CI_param2(1)=zs{param}(1);
                CI_param2(3)=zs{param}(end);
            end
        
        else

            if size(zs{param},2) == 1

                if(zs{param} > overall_minimizer(1,param))
                    if param==1
                        CI_param1(1)=-0.1;
                        CI_param1(3)=zs{param}(1);
                    elseif param==2
                        CI_param2(1)=-0.1;
                        CI_param2(3)=zs{param}(1);
                    end

                else

                    if param==1
                        CI_param1(1)=zs{param}(1);
                        CI_param1(3)=2;
                    elseif param==2
                        CI_param2(1)=zs{param}(1);
                        CI_param2(3)=210;
                    end

                end

            else

                if param==1
                    CI_param1(1)=-0.1;
                    CI_param1(3)=2;
                elseif param==2
                    CI_param2(1)=-0.1;
                    CI_param2(3)=210;
                end
                
            end
        end

    end
end
