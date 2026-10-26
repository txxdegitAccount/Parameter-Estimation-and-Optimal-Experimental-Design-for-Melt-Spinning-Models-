%%
clear;

W=0.009; % mass flow rate(g/s) 
T_0=298; % °C
A_0=pi*(0.22*0.1)^2/4; % mm --> cm
T_a=15; % Air Conditioning Temperature 
u_a=30; % Air Conditioning wind velocity (cm/s)

rho_0=1.356-5*10^(-4)*T_0; % g/cm3
u_0=W/(A_0*rho_0); % cm/s 
L=150;
IV=0.67;
miu_0=3*(IV)^(5.15)*exp(2.303*(3280/(T_0+273)-1.54)); % g/(cm*s)

T_a=T_a/T_0; 
u_a=u_a/u_0;

numPoints=1001;
u_target =  [3480,3730,3980];
numeric_param=[W,T_0,A_0,T_a,u_a,numPoints,u_0];

load('output1/sampleVelocity_21.mat', 'x', 'u_vars');
optimal_u_target=x(1:u_vars)*10;
optimal_numeric_param = numeric_param;
optimal_numeric_param(1) = x(end) * 10 ^(-5);
optimal_numeric_param(2) = x(u_vars + 1); % T_0
optimal_numeric_param(4) = x(u_vars + 2) / x(u_vars +  1); % T_a / T_0
optimal_numeric_param(end) =  optimal_numeric_param(1) / ( optimal_numeric_param(3) * (1.356-5*10^(-4) *optimal_numeric_param(2) ) ); % u_0
optimal_numeric_param(5) = x(u_vars + 3) / optimal_numeric_param(end); % u_a / u_0

% A_op A_c
param_value=[1.41,105.24];
u_target_matrix  = cell(1,2);
u_target_matrix{1}= optimal_u_target;
u_target_matrix{2}= u_target;

numeric_param_matrix  = cell(1,2);
numeric_param_matrix{1} = optimal_numeric_param;
numeric_param_matrix{2} = numeric_param;

F_0=get_force(numeric_param_matrix{2},param_value,u_target_matrix{2},270);
 odeOptions = odeset('AbsTol', 1e-9, 'RelTol', 1e-8);

 data_matrix = zeros(length(u_target_matrix{2}), 3);
 for index = 1:length(F_0)
    [~,datas]=ode45(@steady_solution,linspace(0,1,numPoints),[1 1 1 F_0(index) 0 0],odeOptions,numeric_param_matrix{2},param_value);
    data_matrix(index, :) = [datas(end, 1)*A_0 datas(end,5)*100 datas(end,6)];
 end

F_0=get_force(numeric_param_matrix{1},param_value,u_target_matrix{1},270);

[~,datas]=ode45(@steady_solution,linspace(0,1,numPoints),[1 1 1 F_0 0 0],odeOptions,numeric_param_matrix{1},param_value);
optimal_data_matrix = [datas(end, 1)*optimal_numeric_param(3) datas(end,5)*100 datas(end,6)];

datas_matrix = cell(1, 2);
datas_matrix{1} = optimal_data_matrix;
datas_matrix{2} = data_matrix;

sigma=0.1750;
sigma_matrix = cell(1,2);
sigma_matrix{1} = sigma * max(datas_matrix{1}, [], 1);
sigma_matrix{2} = sigma * max(datas_matrix{2}, [], 1);

%%
param_value=[1.46,105.29];

bestVal = Inf;
bestIndex = -1;

for index = 1 : 1
   
    u_vars=1;
    lb = [200*ones(1,u_vars),285, 15, 30, 675];
    ub =[400*ones(1,u_vars),300, 30, 50, 900];
    % nonlconFun = @(x) nonlcon(x);
    f=@(x) obj_function(param_value,numeric_param,x,sigma, u_vars);
    n_vars = u_vars +4;
    intcon=1:n_vars;
    options = optimoptions('ga', 'Display', 'iter', 'UseParallel', true, 'MaxGenerations', 10000,'MaxTime',3600,...
        'FunctionTolerance',1e-8,'MaxStallGenerations',50,'InitialPopulationRange',[lb;ub],'FitnessScalingFcn',@fitscalingrank);
    % [x, ~] = ga(f, n_vars, [], [], [], [], lb, ub, nonlconFun, intcon, options);
    [x, fval,exitflag,~] = ga(f, n_vars, [], [], [], [], lb, ub, [], intcon, options);

    mondate = datestr(now,'mm-dd HH:MM');
    x_target=[x(1)*10 x(2:end)];

    if(fval  < bestVal)
        bestIndex = index;
        bestVal = fval;
    end

    if(exitflag ~= 0)
        save("output/sampleVelocity_"+index+"_"+mondate);
    end
end

%%

% noisy=squeeze(noisy_matrix(randi([1,100]),:,:));
index=randi([1,100]);
noisy=squeeze(noisy_matrix(96,:,:));
noisy_datas=datas+noisy;

% rand_index = 50;86;89;
selectedIndices=find(weights(1,2,:));
measurement_datas=noisy_datas(selectedIndices,[1,2,5,6]);

[overall_minimizer,max_l,param_str,min_sq_err,~,hessian] = optimize_likelihood(fixed,fixed_param_val,lb,ub,measurement_datas,numeric_param,sigma_list,sigma,selectedIndices);


% 赤池信息准则和贝叶斯信息准则，越小说明模型拟合效果越好
aic = -2*max_l + 2*num_free_params;
% N=sampleNum;
% bic = -2*max_l + log(N)*num_free_params;
% fprintf('AIC=%.3f,BIC=%.3f\n',aic,bic);


%% profile likelihood
% 参数空间的最大似然分析

param_names={'A_o_p','A_c','x_4(0)'};

numpts=100;
param_vals = zeros(num_params, numpts+1);
max_ls = zeros(num_params, numpts+1);
minimizers = cell(num_params, numpts+1);

parfor param=1:num_params

    if fixed(param)
        continue;
    end

    % 86: 0.40 0.38 
    % 96: 1.12 3 3
    lb=[overall_minimizer(1)-1.12,overall_minimizer(2)-3,overall_minimizer(3)-3];
    ub=[overall_minimizer(1)+1.12,overall_minimizer(2)+3,overall_minimizer(3)+3];

    local_param_vals = linspace(lb(param), ub(param), numpts);
    local_param_vals = [local_param_vals, overall_minimizer(param)];
    local_param_vals = sort(local_param_vals);
    [~,mle_idx] = min(abs(local_param_vals - overall_minimizer(param)));

    opt_fixed=linspace(0,0,num_params);
    opt_fixed(param)=1;
    initial=[1.0,150,260];

    local_minimizers = cell(1, numpts+1);
    local_max_ls = zeros(1, numpts+1);
    local_minimizers{mle_idx} = overall_minimizer(opt_fixed == 0);
    local_max_ls(mle_idx) = max_l;

    % 在 minimizers 数组中的相应位置保存当前参数的最优参数值
    % 在 max_ls 数组中的相应位置保存最大似然值

    % 在参数空间中对目标函数进行优化：使用一个 profile likelihood 的方法，在最大似然估计 (MLE) 点附近对目标函数进行优化
    for i=mle_idx+1:numpts+1
        fprintf('Optimizing for %s=%.3f\n',param_names{param},local_param_vals(i));
        initial(opt_fixed==0)=local_minimizers{i-1};
        initial(param)=local_param_vals(i);

        [minimizer,local_maxls,~,~,~,~] = optimize_likelihood(opt_fixed,initial,lb,ub,measurement_datas,numeric_param,sigma_list,sigma,selectedIndices);

        local_max_ls(i) = local_maxls;
        local_minimizers{i}=minimizer;
    end

    for i=mle_idx-1:-1:1
        fprintf('Optimizing for %s=%.3f\n',param_names{param},local_param_vals(i));
        initial(opt_fixed==0)=local_minimizers{i+1};
        initial(param)=local_param_vals(i);

        [minimizer,local_maxls,~,~,~,~] = optimize_likelihood(opt_fixed,initial,lb,ub,measurement_datas,numeric_param,sigma_list,sigma,selectedIndices);
        local_max_ls(i) = local_maxls;
        local_minimizers{i}=minimizer;
    end


    param_vals(param, :) = local_param_vals;
    minimizers(param, :) = local_minimizers;
    max_ls(param, :) = local_max_ls;

end

%%

% 对每个可变参数进行分析，绘制 Profile Likelihood 图
fig=figure('Position',[100 100 1400 400],'color','w');
free_param_count=0;
% 存储Profile Likelihood 曲线的零点
zs = cell(num_params,1);
% 存储存储置信区间的宽度
conf_interval=nan(num_params,1);

for param=1:num_params

    if fixed(param)
        continue;
    end

    free_param_count = free_param_count+1;
    subplot(1,num_free_params,free_param_count);
    hold on;
    xx=param_vals(param,:);
    yy=max_ls(param,:)-max(max_ls(param,:));
    plot(xx,yy,'color',"#6F80BE",'LineWidth',3);
    xline(overall_minimizer(param),'color',"#F46E62",'LineWidth',3)
    % 绘制y=-1.92的水平线 (-1.92对应95%置信区间)
    % plot([min(param_vals(param,:)),max(param_vals(param,:))],[-1.92,-1.92]);
    plot([0,300],[-1.92,-1.92],'--','color',"#BD8EC0",'LineWidth',3);

    ax=gca;
    set(ax, 'FontSize', 18);  % 字体大小
    set(ax, 'FontName', 'Times New Roman');  % 字体
    
    xlabel(param_names{param},'FontSize', 20, 'FontName', 'Times New Roman');
    % ylabel('log(L)','FontSize', 20, 'FontName', 'Times New Roman');
    % ylabel("l("+param_names(param)+" | y^o )",'FontAngle', 'italic');
    ylabel("\itl\rm(" + param_names(param) + " | \bf\it{y}\rm^o)", 'Interpreter', 'tex');
    % ylabel('Normal \bf Bold \it Italic \rm Normal again', 'Interpreter', 'tex');
    axis('square');
    xlim([min(param_vals(param,:)),max(param_vals(param,:))]);
    ylim([-2.5,0]);
    hold off;

    % 使用 interp_zero 函数找到 Profile Likelihood 曲线的零点（线形插值：零点存在定理）
    zs{param}=interp_zero(xx,yy+1.92);
    % 零点个数为2，打印置信区间；零点个数为1，打印零点信息
    if size(zs{param},2) >= 2
        conf_interval(param)=zs{param}(end)-zs{param}(1);
        fprintf('95%% Confidence interval for param %s is: (intercept at -1.92)\n',param_names{param});
        fprintf('width=%.4f: [%.4f,%.4f]\n',conf_interval(param),zs{param}(1),zs{param}(end));
        % xline(zs{param}(1), '--','color',"#A8D08D",'LineWidth',3);  % 下界用绿色标示
        % xline(zs{param}(end),'--','color',"#5CA0CF",'LineWidth',3);  % 上界用青色标示
        xlim([zs{param}(1)-conf_interval(param),zs{param}(end)+conf_interval(param)]);
    else
        fprintf('Do not have 2 intercepts for param %s, they are:\n',param_names{param});
        disp(zs{param});
        if size(zs{param},2) == 1
            if(zs{param} > overall_minimizer(param))
                % xline(zs{param}(1), 'color',"#5CA0CF",'LineWidth',3);  % 上界用青色标示

            else
                % xline(zs{param}(1), 'color',"A8D08D",'LineWidth',3);  % 下界用绿色标示

            end
        end
    end

end


%%

colors = ['r','w','r', 'g', 'b', 'c', 'm', 'y', 'k'];  
symbols = ['o', 'p', '+', '^', 'x', 's', 'd',  'v', '<', '>', 'h'];

weights=ones(9,numPoints);
for index=2:10
    optimazation=load("output2/sampleNum_"+index+".mat",'w');
    optimazation_weight=optimazation.w;
    weights(index-1,:)=optimazation_weight;
end


fig = figure('Position', [100, 100, 1200, 800]);
hold on;

num_row = size(weights,1);
for row = 1:num_row
    weights(row,end) = 0;
    x=(find(weights(row,:))-1)/100;
    y=ones(1,size(x,2))*1+row - 1;
    symbol=symbols(row);
    color=colors(row);
    scatter(y, x, symbol, 'Color', color, "SizeData",380,'LineWidth',2); 
end


ylim([0, 1.05]); 
ylabel('Observed points location on the space axis','FontName', 'Times New Roman', 'FontSize', 32);  
xlim([0.7, 9.3]);
set(gca, 'FontSize', 28); % 设置坐标轴标签字体大小为 14
xlabel('Number of points','FontName', 'Times New Roman', 'FontSize', 32);

hold on;
point_s=[1,2,3,4,5,6,7,8,9]';
plot(point_s', ones(size(point_s)),'--', 'MarkerSize',10,'LineWidth', 3,'Color','#91baa2');
hold off;
