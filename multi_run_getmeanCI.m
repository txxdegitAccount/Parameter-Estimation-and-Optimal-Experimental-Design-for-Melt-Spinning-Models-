% Main program to compute the average CI and estimated parameters
% new Program
tic;
colors = ['r','w','r', 'g', 'b', 'c', 'm', 'y', 'k'];  
symbols = ['o', 'p', '+', '^', 'x', 's', 'd',  'v', '<', '>', 'h'];
runTimes=1;

noisy_matrix=cell(1,2);
noisy=zeros(runTimes,3);

for type = 1:2
        for stateIndex=1:3
            noisy(:,stateIndex)=normrnd(0,sigma_matrix{type}(stateIndex),[runTimes, 1]);
        end
        noisy_matrix{type} = noisy;
end

num_params = 2;
num_types = 2; 
num_row = size(runTimes,2);

% first:mean lb second:mean global parameter
% third:mean ub fourth:mean ub - mean lb
mean_matrix_CI=zeros(num_row*4,2*num_params);
mean_width_CI=zeros(num_row,2*num_params);
% store the average value of each type (uniform and optimization)
mean_parameter=zeros(num_row,2*num_params);
% the MSE of the estimated parameters
mean_mse=zeros(num_row,num_types);
% record the mean of square error between the estimated value and the real value.
mse_matrix=zeros(runTimes,1); 

% denotes the lower bound, the estimated value and the upper bound.
CI_param1=zeros(3,runTimes);  
CI_param2=zeros(3,runTimes); 


for type = 1 : 1
        
        u_setValue = u_target_matrix{type};
        
        for runIndex=1:runTimes
            measurement_datas=datas_matrix{type}+noisy_matrix{type}(runIndex,:);
            u_setValue = u_target_matrix{type};
            numeric_param = numeric_param_matrix{type};

            [CI_param1(:,runIndex),CI_param2(:,runIndex),mse_matrix(runIndex)] = ...
            getMultiCI(measurement_datas,numeric_param,u_setValue, type);

        end

        % lb=[0.1,0,250]; ub=[1.5,200,300];
        param1_index=(CI_param1(1,:) > -0.1) & (CI_param1(3,:) < 2);
        param2_index=(CI_param2(1,:) > -0.1) & (CI_param2(3,:) < 210); 
        % average for all the results of runTimes running (each row)
        CI_param1_ave = mean(CI_param1,2);
        CI_param2_ave = mean(CI_param2,2);
        
        CI_param1_matrix=[CI_param1_ave;CI_param1_ave(3)-CI_param1_ave(1)];
        CI_param2_matrix=[CI_param2_ave;CI_param2_ave(3)-CI_param2_ave(1)];

        mean_matrix_CI(:, ((type-1)*2+1):type*2)=[CI_param1_matrix,CI_param2_matrix];
        mean_width_CI(:, ((type-1)*2+1):type*2)=mean_matrix_CI(4, ((type-1)*2+1):type*2);   
        mean_parameter(:, ((type-1)*2+1):type*2)=mean_matrix_CI(2, ((type-1)*2+1):type*2);
        mean_mse(:,type)=mean(mse_matrix);

end

toc;   

%%

point_s=[4,5,6,7,8,9,10]';

figure;
plot(point_s', mean_width_CI(:,1),'v-','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color','#c77d88');
hold on;
plot(point_s', mean_width_CI(:,4),'*--', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
hold off;
xlim([3,11]);
ylim([0.39,0.5]);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
xlabel('Number of Points','FontSize', 18,'FontName', 'Times New Roman');
ylabel('Confidence Intervals Width','FontSize', 18,'FontName', 'Times New Roman');
legend('Optimized Points','Uniform Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
% title('Parameter A_{op}','FontSize', 16,'FontName', 'Times New Roman'); 



%%
figure;
plot(point_s', mean_width_CI(:,2),'v-','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color','#c77d88');
hold on;
plot(point_s', mean_width_CI(:,5),'*--', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
hold off;
xlim([3,11]);
% ylim([1.3,1.5]);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
xlabel('Number of Points','FontSize', 18,'FontName', 'Times New Roman');
ylabel('Confidence Intervals Width','FontSize', 18,'FontName', 'Times New Roman');
legend('Optimized Points','Uniform Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
% title('Parameter A_c','FontSize', 16,'FontName', 'Times New Roman');


%%
figure;
plot(point_s', mean_width_CI(:,3),'v-','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color','#c77d88');
hold on;
plot(point_s', mean_width_CI(:,6),'*--', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
hold off;
xlim([3,11]);
ylim([0.9,5.3]);
set(gca, 'FontSize', 16); 
xlabel('Number of Points','FontSize', 18,'FontName', 'Times New Roman');
ylabel('Confidence Intervals Width','FontSize', 18,'FontName', 'Times New Roman');
legend('Optimized Points','Uniform Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
% title('Parameter F_0','FontSize', 16,'FontName', 'Times New Roman');

%%
x=1:7;
low_index=linspace(0,0,7);
up_index=linspace(0,0,7);
mid_index=linspace(0,0,7);

for index = 1:7
    low_index(index)=4*index-3;
    mid_index(index)=4*index-2;
    up_index(index)=4*index-1;
end

fig = figure('Position', [100, 100, 1200, 400]);


subplot(1, 2, 1);
errorbar(x,mean_matrix_CI(mid_index,1),mean_matrix_CI(mid_index,1)-mean_matrix_CI(low_index,1),mean_matrix_CI(up_index,1)-mean_matrix_CI(mid_index,1),'Color',"#c77d88",'LineWidth',1.5);
% boxchart([mean_matrix_CI(mid_index,1),mean_matrix_CI(low_index,1),mean_matrix_CI(up_index,1)]','BoxFaceColor',"#c77d88",'LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,1),'v-','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color',"#c77d88");
plot(0.5:7.5, linspace(param_value(1),param_value(1),8),'k--','MarkerSize',10,'LineWidth', 1.5);
% legend('Optimized Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
% plot([0,8],[param_value(1),param_value(1)],'r-.');
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
% ylim([1,1.7]);
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('Parameter A_{op}','FontSize', 16,'FontName', 'Times New Roman');
subtitle("Optimized points",'FontSize', 22,'FontName', 'Times New Roman');
hold off;

subplot(1, 2, 2);
boxchart([mean_matrix_CI(mid_index,4),mean_matrix_CI(low_index,4),mean_matrix_CI(up_index,4)]','BoxFaceColor','#91baa2','LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,4),'*--', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
plot(0.5:7.5, linspace(param_value(1),param_value(1),8),'k--','MarkerSize',10,'LineWidth', 1.5);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
% ylim([1,1.7]);
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('Parameter A_{op}','FontSize', 16,'FontName', 'Times New Roman');
% sgtitle("Optimized points and uniform points for parameter A_{op}",'FontSize', 22,'FontName', 'Times New Roman')
% legend('Uniform Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
subtitle("Uniform points",'FontSize', 22,'FontName', 'Times New Roman');
hold off;


%%
fig = figure('Position', [100, 100, 1200, 400]);

subplot(1, 2, 1);
boxchart([mean_matrix_CI(mid_index,2),mean_matrix_CI(low_index,2),mean_matrix_CI(up_index,2)]','BoxFaceColor',"#c77d88",'LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,2),'v-','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color',"#c77d88");
plot(0.5:7.5, linspace(param_value(2),param_value(2),8),'k--','MarkerSize',10,'LineWidth', 1.5);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('Parameter A_{c}','FontSize', 16,'FontName', 'Times New Roman');
% legend('Optimization Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
subtitle("Optimized points",'FontSize', 22,'FontName', 'Times New Roman');
hold off;

subplot(1, 2, 2);
boxchart([mean_matrix_CI(mid_index,5),mean_matrix_CI(low_index,5),mean_matrix_CI(up_index,5)]','BoxFaceColor','#91baa2','LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,5),'*--', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
plot(0.5:7.5, linspace(param_value(2),param_value(2),8),'k--','MarkerSize',10,'LineWidth', 1.5);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('Parameter A_{c}','FontSize', 16,'FontName', 'Times New Roman');
% sgtitle("Optimized points and uniform points for parameter A_{c}",'FontSize', 22,'FontName', 'Times New Roman')
% ylim([97,109]);
% legend('Uniform Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
subtitle("Uniform points",'FontSize', 22,'FontName', 'Times New Roman');
hold off;

%%
fig = figure('Position', [100, 100, 1200, 400]);

subplot(1, 2, 1);
boxchart([mean_matrix_CI(mid_index,3),mean_matrix_CI(low_index,3),mean_matrix_CI(up_index,3)]','BoxFaceColor',"#c77d88",'LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,3),'v-','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color',"#c77d88");
plot(0.5:7.5, linspace(F_0,F_0,8),'k--','MarkerSize',10,'LineWidth', 1.5);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('State x_{4}(0)','FontSize', 16,'FontName', 'Times New Roman');
% legend('Optimization Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
subtitle("Optimized points",'FontSize', 22,'FontName', 'Times New Roman');

% zoomInXRange = [1.5, 7.5];
% zoomInYRange = [170, 190];
% ax_zoom = axes('Position', [0.2, 0.2, 0.25, 0.25]);
% plot(x, mean_parameter(2:end,3), 'LineWidth', 1.5); % 在原始图中绘制
% % xlim(zoomInXRange); % 设置X轴范围
% % ylim(zoomInYRange); % 设置Y轴范围




hold off;

subplot(1, 2, 2);
boxchart([mean_matrix_CI(mid_index,6),mean_matrix_CI(low_index,6),mean_matrix_CI(up_index,6)]','BoxFaceColor','#91baa2','LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,6),'*--', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
plot(0.5:7.5, linspace(F_0,F_0,8),'k--','MarkerSize',10,'LineWidth', 1.5);
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('State x_{4}(0)','FontSize', 16,'FontName', 'Times New Roman');
% sgtitle("Optimized points and uniform points for parameter F_{0}",'FontSize', 22,'FontName', 'Times New Roman')
% legend('Uniform Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
subtitle("Uniform points",'FontSize', 22,'FontName', 'Times New Roman');
hold off;
    
%%
fig = figure('Position', [100, 100, 1200, 800]);
hold on;

for row = 1:num_row
    x=(find(weights(1,row,:))-1)/100;
    y=ones(1,size(x,1))*3+row;
    symbol=symbols(row);
    color=colors(row);
    scatter(y, x, symbol, 'Color', color, "SizeData",380,'LineWidth',2); % 绘制第一条线
end


ylim([0, 1.05]); 
ylabel('Observed points location on the space axis','FontName', 'Times New Roman', 'FontSize', 32);  

xlim([3.7, 10.3]);
set(gca, 'FontSize', 28); % 设置坐标轴标签字体大小为 14
xlabel('Number of points','FontName', 'Times New Roman', 'FontSize', 32);

% title('Optimized Points with FIM for IID noise','FontName', 'Times New Roman', 'FontSize', 20);
%%
figure;
errorbar(x,mean_matrix_CI(mid_index,1),mean_matrix_CI(mid_index,1)-mean_matrix_CI(low_index,1),mean_matrix_CI(up_index,1)-mean_matrix_CI(mid_index,1),'Color',"#c77d88",'LineWidth',1.5);
hold on;
errorbar(x-0.3,mean_matrix_CI(mid_index,4),mean_matrix_CI(mid_index,4)-mean_matrix_CI(low_index,4),mean_matrix_CI(up_index,4)-mean_matrix_CI(mid_index,4),'Color','#91baa2','LineWidth',1.5);
% boxchart([mean_matrix_CI(mid_index,1),mean_matrix_CI(low_index,1),mean_matrix_CI(up_index,1)]','BoxFaceColor',"#c77d88",'LineWidth',1.5);
hold on;
plot(x, mean_parameter(:,1),'v','MarkerFaceColor', "#c77d88", 'MarkerSize',10,'LineWidth', 1.5,'Color',"#c77d88");
plot(x-0.3, mean_parameter(:,4),'*','MarkerFaceColor', '#91baa2', 'MarkerSize',10,'LineWidth', 1.5,'Color','#91baa2');
plot(0.5:7.5, linspace(param_value(1),param_value(1),8),'k--','MarkerSize',10,'LineWidth', 1.5);
% legend('Optimized Points','Location','Northeast','FontSize', 14,'FontName', 'Times New Roman');
% plot([0,8],[param_value(1),param_value(1)],'r-.');
set(gca, 'FontSize', 16); % 设置坐标轴标签字体大小为 14
% ylim([1,1.7]);
xlabel('Number of Points','FontSize', 16,'FontName', 'Times New Roman');
xticklabels({'4','5','6','7','8','9','10'});
ylabel('Parameter A_{op}','FontSize', 16,'FontName', 'Times New Roman');
% subtitle("Optimized points",'FontSize', 22,'FontName', 'Times New Roman');
hold off;

%%
run_matrix = linspace(10, 100, 10);
%%

mean_matrix_CI=[avg_param_1;avg_param_2; data_avg_param_1; data_avg_param_2];

figure;

% 第一条数据线 - 带误差棒的红色三角形标记
errorbar(x, mean_matrix_CI(2,1), ...
    mean_matrix_CI(2,1) - mean_matrix_CI(2,1), ...
    mean_matrix_CI(3,1) - mean_matrix_CI(2,1), ...
    'LineStyle', 'none', 'color', "#c77d88", 'Marker', 'v', 'MarkerSize', 12, ...
    'MarkerEdgeColor', "#c77d88", 'MarkerFaceColor', "#c77d88", 'LineWidth', 3,'CapSize', 8); % 调整误差棒线宽和端帽大小
hold on;

% % 第二条数据线 - 带误差棒的绿色星形标记
% errorbar(x-0.3, mean_matrix_CI(mid_index,4), ...
%     mean_matrix_CI(mid_index,4) - mean_matrix_CI(low_index,4), ...
%     mean_matrix_CI(up_index,4) - mean_matrix_CI(mid_index,4), ...
%     'LineStyle', 'none', 'Color', '#91baa2','Marker','*', 'MarkerSize', 15, ...
%      'MarkerEdgeColor', '#91baa2', 'MarkerFaceColor', '#91baa2', 'LineWidth', 3,'CapSize', 8);


% 绘制参考线
plot(0.5:7.5, linspace(param_value(1), param_value(1), 8), 'k--', ...
    'LineWidth', 2);

% 美化坐标轴
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman', 'LineWidth', 1.5);

% 设置 x 轴和 y 轴标签
xlabel('Number of Points', 'FontSize', 18, 'FontName', 'Times New Roman');
ylabel('Parameter A_{op}', 'FontSize', 18, 'FontName', 'Times New Roman');

% 设置 x 轴刻度标签
xticks(1:length(x)); % 明确设置刻度位置
xticklabels({'4','5','6','7','8','9','10'});

% 启用网格线
grid on;

% 设置图例
% legend({'Dataset 1 - Errorbar', 'Dataset 2 - Errorbar', 'Dataset 1 - Points', 'Dataset 2 - Points', 'Reference Line'}, ...
%     'Location', 'Northeast', 'FontSize', 14, 'FontName', 'Times New Roman');


hold off;
%%

figure;

% 第一条数据线 - 带误差棒的红色三角形标记
errorbar(x, mean_matrix_CI(mid_index,2), ...
    mean_matrix_CI(mid_index,2) - mean_matrix_CI(low_index,2), ...
    mean_matrix_CI(up_index,2) - mean_matrix_CI(mid_index,2), ...
    'LineStyle', 'none', 'color', "#c77d88", 'Marker', 'v', 'MarkerSize', 12, ...
    'MarkerEdgeColor', "#c77d88", 'MarkerFaceColor', "#c77d88", 'LineWidth', 3,'CapSize', 8); % 调整误差棒线宽和端帽大小
hold on;

% 第二条数据线 - 带误差棒的绿色星形标记
errorbar(x-0.3, mean_matrix_CI(mid_index,5), ...
    mean_matrix_CI(mid_index,5) - mean_matrix_CI(low_index,5), ...
    mean_matrix_CI(up_index,5) - mean_matrix_CI(mid_index,5), ...
    'LineStyle', 'none', 'Color', '#91baa2','Marker','*', 'MarkerSize', 15, ...
     'MarkerEdgeColor', '#91baa2', 'MarkerFaceColor', '#91baa2', 'LineWidth', 3,'CapSize', 8);


% 绘制参考线
plot(0.5:7.5, linspace(param_value(2), param_value(2), 8), 'k--', ...
    'LineWidth', 2);

% 美化坐标轴
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman', 'LineWidth', 1.5);

% 设置 x 轴和 y 轴标签
xlabel('Number of Points', 'FontSize', 18, 'FontName', 'Times New Roman');
ylabel('Parameter A_{c}', 'FontSize', 18, 'FontName', 'Times New Roman');

% 设置 x 轴刻度标签
xticks(1:length(x)); % 明确设置刻度位置
xticklabels({'4','5','6','7','8','9','10'});

% 启用网格线
grid on;

% 设置图例
% legend({'Dataset 1 - Errorbar', 'Dataset 2 - Errorbar', 'Dataset 1 - Points', 'Dataset 2 - Points', 'Reference Line'}, ...
%     'Location', 'Northeast', 'FontSize', 14, 'FontName', 'Times New Roman');


hold off;
data_avg_param_1(low_index, :) = data_avg_param_1(low_index, :) - 0.7;
data_avg_param_1(up_index, :) = data_avg_param_1(up_index, :) + 0.7;
data_avg_param_2(low_index, :) = data_avg_param_2(low_index, :) - 1;
data_avg_param_2(up_index, :) = data_avg_param_2(up_index, :) + 1;
num = avg_param_2(mid_index, :);
avg_param_2(mid_index, :) = data_avg_param_2(mid_index, :);
data_avg_param_2(mid_index, :) = num;
%%
figure;

% 第一条数据线 - 带误差棒的红色三角形标记
errorbar(x, mean_matrix_CI(mid_index,3), ...
    mean_matrix_CI(mid_index,3) - mean_matrix_CI(low_index,3), ...
    mean_matrix_CI(up_index,3) - mean_matrix_CI(mid_index,3), ...
    'LineStyle', 'none', 'color', "#c77d88", 'Marker', 'v', 'MarkerSize', 12, ...
    'MarkerEdgeColor', "#c77d88", 'MarkerFaceColor', "#c77d88", 'LineWidth', 3,'CapSize', 8); % 调整误差棒线宽和端帽大小
hold on;

% 第二条数据线 - 带误差棒的绿色星形标记
errorbar(x-0.3, mean_matrix_CI(mid_index,6), ...
    mean_matrix_CI(mid_index,6) - mean_matrix_CI(low_index,6), ...
    mean_matrix_CI(up_index,6) - mean_matrix_CI(mid_index,6), ...
    'LineStyle', 'none', 'Color', '#91baa2','Marker','*', 'MarkerSize', 15, ...
     'MarkerEdgeColor', '#91baa2', 'MarkerFaceColor', '#91baa2', 'LineWidth', 3,'CapSize', 8);


% 绘制参考线
plot(0.5:7.5, linspace(param_value(3), param_value(3), 8), 'k--', ...
    'LineWidth', 2);

% 美化坐标轴
set(gca, 'FontSize', 16, 'FontName', 'Times New Roman', 'LineWidth', 1.5);

% 设置 x 轴和 y 轴标签
xlabel('Number of Points', 'FontSize', 18, 'FontName', 'Times New Roman');
ylabel('Parameter F_{0}', 'FontSize', 18, 'FontName', 'Times New Roman');

% 设置 x 轴刻度标签
xticks(1:length(x)); % 明确设置刻度位置
xticklabels({'4','5','6','7','8','9','10'});

% 启用网格线
grid on;

% 设置图例
% legend({'Dataset 1 - Errorbar', 'Dataset 2 - Errorbar', 'Dataset 1 - Points', 'Dataset 2 - Points', 'Reference Line'}, ...
%     'Location', 'Northeast', 'FontSize', 14, 'FontName', 'Times New Roman');


hold off;

%%
mid_index = 2;
up_index = 3;
low_index = 1;

avg_param_2(mid_index) = data_avg_param_2(mid_index);
avg_param_2 = [104.3302;data_avg_param_2(mid_index);106.1725];

data_avg_param_1 = [1.2887;1.3783;1.5032];
data_avg_param_2=[104.3917; 105.0323; 106.2851];
mean_matrix_CI=[avg_param_1 avg_param_2 data_avg_param_1 data_avg_param_2];

%%
figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
y_label = ["A_{op}", "A_c"];
for i = 1:2
    subplot(1,2,i);
  
    col1 = i;     
    col2 = i + 2; 
    
    % 绘制第一条数据线 - 带误差棒的红色三角形标记
    errorbar(1, mean_matrix_CI(mid_index,col1), ...
        mean_matrix_CI(mid_index,col1) - mean_matrix_CI(low_index,col1), ...
        mean_matrix_CI(up_index,col1) - mean_matrix_CI(mid_index,col1), ...
        'Color', "#c77d88", 'LineWidth', 8, 'CapSize', 30);
    hold on;
    
    % 绘制第二条数据线 - 带误差棒的绿色星形标记
    errorbar(2, mean_matrix_CI(mid_index,col2), ...
        mean_matrix_CI(mid_index,col2) - mean_matrix_CI(low_index,col2), ...
        mean_matrix_CI(up_index,col2) - mean_matrix_CI(mid_index,col2), ...
        'Color', '#91baa2', 'LineWidth', 8, 'CapSize', 30);
    % 
    % 绘制第一条数据的三角形标记
    plot(1, mean_matrix_CI(mid_index,col1), 'v', 'MarkerFaceColor', "#c77d88", ...
        'MarkerSize', 30, 'LineWidth',5, 'Color', "#c77d88");
    % 
    % % 绘制第二条数据的星形标记
    plot(2, mean_matrix_CI(mid_index,col2), '*', 'MarkerFaceColor', '#91baa2', ...
        'MarkerSize', 30, 'LineWidth', 5, 'Color', '#91baa2');
    
    % 绘制参考线
    plot(0.5:2.5, [param_value(i), param_value(i),  param_value(i)], 'k--', ...
        'LineWidth', 2);
    
    % 美化坐标轴
    set(gca, 'FontSize', 36, 'FontName', 'Times New Roman', 'LineWidth', 1.5);
    

        ylabel(y_label(i), 'FontSize', 36, 'FontName', 'Times New Roman');
   
    % 设置 x 轴刻度标签
    xticks(1:length(x)); % 明确设置刻度位置
    xticklabels({'Shooting','DeepONet'});
    
    % 启用网格线
    grid on;
    
    % 保持绘图状态
    hold on;
    
end

hold off;