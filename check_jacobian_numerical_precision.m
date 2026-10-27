clear; clc;
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

result_path = fullfile(script_dir,'sampling_refinement_interior_results.mat');
assert(isfile(result_path), 'Cannot find: %s', result_path);
assert(exist('steady_solution','file')==2, ...
    'steady_solution.m must be accessible on the MATLAB path.');
S = load(result_path,'theta_all','sigma_all','ok_all','boundary_mask');

P = S.theta_all;                  % rows: [A_op, A_c]
sv = S.sigma_all(:);
valid = logical(S.ok_all(:)) & isfinite(sv);
boundary = logical(S.boundary_mask(:));
assert(size(P,2)==2 && size(P,1)==numel(sv), ...
    'Parameter points and singular value arrays do not match.');
assert(any(valid & ~boundary), 'No valid interior samples in MAT file.');

valid_indices = find(valid);
[~,i_min] = min(sv(valid_indices));
k_global = valid_indices(i_min);
interior_indices = find(valid & ~boundary);
[~,i_min] = min(sv(interior_indices));
k_interior = interior_indices(i_min);
selected_rows = unique([k_global; k_interior], 'stable');

W = 0.009;
T_0 = 298;
A_0 = pi*(0.22*0.1)^2/4;
T_a_abs = 15;
u_a_abs = 30;
rho_0 = 1.356 - 5e-4*T_0;
u_0 = W/(A_0*rho_0);
T_a = T_a_abs/T_0;
u_a = u_a_abs/u_0;

param_base = [W, T_0, A_0, T_a, u_a, 1000, u_0];
u_target = 3730;
F_init = 270;


mesh_intervals = [1000; 2000; 4000];
rel_tols       = [1e-8; 1e-9; 1e-10];
abs_tols       = [1e-9; 1e-10; 1e-11];
finite_dx      = 1e-8;    
finite_dtheta  = 1e-7;    
n_levels = numel(mesh_intervals);

R = struct('PointType',{}, 'SampleIndex',{}, 'Aop',{}, 'Ac',{}, ...
    'Intervals',{}, 'RelTol',{}, 'AbsTol',{}, 'ShootResidual',{}, ...
    'SigmaMin',{}, 'SigmaMax',{}, 'Rank',{}, 'NormJ',{}, ...
    'DeltaJ',{}, 'RelDeltaJ',{}, 'DeltaSigmaMin',{}, ...
    'DeltaJoverMinSigma',{});
J_by_point = cell(numel(selected_rows), n_levels);

for ip = 1:numel(selected_rows)
    index = selected_rows(ip);
    theta = P(index,:);
    if boundary(index)
        type = 'GlobalMinimumBoundary';
    else
        type = 'InteriorMinimum';
    end
    fprintf('\n==== %s (saved index %d) ====\n',type,index);
    fprintf('[A_op,A_c] = [%.12g, %.12g]; saved min sigma = %.12e\n', ...
        theta(1),theta(2),sv(index));

    shoot_opts = odeset('RelTol',1e-11,'AbsTol',1e-12);
    [F, residual_best] = solve_shooting(theta,param_base, ...
        u_target,F_init,shoot_opts);
    fprintf('Polished shooting variable F = %.14g; residual = %.6e\n', ...
        F,residual_best);
    if ~isfinite(residual_best) || abs(residual_best)>1e-3
        error('Shooting boundary residual is too large; inspect the solver.');
    end

    prev_J = [];
    prev_sigma = NaN;
    for lev = 1:n_levels
        N = mesh_intervals(lev);
        numeric_param = param_base;
        numeric_param(6) = N;
        opt = odeset('RelTol',rel_tols(lev),'AbsTol',abs_tols(lev));
        [J, rp] = reduced_jacobian(theta,F,numeric_param,opt, ...
            finite_dx,finite_dtheta);
        assert(all(isfinite(J(:))), 'Jacobian has NaN/Inf at N=%d',N);
        assert(isfinite(rp) && abs(rp)>1e-8, ...
            'Implicit function condition dr/dF may fail at N=%d',N);

        svals = svd(J);
        s_min = svals(end);
        s_max = svals(1);
        residual = shooting_residual(F,theta,numeric_param, ...
            u_target,opt);
        deltaJ = NaN; relDeltaJ = NaN; deltaSigma=NaN; ratio=NaN;
        if ~isempty(prev_J)
            deltaJ = norm(J-prev_J,2);
            relDeltaJ = deltaJ/max(norm(J,2),eps);
            deltaSigma = abs(s_min-prev_sigma);
            ratio = deltaJ/max(s_min,eps);
        end
        fprintf(['N=%-4d  RelTol=%-7.1e AbsTol=%-7.1e', ...
            ' minSV=%-13.7e rank=%d |r|=%.3e', ...
            ' ||dJ||2=%.5e\n'],N,rel_tols(lev),abs_tols(lev), ...
            s_min,rank(J,1e-8),abs(residual),deltaJ);
        fprintf('J =\n'); disp(J);

        entry.PointType = type;
        entry.SampleIndex = index;
        entry.Aop = theta(1);
        entry.Ac = theta(2);
        entry.Intervals = N;
        entry.RelTol = rel_tols(lev);
        entry.AbsTol = abs_tols(lev);
        entry.ShootResidual = residual;
        entry.SigmaMin = s_min;
        entry.SigmaMax = s_max;
        entry.Rank = rank(J,1e-8);
        entry.NormJ = norm(J,2);
        entry.DeltaJ = deltaJ;
        entry.RelDeltaJ = relDeltaJ;
        entry.DeltaSigmaMin = deltaSigma;
        entry.DeltaJoverMinSigma = ratio;
        R(end+1) = entry; 

        J_by_point{ip,lev} = J;
        prev_J = J;
        prev_sigma = s_min;
    end

    numeric_param = param_base;
    numeric_param(6)=4000;
    opt = odeset('RelTol',1e-10,'AbsTol',1e-11);
    [J_step,~] = reduced_jacobian(theta,F,numeric_param,opt, ...
        1e-7,5e-7);
    J_finest = J_by_point{ip,end};
    fprintf(['Finite-difference step check: ', ...
        '||J(h_x=1e-7,h_theta=5e-7)-J_baseline||2 = %.8e\n'], ...
        norm(J_step-J_finest,2));
    fd_results(ip).PointType=type; %#ok<SAGROW>
    fd_results(ip).theta=theta;
    fd_results(ip).JacobianModifiedSteps=J_step;
    fd_results(ip).StepDifference=norm(J_step-J_finest,2);
end

summary = struct2table(R);
res_dir = script_dir;
csv_path = fullfile(res_dir,'jacobian_precision_summary.csv');
mat_path = fullfile(res_dir,'jacobian_precision_results.mat');
writetable(summary,csv_path);
save(mat_path,'summary','J_by_point','fd_results','selected_rows', ...
    'mesh_intervals','rel_tols','abs_tols','-v7.3');

fprintf('\n============= FINAL SUMMARY =============\n');
disp(summary(:,{'PointType','Intervals','SigmaMin','Rank', ...
    'DeltaJ','RelDeltaJ','DeltaJoverMinSigma'}));
fprintf('Saved CSV: %s\nSaved MAT: %s\n',csv_path,mat_path);
fprintf(['Note: successive solver differences are empirical stability ', ...
    'indicators, NOT certified truncation-error bounds.\n']);

function [F, residual] = solve_shooting(theta, numeric_param, ...
    target, F_init, ode_opts)
    objective = @(x) shooting_residual(x,theta,numeric_param, ...
        target,ode_opts).^2;
    options = optimoptions('fmincon', ...
        'MaxIterations',6000,'Display','off','UseParallel',false);
    [F,~] = fmincon(objective,F_init,[],[],[],[],[],[], ...
        @(x) constraint_shoot(x,theta,numeric_param,target,ode_opts),options);
    residual = shooting_residual(F,theta,numeric_param,target,ode_opts);
    
    try
        F_candidate = fzero(@(x) shooting_residual(x,theta,numeric_param, ...
            target,ode_opts), F, optimset('Display','off','TolX',1e-10));
        r_candidate = shooting_residual(F_candidate,theta,numeric_param, ...
            target,ode_opts);
        if isfinite(r_candidate) && abs(r_candidate)<abs(residual)
            F = F_candidate;
            residual = r_candidate;
        end
    catch ME
        fprintf('fzero polishing unavailable: %s\n',ME.message);
    end
end

function [c,ceq] = constraint_shoot(F,theta,numeric_param,target,ode_opts)
    r = shooting_residual(F,theta,numeric_param,target,ode_opts);
    c = r.^2-1e-6;
    ceq = [];
end

function r = shooting_residual(F,theta,numeric_param,target,ode_opts)
    [~,X] = ode45(@steady_solution,[0 1],[1 1 1 F 0 0], ...
        ode_opts,numeric_param,theta);
    u0 = numeric_param(7);
    r = X(end,2)*u0*60/100-target;
end

function [J,rp] = reduced_jacobian(theta,F,numeric_param,ode_opts,fd_x,fd_theta)
    z_grid=linspace(0,1,numeric_param(6)+1);
    y0 = [[1;1;1;F;0;0]; zeros(12,1); [0;0;0;1;0;0]];
    rhs=@(z,y) variational_rhs(z,y,theta,numeric_param,fd_x,fd_theta);
    [~,Y]=ode45(rhs,z_grid,y0,ode_opts);
    ye=Y(end,:)';
    d1 = ye(7:12);
    d2 = ye(13:18);
    ds = ye(19:24);
    rp=ds(2);
    if ~isfinite(rp) || abs(rp)<=1e-8
        J=nan(3,2);
        return;
    end
    dF1=-d1(2)/rp;
    dF2=-d2(2)/rp;
    v1=d1+ds*dF1;
    v2=d2+ds*dF2;
    J=[v1(1),v2(1);v1(5),v2(5);v1(6),v2(6)];
end

function dydz = variational_rhs(~,y,theta,numeric_param,fd_x,fd_theta)
    x = y(1:6);
    s1 = y(7:12);
    s2 = y(13:18);
    sF = y(19:24);
    f = steady_solution([],x,numeric_param,theta);
    f=f(:);
    A=zeros(6,6);
    for i=1:6
        xp=x; xm=x;
        xp(i)=xp(i)+fd_x;
        xm(i)=xm(i)-fd_x;
        fp=steady_solution([],xp,numeric_param,theta);
        fm=steady_solution([],xm,numeric_param,theta);
        A(:,i)=(fp(:)-fm(:))/(2*fd_x);
    end
    b=zeros(6,2);
    for j=1:2
        hj=fd_theta*(abs(theta(j))+1e-10);
        if hj==0, hj=fd_theta; end
        pp=theta; pm=theta;
        pp(j)=pp(j)+hj;
        pm(j)=pm(j)-hj;
        fp=steady_solution([],x,numeric_param,pp);
        fm=steady_solution([],x,numeric_param,pm);
        b(:,j)=(fp(:)-fm(:))/(2*hj);
    end
    dydz=[f;A*s1+b(:,1);A*s2+b(:,2);A*sF];
end
