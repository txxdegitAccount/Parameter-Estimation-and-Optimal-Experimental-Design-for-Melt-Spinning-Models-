%% diagnose_jacobian_convergence.m
% Diagnostic for the melt-spinning reduced 3-by-2 Jacobian.
% Required in the SAME directory: steady_solution.m
% Uses the TWO already-polished shooting values printed by the previous run;
% does NOT perform fzero, does NOT redo the 20304 samples.
%
% Why a new diagnostic?
% The previous test changed the output-grid size and ode45 tolerances
% simultaneously, so its ||dJ|| values cannot identify the source of error.
%
% Outputs:
%   jacobian_diagnostic_summary.csv
%   jacobian_diagnostic_results.mat
%
% NOTE: Differences between numerical results are NOT rigorous error bounds.

clear; clc;
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
assert(exist('steady_solution','file') == 2, ...
    'Please put steady_solution.m in the same folder as this script.');

% Points and polished F values FROM THE USER'S PREVIOUS MATLAB LOG.
theta_list = [1.96052631579, 90; ...
              1.50697406141, 90.0022904285];
F_list = [277.70514930754; 277.70568418116];
point_names = {'BoundaryMinimum','InteriorMinimum'};

W=0.009; T_0=298; A_0=pi*(0.22*0.1)^2/4;
T_a_abs=15; u_a_abs=30;
rho_0 = 1.356-5e-4*T_0;
u_0 = W/(A_0*rho_0);
T_a = T_a_abs/T_0;
u_a = u_a_abs/u_0;
base_param = [W,T_0,A_0,T_a,u_a,2000,u_0];
target=3730;

% Baseline settings, SAME for all three isolated comparisons.
N0=2000; rt0=1e-9; at0=1e-10;
hx0=1e-8; ht0=1e-7;

% Each column specifies [intervals, RelTol, AbsTol, stateFD, paramRelativeFD]
cases = [
  2000, 1e-8,  1e-9,  hx0,   ht0;   % Tolerance only
  2000, 1e-9,  1e-10, hx0,   ht0;
  2000, 1e-10, 1e-11, hx0,   ht0;
  1000, rt0,   at0,   hx0,   ht0;   % Output-grid only
  2000, rt0,   at0,   hx0,   ht0;
  4000, rt0,   at0,   hx0,   ht0;
  N0,   rt0,   at0,   1e-7,  1e-6;  % Finite-difference steps only
  N0,   rt0,   at0,   1e-8,  1e-7;
  N0,   rt0,   at0,   1e-9,  1e-8];
case_group = [repmat({'ToleranceOnly'},3,1); ...
              repmat({'OutputGridOnly'},3,1); ...
              repmat({'FDStepsOnly'},3,1)];

results = struct('Point',{},'Test',{},'Intervals',{}, ...
    'RelTol',{},'AbsTol',{},'StateFD',{},'ParamFD',{}, ...
    'BoundaryResidual',{},'dRdF',{},'Rank',{}, ...
    'SigmaMin',{},'NormJ',{},'DeltaFromBaseline',{}, ...
    'DeltaOverSigmaMin',{},'Success',{},'ErrorMessage',{});
Jacobian = cell(2,size(cases,1));

for ip=1:size(theta_list,1)
    theta = theta_list(ip,:);
    F = F_list(ip);
    fprintf('\n========== %s ==========',point_names{ip});
    fprintf('\nA_op=%.12g, A_c=%.12g, fixed F=%.14g\n',theta(1),theta(2),F);

    % Check whether the physical temperature trajectory crosses thresholds
    % used by the piecewise laws in steady_solution.m.
    param=base_param;
    opts=odeset('RelTol',rt0,'AbsTol',at0);
    try
        [zt,xx]=ode45(@steady_solution,linspace(0,1,1501), ...
            [1;1;1;F;0;0],opts,param,theta);
        require_endpoint(zt,'Temperature trajectory');
        T_c = T_0 * xx(:,3);
        crossing267 = any((T_c(1:end-1)-267).*(T_c(2:end)-267)<=0);
        crossing67  = any((T_c(1:end-1)-67).*(T_c(2:end)-67)<=0);
        fprintf('Temperature range: %.6f to %.6f C; crosses 267C: %d; crosses 67C: %d\n', ...
            min(T_c),max(T_c),crossing267,crossing67);
        if crossing267 || crossing67
            fprintf('CAUTION: state dynamics are piecewise at threshold; sensitivities may be affected by switching events.\n');
        end
    catch ME
        fprintf('Temperature trajectory test failed: %s\n',ME.message);
    end

    Jbase=nan(3,2); svbase=nan;
    try
        [Jbase,~]=get_J(theta,F,param,opts,hx0,ht0);
        svbase=min(svd(Jbase));
        fprintf('Reference (N=2000, Rt=1e-9, At=1e-10): minSV=%.12e\n',svbase);
    catch ME
        fprintf('Reference calculation failed: %s\n',ME.message);
    end

    for k=1:size(cases,1)
        v=cases(k,:);
        N=v(1); rt=v(2); at=v(3); hx=v(4); ht=v(5);
        param=base_param; param(6)=N;
        opts=odeset('RelTol',rt,'AbsTol',at);
        good=false; errmsg=''; dr=nan; rp=nan; r=nan;
        smin=nan; jnorm=nan; J=nan(3,2); jr=nan;
        try
            [J,rp]=get_J(theta,F,param,opts,hx,ht);
            r=get_residual(F,theta,param,target,opts);
            singular_vals=svd(J);
            smin=singular_vals(end);
            jnorm=norm(J,2);
            jr=rank(J,1e-8);
            if all(isfinite(J(:))) && all(isfinite(singular_vals))
                good=true;
                if all(isfinite(Jbase(:)))
                    dr=norm(J-Jbase,2);
                end
            end
        catch ME
            errmsg=ME.message;
        end
        ratio=nan;
        if good && isfinite(svbase) && svbase>0
            ratio=dr/svbase;
        end
        fprintf('%-16s N=%-4d Rt=%-8.1e At=%-8.1e hx=%-8.1e ht=%-8.1e |r|=%-10.3e |dJ|=%-10.3e minSV=%-10.4e\n', ...
           case_group{k},N,rt,at,hx,ht,abs(r),dr,smin);
        if ~good, fprintf('    ERROR: %s\n',errmsg); end
        Jacobian{ip,k}=J;
        results(end+1)=struct( ...
            'Point',point_names{ip},'Test',case_group{k}, ...
            'Intervals',N,'RelTol',rt,'AbsTol',at, ...
            'StateFD',hx,'ParamFD',ht, ...
            'BoundaryResidual',r,'dRdF',rp,'Rank',jr, ...
            'SigmaMin',smin,'NormJ',jnorm, ...
            'DeltaFromBaseline',dr,'DeltaOverSigmaMin',ratio, ...
            'Success',good,'ErrorMessage',errmsg); %#ok<SAGROW>
    end
end

summary=struct2table(results);
outcsv=fullfile(script_dir,'jacobian_diagnostic_summary.csv');
outmat=fullfile(script_dir,'jacobian_diagnostic_results.mat');
writetable(summary,outcsv);
save(outmat,'summary','Jacobian','theta_list','F_list','cases');
fprintf('\nSaved: %s\nSaved: %s\n',outcsv,outmat);
fprintf('Note: These are empirical comparison errors, not certified error bounds.\n');

%% Local helpers
function r=get_residual(F,theta,param,target,opts)
    [t,X]=ode45(@steady_solution,[0,1],[1;1;1;F;0;0],opts,param,theta);
    require_endpoint(t,'Shooting residual');
    r=X(end,2)*param(7)*60/100-target;
end

function [J,rp]=get_J(theta,F,param,odeopts,hx,ht)
    z=linspace(0,1,param(6)+1);
    y0=[[1;1;1;F;0;0]; zeros(12,1); [0;0;0;1;0;0]];
    rhs=@(t,y) augmented_rhs(t,y,theta,param,hx,ht);
    [t,Y]=ode45(rhs,z,y0,odeopts);
    require_endpoint(t,'Augmented sensitivity solution');
    ye=Y(end,:)';
    d1=ye(7:12);d2=ye(13:18);ds=ye(19:24);
    rp=ds(2);
    if ~isfinite(rp) || abs(rp)<=1e-8
        error('Implicit-function denominator dr/dF = %.5g is too small.',rp);
    end
    v1=d1-ds*(d1(2)/rp);
    v2=d2-ds*(d2(2)/rp);
    J=[v1(1),v2(1);v1(5),v2(5);v1(6),v2(6)];
    if ~all(isfinite(J(:)))
        error('Nonfinite Jacobian.');
    end
end

function dy=augmented_rhs(~,y,theta,param,hx,ht)
    x=y(1:6);s1=y(7:12);s2=y(13:18);sF=y(19:24);
    f=steady_solution([],x,param,theta);f=f(:);
    A=zeros(6,6);
    for i=1:6
        xp=x;xm=x;
        xp(i)=xp(i)+hx;xm(i)=xm(i)-hx;
        fp=steady_solution([],xp,param,theta);
        fm=steady_solution([],xm,param,theta);
        A(:,i)=(fp(:)-fm(:))/(2*hx);
    end
    B=zeros(6,2);
    for j=1:2
        h=ht*(abs(theta(j))+1e-10);
        pp=theta;pm=theta;
        pp(j)=pp(j)+h;pm(j)=pm(j)-h;
        fp=steady_solution([],x,param,pp);
        fm=steady_solution([],x,param,pm);
        B(:,j)=(fp(:)-fm(:))/(2*h);
    end
    dy=[f;A*s1+B(:,1);A*s2+B(:,2);A*sF];
end

function require_endpoint(t,label)
    if isempty(t) || abs(t(end)-1)>1e-9
        if isempty(t), lastt=nan; else, lastt=t(end);end
        error('%s stopped before z=1 (last z=%.12g).',label,lastt);
    end
end
