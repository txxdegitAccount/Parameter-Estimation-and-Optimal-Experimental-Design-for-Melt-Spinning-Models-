function dcoeff = steady_solution_fast(~, coeff, numeric_param, param_value)
%STEADY_SOLUTION_FAST  与原 steady_solution.m 数学模型一致的加速版本。
%
% 主要优化：
% 1) 去掉每次 RHS 调用时重复创建的匿名函数；
% 2) 对只依赖 numeric_param 的常数使用 worker-local persistent 缓存；
% 3) 保持原 6 个状态方程和参数 A_op、A_c 的定义不变。
%
% 注意：parallel process pool 中每个 worker 都有自己的 persistent 缓存，
% 不会在不同 worker 之间共享可变状态。

    % ---------------------------------------------------------
    % 仅当 numeric_param 改变时重新计算常数
    % ---------------------------------------------------------
    key = numeric_param([2 3 4 5 7]);

    persistent last_key ...
               T0 A0 Ta ua u0 ...
               rho_a mu_a mu0 ...
               St_pref drag_pref q_pref inert_pref Fder_pref visc_pref

    if isempty(last_key) || ~isequal(last_key, key)

        T0 = numeric_param(2);
        A0 = numeric_param(3);
        Ta = numeric_param(4);
        ua = numeric_param(5);
        u0 = numeric_param(7);

        Kd = 0.37;
        IV = 0.67;

        L = 150;
        G = 1e9;

        rho_a = 0.351/(Ta + 273);
        mu_a  = 1.446e-5*(Ta + 273)^(1.5)/(Ta + 386.9);

        n_drag = 0.61;

        mu0 = 3*(IV)^(5.15) * ...
              exp(2.303*(3280/(T0+273) - 1.54));

        % S_t = St_pref/(rho*C_p)
        St_pref = 2*sqrt(pi)*0.473e-4*L / ...
                  (u0^(0.667)*A0^(0.833));

        % 空气阻力项中与状态无关的部分
        drag_pref = Kd * ...
                    rho_a^(1-n_drag) * ...
                    mu_a^(n_drag) * ...
                    2^(-n_drag) * ...
                    pi^((1+n_drag)/2) * ...
                    A0^((1-n_drag)/2) * ...
                    u0^(2-n_drag);

        q_pref = mu0*u0/(L*G);
        inert_pref = u0^2/G;
        Fder_pref = L^2/(A0*mu0*u0);

        visc_pref = 2.303*3280/(T0+273);

        last_key = key;
    end

    % ---------------------------------------------------------
    % 当前状态与待研究参数
    % ---------------------------------------------------------
    A     = coeff(1);
    u     = coeff(2);
    T     = coeff(3);
    F     = coeff(4);
    Theta = coeff(5);
    n_bir = coeff(6);

    A_op = param_value(1);
    A_c  = param_value(2);

    % ---------------------------------------------------------
    % 固定模型常数
    % ---------------------------------------------------------
    L = 150;
    g = 980;
    G = 1e9;

    n_drag = 0.61;
    n0 = 0.275;

    d     = 64;
    T1    = 267;
    Tg    = 67;
    Kmax  = 0.016;
    Tmax  = 190;

    % ---------------------------------------------------------
    % 当前状态下的物性量
    % ---------------------------------------------------------
    rho = 1.356 - 5e-4*T0*T;
    Cp  = 0.3 + 6e-4*T0*T;

    mu_ratio = exp( ...
        visc_pref*(1-T)/(T + 273/T0) ) * ...
        (1 + 99*Theta);

    rel = mu0*mu_ratio/G;

    active = double(T*T0 < T1 && T*T0 > Tg);

    K = Kmax * exp( ...
        -4*log(2)*((T*T0 - Tmax)/d)^2 ...
        + A_c*q_pref*(F/A) ) * active;

    St = St_pref/(rho*Cp);

    drag = drag_pref * ...
           A^((1-n_drag)/2) * ...
           u^(2-n_drag);

    % ---------------------------------------------------------
    % 六个状态方程
    % ---------------------------------------------------------
    dcoeff = zeros(6,1);

    dcoeff(3) = -St * ...
                A^(-0.833) * ...
                (u^2 + 64*ua^2)^(0.167) * ...
                (T-Ta)/u;

    denom = 1 ...
            - q_pref*u*F ...
            - rho*inert_pref*u^2;

    dcoeff(2) = ( ...
        F*u/mu_ratio ...
        + (L*u^2/(A0*G)) * ...
          (-rho*g*A0*A + drag) ...
        ) / denom;

    dcoeff(1) = -dcoeff(2)*A/u;

    dcoeff(4) = Fder_pref * ( ...
        rho*A0*u0^2*A*u*dcoeff(2)/L ...
        - rho*g*A0*A ...
        + drag );

    dcoeff(5) = K*L*(1-Theta)/(u*u0);

    dcoeff(6) = ( ...
        (A_op/(u*n0))*dcoeff(2) ...
        - n_bir*L/(u0*u*rel) ...
        ) * active;
end
