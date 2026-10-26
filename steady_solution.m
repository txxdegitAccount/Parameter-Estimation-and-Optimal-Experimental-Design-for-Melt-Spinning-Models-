function dcoeff = steady_solution(~,coeff,numeric_param,param_value)
 % UCM model

    % numeric_param=[W,T_0,A_0,T_a,u_a,F_0]
    W=numeric_param(1);
    T_0=numeric_param(2);
    A_0=numeric_param(3);
    T_a=numeric_param(4);
    u_a=numeric_param(5);

    % param_value=[A_op A_c]
    A_op=param_value(1);
    A_c=param_value(2);
    K_d=0.37;
    
    dcoeff=zeros(6,1);
    
    IV=0.67; % characteristic viscosity 
    rho_0=1.356-5*10^(-4)*T_0; % g/cm3
    u_0=W/(A_0*rho_0); % cm/s 

    L=150; % cm
    g=980; % cm/s2
    G=10^9; % dyns/cm2

    rho_a=0.351/(T_a+273); % g/cm3
    miu_a=1.446*10^(-5)*(T_a+273)^(1.5)/(T_a+386.9);
    n=0.61;

    miu_0=3*(IV)^(5.15)*exp(2.303*(3280/(T_0+273)-1.54)); % g/(cm*s)
    miu=@(T,Theta)exp(2.303*3280*(1-T)/((T+273/T_0)*(T_0+273)))*(1+99*Theta);

    d=64; %  Kinetic half-width value (°C)
    T_1=267; % melt Temperature
    T_g=67; % Glass Transition Temperature
    K_max=0.016; % s-1
    T_max=190; % °C
    rel=@(T,Theta)miu_0*miu(T,Theta)/G;
    flag=@(T)1.*(T*T_0<T_1 & T*T_0>T_g);

    rho=@(T)1.356-5*10^(-4)*T_0*T; % g/cm3
    C_p=@(T)0.3+6*10^(-4)*T_0*T; % cal/g*°C

    n_0=0.275; % the amorphous birefringence of ideally oriented  chains 

    K=@(T,F,A)K_max*exp( -4*log(2)*((T*T_0-T_max)/d)^2 + A_c*((miu_0*u_0/(L*G))*(F/A)) ).*(T*T_0<T_1 & T*T_0>T_g);

    % coeff(1)=A coeff(2)=u coeff(3)=T coeff(4)=F coeff(5)=Theta coeff(6)=n
    S_t=2*sqrt(pi)*0.473e-4*L/(rho(coeff(3))*C_p(coeff(3))*u_0^(0.667)*A_0^(0.833));

    dcoeff(3)=-S_t*coeff(1)^(-0.833)*(coeff(2)^2+64*u_a^2)^(0.167)*(coeff(3)-T_a)/coeff(2);
  
    dcoeff(2)=(coeff(4)*coeff(2)/miu(coeff(3),coeff(5))+(L*coeff(2)^2/(A_0*G))*(-rho(coeff(3))*g*A_0*coeff(1) ...   
        +K_d*rho_a^(1-n)*miu_a^(n)*2^(-n)*pi^((1+n)/2)*A_0^((1-n)/2)*u_0^(2-n)*coeff(1)^((1-n)/2)*coeff(2)^(2-n)))...
        /(1-(miu_0*u_0/(L*G))*coeff(2)*coeff(4)-rho(coeff(3))*u_0^2*coeff(2)^2/G);


    dcoeff(1)=-dcoeff(2)*coeff(1)/coeff(2);
    
    dcoeff(4)=(L^2/(A_0*miu_0*u_0))*(rho(coeff(3))*A_0*u_0^2*coeff(1)*coeff(2)*dcoeff(2)/L ...
        -rho(coeff(3))*g*A_0*coeff(1) ...   
        +K_d*rho_a^(1-n)*miu_a^(n)*2^(-n)*pi^((1+n)/2)*A_0^((1-n)/2)*u_0^(2-n)*coeff(1)^((1-n)/2)*coeff(2)^(2-n));

    dcoeff(5)=K(coeff(3),coeff(4),coeff(1))*L*(1-coeff(5))/(coeff(2)*u_0);
    
    dcoeff(6)=((A_op/(coeff(2)*n_0))*dcoeff(2)-coeff(6)*L/(u_0*coeff(2)*rel(coeff(3),coeff(5))))*flag(coeff(3));

end