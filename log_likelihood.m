function [l,sigma] = log_likelihood(err,measurement_datas)

    normalization_factors = max(measurement_datas, [], 1); 
    normalization_factors(normalization_factors == 0) = 1;

    sigma2=err/(3*length(measurement_datas));
    sigma=sigma2^(0.5);

    if err==0
        l=Inf;
    else
        
        l = -(length(measurement_datas(:,1))*3/2)*log(2*pi*sigma2)-length(measurement_datas(:,1))/log(prod(normalization_factors,"all"))-err/(2*sigma2);
    end

end

