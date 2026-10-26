function [c, ceq] = nonlcon(x)

    index_diff=5;
    c = [];
    end_index =length(x) - 3;
    
    for index = 2:end_index
        dist = x(index) - x(index  - 1);
        c = [c; index_diff - dist];
    end
    
    ceq = [];
end