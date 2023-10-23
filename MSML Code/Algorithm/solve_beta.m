function new_beta = solve_beta(results,M,beta0)
    function B = func(beta) 
        [m,k] = size(M);
        E =results;
        rank_triangle = tril(ones(m,m-1),-1);
        A = repmat(exp(M*beta),1,m-1).*rank_triangle;  
        B = -sum(repmat(E(1:end-1)',k,1).*(M(1:end-1,:)' - ((A')*M)'./(repmat(sum((A),1),k,1))),2);
    end
new_beta = fsolve(@func,beta0,options);
end
