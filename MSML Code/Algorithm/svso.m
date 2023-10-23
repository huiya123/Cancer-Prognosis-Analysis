function B = svso( t,A )
% Singular Value Shrinkage Operator
[ U_svso,s,V ] = svd( A );
s = s - t;                 
ZeroLog = s > 0;
s = s.*ZeroLog;
B = U_svso * s * V';   
end
