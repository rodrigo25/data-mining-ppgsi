function [ Y ] = MLPsaida( X, L, A, B )
    N = size(X,1);    
    if L==1 && iscell(A) == 0
       X = [ones(N,1),X];
       Zin = X*A';
       Z = 1./(1+exp(-Zin));
       Yin = [ones(N,1),Z]*B';
       Y = 1./(1+exp(-Yin));
       return;
    end
    
    for i=1:L
       X = [ones(N,1),X];
       Zin{i} = X*A{i}';
       Z{i} = 1./(1+exp(-Zin{i}));
       X = Z{i};
    end
    Yin = [ones(N,1),Z{i}]*B';
    Y = 1./(1+exp(-Yin));
end