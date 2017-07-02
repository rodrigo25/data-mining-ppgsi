function [ Y, Z ] = MLPsaida( X, A, B )
    N = size(X,1);    
    Zin = [ones(N,1) X]*A';
    Z = 1./(1+exp(-Zin));
    Yin = [ones(N,1) Z]*B';
    Y = 1./(1+exp(-Yin));
end