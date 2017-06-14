function [ Y ] = saidaMLP( X, A, B )
N = size(X,1);
X = [ones(N,1),X];

Zin = X*A';
Z = 1./(1+exp(-Zin)); %Z = func(Zin, 'log');
Yin = [ones(N,1),Z]*B';
Y = 1./(1+exp(-Yin));

end