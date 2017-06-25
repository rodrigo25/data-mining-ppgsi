function alfa_m = calc_alfa_clodoaldo(X,Yd, A, B,gA, gB, d, N)
alfa_l= 0;
alfa_u = rand;
Aaux = A - alfa_u*gA;
Baux = B - alfa_u*gB;
[g,~,~]=calc_grad(X,Yd,Aaux,Baux,N);
hl = g'*d;

if abs(hl)< 1e-8
    alfa_m = alfa_u;
end

while hl<0 
    alfa_u = alfa_u*2;
    Aaux = A - alfa_u*gA;
    Baux = B - alfa_u*gB;
    [g,~,~]=calc_grad(X,Yd,Aaux,Baux,N);
    hl = g'*d;
end
epsilon = 1e-8;
alfa_m = alfa_u;
nitmax = ceil(log(alfa_u/epsilon));
nit = 0;

while abs(hl)>epsilon & nit < nitmax
     nit = nit + 1;
    alfa_m = (alfa_l+alfa_u)/2;
    Aaux = A - alfa_m*gA;
    Baux = B - alfa_m*gB;
    [g,~,~]=calc_grad(X,Yd,Aaux,Baux,N);
    hl = g'*d;
    
    if hl<0
        alfa_l = alfa_m;
    else 
        alfa_u = alfa_m;
    end
end

%alfa_m

end





function [g,gradA, gradB]=calc_grad(X,Yd,A,B,N)
  Zin = X*A';
  Z = 1./(1+exp(-Zin));
  Yin=[ones(N,1),Z]*B';
  Y = 1./(1+exp(-Yin));

  erro = Y-Yd;
  gradB = 1/N*(erro.*(Y.*(1-Y)))'*[ones(N,1),Z];
  DJDZ = (erro.*(Y.*(1-Y)))*B(:,2:end);
  gradA = 1/N*(DJDZ.*(Z.*(1-Z)))'*X;
  g = [gradA(:);gradB(:)];
end

