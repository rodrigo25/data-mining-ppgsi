function [A,B]=MLP_clodoaldo(X,Yd,h,maxIt)
if ~exist('maxIt','var')
  nepocasmax = 10000;
else
  nepocasmax = maxIt;
end

[N,ne] = size(X);
[N,ns]=size(Yd);
X = [ones(N,1),X];
% Inicializa
A = rands(h,ne+1)/10;
B = rands(ns, h+1)/10;
% Calcula a saida
Yr = calc_saida(X, A, B, N);
erro = (Yr - Yd);
EQM = sum(sum(erro.*erro))/N;
[g,gA, gB] = calc_grad(X,Yd,A,B,N);
nepocas = 0;
veterro = [];
veterro = [veterro;EQM];
%alfa = 0.1;

fprintf('Iterações: ')
while norm(g)>1e-8 & nepocas < nepocasmax
    nepocas = nepocas +1;
    
    if mod(nepocas,100)==0
      fprintf(' %d ',nepocas)
      if mod(nepocas,1000)==0
        fprintf(' \n')
      end
    end
    
    d = -g;
    alfa = calc_alfa(X,Yd, A, B,gA, gB, d, N);
    A = A - alfa*gA;
    B = B - alfa*gB;
    [g,gA, gB] = calc_grad(X, Yd, A, B, N); 
    Yr = calc_saida(X, A, B, N);
    erro = (Yr - Yd);
    EQM = sum(sum(erro.*erro))/N;
    veterro = [veterro;EQM];
    %disp(sprintf('EQM =%2.7f, nepocas=%d, alfa = %1.3f',EQM,nepocas,alfa))
end
%plot(veterro)
    
end



function alfa_m = calc_alfa(X,Yd, A, B,gA, gB, d, N)
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



function Y=calc_saida(X,A,B,N)
  Zin = X*A';
  Z = 1./(1+exp(-Zin));
  Yin=[ones(N,1),Z]*B';
  Y = 1./(1+exp(-Yin));
end





