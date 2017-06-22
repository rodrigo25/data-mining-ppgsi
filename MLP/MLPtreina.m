function [ A, B ] = MLPtreina( Xtr, Ytr, Xval, Yval, mA, maxIt )
% MLPtreina treina uma MLP
% Xtr, Ytr - Conjunto de treinamento
% Xval, Yval - Conjunto de validacao
% mA - Numero de neuronios na camada escondida
if nargin < 6
    maxIt = 10000; %máximo de iteracoes
end

[N, m0] = size(Xtr);
mB = size(Ytr,2);
Xtr = [ones(N,1),Xtr];

%Inicializando pesos
A = rand(mA, m0+1); %Pesos camada de escondida
B = rand(mB, mA+1); %Pesos camada de saida

alfa = .01; %taxa de aprendizado

it = 0; %contador de iteracoes

maxErr = 1e-5;

ERRO = [];
alfas = [ alfa ];
alfa0 = alfa;

fig=figure;
while it<maxIt
  it = it+1;
  
  %Feedfoward
  Zin = Xtr*A';
  Z = 1./(1+exp(-Zin));
  Yin = [ones(N,1),Z]*B';
  Y = 1./(1+exp(-Yin)); %func(Yin);
  
  %calcula o erro
  err = (Ytr - Y);
  EQM = sum(sum(err.^2))/N;
  if EQM<maxErr
    break
  end;
  
  %Backpropagation
  gradB = err.*(1-Y).*Y;
  gradA = (gradB*B(:,2:end)).*(Z.*(1-Z));
  
  deltaA = alfa * gradA' * Xtr;
  deltaB = alfa * gradB' * [ones(N,1),Z];
  
  %Weights update
  A = A + deltaA;
  B = B + deltaB;
  
  %Learning rate update
  %alfa = alfa0/( 1 + it*0.001);
  alfa = alfa0;
  %Stores some values for plotting
  alfas = [alfas;alfa];
  ERRO = [ERRO;EQM];

   clf(fig)
   %plot(Ytr,'r--')
   %hold on
   %plot(Y,'b')
   %grid
   plot(ERRO);
   title('EQM X Epocas');
   xlabel('Epocas');
   ylabel('Erro Quadratico Medio');
   text(it, EQM, sprintf('it=%d',it));
end

clf(fig);
plot(ERRO);
title('EQM X Epocas');
xlabel('Epocas');
ylabel('Erro Quadratico Medio');

%figure;
%plot(alfas);
%title('Alfa');

fprintf('Iterations: %d\n', it);

end