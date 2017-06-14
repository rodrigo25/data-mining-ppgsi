% Xtr, Ytr - Conjunto de treinamento
% Xval, Yval - Conjunto de validacao
% mHL - Numero de neuronios na camada escondida
function [ A, B ] = MLP( Xtr, Ytr, Xval, Yval, mA )

[N, m0] = size(Xtr);
mB = size(Ytr,2);
Xtr = [ones(N,1),Xtr];

%Inicializando pesos
A = rand(mA, m0+1); %Pesos camada de escondida
B = rand(mB, mA+1); %Pesos camada de saida

alfa = .1; %taxa de aprendizado

it = 0; %contador de iteracoes
maxIt = 10000; %máximo de iteracoes

EQM = inf;
maxErr = 1e-5;

ERRO = [];

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
  
  ERRO = [ERRO;EQM];
end
plot(ERRO);

end