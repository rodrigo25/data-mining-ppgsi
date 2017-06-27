function [ A, B, ERROtr, ERROval, ALFA ] = MLP_alfaAdaptativo( Xtr, Ytr, Xval, Yval, mA, maxIt, earlyStopping )
% MLPtreina treina uma MLP
% Xtr, Ytr - Conjunto de treinamento
% Xval, Yval - Conjunto de validacao
% mA - Numero de neuronios na camada escondida
% maxIt - Numero maximo de iteracoes
if ~exist('maxIt','var')
  maxIt = 10000; %maximo de iteracoes quando nao definida
end

[N, m0] = size(Xtr); % N - qtd exemplos de dados // m0 - Numero de neuronios na camada de entrada
mB = size(Ytr,2); % mb - Numero de neuronios na camada de saida
Nval = size(Xval,1);
%Xtr = [ones(N,1),Xtr]; 

%INICIALIZANDO PESOS
A = rand(mA, m0+1); %Pesos randomicos na camada de escondida
B = rand(mB, mA+1); %Pesos randomicos na camada de saida

%INICIALIZANDO VARIAVEIS
alfa = .1; %taxa de aprendizado inicial
it = 0; %contador de iteracoes
maxErr = 1e-5; %Valor de erro aceito para parar o processo

%CALCULA ERRO INICIAL
[ Y ] = MLPsaida( Xtr, (A), (B) );
EQM = sum(sum((Ytr - Y).^2))/N;

[ Y ] = MLPsaida( Xval, (A), (B) );
EQMval = sum(sum((Yval - Y).^2))/Nval;

ERROtr = [EQM]; %Vetor com historico de erros
ERROval = [EQMval]; %Vetor com historico de erros de validacao
ALFA = [alfa]; %Vetor com historico de alfas

fprintf('Iterações: ')
%PROCESSO ITERATIVO
while it<maxIt 

    
  %FEEDFOWARD
  Zin = [ones(N,1),Xtr]*A';
  Z = 1./(1+exp(-Zin));
  Yin = [ones(N,1),Z]*B';
  Y = 1./(1+exp(-Yin)); %func(Yin);
  
  %CALCULO DO ERRO
  err = (Ytr - Y);
  EQM = sum(sum(err.^2))/N;
  if EQM<maxErr
    break
  end;
  
  %BACKPROPAGATION
  gradB = err.*(1-Y ).*Y;
  gradA = (gradB*B(:,2:end)).*(Z.*(1-Z));
  
  deltaA = alfa * gradA' * [ones(N,1),Xtr];
  deltaB = alfa * gradB' * [ones(N,1),Z];

  
  %ATUALIZACAO DOS PESOS E ADAPTACAO DO ALFA
  
  % Calcula novo erro da rede se os pesos forem atualizados
  [ newY ] = MLPsaida( Xtr, (A+deltaA), (B+deltaB) );
  newEQM = sum(sum((Ytr - newY).^2))/N;
  
  if (newEQM<EQM) %Se o novo erro diminuir (referente a ultima atualizacao)
    A = A + deltaA; %Atualiza os pesos da camada escondida A
    B = B + deltaB; %Atualiza os pesos da camada de saida B
    alfa = alfa*1.1; %Aumenta o alfa em 10%
    ERROtr = [ERROtr;newEQM]; %Armazena o erro
    
    %Calcula erro de validacao
    [ newYv ] = MLPsaida( Xval, (A), (B) );
    newEQMval = sum(sum((Yval - newYv).^2))/Nval;
    ERROval = [ERROval;newEQMval]; %Armazena o erro de validacao
    
    it = it+1;
  
    if mod(it,100)==0
      fprintf(' %d ',it)
      if mod(it,1000)==0
        fprintf(' \n')
      end
    end
    
    if earlyStopping
      if it > 40
        if ~any(newEQMval < ERROval(it-40:end))
          fprintf(' %d ',it)
          return
        end
      end
    end
    
  else %Senao
    alfa = alfa*0.9; %Diminui o alfa em 10%
  end
  
  ALFA = [ALFA;alfa];
  
end

end