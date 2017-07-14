% DATA  - Conj de dados
% dim - Dimensão do Grid de saida - dimention[Nx Ny] (Nx*Ny=Ns)
% alphaIni - Taxa de aprendizado inicial
% radiusIni - Raio inicial delimitador da vizinhança de um neuronio
% lambda - Taxa de decaimento do alfa
% tau - Taxa de decaimento do raio
% fDist - Função de distancia ('e' euclidiana, 'm' manhattan )
% itMax - Num maximo de iterações
function [ W, WHist, QuantErrorHist, TopolErrorHist, DeltaMeanHist, DeltaMaxHist, TimeTraining ] = SOM( DATA, dim, alphaIni, radiusIni, lambda, tau, fDist, itMax )
tic

Ns = dim(1)*dim(2); %Ns - Num total de neuronios no Grid de saida

neuronsGrid = [ceil((1:Ns)/dim(2));mod((0:Ns-1),dim(2))+1]'; % Vetor de indices dos Ns neuronios no grid

% DEFINIÇÃO DE VARIÁVEIS
[N,D] = size(DATA); % N - qtd exemplos de dados // D - Dimensão original do problema
alpha = alphaIni; % Taxa de aprendizado atual
radius = radiusIni; % Raio atual
it = 1; % Contador de iterações

% INICIA PROTÓTIPOS RANDOMICAMENTE
maximo = max(DATA);   %valores máximos de cada dimensão original
minimo = min(DATA);   %valores minimos de cada dimensão original
%W - Pesos de cada neuronio. Localização no espaço dimensional D original
W = repmat(minimo,Ns,1)+repmat(maximo-minimo,Ns,1).*rand(Ns,D);

% INICIA ESTRUTURAS PARA ARMAZENAR HISTORICO
QuantErrorHist = zeros(itMax,1);
TopolErrorHist = zeros(itMax,1);
DeltaMeanHist = zeros(itMax,D);
DeltaMaxHist = zeros(itMax,D);
WHist = cell(4,2);
WHist{1,1} = W;
WHist{1,2} = 0;
indWHist = [round(itMax*0.33) round(itMax*0.66)];

fprintf('Iterations:\n');

% PROCESSO ITERATIVO
while (it<=itMax)
  %imprime a iteracao
  if mod(it,25) == 0
    fprintf('%d ', it);
    if mod(it,100) == 0
      fprintf('\n'); 
    end
  end
  
  %PERMUTAÇÃO DOS DADOS
  rp = randperm(size(DATA,1)); % permuta os indices
  DATA = DATA(rp,:); % aplica permutacao em X
  
  quantErrIt = 0;
  topolErrIt = 0;
  deltaMeanIt = 0;
  deltaMaxIt = zeros(1,D);
  
  for i=1:N %iteração em todos as instancias
    
    % CALCULA O BMU (PROCESSO DE COMPETICAO)
    p = DATA(i,:); % ponto vetorial da instancia i
    Dist = distance(p,W,fDist); % Distancia do ponto p para a posição dos neuronios W
    
    [~,BMU] = min(Dist); %BMU - Neuronio mais próximo de p
    
    
    % (CALCULO DOS ERROS DA ITERACAO)
    %erro de quantizacao da iteracao
    quantErrIt = quantErrIt + Dist(BMU);
    %erro topologico da iteracao
    Dist(BMU) = Inf;
    [~,secondBMU] = min(Dist);
    BMUsDistances = distance(neuronsGrid(BMU,:), neuronsGrid(secondBMU,:), 'm');
    if BMUsDistances > 1%sqrt(2)
      %topolErrIt = topolErrIt + BMUsDistances;
      topolErrIt = topolErrIt + 1;
    end
    
    % CALCULA A INFLUENCIA DA ATUALIZAÇÃO NOS VIZINHOS DO BMU (PROCESSO DE COOPERACAO)
    theta = neighborhood(neuronsGrid,radius,BMU); %Matriz de influencia sobre a vizinhança do BMU
    
    % ATUALIZA OS PESOS W (PROCESSO DE ADAPTACAO SINAPTICA)
    P = repmat(p,Ns,1);
    delta = repmat(theta,1,D).*alpha.*(P-W);
    W = W + delta;
    
    % (ARMAZENA O DELTA DA ITERACAO)
    %delta medio da iteracao
    %deltaMeanIt = deltaMeanIt + sum(repmat(theta,1,D).*delta)/sum(theta);
    deltaMeanIt = deltaMeanIt + sum(abs(delta))/Ns;
    
    %delta maximo da iteracao
    valores = max(abs(delta));
    deltaMaxIt( valores>deltaMaxIt ) = valores(valores>deltaMaxIt);
    
  end
  
  % ALTERA A TAXA  DE APRENDIZADO E O RAIO
  alpha = alphaIni*exp(-it/lambda); %diminuição do alfa
  radius = radiusIni*exp(-it/tau); %diminuição do raio
  
  
  QuantErrorHist(it,1) = quantErrIt/N;
  TopolErrorHist(it,1) = topolErrIt/N;
  DeltaMeanHist(it,:) = deltaMeanIt/N;
  DeltaMaxHist(it,:) = deltaMaxIt;
  
  if it==indWHist(1)
    WHist{2,1} = W;  
    WHist{2,2} = it;
  end
  if it==indWHist(2)
    WHist{3,1} = W;  
    WHist{3,2} = it;
  end
  
  it = it + 1;
end

WHist{4,1} = W;  
WHist{4,2} = it-1;

TimeTraining = toc;
fprintf('Tempo de treinamento (s): \t%7.7f\n', TimeTraining);
end



function [theta] = neighborhood(neuronsGrid,radius,BMU)
  %Calcula a distancia de todos os neuronios ao BMU no Grid
  D = distance(neuronsGrid(BMU,:),neuronsGrid,'e');
  %Calcula a influencia da atualizacao sobre os neuronios pelo raio
  theta = exp(-D./(2*radius^2));
end


% FUNÇÃO DE DISTÂNCIAS
% calcula a distancia do ponto p aos pontos em POINTS
% de acordo com uma distancia (dist) euclidiana ('e') ou manhattan ('m')
function D = distance(p, POINTS, fDist)
  [n,~] = size(POINTS);
  P = repmat(p,n,1);
  if strcmp(fDist,'e') %distancia euclidiana
    D = sqrt(sum((P-POINTS).^2, 2));
  elseif strcmp(fDist,'m') %distancia manhattan
    D = sum(abs(P-POINTS), 2);
  end
end
