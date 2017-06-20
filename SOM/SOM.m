% DATA  - Conj de dados
% dim - Dimens�o do Grid de saida - dimention[Nx Ny] (Nx*Ny=Ns)
% alfaIni - Taxa de aprendizado inicial
% radiusIni - Raio inicial delimitador da vizinhan�a de um neuronio
% lambda - Desvio padrao da funcao de decaimento do alfa
% tau - Desvio padrao da funcao de decaimento do raio
% fDist - Fun��o de distancia ('e' euclidiana, 'm' manhattan )
% itMax - Num maximo de itera��es
function [ W, quantError, topologicError ] = SOM( DATA, dim, alfaIni, radiusIni, lambda, tau, fDist, itMax )

Ns = dim(1)*dim(2); %Ns - Num total de neuronios no Grid de saida
neuronsGrid = [ceil((1:Ns)/dim(2));mod((0:Ns-1),dim(2))+1]'; % Vetor de indices dos Ns neuronios no grid

%Ns = neur^dim;
%neuronsGrid = [ceil((1:Ns)/neur);mod((0:Ns-1),neur)+1]'; %indice dos neuronios no grid

% DEFINI��O DE VARI�VEIS
[N,D] = size(DATA); % N - qtd exemplos de dados // D - Dimens�o original do problema
alfa = alfaIni; % Taxa de aprendizado atual
radius = radiusIni; % Raio atual
it = 1; % Contador de itera��es

% INICIA PROT�TIPOS RANDOMICAMENTE
maximo = max(DATA);   %valores m�ximos de cada dimens�o original
minimo = min(DATA);   %valores minimos de cada dimens�o original
%W - Pesos de cada neuronio. Localiza��o no espa�o dimensional D original
W = repmat(minimo,Ns,1)+repmat(maximo-minimo,Ns,1).*rand(Ns,D);

fprintf('Iterations:\n');

% PROCESSO ITERATIVO
while (it<=itMax)
  if mod(it,25) == 0
    fprintf('%d ', it);
    if mod(it,100) == 0
      fprintf('\n'); 
    end
  end
  
  %PERMUTA��O DOS DADOS
  rp = randperm(size(DATA,1)); % permuta os indices
  DATA = DATA(rp,:); % aplica permutacao em X
  
  for i=1:N %itera��o em todos as instancias
    
    p = DATA(i,:); % ponto vetorial da instancia i
    Dist = distance(p,W,fDist); % Distancia do ponto p para a posi��o dos neuronios W
    
    [~,BMU] = min(Dist); %BMU - Neuronio mais pr�ximo de p
    
    %theta - Matriz de pesos para atualiza��o dos vizinhos do BMU
    theta = neighborhood(neuronsGrid,radius,BMU);
    
    % ATUALIZA OS PESOS W
    P = repmat(p,Ns,1);
    W = W + repmat(theta,1,D).*alfa.*(P-W);
    
  end
  
  % ALTERA A TAXA  DE APRENDIZADO E O RAIO
  alfa = alfaIni*exp(-it/lambda); %diminui��o do alfa
  radius = radiusIni*exp(-it/tau); %diminui��o do raio
  
  it = it + 1;
end

                  BMUs = zeros(N,D);
                  topologicErrorInd = zeros(N,1);
                  for i=1:N
                      p = DATA(i,:);
                      Dist = distance(p,W,fDist);
                      [~,BMUindex] = min(Dist);
                      BMUs(i, :) = W(BMUindex, :);

                      Dist(BMUindex,1) = max(Dist);
                      [~,secondBMUindex] = min(Dist);
                      BMUsDistances = distance(neuronsGrid(BMUindex,:), neuronsGrid(secondBMUindex,:), 'e');
                      if BMUsDistances > sqrt(2)
                         topologicErrorInd(i) = 1;
                      end
                  end

                  quantError = sum(distance2(DATA, BMUs, fDist), 1)/N;
                  fprintf('\nQuantization error=%.8f\n', quantError);

                  topologicError = sum(topologicErrorInd)/N;
                  fprintf('\nTopologic error=%.8f\n', topologicError);   
end



function [theta] = neighborhood(neuronsGrid,radius,BMU)
  
  %Calcula a distancia de todos os neuronios ao BMU no Grid
  D = distance(neuronsGrid(BMU,:),neuronsGrid,'e');
  %Calcula a influencia da atualizacao sobre os neuronios pelo raio
  theta = exp(-D./(2*radius^2));
  
end

% FUN��O DE DIST�NCIAS
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

function D = distance2(P, POINTS, fDist)
  if strcmp(fDist,'e') %distancia euclidiana
    D = sqrt(sum((P-POINTS).^2, 2));
  elseif strcmp(fDist,'m') %distancia manhattan
    D = sum(abs(P-POINTS), 2);
  end
end