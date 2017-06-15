% data  - Conj de dados
% Ns - Qtd de neuronios de saida
% dim   - [Nx Ny] (Nx*Ny=Ns)
% topology  - 'hex' ou 'rec'
% alfa - Taxa de aprendizado
% radius - Raio delimitador da vizinhan�a de um neuronio
% fDist - Fun��o de distancia ('e' euclidiana, 'm' manhattan )
% itMax - Num maximo de itera��es
%function [ output_args ] = SOM( data, Ns, dim, topology)
function [ W ] = SOM( data, neur, dim, topology, alfa, radius, fDist, itMax )

Ns = neur^dim;
neuronsGrid = [ceil((1:Ns)/neur);mod((0:Ns-1),neur)+1]'; %indice dos neuronios no grid

% DEFINI��O DE VARI�VEIS
% n - qtd exemplos de dados
% D - Dimens�o original do problema
[n,D] = size(data); 
alfaIni = alfa; % Taxa de aprendizado inicial
lambda = 32;
it = 1; % Contador de itera��es

% INICIA PROT�TIPOS RANDOMICAMENTE
maximo = max(data);   %valores m�ximos de cada dimens�o original
minimo = min(data);   %valores minimos de cada dimens�o original
W = repmat(minimo,Ns,1)+repmat(maximo-minimo,Ns,1).*rand(Ns,D);
%W - Pesos de cada neuronio. Localiza��o no espa�o dimensional D original

% PROCESSO ITERATIVO
while (it<itMax)
  it = it + 1;
  for i=1:n %itera��o em todos as instancias
    
    p = data(i,:); % ponto vetorial da instancia i
    Dist = distance(p,W,fDist); % Distancia do ponto p para a posi��o dos neuronios W
    
    [~,BMU] = min(Dist); %BMU - Neuronio mais pr�ximo de p
    
    %theta - Matriz de pesos para atualiza��o dos vizinhos do BMU
    theta = neighborhood(topology,neuronsGrid,radius,BMU,it);
    
    % ATUALIZA OS PESOS W
    P = repmat(p,Ns,1);
    W = W + repmat(theta,1,D).*alfa.*(P-W);
    
  end
  
  % ALTERA A TAXA  DE APRENDIZADO  
  %20*exp(-7/1)
  %raioIni*exp(-it/1) diminui��o do raio
  alfa = alfaIni*exp(-it/lambda); %diminui��o do alfa
  
end

end

function [theta] = neighborhood(topology,neuronsGrid,radius,BMU,t)
  tau = 32;
  
  if strcmp(topology, 'gauss')
    radius = radius*exp(-t/tau);
    D = distance(neuronsGrid(BMU,:),neuronsGrid,'e');
    theta = exp(-D./(2*radius^2));
  elseif strcmp(topology, 'hex')
  elseif strcmp(topology, 'rec')
    
  end
  
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
