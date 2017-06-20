%CARREGA DADOS
dataset = 't4.8k.mat';
%dataset = 't4.8k-modified.mat';
%dataset = 'path-based2';
%dataset = 'test_rand1k';
%dataset = 'test_uniform';
%dataset = 'test_points2';
%dataset = 'test_S1';
%dataset = 'test_chainlink';

load(['data/data_' dataset])



%NORMALIZACAO DOS DADOS
[X, mean_val, std_val] = normalization( X, 'zscore' ); % z-score
%[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' ); % min-max



%PARAMETRIZAÇÃO
dim = [30 30]; % Dimensão do Grid de saida - dimention[Nx Ny] (Nx*Ny=Ns)
alfaIni = .9; % Taxa de aprendizado inicial
raioIni = 32; % Raio inicial delimitador da vizinhança de um neuronio
lambda = 32; % Desvio padrao da funcao de decaimento do alfa
tau = 55; % Desvio padrao da funcao de decaimento do raio
fDist = 'e'; % Função de distancia ('e' euclidiana, 'm' manhattan )
itMax = 300; % Num maximo de iterações



%TREINAMENTO DO SOM
[W, ~, ~] = SOM( X, dim, alfaIni, raioIni, lambda, tau, fDist, itMax );


%CALCULA PROXIMIDADE DE PONTOS
[ H, BMUfinal ] = calcBMUfinal( X, W, fDist );

%RESULTADOS
%cria diretorio para salvar os arquivos do teste
%timestamp = sprintf('%sT%d%d%d',datestr(date,'yymmdd'),hour(now),minute(now),round(second(now)));
%dirName = ['Resultados/' dataset '/' timestamp '/'];
dt = datetime;
dt.Format = 'yyMMdd''T''HHmmss';
dirName = ['Resultados/' dataset '/' char(dt) '/'];
mkdir(dirName);

% Plota dados originais
plotOriginalData(X, dirName);

%Plota Grid
plotGrid2(W, dim, dirName);

%Plota Influencia
inflMin = 5;
plotInfluencia( X, W, H, BMUfinal, inflMin, dirName);

%Plota U-Matrix
%plotUMatix( W, Nx, Ns, dirName );

