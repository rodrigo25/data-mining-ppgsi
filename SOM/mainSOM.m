%CARREGA DADOS
%dataset = 't4.8k.mat';
%dataset = 't4.8k-modified.mat';
dataset = 'path-based2';
%dataset = 'test_rand1k';
%dataset = 'test_uniform';
%dataset = 'test_points2';
%dataset = 'test_points3';
%dataset = 'test_S1';
%dataset = 'test_chainlink';

%load fisheriris.mat
%X = meas;

load(['data/data_' dataset])



%NORMALIZACAO DOS DADOS
typeNormalization = 'zscore'; % zscore ou min-max
[X, mean_val, std_val] = normalization( X, typeNormalization ); % z-score
%[X, ~, ~,min_val, max_val] = normalization( X, typeNormalization ); % min-max



%PARAMETRIZA��O
dim = [17 17]; % Dimens�o do Grid de saida - dimention[Nx Ny] (Nx*Ny=Ns)
alphaIni = .9; % Taxa de aprendizado inicial
raioIni = 15; % Raio inicial delimitador da vizinhan�a de um neuronio
lambda = 50; % Taxa de decaimento do alfa
tau = 50; % Taxa de decaimento do raio
fDist = 'e'; % Fun��o de distancia ('e' euclidiana, 'm' manhattan )
itMax = 150; % Num maximo de itera��es



%TREINAMENTO DO SOM
tic
[W, WHist, QuantErrorHist, TopolErrorHist, DeltaMeanHist, DeltaMaxHist, TimeTrain] = SOM( X, dim, alphaIni, raioIni, lambda, tau, fDist, itMax );
%W - Pesos finais dos neuronios da rede
%WHist - Historico dos pesos dos neuronios no inicio, em 33% e 66% de iteracoes e no fim
%QuantErrorHist - Historico do Erro de Quantizacao a cada iteracao
%TopolErrorHist - Historico do Erro Topologico a cada iteracao
%DeltaMeanHist - Historico do Delta Medio a cada iteracao
%DeltaMaxHist - Historico do Delta Maximo a cada iteracao
%TimeTrain - Tempo de Treinamento


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
%plotOriginalData(X, dirName);

%Plota Grid
plotGrid(W, dim, dirName);

%Plota Influencia
inflMin = 0;
%plotInfluencia( X, W, H, BMUfinal, inflMin, dirName);

inflMin = 5;
%plotInfluencia( X, W, H, BMUfinal, inflMin, dirName);

%Plota Erros
plotErros(QuantErrorHist, TopolErrorHist, dirName);

%Plota Deltas
%plotDeltas(DeltaMeanHist, DeltaMaxHist, dirName);

%Cria log do SOM
createLog( dataset, size(X), typeNormalization,  dim, alphaIni, raioIni, lambda, tau, fDist, itMax, TimeTrain, QuantErrorHist(end), TopolErrorHist(end), dirName );

%Plota Grid Evolution
%plotGridEvolution( WHist, dim, dirName );

%Plota U-Matrix
plotUMatix( W, dim, dirName );

%Salva Workspace
save([dirName '/parameters'], 'dataset', 'typeNormalization', 'dim', 'alphaIni', 'raioIni', 'lambda', 'tau', 'fDist', 'itMax');
save([dirName '/results'], 'W', 'WHist', 'QuantErrorHist', 'TopolErrorHist', 'DeltaMeanHist', 'DeltaMaxHist', 'TimeTrain');