%CARREGA DADOS
%dataset = 't4.8k.mat';
%dataset = 't4.8k-modified.mat';
%dataset = 'path-based2';
%dataset = 'test_rand1k';
%dataset = 'test_uniform';
dataset = 'test_points2';
%dataset = 'test_S1';
%dataset = 'test_chainlink';

load(['data/data_' dataset])

%NORMALIZACAO DOS DADOS
[X, mean_val, std_val] = normalization( X, 'zscore' ); % z-score
%[X, ~, ~, min_val, max_val] = normalization( X, 'minmax' ); % min-max
 
%PERMUTAÇÃO DOS DADOS
rp = randperm(size(X,1)); % permuta os indices
X = X(rp,:); % aplica permutacao em X
%Y = Y(rp,:); % aplica permutacao em Y

%PARAMETRIZAÇÃO
Nx = 10;
dim = 2;

%TREINAMENTO DO SOM
[W, Ns, ~] = SOM( X, Nx, dim, 'gauss', .9, 30, 'e', 300 );

%RESULTADOS
%cria diretorio para salvar os arquivos do teste
dt = datetime;
dt.Format = 'yyMMdd''T''HHmmss';
dirName = ['Resultados/' dataset '/' char(dt) '/'];
mkdir(dirName);

% Plota dados originais
figure
scatter(X(:,1),X(:,2)) 

%Plota Grid
plotGrid(W, Ns, Nx, dirName);

%Plota U-Matrix
umatix( W, Nx, Ns, dirName );

