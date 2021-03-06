
%DEFINE NOME DO TESTE
%testName = 'Test1_MLPclodoaldo';
%testName = 'Test2_MLPalfaAdaptativo';
testName = 'Test1_MLPalfaFixo_0.1';

%SELECIONA ALGORITMOS PARA EXECUTAR
%Algoritmos = {'MLP_clodoaldo', 'MLP_alfaAdaptativo', 'MLP_alfaFixo', 'MLP_alfaDecay'};
%Algoritmos = {'MLP_clodoaldo'};
Algoritmos = {'MLP_alfaFixo'};
totalAlgoritmos = length(Algoritmos);

%SELECIONA VARIACAO DE NEURONIOS NA CAMADA OCULTA
Neuronios = [10 30 50 70 90 110];
%Neuronios = 20:20:200;
totalNeuronios = length(Neuronios);

%imprimi cabecalho
fprintf(['\n\n--------------------------------------------------------------\n']);
fprintf(['                      Testes de An�lise ROC                   \n']);
fprintf(['--------------------------------------------------------------\n']);

%cria diretorio para o teste
dirName = ['Resultados/' testName '/'];
if ~exist(dirName, 'dir') % se o dir nao existe, ele eh criado
  mkdir(dirName);
end

im_roc = figure; %cria o grafico ROC
plot([0 1],[0 1],'b','DisplayName','Diagonal'); %plota a diagonal ascendente
hold on

ROCdata = cell(totalAlgoritmos, totalNeuronios);

for a=1:totalAlgoritmos % iteracao dos algoritmos
  fprintf(['\n\n############### Testes com ' Algoritmos{a} ' ###############']);
  
  for n=1:totalNeuronios %iteracao dos neuronios
    
    fprintf(['\n\n\n#### ' num2str(Neuronios(n)) ' Neur�nios na Camada Econdida ####\n']);
    
    [TPR, FPR] = spambase_kfoldCV(Algoritmos{a}, Neuronios(n), dirName);
    ROCdata{a,n} = {TPR FPR};
    plot( FPR, TPR, 'DisplayName',[Algoritmos{a} ' ' num2str(Neuronios(n)) ' neur']); %Plota a curva ROC do componente
    %plot( FPR, TPR, 'DisplayName',['MLP Clodoaldo com ' num2str(Neuronios(n))]); %Plota a curva ROC do componente
    %plot( FPR, TPR, 'DisplayName',['MLP Adaptativo ' num2str(Neuronios(n)) ' neur']); %Plota a curva ROC do componente
    
  end
end

fprintf('\n');
ylabel('True Positive Rate');
xlabel('False Positive Rate');
legend('show')
title('ROC')

print(im_roc,[dirName 'ROC'],'-dpng');
save([dirName 'ROCdata'],'ROCdata','Algoritmos','Neuronios');