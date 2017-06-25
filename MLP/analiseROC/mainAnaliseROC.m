
%DEFINE NOME DO TESTE
%testName = 'Test1_MLPclodoaldo';
testName = 'Test1_MLPalfaAdaptativo';

%SELECIONA ALGORITMOS PARA EXECUTAR
%Algoritmos = {'MLP_clodoaldo'};
Algoritmos = {'MLP_alfaAdaptativo'};
totalAlgoritmos = length(Algoritmos);

%SELECIONA VARIACAO DE NEURONIOS NA CAMADA OCULTA
Neuronios = [10 20];
totalNeuronios = length(Neuronios);

fprintf(['\n\n--------------------------------------------------------------\n']);
fprintf(['                      Testes de Análise ROC                   \n']);
fprintf(['--------------------------------------------------------------\n']);


im_roc = figure; %cria o grafico ROC
plot([0 1],[0 1],'b','DisplayName','Diagonal'); %plota a diagonal ascendente
hold on

for a=1:totalAlgoritmos % iteracao dos algoritmos
  fprintf(['\n\n############### Testes com ' Algoritmos{a} ' ###############\n']);
  
  for n=1:totalNeuronios %iteracao dos neuronios
    
    fprintf(['\n#### ' num2str(Neuronios(n)) ' Neurônios na Camada Econdida ####\n']);
    
    [TPR, FPR] = spambase_holdout(Algoritmos{a}, Neuronios(n), testName);
    plot( FPR, TPR, 'DisplayName',[Algoritmos{a} ' com ' num2str(Neuronios(n))]); %Plota a curva ROC do componente
    
  end
end

ylabel('True Positive Rate');
xlabel('False Positive Rate');
legend('show')
title('ROC')

