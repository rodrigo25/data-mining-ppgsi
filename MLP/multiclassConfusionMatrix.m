function [accuracy,matrix] = multiclassConfusionMatrix( Yd, Y, classes, resultPath  )
%multiclassConfusionMatrix cria e exibe uma matriz de confusao multiclasse.
%	Yd - matriz (n x 1) com os rotulos do conjunto de dados
%	Y - matriz (n x 1) com os rotulos preditos
%	classes - as classes presentes em Yd
%	resultPath - parametro opcional, quando definido implica na persistencia da
%	imagem da matriz de confusao
   matrix = confusionmat( Yd, Y' );
   if nargin > 3
      plotConfusionMatrix( classes, matrix, resultPath );
   else
      plotConfusionMatrix( classes, matrix );
   end
   
   %classesMetrics = getConfusionMetrics( matrix, classes );
   accuracy = sum(diag(matrix)) / sum(sum(matrix));
   fprintf('Accuracy is %.6f\n', accuracy);
end

function [] = plotConfusionMatrix( classes, matrix, resultPath )
    
    imagesc(matrix);            
    colormap(flipud(gray));  

    textStrings = num2str(matrix(:),'%0.2f');  
    textStrings = strtrim(cellstr(textStrings));  
    for idx=1:length(textStrings)
        if isequal(textStrings{idx},'0.00')
            textStrings{idx} = '';
        end
    end    
    
    [x,y] = meshgrid(1:size(matrix,1));
    hStrings = text(x(:),y(:),textStrings(:), 'HorizontalAlignment','center');
    midValue = mean(get(gca,'CLim'));  
    textColors = repmat(matrix(:) > midValue,1,3);
    
    set(hStrings,{'Color'},num2cell(textColors,2)); 
    
    set(gca,'XTick',1:size(matrix,1),...                         
            'XTickLabel',arrayfun(@( c ) strcat('C', num2str(c)), classes, 'UniformOutput', false ),...  
            'YTick',1:size(matrix,1),...
            'YTickLabel',arrayfun(@( c ) strcat('C', num2str(c)), classes,'UniformOutput', false ),...
            'TickLength',[0 0] );
	xlabel( 'Predito' );
    ylabel( 'Real' );
        
    title('Matriz de confusão');
    
    if nargin > 2
        confusionMatrixFileName = strcat( resultPath, '\confusionMatrix');
        pause
        if ~exist(strcat(confusionMatrixFileName,'.png'), 'file')
            set(gcf, 'PaperPositionMode', 'auto');
            print(confusionMatrixFileName, '-dpng'); 
        end
        close all;
    end
end

%{
function classesMetricsMap = getConfusionMetrics( matrix, classes )
    
    classesMetrics = cell( 1, length(classes) );
    for i=1:length(classes)
        c = classes(i);
        metrics = containers.Map();
        
        TP = matrix(c,c);
        metrics( 'TP' ) = TP;

        FN = sum(matrix(c,:)) - matrix(c,c);
        metrics( 'FN' ) = FN;

        FP = sum(matrix(:,c)) - matrix(c,c);
        metrics( 'FP' ) = FP;

        TN = sum(sum(matrix)) - sum(matrix(c,:)) - sum(matrix(:,c));
        metrics( 'TN' ) = TN;

        TPR = FP / (TN + FP);
        metrics( 'TPR' ) = TPR;

        FPR = FP / (TN + FP);
        metrics( 'FPR' ) = FPR;

        SPC = TN / (FP + TN);
        metrics( 'SPC' ) = SPC;

        PPV = TP / (TP + FP);
        metrics( 'PPV' ) = PPV;

        NPV = TN / (TN + FN);
        metrics( 'NPV' ) = NPV;

        FDR = FP / (TP + FP);
        metrics( 'FDR' ) = FDR;

        accuracy = sum(diag(matrix)) / sum(sum(matrix));
        metrics( 'accuracy' ) = accuracy;

        error = ( sum(sum(matrix)) - sum(diag(matrix)) ) / sum(sum(matrix));
        metrics( 'error' ) = error;
        %{'sensibility' 'FPR' 'SPC' 'PPV' 'NPV' 'FDR' 'accuracy' 'error'}, ...
    
        classesMetrics{ c } = metrics;
    end
    
    classesMetricsMap = containers.Map( num2cell(classes), classesMetrics );
end
%}
