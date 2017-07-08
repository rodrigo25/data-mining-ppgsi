function [accuracy,matrix] = multiclassConfusionMatrix( Yd, Y, classes, figureHandle, title, resultFileName  )
%multiclassConfusionMatrix cria e exibe uma matriz de confusao multiclasse.
%	Yd - matriz (n x 1) com os rotulos do conjunto de dados
%	Y - matriz (n x 1) com os rotulos preditos
%	classes - as classes presentes em Yd
%	resultPath - parametro opcional, quando definido implica na persistencia da
%	imagem da matriz de confusao
   matrix = confusionmat( Yd, Y' );
   if nargin < 6
      if nargin == 4
          plotConfusionMatrix( classes, matrix, figureHandle );          
      elseif nargin == 5
          plotConfusionMatrix( classes, matrix, figureHandle, title );
          close all;
      end
   else
        plotConfusionMatrix( classes, matrix, figureHandle, title, resultFileName );
        if numel(classes) == 2
            classesMetricsMap = getConfusionMetrics( matrix, classes );
            save(resultFileName,'classesMetricsMap');
        end
        close all;
   end
   
   accuracy = sum(diag(matrix)) / sum(sum(matrix));

   %fprintf('Accuracy is %.6f\n', accuracy);
end

function [] = plotConfusionMatrix( classes, matrix, figureHandle, matrixTitle, resultPath )
    
    if nargin < 3
       figureHandle = 1; 
    end
    figure(figureHandle);
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
    
    if nargin > 3
        title(sprintf('Matriz de confusão - %s',matrixTitle));
        if nargin > 4
            confusionMatrixFileName = resultPath;
            %pause
            if ~exist(strcat(confusionMatrixFileName,'.png'), 'file')
                set(gcf, 'PaperPositionMode', 'auto');
                print(confusionMatrixFileName, '-dpng'); 
            end
        end
    else
        title('Matriz de confusão');
    end
end

function classesMetricsMap = getConfusionMetrics( matrix, classes )
    zeroBasedClasses = 0;
    if min(classes) == 0
       classes = classes + 1; 
       zeroBasedClasses = 1;
    end
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
    if zeroBasedClasses
       classes = classes - 1; 
    end
    classesMetricsMap = containers.Map( num2cell(classes), classesMetrics );
end