clc; 
clear;
close all
fclose('all');


fid_label=fopen('experiment_label.txt','r');

%camera parameters
pixelSize=5.86e-04; %cm
fx=1.4135021364891049e+03; %pixel
fc=15.24; %cm; =6 inch
scalFactor= fc/(fx*pixelSize)*pixelSize; %cm/pixel

tablerow=0;

while ~feof(fid_label)    
    line=fgets(fid_label);

    pot_id=regexprep(line,'\s+',''); %replace new line

    firstday=1;
    date_count=0;
    fid_date=fopen('experiment_date.txt','r');
    empty_pot=0;
    area_ref=0;
    convex_area_ref=0;
    perimeter_ref=0;
    day_ref=0;
    while ~feof(fid_date)  
        close all;
        genotype=pot_id(1:3);
        date=fgets(fid_date);
        date=regexprep(date,'\s+',''); %replace new line
        date_count=date_count+1;
        
        str=strcat('2d_images_deeplab/',pot_id,'/processed/',pot_id,'_',date,'_plant.png');

  
        if (exist(str) == 0) 
            disp(strcat(str,' not exist'))
            continue;
        end
        
        a=imread(str);    
        hsv = rgb2hsv(a);
        lab = rgb2lab(a);
        final = imbinarize(rgb2gray(a), 0.0001);
          
        if (nnz(final))==0
            continue
        end
        
        tablerow=tablerow+1;
        T = regionprops('Table',bwareafilt(final, 1, 'largest'),'Area','ConvexArea','Solidity','MajorAxisLength','MinorAxisLength','Perimeter', 'BoundingBox');
        T.BoundingboxArea(1)= T.BoundingBox(3)*T.BoundingBox(4);
        width=min(T.BoundingBox(4),T.BoundingBox(3));
        length=max(T.BoundingBox(4),T.BoundingBox(3));
        T.AspectRatio(1)=width/length;

        [L,num]=bwlabel(final);
        
        if num>1
            T.Area(1)=nnz(final); %a3 none zero pixel
            finalconvex=bwconvhull(final);
            T.ConvexArea(1)=nnz(finalconvex);
                
            AxisLength=regionprops(finalconvex,'MajorAxisLength','MinorAxisLength','Perimeter', 'BoundingBox');
            T.MajorAxisLength(1)=AxisLength.MajorAxisLength(1);
            T.MinorAxisLength(1)=AxisLength.MajorAxisLength(1);
                
            T.Perimeter(1)=AxisLength.Perimeter(1);
            T.BoundingboxArea(1)= AxisLength.BoundingBox(3)*AxisLength.BoundingBox(4);
                
            width=min(AxisLength.BoundingBox(4),AxisLength.BoundingBox(3));
            length=max(AxisLength.BoundingBox(4),AxisLength.BoundingBox(3));
            T.AspectRatio(1)=width/length;
        end
         
        %pixel to cm
        T.Area(1)=T.Area(1)*scalFactor*scalFactor;
        T.ConvexArea(1)=T.ConvexArea(1)*scalFactor*scalFactor;
        T.MajorAxisLength(1)=T.MajorAxisLength(1)*scalFactor;
        T.MinorAxisLength(1)=T.MinorAxisLength(1)*scalFactor;
        T.Perimeter(1)=T.Perimeter(1)*scalFactor;
        T.BoundingboxArea(1)=T.BoundingboxArea(1)*scalFactor*scalFactor;
        %update
        T.Solidity(1)=T.Area(1)/T.ConvexArea(1);
        T.Rectangularity(1)=T.Area(1)/T.BoundingboxArea(1);
        T.Circularity(1)=4*3.14159*T.Area(1)/T.Perimeter(1)/T.Perimeter(1);
        
       T.averageR= mean( nonzeros(a(:,:,1)) );
       T.averageG= mean( nonzeros(a(:,:,2)) );
       T.averageB= mean( nonzeros(a(:,:,3)) );
       T.hsvH= mean( nonzeros(hsv(:,:,1)) );
       T.hsvS= mean( nonzeros(hsv(:,:,2)) );
       T.hsvV= mean( nonzeros(hsv(:,:,3)) );
       T.labL= mean( nonzeros(lab(:,:,1)) );
       T.labA= mean( nonzeros(lab(:,:,2)) );
       T.labB= mean( nonzeros(lab(:,:,3)) );
       
       T.potId=pot_id;
       T.sampleDate=date;

       if firstday
           area_ref=T.Area(1);
           convex_area_ref=T.ConvexArea(1);
           perimeter_ref=T.Perimeter(1);
           day_ref=date_count;
           T.absArea=0;
           T.absConvexArea=0;
           T.absPerimeter=0;
           T.absDay=0;
       else
           T.absArea=(T.Area(1)-area_ref)/(date_count-day_ref);
           T.absConvexArea=(T.ConvexArea(1)-convex_area_ref)/(date_count-day_ref);
           T.absPerimeter=(T.Perimeter(1)-perimeter_ref)/(date_count-day_ref);
           T.absDay=date_count-day_ref;
       end
            
       
%        T=T(:,[15 16 1 3:14 17:20]);
       T=T(:,[21 22 1 3:20 23:26]);
       
       tRange=strcat('A',int2str(tablerow),':Y',int2str(tablerow+1));
       
       %write to csv file

       disp(strcat(pot_id,'_',date,'_plant.bmp'))
       if tablerow==1
           writetable(T,'2d_traits.xls','WriteVariableNames',true,'Range',tRange);
           tablerow=tablerow+1;
       else
           writetable(T,'2d_traits.xls','WriteVariableNames',false,'Range',tRange);
       end
      firstday=0;
    end
    fclose(fid_date)
        
end
    
fclose('all');
   


