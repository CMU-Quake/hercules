%This function is useful to plot meshing of any region  within domain in 3d
%It reads data from binary files created by Hercules.One can plot domain in
%3d and color the elements according to Vp , Vs or rho values. It is also
%useful when it is desired to see which processor posseses which element.
%All these options are available through parameters_for_matlab.in. Usage of
%parameters file is added below.

%A sample parameters_for_matlab.in file

%1-x dimension in m : 800 ** These are dimensions of whole domain
%2-y dimension in m : 800
%3-z dimension in m : 800
%4-where to start plotting (x coordinate) : 0 ** These are coordinates of
%5-where to end plotting (x coordinate) : 800    domain where you want to
%6-where to start plotting (y coordinate) : 0    plot.
%7-where to end plotting (y coordinate) : 800
%8-where to start plotting (z coordinate) : 0
%9-where to end plotting (z coordinate) : 800
%10-which of these to plot as 4th dimension Vs(1) Vp(2) or Rho(3): 1 ** Put 1,2 or 3 only
%11-number of processors used : 8  **This is essential
%12-path to directory where coordinates are stored :/Users/testrun/outputfiles/For_Matlab
%** path to your binary files created by Hercules
%13-path to directory where data are stored : /Users/testrun/outputfiles/For_Matlab
%14-plot processor(p) or data(d) : d
%** If you put p, colors will show which processor posseses which element
%and parameter 10 is disregarded.
%If you put d, colors will show Vs, Vp or Rho according to  parameter 10.

%The ordering of parameters are important and should not be changed.You are
%free to change the names of parameters as long as followed by a colon(:).
%Finally, each parameter should appear in a single line and there should not
%be any blank lines in between parameters.When calling fuction, one must
%include the name of path to parameters between quotation marks i.e.
%plot3d_Hercules('/Users/Desktop/parameters_for_matlab.in')


function  plot3d_Hercules_v2( path_to_parameters )

%Parsing parameters from parameters_for_matlab.in
fid = fopen( path_to_parameters,'r');
i=0;
temp_numbers=zeros(14,1);
while 1
    i=i+1;
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    if (i ~= 12 && i ~= 13 && i~=14)
        [ temp_string temp_numbers(i) ] = strread(tline, ' %s %d ' , 'delimiter',':');
    else
        [ temp_string temp_paths(i-11) ] = strread(tline, ' %s %s ' , 'delimiter',':');
    end
    
end
fclose(fid);

%Initiliazing of variables
x_dimension = temp_numbers(1);
y_dimension = temp_numbers(2);
z_dimension = temp_numbers(3);

%One can plot any region within domain with these inputs
x_min = temp_numbers(4);
x_max = temp_numbers(5);
y_min = temp_numbers(6);
y_max = temp_numbers(7);
z_min = temp_numbers(8);
z_max = temp_numbers(9);
Forth_dim = temp_numbers(10);

%This is done to grab data from different text files written by different
%processors.Need to know how many processors are used.

number_processors = temp_numbers(11);

directory_coord = char(temp_paths(1));
directory_data  = char(temp_paths(2));

%plot processors or data
plot_processor_p_or_data_d = char(temp_paths(3));

if (strcmp(plot_processor_p_or_data_d,'p'))
    plot_processors = 1;
else
    plot_processors = 0;
end


%-----------------------Parsing and Initiliazing Ends Here-----------------

%There are 24 coordinates and 3 data values.A is coordinate matrix.(number
%of points by 24). B is data matrix(number of points by 3)

j = 0 ;

for i=0:number_processors-1,
    
    if ( exist([ directory_data '/mesh_coordinates.' num2str(i) ] ,'file'))
       
        fid1 = fopen([ directory_data '/mesh_coordinates.' num2str(i) ]);
        %This is a auxillary matrix for holding coordinates and data.
        Coord_Matrix = fread(fid1,[24,inf],'int32');
        
        %Converting Coord_Matrix to a (number of points) by (24) matrix
        
        auxilary_coord = Coord_Matrix';
        if ( j == 0 )
            A = auxilary_coord;
        else
            A = [A ; auxilary_coord];
        end
        fclose(fid1);
        [row_size column_size] = size(auxilary_coord);
        %Repeat same procedure for data matrix if necessaery
        
        if (plot_processors == 0)
            fid2 = fopen([ directory_data '/mesh_data.' num2str(i) ]);
            Data_Matrix =  fread(fid2,[3,inf], 'float');
            fclose(fid2);
            auxilary_data  = Data_Matrix';
            if ( j == 0 )
                B = auxilary_data;
            else
                B = [B ; auxilary_data];
            end
        end
        
        %Note that B is row_size by 3
        if (plot_processors == 1)
            if ( j == 0 )
                B = i*ones(row_size,3);
            else
                B = [B ; i*ones(row_size,3)];
            end
            
        end
        j=j+1 ;
    end
end

%We need to convert them to physical coordinates

max_dimension = max ( [x_dimension y_dimension z_dimension] );

max_x_etree = 2^30 * x_dimension / max_dimension ;
max_y_etree = 2^30 * y_dimension / max_dimension ;
max_z_etree = 2^30 * z_dimension / max_dimension ;

for i=0:7,
    
    A(:,3*i+1)  = A(:,3*i+1) / ( max_x_etree / x_dimension);
    A(:,3*i+2)  = A(:,3*i+2) / ( max_y_etree / y_dimension);
    A(:,3*i+3)  = A(:,3*i+3) / ( max_z_etree / z_dimension);
    
end

[r c]=size(A);

faces_matrix=[1 3 4 2;
    5 7 8 6;
    7 8 4 3;
    5 6 2 1;
    5 7 3 1;
    6 8 4 2;];

%If the coordinates are in the max and min limits.Comparasion with left corner coordinates
for i=1:r,
    if( A(i,1) >= x_min && A(i,1) < x_max &&...
            A(i,2) >= y_min && A(i,2) < y_max &&...
            A(i,3) >= z_min && A(i,3) < z_max )
        vertex_matrix=zeros(8,3);
        
        for j=0:7
            
            vertex_matrix(j+1,1)=A(i,j*3+1);
            vertex_matrix(j+1,2)=A(i,j*3+2);
            vertex_matrix(j+1,3)=-A(i,j*3+3);%z_downwards
        end
        
        patch('Vertices',vertex_matrix,'Faces',faces_matrix,'FaceColor','flat','FaceVertexCData',B(i,Forth_dim));
        
        patch('Vertices',vertex_matrix,'Faces',faces_matrix,'FaceColor','flat','FaceVertexCData',B(i,Forth_dim));
        hold on;
    end
end
%set(gcf,'Menubar','none','Name','Cube', ...
%'NumberTitle','off','Position',[10,350,1000,1000], ...
%'Color',[0.5 0.8 0.3]);
set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1])
colorbar('EastOutside');
%light('Position',[100 100 0],'Style','local');
light('Position',[0 0 -100]);
material shiny;
alpha(0.4);
alphamap('rampdown');
camlight(45,45);
lighting phong
view(30,30);
%zoom(2);
axis([x_min x_max y_min y_max -z_max -z_min]); %z downwards in the Hercules

if(plot_processors==1)
    colors = 'id of processors';
else
    if(Forth_dim==1)
        colors = 'Vs';
    elseif(Forth_dim==2)
        colors = 'Vp';
    elseif(Forth_dim==3)
        colors = 'Rho';
    end
end
title(['Colors represent ' colors]);

end

