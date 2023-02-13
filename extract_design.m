function [design] = extract_design(design_name, path)
  file_name = strcat('./', design_name, '_design_data.mat');
  if isfile(file_name)
    load(file_name, 'design');
    design.cell_names = cellfun(@strtrim, cellstr(design.cell_names), 'UniformOutput', false);
    design.net_names  = cellfun(@strtrim, cellstr(design.net_names),  'UniformOutput', false);
  else
    dir  = [path design_name '/' design_name];
    np = py.importlib.import_module('numpy');
    
    % Instance dimensions and initial placement extraction
    cell_file = gunzip([path 'cells.json.gz']);
    spec_file = gunzip([dir '.json.gz']);
  
    cell_specs   = jsondecode(fileread(cell_file{1}));
    design_specs = jsondecode(fileread(spec_file{1}));
    
    design.n_cells = length(design_specs.instances);
    design.n_nets  = length(design_specs.nets);
  
    design.x0 = zeros(design.n_cells, 1);
    design.y0 = zeros(design.n_cells, 1);
    design.dx = zeros(design.n_cells, 1);
    design.dy = zeros(design.n_cells, 1);
    design.px = zeros(design.n_cells, 1);
    design.py = zeros(design.n_cells, 1);
  
    design.cell_names = cell(design.n_cells, 1);
  
    for i = 1:design.n_cells
      instance = design_specs.instances(i);
      xloc   = instance.xloc;
      yloc   = instance.yloc;
      index  = instance.cell + 1; % For looking up dimensions of given cell type
      orient = instance.orient;
      id     = instance.id + 1; % To account for zero-indexing
  
      width  = cell_specs(index).width;
      height = cell_specs(index).height;
      
      if ismember(orient, [1, 3, 5, 7])
        dx = height;
        dy = width;
      else
        dx = width;
        dy = height;
      end
      if ismember(orient, [1 2 4 5])
        xloc = xloc - dx; % lower x coordinate after rotation/relection
      end
      if ismember(orient, [2 3 5 6])
        yloc = yloc - dy; % lower y coordinate after rotation/reflection
      end
      if isfield(cell_specs(index).terms, 'xloc') && isfield(cell_specs(index).terms, 'yloc')
        xpin = mean([cell_specs(index).terms.xloc]);
        ypin = mean([cell_specs(index).terms.yloc]);
      else
        xpin = 0.50*dx
        ypin = 0.50*dy
      end
      design.x0(id) = xloc;
      design.y0(id) = yloc;
      design.dx(id) = dx;
      design.dy(id) = dy;
      design.px(id) = xpin;
      design.py(id) = ypin;
      design.cell_names{id} = instance.name;
    end
  
    % Check for .def file - if it exists, read the 
    % placement region coordinates from the .def file
    if isfile([dir '_route_opt.def'])
      def = fopen([dir '_route_opt.def'], "r");
      while ~feof(def)
        line = fgetl(def);
        if startsWith(line, 'DIEAREA')
          die = str2double(regexp(line, '\d*','match'));
          R.x = die(1);
          R.y = die(2);
          R.dx = die(5) - R.x;
          R.dy = die(6) - R.y;
        end
      end
      fclose(def);
    else
      R.x = 0;
      R.y = 0;
      R.dx = 1.25*max(design.x0);
      R.dy = 1.25*max(design.y0);
    end
    design.R = R;

    % Netlist extraction
    connectivity = np.load([dir '_connectivity.npz']);
    rows  = double(pyrun("rows = conn['row']", "rows", conn = connectivity));
    cols  = double(pyrun("cols = conn['col']", "cols", conn = connectivity));
    
    design.netlist   = cell(design.n_nets, 1);
    design.net_names = cell(design.n_nets, 1);
    for j = 1:design.n_nets
      design.net_names{j} = design_specs.nets(j).name;
      design.netlist{j} = rows(cols == j - 1) + 1; % zero-indexing
    end

    save(file_name, 'design');
  end
end