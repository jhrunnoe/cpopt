function [design] = generate_design()
  n     = 2000;
  n_mac = 10;
  n_std = n  -  n_mac; % number of standard cells

  n_groups       = round(n_mac/2);       % number of disconnected groups
  max_group_size = round(n/n_groups);
  n_nets         = 10*n_mac;
  net_size_limit = max_group_size;

  design.R = struct('x', 0, 'y', 0, 'dx', 17640, 'dy', 15120);

  % Create a cell bank to choose macro shapes from
  mcr_types = 3;
  mcr_bank = cell(mcr_types, 1);
  for i = 1:mcr_types
    mcr_bank{i}.x = design.R.dx/8 + randi(round(design.R.dx/16));
    mcr_bank{i}.y = design.R.dy/8 + randi(round(design.R.dy/16));
  end

  % Create a cell bank to choose standard shapes from
  std_types = 3;
  std_bank = cell(std_types, 1);
  for i = 1:std_types
    std_bank{i}.x = (64 + randi(round(design.R.dx/420)));
    std_bank{i}.y = (64 + randi(round(design.R.dy/420)));
  end
  
  x  = zeros(n, 1);
  y  = zeros(n, 1);
  dx = zeros(n, 1);
  dy = zeros(n, 1);
  
  area = 0;

  % Place the macros
  sx = 0;
  sy = 0;
  maxy = 0;

  for i = 1:n_mac
    m     = randi(mcr_types);
    dx(i) = mcr_bank{m}.x;
    dy(i) = mcr_bank{m}.y;

    if sx + dx(i) >= design.R.dx
      sx = 0;
      sy = maxy + 32;
    end

    maxy = max(maxy, sy + dy(i));

    x(i) = sx + 0.50*dx(i);
    y(i) = sy + 0.50*dy(i);

    sx = sx + dx(i) + 32;
    
    area = area + dx(i)*dy(i);
  end

  % Place the standard cells
  for i = n_mac + 1:n
    s = randi(std_types);
    dx(i) = std_bank{s}.x;
    dy(i) = std_bank{s}.y;
    x(i) = 0.50*dx(i) + randi(design.R.dx - dx(i)) - 1;
    y(i) = 0.50*dy(i) + randi(design.R.dy - dy(i)) - 1;
    area = area + dx(i)*dy(i);
  end

  % The cells are first divided into groups and then the nets
  % are chosen from within the groups
  groups = cell(n_groups, 1);
  idx    = randperm(n);
  for i = 1:n_groups
    groups{i}.cells = idx(1 + (i - 1)*max_group_size:i*max_group_size);
    groups{i}.nets  = [];
  end
  
  netlist = cell(n_nets, 1);
  for i = 1:n_nets
    net_size = 1 + randi(net_size_limit - 1);
    j = randi(n_groups);
    netlist{i} = randsample(groups{j}.cells, net_size);
    groups{j}.nets = [groups{j}.nets i];
  end

  % Compile design data into single structure
  design.utilization = area/(design.R.dx*design.R.dy);
  
  design.x0 = x;
  design.y0 = y;
  design.dx = dx;
  design.dy = dy;

  design.netlist = netlist;
  design.groups  = groups;
  
  design.mac_idx = 1:n_mac;
  design.std_idx = n_mac + 1:n;
  
  design.n_cells  = n;
  design.n_nets   = n_nets;
  design.n_groups = n_groups;
  design.n_mac    = n_mac;
  design.n_std    = n_std;
end