classdef Placer
  properties
    design;
    nets;
    grid;
    masks;

    % Electrostatic quantities
    charge;
    density;
    potential;
    field;
    
    xind;
    yind;

    mu = 1;
  end
  properties(Constant)
    grid_resolution = 128;
    default_n_mac   = 10;
    default_n_cell  = 1000;
  end
  
  methods
    function [obj] = Placer()
      obj.design = generate_design(obj.default_n_cell, obj.default_n_mac);
      
      obj.design.z0 = [obj.design.x0; obj.design.y0];
      obj.design.dz = [obj.design.dx; obj.design.dy];

      obj.nets = cell(obj.design.n_nets, 1);
      for j = 1:obj.design.n_nets
        obj.nets{j} = Net(j, obj.design.netlist{j});
      end

      obj.grid.nx   = obj.grid_resolution;
      obj.grid.ny   = obj.grid_resolution;
      obj.grid.n    = obj.grid.nx*obj.grid.ny;
      obj.grid.dx   = ceil(obj.design.R.dx/obj.grid.nx);
      obj.grid.dy   = ceil(obj.design.R.dy/obj.grid.ny);
      obj.grid.area = obj.grid.dx*obj.grid.dy;

      [obj.density, obj.masks] = obj.compute_density(obj.design.x0, obj.design.y0);
      obj.charge    = obj.design.dx.*obj.design.dy;
      obj.potential = zeros(obj.grid.nx, obj.grid.ny);
      obj.field.x   = zeros(obj.grid.nx, obj.grid.ny);
      obj.field.y   = zeros(obj.grid.nx, obj.grid.ny);

      obj.xind = 1:obj.design.n_cells;
      obj.yind = (obj.design.n_cells + 1):2*obj.design.n_cells;
    end

    function [] = plot_cells(obj, x, y, cells)
      if isempty(cells)
        cells = 1:obj.design.n_cells;
      end
      hold on;
      for i = cells
        rectangle('Position', [x(i) - 0.50*obj.design.dx(i), ...
                               y(i) - 0.50*obj.design.dy(i), ...
                               obj.design.dx(i), ...
                               obj.design.dy(i)]);
      end
      axis([0, (obj.grid.nx - 1)*obj.grid.dx, 0, (obj.grid.ny - 1)*obj.grid.dy]);
      hold off;
    end

    function [hm] = plot_heatmap(obj, M)
      hm = imagesc(M');
      set(gca, 'Ydir', 'normal');
      hm.XData = (hm.XData - 1)*obj.grid.dx;
      hm.YData = (hm.YData - 1)*obj.grid.dy;
      axis([0, (obj.grid.nx - 1)*obj.grid.dx, 0, (obj.grid.ny - 1)*obj.grid.dy]);
    end

    function [clearance] = compute_clearance(obj, x, y, cells)
      %% ----------------- compute_clearance()
      %  Computation of the minimum pairwise clearance for the provided
      %  group of cells. The result is positive if there is no overlap,
      %  zero when two or more cells are touching, and negative if two or
      %  more cells intersect.
      
      pairs   = nchoosek(cells, 2);
      n_pairs = size(pairs, 1);
      deltas  = zeros(n_pairs, 1);

      for k = 1:n_pairs
        xi = x(pairs(k, 1)); 
        xj = x(pairs(k, 2));
        yi = y(pairs(k, 1)); 
        yj = y(pairs(k, 2));

        dxi = obj.design.dx(pairs(k, 1));
        dxj = obj.design.dx(pairs(k, 2));
        dyi = obj.design.dy(pairs(k, 1));
        dyj = obj.design.dy(pairs(k, 2));
        
        dijx = max(xi - 0.50*dxi, xj - 0.50*dxj) - min(xi + 0.50*dxi, xj + 0.50*dxj);
        dijy = max(yi - 0.50*dyi, yj - 0.50*dyj) - min(yi + 0.50*dyi, yj + 0.50*dyj);

        deltas(k) = max(dijx, dijy);
      end
      clearance = min(deltas);
    end

    function [result] = solve(obj)
      x = obj.design.x0;
      y = obj.design.y0;
      z = obj.design.z0;
    end

    function [f, g, D, M] = evaluate(obj, z)
      %% ------------ evaluate()
      %
      %  DESCRIPTION
      %  This computes the objective function 
      %       f(z) = W(z) + mu*E(z)
      %  and its gradient, where W is some measure of the
      %  wire length and E is the system electrostatic energy.
      %  Possible choices for W(z) include the weighted average
      %  smooth approximation of half-perimeter wire length, or
      %  the pin coordinate variances.
      %  
      %  Parameters
      %  z: the current placement coordinates z = [x; y].
      %  D: the previous density matrix  - will be updated to reflect the new placement z.
      %  M: the previous occupancy masks - will be updated to reflect the new placement z.
      % 
      %  Returns
      %  f: the function value.
      %  g: the gradient of f at z.
      %  D: updated density matrix.
      %  M: updated occupancy masks.
      
      x = z(obj.xind); 
      y = z(obj.yind);

      [D, M]    = obj.update_density(x, y, 1:obj.design.n_cells);
      [P, F, E] = obj.compute_electrostatics(D);

      % Determine in which grid elements each cell locates, extract 
      % the electric field at those locations, and scale by the cell 
      % charges (area):
      dE.x = zeros(size(x));
      dE.y = zeros(size(y));

      for k = 1:obj.design.n_cells
        [I, J] = find(M{k});
        % The total force on cell k will be the sum of the forces
        % due to the electrostatic field at each grid element scaled by the overlap
        % area there.
        dE.x(k) = -sum(F.x(sub2ind(size(F.x), I, J)), "all");
        dE.y(k) = -sum(F.y(sub2ind(size(F.y), I, J)), "all");
      end

      % The wirelength gradient is the sum over nets 
      dW.x = zeros(size(x));
      dW.y = zeros(size(y));
      W = 0;
      for j = 1:obj.design.n_nets
        [nW, ndW] = obj.nets{j}.pin_spread(x, y);
        W = W + nW;
        dW.x(obj.nets{j}.cells) = dW.x(obj.nets{j}.cells) + ndW.x;
        dW.y(obj.nets{j}.cells) = dW.y(obj.nets{j}.cells) + ndW.y;
      end
      
      f = W + obj.mu*E;
      g = [dW.x + obj.mu*dE.x;
           dW.y + obj.mu*dE.y];
    end

    function [D, M] = update_density(obj, x, y, cells)
      %% ------------ update_density()
      %
      %  Description
      %  Updates the density matrix D and occupancy masks M based
      %  on new coordinates x, y. It is assumed only cells listed
      %  change position. The update is done by subtracting out
      %  the density contribution of the moved cells, updating
      %  their sparse density mask, and adding the new
      %  contribution back into D. 
      %
      %  Parameters
      %  x    : new cell coordinates
      %  y    : new cell coordinates
      %  D    : (sparse double) current density matrix.
      %  M    : (cell array) cell array of sparse occupancy masks.
      %  cells: list of cells to update.
      %
      %  Returns
      %  D: (sparse double) Density matrix
      %  M: (cell array) Grid-cell occupancy masks
      %
      D = obj.density;
      M = obj.masks;
      for k = cells
        [I, J, v] = find(obj.masks{k});
        di = floor((x(k) - 0.50*obj.design.dx(k))/obj.grid.dx) - I(1) + 1; 
        dj = floor((y(k) - 0.50*obj.design.dy(k))/obj.grid.dy) - J(1) + 1; 
        if di ~= 0 || dj ~= 0
          D    = D - obj.masks{k};
          M{k} = sparse(I + di, J + dj, v, obj.grid.nx, obj.grid.ny);
          D    = D + M{k};
        end
      end
    end

    function [D, M] = compute_density(obj, x, y)
      %% ------------ compute_density()
      %
      %  Description
      %  Compute the density based on the placement coordinates
      %  x, y. The instances listed in the cells parameter are
      %  assumed to have moved relative to the coordinates used
      %  to compute the provided D, M. If cells lists all design
      %  cells, the density is computed from scratch, otherwise
      %  the density is updated.
      %
      %  Parameters
      %  x    : new cell coordinates
      %  y    : new cell coordinates
      %  D    : (sparse double) current density matrix.
      %  M    : (cell array) cell array of sparse occupancy masks.
      %  cells: list of cells to update.
      %
      %  Returns
      %  D: (sparse double) Density matrix
      %  M: (cell array) Grid-cell occupancy masks
      %

      D = sparse(obj.grid.nx, obj.grid.ny);
      M = cell(obj.design.n_cells, 1);
      for k = 1:obj.design.n_cells
        li = floor((x(k) - 0.50*obj.design.dx(k))/obj.grid.dx) + 1;
        ui = floor((x(k) + 0.50*obj.design.dx(k))/obj.grid.dx) + 1;
        lj = floor((y(k) - 0.50*obj.design.dy(k))/obj.grid.dy) + 1;
        uj = floor((y(k) + 0.50*obj.design.dy(k))/obj.grid.dy) + 1;
        M{k} = sparse(obj.grid.nx, obj.grid.ny);
        M{k}(li:ui, lj:uj) = 1;
        % The density is the sum over individual grid-cell occupancy masks
        D = D + M{k};
      end
    end

    function [P, F, E] = compute_electrostatics(obj, D)
      %% --------------- compute_electrostatics()
      %
      %  Description
      %  Computes the electrostatic potential map, horizontal and
      %  vertical field maps, and total energy using fast Fourier
      %  transforms
      %
      %  Parameters
      %  D: density matrix
      %
      %  Returns
      %  P: (double) potential map such that P, D satisfy Poisson's equation
      %  F: (struct) electric field maps F.x and F.y
      %  E: (scalar) total system energy

      if issparse(D)
        D = full(D);
      end

      % Generate the mesh of support for the cosine basis
      u  = (pi/obj.grid.nx)*(0:(obj.grid.nx - 1));
      v  = (pi/obj.grid.ny)*(0:(obj.grid.ny - 1));

      % Compute the Fourier coefficients to express the density D in the
      % cosine basis. The 2D DCT is achieved by iterating 1D DCT
      A = dct(dct(D, [], dim=1), [], dim=2);

      % Compute the electrostatic potential: (the coefficients
      % A./S come from rho(x, y) = -laplacian(psi(x, y)), writing the expansion
      % of the density in the cosine basis, and integrating twice
      % in each variable.
      S       = (u.^2)' + v.^2; % (S)um of squares
      S(1, 1) = 1;
      C       = A./S;
      C(1, 1) = 0;

      P = idct(idct(C, [], dim=1), [], dim=2);

      % The force calculation can also be done using the idst function:
      % T   = idct(Cu')';
      % F.x = idst([T(2:end, :); zeros(1, obj.grid.ny)]);
      % F.y = idct(idst([Cv(:, 2:end) zeros(obj.grid.nx, 1)]')');

      % Differentiating the potential map with respect to x gives the 
      % horizontal component of the electrostatic field. This multiplies
      % each row of the coefficient matrix C by elements of u.
      Cu  = u'.*C;
      F.x = obj.idxst(idct(Cu, [], "dim", 2));

      % Differentiating the potential map with respect to y gives the
      % vertical component of the electrostatic field.
      Cv  = C.*v;
      F.y = idct(obj.idxst(Cv, [], "dim", 2));

      % Total system energy is half the sum over all charge-potential
      % products (charge represented by area)
      E = 0.50*sum(P.*D, "all");
    end
  end

  methods(Static)
    function [area] = compute_overlap(r1, r2)
      %% ------------ compute_overlap(r1, r2)
      % Compute the area of the overlap of two rectangles r1, r2. 
      dx = max(0, min(r1.x + r1.dx, r2.x + r2.dx) - max(r1.x, r2.x));
      dy = max(0, min(r1.y + r1.dy, r2.y + r2.dy) - max(r1.y, r2.y));
      area = dx*dy;
    end

    function [Y] = idxst(X, ~, varargin)
      %% --------- idxst(X, ~, varargin)
      % Purpose: 
      % Compute the inverse discrete sine transform of the input
      % transformed by [a_1, a_2, a_3,..., a_n] -> [a_2, a_3,..., a_n, 0]
      % in terms of the idct function (by reversing the input and flipping 
      % the sign of every other output).
      if nargin > 2
        dim = varargin{2};
      else
        dim = 1;
      end
      if dim == 1
        X = X';
      end
      [m, ~] = size(X);
      Y = zeros(size(X));
      for i = 1:m
        y = idct(flip([X(i, 2:end) 0]));
        y(2:2:end) = -y(2:2:end);
        Y(i, :) = y;
      end
      if dim == 1
        Y = Y';
      end
    end
  end
end