classdef Net < handle
  properties
    id;
    cells;
    net_size;
  end

  properties(Constant)
    smooth_param = 256
  end

  methods
    function [obj] = Net(id, cells)
      obj.id       = id;
      obj.cells    = cells;
      obj.net_size = length(cells);
    end

    function [pins] = locate_pins(obj, x, y)
      pins.x = x(obj.cells);
      pins.y = y(obj.cells);
    end

    function [W, dW] = WAWL(obj, x, y)
      %% ------------- WAWL()
      %  Computation of value and gradient of the weighted-average smooth 
      %  approximation of the net's half-perimeter wirelength. 
      pins = obj.locate_pins(x, y);

      [maxSx, maxdSx] = obj.smooth_extrema(pins.x,  obj.smooth_param);
      [maxSy, maxdSy] = obj.smooth_extrema(pins.y,  obj.smooth_param);

      [minSx, mindSx] = obj.smooth_extrema(pins.x, -obj.smooth_param);
      [minSy, mindSy] = obj.smooth_extrema(pins.y, -obj.smooth_param);

      W   = maxSx  - minSx  + maxSy  - minSy;
      dW.x = maxdSx - mindSx;
      dW.y = maxdSy - mindSy;
    end

    function [V, dV] = pin_spread(obj, x, y)
      %% ------------- pin_spread()
      %  Computation of the value and gradient for sum of the variances of 
      %  the net's x and y coordinates.  
      pins  = obj.locate_pins(x, y);
      V     = var(pins.x) + var(pins.y);
      dV.x  = (2/obj.net_size)*(pins.x - mean(pins.x));
      dV.y  = (2/obj.net_size)*(pins.y - mean(pins.y));
    end

  end
  
  methods(Static)
    function [S_, dS_] = smooth_extrema(z, t)
      %% --------------- smooth_extrema()
      % The smoothing paramater determines whether the minimum (t < 0) or
      % the maximum (t > 0) of w is approximated. Note that the argument z
      % is normalized, the approximate maximum is computed, and then
      % rescaled. This is to prevent overflow in v = exp(t*u).
      z_norm = norm(z);
      
      u = z/z_norm;
      v = exp(t*u); 
      w = v/sum(v);

      S = w'*u; % Smooth max of the unit vector u
      q = w.*(1 + t*(u - S));
      
      S_  = z_norm*S;  
      dS_ = q + (S - q'*u)*u;
    end
  end
end