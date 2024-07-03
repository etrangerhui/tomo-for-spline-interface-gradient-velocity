# tomo-for-spline-interface-gradient-velocity
This inversion method is well-founded theoretically, and the algorithm implementation utilizes many techniques, which makes the code extremely concise without much loss of functionality.
It can reconstruct the interface depth nodes while performing gradient velocity or constant velocity inversion for P and S wave velocities, effectively solving a series of problems in the wide-angle reflection field.
Compared with the tomo2d software, this method provides more ray encodings and includes more seismic phases with shear waves, providing another effective tool for OBS tomographic exploration.
Relative to grid discretization tomography, the model used in this inversion appears relatively simple and can be further complicated, such as adopting a triangular mesh gradient model, which is the content that needs to be continued in future research.
