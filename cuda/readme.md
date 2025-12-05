CUDA 版本流程：

Host: OpenCV Canny + Sobel → edge list with normals

Copy edges (x,y,nx,ny) to device

For line：allocate device accumulator (rho_bins * theta_bins), launch kernel where each thread handles one edge and loops over θ candidates (only small ±window around gradient-based angle), atomicAdd into device accumulator; copy back and detect peaks.

For circle：for each radius r, allocate device W*H int accumulator, kernel: each thread per edge computes candidate center(s) and atomicAdd into device acc; after kernel, copy acc back and find best center for that r; free device acc.

Timing: use cudaEvent_t for kernel timing; host uses chrono for Canny/Grad/host parts.