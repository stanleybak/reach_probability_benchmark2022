# reach_probability_benchmark2022
Benchmark for VNNCOMP 2022 related to reachability probability density networks

Meng, Yue, Dawei Sun, Zeng Qiu, Md Tawhid Bin Waez, and Chuchu Fan. "Learning Density Distribution of Reachable States for Autonomous Systems." In Conference on Robot Learning, pp. 124-136. PMLR, 2022.

## Vanderpol
vdp.onnx

rad 0.2 to 0.8
log_prob 0.15 to 0.3

trange = [0.0, 5.0]
init_box = np.array([[-2.5, 2.5], [-2.5, 2.5], trange], dtype=float)

## Robot
robot.onnx

rad = 0.0 to 0.3
log_prob between 0.05 and 0.3

 mat = np.array([[0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0], [-1, 0, 0, 0, 0]], dtype=float)
 
 
    rhs = np.array([rad, rad, rad, rad, -log_prob], dtype=float)
    spec = Specification(mat, rhs)
    
## GCAS

