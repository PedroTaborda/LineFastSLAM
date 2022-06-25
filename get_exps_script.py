import numpy as np

N_default = 50
r_odom_default = 0.1
theta_odom_default = 0.5
r_bias_default = 1
theta_bias_default = 1
r_line_default = 0.1
phi_line_default = 10*np.pi/180

N_exp = " ".join([str(n) for n in [10, 20, 50, 100, 200]])

action_model_noise_theta_bias_exp = [
    [r_bias_default, theta_bias]
    for theta_bias in np.linspace(0.1, 1.0, 10, endpoint=True)
]

action_model_noise_r_bias_exp = [
    [r_bias, theta_bias_default]
    for r_bias in np.linspace(0.05, 0.5, 10, endpoint=True)
]

action_model_noise_var_r_exp = [
    [r_var, theta_odom_default]
    for r_var in np.linspace(0.8, 1.2, 5, endpoint=True)
]
action_model_noise_var_theta_exp = [
    [r_odom_default, theta_var]
    for theta_var in np.linspace(0.8, 1.2, 5, endpoint=True)
]

r_line = " ".join([f"{r_line_val:.7f}" for r_line_val in np.linspace(0.8, 1.2, 5, endpoint=True)])

phi_line = " ".join([f"{phi_line_val:.7f}" for phi_line_val in np.linspace(0.8, 1.2, 5, endpoint=True)])

run_str_exp1 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_exp} --action-model-noise-cov "[[{r_odom_default}, {theta_odom_default}]]" \
--action-model-noise-bias "[[1, 1]]" \
-r-std-line {r_line_default} -phi-std-line {phi_line_default} --repeats 5'

run_str_exp2 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_default} --action-model-noise-cov "[[{r_odom_default}, {theta_odom_default}]]" \
--action-model-noise-bias "{action_model_noise_theta_bias_exp}" \
-r-std-line {r_line_default} -phi-std-line {phi_line_default} --repeats 5'

run_str_exp3 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_default} --action-model-noise-cov "[[{r_odom_default}, {theta_odom_default}]]" \
--action-model-noise-bias "{action_model_noise_r_bias_exp}" \
-r-std-line {r_line_default} -phi-std-line {phi_line_default} --repeats 5'

run_str_exp4 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_default} --action-model-noise-cov "{action_model_noise_var_r_exp}" \
--action-model-noise-bias "[[1, 1]]" \
-r-std-line {r_line_default} -phi-std-line {phi_line_default} --repeats 5'

run_str_exp5 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_default} --action-model-noise-cov "{action_model_noise_var_theta_exp}" \
--action-model-noise-bias "[[1, 1]]" \
-r-std-line {r_line_default} -phi-std-line {phi_line_default} --repeats 5'

run_str_exp6 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_default} --action-model-noise-cov "[[{r_odom_default}, {theta_odom_default}]]" \
--action-model-noise-bias "[[1, 1]]" \
-r-std-line {r_line} -phi-std-line {phi_line_default} --repeats 5'

run_str_exp7 = f'python3 -m slam.mass --sensor-data corridor-w-light.xz -t0 20 \
-N {N_default} --action-model-noise-cov "[[{r_odom_default}, {theta_odom_default}]]" \
--action-model-noise-bias "[[1, 1]]" \
-r-std-line {r_line_default} -phi-std-line {phi_line} --repeats 5'

print("Exps:")

print(f"{run_str_exp1}\n")

print(f"{run_str_exp2}\n")

print(f"{run_str_exp3}\n")

print(f"{run_str_exp4}\n")

print(f"{run_str_exp5}\n")

print(f"{run_str_exp6}\n")

print(f"{run_str_exp7}\n")

