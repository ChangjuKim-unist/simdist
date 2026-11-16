# 설계 27 설명용: 미끄러진 순간의 상태 (판정에 쓰지 않음)

## broad: slip-event state, planner-made vs policy-made (mean, ratio) | AUC of feature for policy-head recovery fall (pooled)
| feature | planner-made | policy-made | ratio | AUC policy fall | AUC planner fall |
|---|---|---|---|---|---|
| slip_ma | 1.348 | 1.140 | 1.18 | 0.75 | 0.70 |
| tilt | 0.252 | 0.221 | 1.14 | 0.83 | 0.70 |
| rp_rate | 1.804 | 1.671 | 1.08 | 0.68 | 0.60 |
| yaw_rate | 0.321 | 0.323 | 0.99 | 0.58 | 0.60 |
| vx | 0.811 | 0.806 | 1.01 | 0.39 | 0.39 |
| abs_vy | 0.120 | 0.095 | 1.27 | 0.61 | 0.56 |
| vz | 0.171 | 0.136 | 1.26 | 0.68 | 0.66 |
| height | 0.354 | 0.358 | 0.99 | 0.45 | 0.49 |
| qd | 13.943 | 10.159 | 1.37 | 0.69 | 0.63 |
| hist_jerk | 0.998 | 0.668 | 1.50 | 0.76 | 0.64 |
| hist_rate | 1.318 | 1.093 | 1.21 | 0.75 | 0.64 |
| cmd_vx | 0.960 | 0.970 | 0.99 | 0.67 | 0.67 |
| since | 1153.410 | 1915.410 | 0.60 | 0.46 | 0.48 |
falls (n points with clone-mean>0.5): policy 64 planner 89 of 400

## paper: slip-event state, planner-made vs policy-made (mean, ratio) | AUC of feature for policy-head recovery fall (pooled)
| feature | planner-made | policy-made | ratio | AUC policy fall | AUC planner fall |
|---|---|---|---|---|---|
| slip_ma | 1.222 | 1.167 | 1.05 | 0.67 | 0.66 |
| tilt | 0.130 | 0.103 | 1.26 | 0.84 | 0.70 |
| rp_rate | 1.544 | 1.062 | 1.45 | 0.75 | 0.71 |
| yaw_rate | 0.212 | 0.331 | 0.64 | 0.59 | 0.66 |
| vx | 0.863 | 0.862 | 1.00 | 0.26 | 0.27 |
| abs_vy | 0.226 | 0.217 | 1.04 | 0.25 | 0.30 |
| vz | 0.107 | 0.087 | 1.24 | 0.56 | 0.51 |
| height | 0.322 | 0.305 | 1.06 | 0.53 | 0.56 |
| qd | 9.924 | 9.471 | 1.05 | 0.63 | 0.60 |
| hist_jerk | 0.933 | 0.749 | 1.24 | 0.51 | 0.49 |
| hist_rate | 1.215 | 1.163 | 1.05 | 0.66 | 0.65 |
| cmd_vx | 0.859 | 0.869 | 0.99 | 0.71 | 0.67 |
| since | 1287.710 | 1146.850 | 1.12 | 0.46 | 0.47 |
falls (n points with clone-mean>0.5): policy 38 planner 43 of 400
