import numpy as np
import matplotlib.pyplot as plt

# 시스템 파라미터
J = 0.01  # 관성
K = 0.01  # 토크 상수

# 관측기 이득
L1 = 50
L2 = 500

# 파라미터 적응 이득
gamma = 100

dt = 0.001  # 시간 간격
T = 5       # 전체 시뮬레이션 시간
N = int(T/dt)

# 실제 상태 및 추정값 초기화
theta = 0
omega = 0
b_true = 0.1  # 실제 마찰계수

theta_hat = 0
omega_hat = 0
b_hat = 0.05  # 마찰계수 초기 추정값

# 결과 저장용 리스트
theta_hist = []
omega_hist = []
b_true_hist = []
theta_hat_hist = []
omega_hat_hist = []
b_hat_hist = []

# 입력 신호 (스텝)
u = 1.0

for i in range(N):
    # 실제 시스템 동역학
    dtheta = omega
    domega = (-b_true/J)*omega + (K/J)*u
    theta += dtheta*dt
    omega += domega*dt

    # 출력 측정 (theta)
    y = theta

    # 관측기(ESO) 동역학
    dtheta_hat = omega_hat + L1*(y - theta_hat)
    domega_hat = (-b_hat/J)*omega_hat + (K/J)*u + L2*(y - theta_hat)
    d_b_hat = gamma*(y - theta_hat)*(-omega_hat/J)  # phi(omega_hat) = -omega_hat/J (자코비안)

    theta_hat += dtheta_hat*dt
    omega_hat += domega_hat*dt
    b_hat += d_b_hat*dt

    # Projection: b_hat이 물리적 경계 내에 있도록 제한
    b_hat = np.clip(b_hat, 0.01, 0.2)

    # 결과 저장
    theta_hist.append(theta)
    omega_hist.append(omega)
    b_true_hist.append(b_true)
    theta_hat_hist.append(theta_hat)
    omega_hat_hist.append(omega_hat)
    b_hat_hist.append(b_hat)

# 결과 플롯
plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(np.arange(N)*dt, theta_hist, label='True Theta')
plt.plot(np.arange(N)*dt, theta_hat_hist, '--', label='Estimated Theta')
plt.legend()
plt.ylabel('Theta (rad)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(np.arange(N)*dt, omega_hist, label='True Omega')
plt.plot(np.arange(N)*dt, omega_hat_hist, '--', label='Estimated Omega')
plt.legend()
plt.ylabel('Omega (rad/s)')
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(np.arange(N)*dt, b_true_hist, label='True b')
plt.plot(np.arange(N)*dt, b_hat_hist, '--', label='Estimated b')
plt.legend()
plt.ylabel('Friction Coefficient b')
plt.xlabel('Time (s)')
plt.grid(True)

plt.tight_layout()
plt.show()
