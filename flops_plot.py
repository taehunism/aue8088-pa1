import matplotlib.pyplot as plt

# 데이터 정의
model_names = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
params = [34.63, 51.03, 58.69, 85.14, 131.97, 205.81, 292.28]  # x축
flops = [4.26, 6.77, 7.98, 11, 17.91, 28.75, 41.2]  # y축

# 플롯 그리기
plt.figure(figsize=(8, 6))
plt.plot(params, flops, marker='o', linestyle='-', color='blue')

# 각 점에 모델 이름 표시
for name, x, y in zip(model_names, params, flops):
    plt.text(x, y + 0.5, name, ha='center', fontsize=9)

# 축 라벨 및 제목
plt.xlabel('Params [M]')
plt.ylabel('FLOPs [M]')
plt.title('EfficientNet Family: Params vs FLOPs')
plt.grid(True)
plt.tight_layout()

plt.show()

# import matplotlib.pyplot as plt

# # 데이터 정의
# model_names = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
# params = [34.63, 51.03, 58.69, 85.14, 131.97, 205.81]
# flops = [4.26, 6.77, 7.98, 11, 17.91, 28.75]
# val_accuracy = [0.2929, 0.2798, 0.2711, 0.2871, 0.2808, 0.3092]

# # 플롯 그리기
# plt.figure(figsize=(8, 6))
# plt.scatter(flops, val_accuracy, color='green', s=80)

# # 모델 이름 라벨링
# for name, x, y in zip(model_names, flops, val_accuracy):
#     plt.text(x, y + 0.003, name, ha='center', fontsize=9)

# # 축 라벨 및 제목
# plt.xlabel('FLOPs [M]')
# plt.ylabel('accuracy/val')
# plt.title('EfficientNet Family: FLOPs vs accuracy/val')
# plt.grid(True)
# plt.tight_layout()

# plt.show()
