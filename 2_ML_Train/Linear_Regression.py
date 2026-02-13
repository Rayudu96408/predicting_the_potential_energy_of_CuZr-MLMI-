# ============================================
#  Import required libraries
# ============================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.style.use('default')

# ============================================
#  Load dataset
# ============================================
data = pd.read_csv("data5.csv")

# ============================================
#  Define features and target
# ============================================
X = data[['Temp', 'Vol']]
y = data['PE']

print("Target variable (PE) statistics:")
print(y.describe())
print()

# ============================================
#  Split dataset
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================
#  Train Linear Regression Model
# ============================================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================================
#  Evaluate model performance
# ============================================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(" Linear Regression Model for PE Prediction")
print("------------------------------------------------")
print(f"Intercept (b0): {model.intercept_:.6f}")
print(f"Coefficients (b1 for Temp, b2 for Vol): {model.coef_}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print()

# ============================================
#  Plot 1: Predicted vs Actual PE
# ============================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, s=50)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
plt.xlabel("Actual PE (eV)")
plt.ylabel("Predicted PE (eV)")
plt.title("Predicted vs Actual Potential Energy (PE)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================
#  Plot 2: PE vs Volume with Connected Line
# ============================================
sorted_data = data.sort_values(by='Vol')

plt.figure(figsize=(8, 6))
plt.scatter(sorted_data['Vol'], sorted_data['PE'], color='purple', alpha=0.6, s=50, label='Data Points')
plt.plot(sorted_data['Vol'], sorted_data['PE'], color='orange', linewidth=2, label='Connected Line')
plt.xlabel("Volume (Å³)")
plt.ylabel("Potential Energy (eV)")
plt.title("PE vs Volume (CuZr Simulation)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================
#  Plot 3: 3D Regression Surface
# ============================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(data['Temp'], data['Vol'], data['PE'], 
                    c=data['PE'], cmap='viridis', marker='o', 
                    alpha=0.7, s=50)

temp_range = np.linspace(data['Temp'].min(), data['Temp'].max(), 20)
vol_range = np.linspace(data['Vol'].min(), data['Vol'].max(), 20)
temp_grid, vol_grid = np.meshgrid(temp_range, vol_range)
pe_pred_grid = model.intercept_ + model.coef_[0] * temp_grid + model.coef_[1] * vol_grid

surface = ax.plot_surface(temp_grid, vol_grid, pe_pred_grid, 
                         color='red', alpha=0.3)

ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Volume (Å³)')
ax.set_zlabel('Potential Energy (eV)')
ax.set_title("3D Regression Surface for PE Prediction")

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
cbar.set_label('Potential Energy (eV)')

plt.tight_layout()
plt.show()

# ============================================
#  Gradient Descent with RMSE
# ============================================
X_train_norm = (X_train - X_train.mean()) / X_train.std()
X_test_norm = (X_test - X_train.mean()) / X_train.std() 

X_train_norm.insert(0, 'Intercept', 1)
X_test_norm.insert(0, 'Intercept', 1)

X_train_np = X_train_norm.values
X_test_np = X_test_norm.values
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

np.random.seed(42)
theta = np.random.randn(X_train_np.shape[1], 1)

learning_rate = 0.01
epochs = 100
train_rmse_history = []
test_rmse_history = []

for epoch in range(epochs):
    y_train_pred = X_train_np.dot(theta)
    train_error = y_train_pred - y_train_np
    train_rmse = np.sqrt((1 / len(y_train_np)) * np.sum(train_error ** 2))
    train_rmse_history.append(train_rmse)
    
    y_test_pred = X_test_np.dot(theta)
    test_error = y_test_pred - y_test_np
    test_rmse = np.sqrt((1 / len(y_test_np)) * np.sum(test_error ** 2))
    test_rmse_history.append(test_rmse)
    
    gradient = (2 / len(y_train_np)) * X_train_np.T.dot(train_error)
    theta -= learning_rate * gradient

# ============================================
#  Plot 4: RMSE over Epochs
# ============================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_rmse_history, label='Training RMSE', color='blue', lw=2)
plt.plot(range(1, epochs + 1), test_rmse_history, label='Testing RMSE', color='red', lw=2, linestyle='--')
plt.xlabel("Epochs")
plt.ylabel("Root Mean Squared Error (RMSE)")
plt.title("Training vs Testing RMSE over Epochs (Gradient Descent)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.show()

# ============================================
#  Final Results
# ============================================
print("\nGradient Descent Results:")
print(f"Final Training RMSE: {train_rmse_history[-1]:.6f}")
print(f"Final Testing RMSE: {test_rmse_history[-1]:.6f}")
print(f"Final Theta values: {theta.flatten()}")
