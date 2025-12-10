import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PART 1: RIEMANN SUM TOOL
# ==========================================

def riemann_sum(func, a, b, n, method='midpoint'):
    """
    Computes the Riemann Sum for function `func` over [a, b] with n subintervals.
    """
    dx = (b - a) / n
    # Generate points
    x = np.linspace(a, b, n + 1)

    if method == 'left':
        points = x[:-1]
    elif method == 'right':
        points = x[1:]
    elif method == 'midpoint':
        points = (x[:-1] + x[1:]) / 2
    else:
        raise ValueError("Method must be 'left', 'right', or 'midpoint'")

    # Calculate area: sum(f(ci) * dx)
    area = np.sum(func(points) * dx)
    return area, points, dx


def plot_riemann(func, a, b, n, method, title):
    """
    Plots the function and the Riemann sum rectangles.
    """
    # Create smooth line for graph
    x_dense = np.linspace(a, b, 1000)
    y_dense = func(x_dense)

    # Calculate Riemann Sum
    area, points, dx = riemann_sum(func, a, b, n, method)

    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, y_dense, 'b', label='f(x)')

    # Draw rectangles
    for point in points:
        height = func(point)
        # Determine the x-coordinate of the rectangle's left edge
        if method == 'left':
            x_rect = point
        elif method == 'right':
            x_rect = point - dx
        else:  # midpoint
            x_rect = point - dx / 2

        plt.bar(x_rect, height, width=dx, align='edge', alpha=0.4, edgecolor='black', color='orange')

    # Formatting title without backslash n
    plt.title(f"{title} -- Approx Area: {area:.4f}")
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================================
# EXECUTION: PART 1 (GRAPHS & CALCULATIONS)
# ==========================================

print("--- PART 1: GRAPHS ---")
# 1. Graph sin(x) + 1 over [-pi, pi] with 4 subintervals
f_sin = lambda x: np.sin(x) + 1
plot_riemann(f_sin, -np.pi, np.pi, 4, 'left', 'Left-Hand Sum (n=4)')
plot_riemann(f_sin, -np.pi, np.pi, 4, 'right', 'Right-Hand Sum (n=4)')
plot_riemann(f_sin, -np.pi, np.pi, 4, 'midpoint', 'Midpoint Sum (n=4)')

print("")
print("--- PART 1: NUMERICAL & EXACT CALCULATIONS ---")

# Function 1: 3x + 2x^2 over [0, 1]
# Antiderivative F(x) = (3/2)x^2 + (2/3)x^3
def exact_f1(x): return (1.5 * x ** 2) + ((2 / 3) * x ** 3)

val_exact_1 = exact_f1(1) - exact_f1(0)
# Numerical Check (n=10000)
f1 = lambda x: 3 * x + 2 * x ** 2
val_num_1, _, _ = riemann_sum(f1, 0, 1, 10000, 'midpoint')

print(f"1. Area of 3x + 2x^2 over [0, 1]")
print(f"   - Exact Value (Calculus): {val_exact_1:.6f}")
print(f"   - Numerical Approx (n=10k): {val_num_1:.6f}")


# Function 2: ln(x) over [1, e]
# Antiderivative F(x) = x*ln(x) - x
def exact_f2(x): return x * np.log(x) - x

val_exact_2 = exact_f2(np.e) - exact_f2(1)
# Numerical Check (n=10000)
f2 = lambda x: np.log(x)
val_num_2, _, _ = riemann_sum(f2, 1, np.e, 10000, 'midpoint')

print(f"2. Area of ln(x) over [1, e]")
print(f"   - Exact Value (Calculus): {val_exact_2:.6f}")
print(f"   - Numerical Approx (n=10k): {val_num_2:.6f}")


# Function 3: x^2 - x^3 over [-1, 0]
# Antiderivative F(x) = (1/3)x^3 - (1/4)x^4
def exact_f3(x): return ((1 / 3) * x ** 3) - (0.25 * x ** 4)

val_exact_3 = exact_f3(0) - exact_f3(-1)
# Numerical Check (n=10000)
f3 = lambda x: x ** 2 - x ** 3
val_num_3, _, _ = riemann_sum(f3, -1, 0, 10000, 'midpoint')

print(f"3. Area of x^2 - x^3 over [-1, 0]")
print(f"   - Exact Value (Calculus): {val_exact_3:.6f}")
print(f"   - Numerical Approx (n=10k): {val_num_3:.6f}")


# ==========================================
# PART 1(C): General Solution Graph for ln(x)
# ==========================================
print("")
print("--- Generating Part 1(c) High Granularity Plot for ln(x) ---")

# Define the function and interval
f_ln = lambda x: np.log(x)
a_ln = 1
b_ln = np.e

# 1. Create data with "highest granularity" (many points for a smooth curve)
x_dense_ln = np.linspace(0.5, 3.5, 1000)  # Extended range for better view
y_dense_ln = f_ln(x_dense_ln)

# 2. Setup the plot
plt.figure(figsize=(10, 6))

# Plot the function line (Removed LaTeX formatting backslashes)
plt.plot(x_dense_ln, y_dense_ln, label='f(x) = ln(x)', color='blue', linewidth=2)

# 3. Fill the area under the curve for the definite integral [1, e]
x_fill_ln = np.linspace(a_ln, b_ln, 1000)
y_fill_ln = f_ln(x_fill_ln)
plt.fill_between(x_fill_ln, y_fill_ln, color='lightblue', alpha=0.5, label='Area integral 1 to e of ln(x) dx')

# Add visual markers for the limits
plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=np.e, color='gray', linestyle='--', alpha=0.7)
plt.text(1, -0.2, 'x=1', ha='center')
plt.text(np.e, -0.2, 'x=e', ha='center')

# Labels and formatting (Removed LaTeX formatting backslashes)
plt.title('Part 1(c): General Solution of integral 1 to e of ln(x) dx')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(0, color='black', linewidth=1) # x-axis
plt.axvline(0, color='black', linewidth=1) # y-axis
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ==========================================
# EXECUTION: PART 2 (NETWORK DATA)
# ==========================================
print("")
print("PART 2: NETWORK DATA ANALYSIS")

# 1. Data Input (Minutes, Mbps)
times_min = np.arange(1, 31)
rates_mbps = np.array([
    12.5, 24.8, 45.2, 48.1, 52.4, 51.9, 55.3, 53.7, 49.2, 42.1,
    38.5, 35.6, 36.8, 41.2, 45.9, 47.3, 48.0, 49.5, 50.1, 46.4,
    28.7, 18.2, 22.4, 34.9, 44.1, 49.8, 51.2, 52.6, 50.5, 48.9
])

# 2. Define Continuous Function R(t) via Polynomial Curve Fitting
# Degree 5 gives a good approximation of the curve
coeffs = np.polyfit(times_min, rates_mbps, 5)
R_t = np.poly1d(coeffs)

# 3. Calculate Total Data Transfer
# We integrate the Rate function from t=0 to t=30 minutes.
# Integral of Rate (Mbps) dt = Megabits
t_start = 0
t_end = 30
integral_val, _, _ = riemann_sum(R_t, t_start, t_end, 1000, 'midpoint')

# 4. Unit Conversion
# The integral is in units of (Mbps * minutes).
# We need strictly Megabits, so we convert minutes to seconds (x 60).
total_megabits = integral_val * 60
total_megabytes = total_megabits / 8

print(f"Integral Result (Area under curve): {integral_val:.2f} Mbps*min")
print(f"Total Data Transferred: {total_megabits:.2f} Megabits")
print(f"Total Data Transferred: {total_megabytes:.2f} Megabytes (MB)")

# Visualization of Part 2
t_dense = np.linspace(0, 30, 200)
plt.figure(figsize=(10, 6))
plt.scatter(times_min, rates_mbps, color='red', label='Measured Data Points')
plt.plot(t_dense, R_t(t_dense), 'b-', label='Approximated Rate Function R(t)')
plt.fill_between(t_dense, R_t(t_dense), alpha=0.3, color='blue', label='Total Data (Integral)')
plt.title('Network Download Rate over Time (30 min)')
plt.xlabel('Time (minutes)')
plt.ylabel('Rate (Mbps)')
plt.legend()
plt.grid(True)
plt.show()