import numpy as np
from vrp_solver import VRPSolver
from dyt_solver import DyTSolver
import time
import matplotlib.pyplot as plt
from typing import List, Tuple
import pandas as pd
import matplotlib as mpl
import sys
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import seaborn as sns

# 设置中文字体和样式
plt.style.use('default')  # 使用默认样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.edgecolor'] = 'white'
plt.rcParams['grid.color'] = '#E0E0E0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#666666'
plt.rcParams['ytick.color'] = '#666666'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def create_gradient_colors(n: int, start_color: str = '#FF6B6B', end_color: str = '#4ECDC4'):
    """创建渐变色列表"""
    return [mpl.colors.to_hex(c) for c in mpl.colors.LinearSegmentedColormap.from_list(
        'custom', [start_color, end_color])(np.linspace(0, 1, n))]

def generate_random_points(num_points: int, x_range: tuple = (0, 100), y_range: tuple = (0, 100)):
    """生成随机配送点"""
    return [(np.random.uniform(x_range[0], x_range[1]),
             np.random.uniform(y_range[0], y_range[1])) for _ in range(num_points)]

def generate_time_windows(num_points: int, time_range: tuple = (0, 100)):
    """生成随机时间窗口"""
    return [(np.random.uniform(time_range[0], time_range[1]),
             np.random.uniform(time_range[0], time_range[1]) + 2) for _ in range(num_points)]

def plot_routes(depot: Tuple[float, float], 
                delivery_points: List[Tuple[float, float]], 
                routes: List[List[int]], 
                title: str,
                description: str,
                time_windows: List[Tuple[float, float]] = None,
                service_times: List[float] = None):
    """绘制路线图"""
    # 根据数据范围自动调整图形大小
    x_coords = [p[0] for p in delivery_points] + [depot[0], depot[0] + 20]
    y_coords = [p[1] for p in delivery_points] + [depot[1]]
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    # 计算合适的图形大小
    aspect_ratio = y_range / x_range
    base_width = 12  # 减小基础宽度
    base_height = base_width * aspect_ratio
    
    # 创建图形，使用计算出的尺寸，并增加宽度以容纳图例
    fig = plt.figure(figsize=(base_width * 1.4, base_height))
    
    # 创建子图，根据是否有路线数据调整比例
    if routes:
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.4)  # 增加子图间距
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])
    else:
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.4)  # 增加子图间距
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = None
    
    # 设置背景色
    ax1.set_facecolor('#F8F9FA')
    ax2.set_facecolor('#F8F9FA')
    if ax3:
        ax3.set_facecolor('#F8F9FA')
    
    # 绘制配送中心区域
    depot_circle = Circle(depot, 5, color='#FF6B6B', alpha=0.2)
    ax1.add_patch(depot_circle)
    
    # 绘制配送中心
    ax1.scatter(depot[0], depot[1], c='#FF6B6B', s=200, label='配送中心', zorder=5)
    
    # 绘制配送点
    ax1.scatter(x_coords[:-2], y_coords[:-1], c='#4ECDC4', s=100, label='配送点', zorder=4)
    
    # 计算并绘制所有点到配送中心的连线
    total_distance = 0
    for i, point in enumerate(delivery_points):
        distance = np.sqrt((point[0] - depot[0])**2 + (point[1] - depot[1])**2)
        total_distance += distance
        ax1.plot([depot[0], point[0]], [depot[1], point[1]], 
                 c='#95A5A6', alpha=0.2, linestyle='--', zorder=1)
    
    # 绘制路线
    if routes:
        colors = create_gradient_colors(len(routes))
        for i, route in enumerate(routes):
            if not route:
                continue
                
            # 绘制路线
            route_x = [depot[0]] + [delivery_points[j][0] for j in route] + [depot[0]]
            route_y = [depot[1]] + [delivery_points[j][1] for j in route] + [depot[1]]
            ax1.plot(route_x, route_y, c=colors[i], alpha=0.7, label=f'车辆 {i+1}', zorder=2)
            
            # 添加点编号和时间窗口信息
            for j, point_idx in enumerate(route):
                point = delivery_points[point_idx]
                if time_windows and service_times:
                    window = time_windows[point_idx]
                    service = service_times[point_idx]
                    info = f'{j+1}\n[{window[0]:.1f}-{window[1]:.1f}]'
                else:
                    info = f'{j+1}'
                ax1.annotate(info, 
                            (point[0], point[1]),
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                            fontsize=8)
    
    # 创建对照点（在配送中心右侧20单位处）
    comparison_depot = (depot[0] + 20, depot[1])
    comparison_circle = Circle(comparison_depot, 5, color='#2ECC71', alpha=0.2)
    ax1.add_patch(comparison_circle)
    ax1.scatter(comparison_depot[0], comparison_depot[1], c='#2ECC71', s=200, label='对照配送中心', zorder=5)
    
    # 计算并绘制所有点到对照配送中心的连线
    comparison_total_distance = 0
    for i, point in enumerate(delivery_points):
        distance = np.sqrt((point[0] - comparison_depot[0])**2 + 
                         (point[1] - comparison_depot[1])**2)
        comparison_total_distance += distance
        ax1.plot([comparison_depot[0], point[0]], [comparison_depot[1], point[1]], 
                 c='#95A5A6', alpha=0.2, linestyle='--', zorder=1)
    
    # 设置主图属性
    ax1.set_title(title, fontsize=16, pad=20, fontweight='bold')
    ax1.set_xlabel('X 坐标', fontsize=12)
    ax1.set_ylabel('Y 坐标', fontsize=12)
    
    # 调整图例位置和样式
    legend = ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                       borderaxespad=0., frameon=True, fancybox=True, shadow=True,
                       fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    ax1.grid(True, alpha=0.3)
    
    # 设置坐标轴范围，留出一定边距
    margin = 5
    ax1.set_xlim(min(x_coords) - margin, max(x_coords) + margin)
    ax1.set_ylim(min(y_coords) - margin, max(y_coords) + margin)
    
    # 在中间子图中添加距离统计信息
    ax2.axis('off')
    stats_text = (
        f"原始配送中心总距离: {total_distance:.2f}\n"
        f"对照配送中心总距离: {comparison_total_distance:.2f}\n"
        f"距离差异: {abs(total_distance - comparison_total_distance):.2f}\n"
        f"差异百分比: {abs(total_distance - comparison_total_distance)/total_distance*100:.2f}%"
    )
    ax2.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    # 在底部子图中添加路线统计信息
    if routes and ax3:
        route_lengths = [len(route) for route in routes]
        ax3.bar(range(1, len(routes) + 1), route_lengths, color=colors, alpha=0.7)
        ax3.set_title('各车辆配送点数量分布', fontsize=12, pad=10)
        ax3.set_xlabel('车辆编号')
        ax3.set_ylabel('配送点数量')
        ax3.grid(True, alpha=0.3)
    
    # 添加说明文字，调整位置和样式
    plt.figtext(0.5, -0.08, description, ha='center', fontsize=10, wrap=True,
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8, ec='#CCCCCC'))
    
    # 调整布局，为图例和说明文字留出空间
    plt.subplots_adjust(right=0.85, left=0.1, top=0.92, bottom=0.15)  # 增加底部空间
    
    # 显示图形
    plt.show()

def calculate_route_statistics(routes: List[List[int]], 
                             distances: np.ndarray,
                             time_windows: List[Tuple[float, float]],
                             service_times: List[float]) -> dict:
    """计算路线统计信息"""
    stats = {
        'total_distance': 0,
        'route_lengths': [],
        'num_deliveries': [],
        'time_window_violations': 0
    }
    
    for route in routes:
        if not route:
            continue
            
        # 计算路线长度
        route_length = distances[0, route[0]]  # 从配送中心到第一个点
        for i in range(len(route)-1):
            route_length += distances[route[i], route[i+1]]
        route_length += distances[route[-1], 0]  # 从最后一个点到配送中心
        
        stats['total_distance'] += route_length
        stats['route_lengths'].append(route_length)
        stats['num_deliveries'].append(len(route))
    
    return stats

def print_statistics(stats: dict, title: str):
    """打印路线统计信息"""
    print(f"\n{title}")
    print("-" * 50)
    print(f"总距离: {stats['total_distance']:.2f}")
    print(f"平均路线长度: {np.mean(stats['route_lengths']):.2f}")
    print(f"最长路线长度: {max(stats['route_lengths']):.2f}")
    print(f"最短路线长度: {min(stats['route_lengths']):.2f}")
    print(f"每条路线平均配送点数量: {np.mean(stats['num_deliveries']):.2f}")
    print(f"最长路线配送点数量: {max(stats['num_deliveries'])}")
    print(f"最短路线配送点数量: {min(stats['num_deliveries'])}")

def plot_optimization_comparison(initial_stats: dict, dyt_stats: dict, final_stats: dict):
    """绘制优化效果对比图"""
    # 调整图形大小
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准备数据
    stages = ['初始路线', 'DyT优化', '最终优化']
    distances = [initial_stats['total_distance'], dyt_stats['total_distance'], final_stats['total_distance']]
    times = [initial_stats.get('computation_time', 0), 
             dyt_stats.get('computation_time', 0), 
             final_stats.get('computation_time', 0)]
    
    # 绘制距离对比
    colors = create_gradient_colors(3)
    bars1 = ax1.bar(stages, distances, color=colors, alpha=0.7)
    ax1.set_title('路线总距离对比', fontsize=14, pad=15)
    ax1.set_ylabel('总距离')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 绘制计算时间对比
    bars2 = ax2.bar(stages, times, color=colors, alpha=0.7)
    ax2.set_title('计算时间对比', fontsize=14, pad=15)
    ax2.set_ylabel('计算时间 (秒)')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 调整布局
    plt.subplots_adjust(wspace=0.3, bottom=0.15)  # 增加子图间距和底部空间
    plt.show()

def plot_route_analysis(initial_stats: dict, final_stats: dict):
    """绘制路线分析图"""
    # 调整图形大小
    plt.figure(figsize=(14, 10))
    
    # 创建子图
    gs = plt.GridSpec(2, 2, hspace=0.4, wspace=0.3)  # 增加子图间距
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])
    ax4 = plt.subplot(gs[1, 1])
    
    # 设置背景色
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor('#F8F9FA')
    
    # 1. 路线长度分布
    sns.boxplot(data=[initial_stats['route_lengths'], final_stats['route_lengths']], 
                ax=ax1, palette=['#FF6B6B', '#4ECDC4'])
    ax1.set_title('路线长度分布对比', fontsize=12)
    ax1.set_xticklabels(['初始路线', '最终路线'])
    ax1.set_ylabel('路线长度')
    
    # 2. 配送点数量分布
    sns.boxplot(data=[initial_stats['num_deliveries'], final_stats['num_deliveries']], 
                ax=ax2, palette=['#FF6B6B', '#4ECDC4'])
    ax2.set_title('配送点数量分布对比', fontsize=12)
    ax2.set_xticklabels(['初始路线', '最终路线'])
    ax2.set_ylabel('配送点数量')
    
    # 3. 路线长度直方图
    sns.histplot(data=initial_stats['route_lengths'], ax=ax3, color='#FF6B6B', alpha=0.5, label='初始路线')
    sns.histplot(data=final_stats['route_lengths'], ax=ax3, color='#4ECDC4', alpha=0.5, label='最终路线')
    ax3.set_title('路线长度分布直方图', fontsize=12)
    ax3.set_xlabel('路线长度')
    ax3.set_ylabel('频次')
    ax3.legend()
    
    # 4. 配送点数量直方图
    sns.histplot(data=initial_stats['num_deliveries'], ax=ax4, color='#FF6B6B', alpha=0.5, label='初始路线')
    sns.histplot(data=final_stats['num_deliveries'], ax=ax4, color='#4ECDC4', alpha=0.5, label='最终路线')
    ax4.set_title('配送点数量分布直方图', fontsize=12)
    ax4.set_xlabel('配送点数量')
    ax4.set_ylabel('频次')
    ax4.legend()
    
    # 调整布局
    plt.subplots_adjust(bottom=0.15)  # 增加底部空间
    plt.show()

def generate_summary(initial_stats: dict, dyt_stats: dict, final_stats: dict,
                    total_distance: float, comparison_distance: float):
    """生成总结报告"""
    print("\n" + "=" * 80)
    print("配送路径规划总结报告")
    print("=" * 80)
    
    # 计算关键指标
    improvement = (initial_stats['total_distance'] - final_stats['total_distance']) / initial_stats['total_distance'] * 100
    route_balance = np.std(final_stats['route_lengths']) / np.mean(final_stats['route_lengths'])
    delivery_balance = np.std(final_stats['num_deliveries']) / np.mean(final_stats['num_deliveries'])
    
    print("\n1. 配送中心位置分析")
    print("-" * 50)
    print(f"原始配送中心总距离: {total_distance:.2f}")
    print(f"对照配送中心总距离: {comparison_distance:.2f}")
    print(f"距离差异: {abs(total_distance - comparison_distance):.2f}")
    print(f"差异百分比: {abs(total_distance - comparison_distance)/total_distance*100:.2f}%")
    
    print("\n2. 路线优化效果")
    print("-" * 50)
    print(f"初始路线总距离: {initial_stats['total_distance']:.2f}")
    print(f"DyT优化后总距离: {dyt_stats['total_distance']:.2f}")
    print(f"最终优化后总距离: {final_stats['total_distance']:.2f}")
    print(f"\n优化效果: {improvement:.2f}%")
    
    print("\n3. 路线均衡性分析")
    print("-" * 50)
    print(f"初始路线:")
    print(f"  最长路线: {max(initial_stats['route_lengths']):.2f}")
    print(f"  最短路线: {min(initial_stats['route_lengths']):.2f}")
    print(f"  路线长度标准差: {np.std(initial_stats['route_lengths']):.2f}")
    
    print(f"\n最终路线:")
    print(f"  最长路线: {max(final_stats['route_lengths']):.2f}")
    print(f"  最短路线: {min(final_stats['route_lengths']):.2f}")
    print(f"  路线长度标准差: {np.std(final_stats['route_lengths']):.2f}")
    
    print("\n4. 配送点分布分析")
    print("-" * 50)
    print(f"平均每辆车配送点数量: {np.mean(final_stats['num_deliveries']):.2f}")
    print(f"配送点数量标准差: {np.std(final_stats['num_deliveries']):.2f}")
    
    print("\n5. 综合评估")
    print("-" * 50)
    print(f"路线均衡性指标: {route_balance:.2f}")
    print(f"配送点分布均衡性指标: {delivery_balance:.2f}")
    
    print("\n6. 建议")
    print("-" * 50)
    if total_distance < comparison_distance:
        print("✓ 当前配送中心位置优于对照点，建议保持现有位置")
    else:
        print("⚠ 对照配送中心位置可能更优，建议考虑更换位置")
    
    if improvement > 0:
        print(f"✓ 路线优化效果显著，节省了{improvement:.2f}%的总距离")
    else:
        print("⚠ 路线优化效果不明显，建议检查约束条件是否过于严格")
    
    if route_balance < 0.2:
        print("✓ 路线长度分布较为均衡，工作分配合理")
    else:
        print("⚠ 路线长度差异较大，建议重新平衡各车辆的工作量")
    
    if delivery_balance < 0.2:
        print("✓ 配送点分布较为均衡，任务分配合理")
    else:
        print("⚠ 配送点分布不均衡，建议调整任务分配")

def main():
    try:
        # 初始设置
        num_vehicles = 3
        depot = (50, 50)  # 区域中心
        num_initial_points = 10
        
        print("=" * 80)
        print("动态车辆路径规划问题求解")
        print("=" * 80)
        
        # 生成初始配送点和时间窗口
        print("\n正在生成初始配送点...")
        delivery_points = generate_random_points(num_initial_points)
        time_windows = generate_time_windows(num_initial_points)
        service_times = [1.0] * num_initial_points  # 每个配送点服务时间1小时
        
        print("\n初始问题设置:")
        print("-" * 50)
        print(f"车辆数量: {num_vehicles}")
        print(f"配送点数量: {num_initial_points}")
        print(f"配送中心位置: {depot}")
        
        # 绘制初始配送点
        plot_routes(depot, delivery_points, [], 
                    "初始配送点分布",
                    "红色点表示配送中心，蓝色点表示配送点。\n"
                    "配送点随机分布在100x100的区域内。")
        
        # 求解初始VRP
        print("\n正在求解初始VRP...")
        vrp_solver = VRPSolver(num_vehicles, depot, delivery_points, time_windows, service_times)
        start_time = time.time()
        initial_routes, initial_cost = vrp_solver.solve(time_limit=30)  # 设置30秒时间限制
        
        if initial_routes is None:
            print("\n错误：无法找到可行的初始路线！")
            print("可能的原因：")
            print("1. 时间窗口约束过于严格")
            print("2. 车辆数量不足")
            print("3. 服务时间过长")
            return
            
        vrp_time = time.time() - start_time
        
        # 计算并打印初始统计信息
        initial_stats = calculate_route_statistics(initial_routes, vrp_solver.distances, 
                                                 time_windows, service_times)
        print_statistics(initial_stats, "初始解决方案统计")
        print(f"\n计算时间: {vrp_time:.2f} 秒")
        
        # 绘制初始路线
        plot_routes(depot, delivery_points, initial_routes, 
                    "初始路线规划",
                    "使用Gurobi求解器得到的初始最优路线。\n"
                    "不同颜色的线表示不同车辆的路线，数字表示访问顺序。")
        
        # 初始化DyT求解器
        print("\n初始化DyT求解器...")
        dyt_solver = DyTSolver(initial_routes, vrp_solver.distances)
        
        # 模拟新订单
        num_new_orders = 3
        print(f"\n模拟 {num_new_orders} 个新订单...")
        new_points = generate_random_points(num_new_orders)
        new_time_windows = generate_time_windows(num_new_orders)
        new_service_times = [1.0] * num_new_orders
        
        print("\n" + "=" * 80)
        print(f"新订单信息:")
        print("-" * 50)
        for i, (point, window) in enumerate(zip(new_points, new_time_windows)):
            print(f"订单 {i+1}: 位置 {point}, 时间窗口 {window}")
        
        # 绘制新配送点
        all_points = delivery_points + new_points
        plot_routes(depot, all_points, initial_routes, 
                    "新订单加入后的配送点分布",
                    "绿色点表示新加入的配送点。\n"
                    "需要重新规划路线以适应新的配送需求。")
        
        # 使用DyT进行快速更新
        print("\n使用DyT算法更新路线...")
        start_time = time.time()
        new_point_indices = list(range(len(delivery_points), len(delivery_points) + num_new_orders))
        updated_routes = dyt_solver.optimize_routes(new_point_indices)
        dyt_time = time.time() - start_time
        
        if updated_routes is None:
            print("\n警告：DyT算法无法找到合适的插入位置！")
            print("将使用完整VRP求解器重新规划路线。")
        else:
            # 计算并打印DyT统计信息
            dyt_stats = calculate_route_statistics(updated_routes, vrp_solver.distances,
                                                 time_windows + new_time_windows,
                                                 service_times + new_service_times)
            print_statistics(dyt_stats, "DyT解决方案统计")
            print(f"\n计算时间: {dyt_time:.2f} 秒")
            
            # 绘制DyT路线
            plot_routes(depot, all_points, updated_routes, 
                        "DyT更新后的路线",
                        "使用动态时间规整(DyT)算法快速更新路线。\n"
                        "在保持原有路线结构的基础上，将新订单插入最优位置。")
        
        # 如果DyT解决方案不理想，使用VRP求解器
        if len(new_points) > 5 or updated_routes is None:  # 示例条件：当新订单数量大于5时使用完整VRP
            print("\n" + "=" * 80)
            print("使用完整VRP求解器进行大规模更新")
            print("=" * 80)
            
            vrp_solver.delivery_points.extend(new_points)
            vrp_solver.time_windows.extend(new_time_windows)
            vrp_solver.service_times.extend(new_service_times)
            vrp_solver.distances = vrp_solver._calculate_distance_matrix()
            
            start_time = time.time()
            final_routes, final_cost = vrp_solver.solve(time_limit=60)  # 设置60秒时间限制
            
            if final_routes is None:
                print("\n错误：无法找到可行的最终路线！")
                print("建议：")
                print("1. 增加车辆数量")
                print("2. 放宽时间窗口约束")
                print("3. 减少服务时间")
                return
                
            vrp_time = time.time() - start_time
            
            # 计算并打印最终统计信息
            final_stats = calculate_route_statistics(final_routes, vrp_solver.distances,
                                                   time_windows + new_time_windows,
                                                   service_times + new_service_times)
            print_statistics(final_stats, "最终VRP解决方案统计")
            print(f"\n计算时间: {vrp_time:.2f} 秒")
            
            # 绘制最终路线
            plot_routes(depot, all_points, final_routes, 
                        "最终优化后的路线",
                        "使用Gurobi求解器重新优化所有路线。\n"
                        "这种方法计算时间较长，但能得到全局最优解。")
            
            # 比较解决方案
            print("\n解决方案比较:")
            print("-" * 50)
            print(f"初始总距离: {initial_stats['total_distance']:.2f}")
            print(f"DyT总距离: {dyt_stats['total_distance']:.2f}")
            print(f"最终VRP总距离: {final_stats['total_distance']:.2f}")
            print(f"\n初始计算时间: {vrp_time:.2f} 秒")
            print(f"DyT计算时间: {dyt_time:.2f} 秒")
            print(f"最终VRP计算时间: {vrp_time:.2f} 秒")
            
            # 在绘制路线时计算总距离
            total_distance = sum(np.sqrt((p[0] - depot[0])**2 + (p[1] - depot[1])**2) 
                               for p in delivery_points)
            comparison_distance = sum(np.sqrt((p[0] - (depot[0] + 20))**2 + (p[1] - depot[1])**2) 
                                    for p in delivery_points)
            
            # 添加计算时间到统计信息
            initial_stats['computation_time'] = vrp_time
            dyt_stats['computation_time'] = dyt_time
            final_stats['computation_time'] = vrp_time
            
            # 绘制优化效果对比图
            plot_optimization_comparison(initial_stats, dyt_stats, final_stats)
            
            # 绘制路线分析图
            plot_route_analysis(initial_stats, final_stats)
            
            # 在程序结束前添加总结报告
            generate_summary(initial_stats, dyt_stats, final_stats, 
                            total_distance, comparison_distance)
            
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        print("程序终止。")
        sys.exit(1)

if __name__ == "__main__":
    main() 