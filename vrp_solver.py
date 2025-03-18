import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Tuple, Dict
import time

class VRPSolver:
    def __init__(self, 
                 num_vehicles: int,
                 depot: Tuple[float, float],
                 delivery_points: List[Tuple[float, float]],
                 time_windows: List[Tuple[float, float]],
                 service_times: List[float],
                 max_capacity: float = None):
        """
        初始化VRP求解器
        
        Args:
            num_vehicles: 可用车辆数量
            depot: 配送中心坐标 (x, y)
            delivery_points: 配送点坐标列表
            time_windows: 每个配送点的时间窗口列表
            service_times: 每个配送点的服务时间
            max_capacity: 每辆车的最大容量（可选）
        """
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.delivery_points = delivery_points
        self.time_windows = time_windows
        self.service_times = [0.0] + service_times  # 添加配送中心服务时间为0
        self.max_capacity = max_capacity
        
        # 计算距离矩阵
        self.distances = self._calculate_distance_matrix()
        
        # 检查时间窗口约束
        self._check_time_windows()
        
    def _check_time_windows(self):
        """检查时间窗口约束是否合理"""
        for i, (early, late) in enumerate(self.time_windows):
            if early >= late:
                raise ValueError(f"配送点 {i} 的时间窗口无效：开始时间 {early} 大于等于结束时间 {late}")
            if early < 0:
                raise ValueError(f"配送点 {i} 的时间窗口无效：开始时间 {early} 小于0")
    
    def _calculate_distance_matrix(self) -> np.ndarray:
        """计算所有点之间的距离矩阵"""
        all_points = [self.depot] + self.delivery_points
        n = len(all_points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    distances[i,j] = np.sqrt(
                        (all_points[i][0] - all_points[j][0])**2 +
                        (all_points[i][1] - all_points[j][1])**2
                    )
        return distances
    
    def solve(self, time_limit: float = 60.0) -> Tuple[List[List[int]], float]:
        """
        使用Gurobi求解VRP
        
        Args:
            time_limit: 最大求解时间（秒）
            
        Returns:
            Tuple of (routes, objective_value)
            routes: 路线列表，每条路线是配送点索引的列表
            objective_value: 总距离
        """
        try:
            # 创建模型
            model = gp.Model("VRP")
            
            # 设置时间限制
            model.setParam('TimeLimit', time_limit)
            
            # 决策变量
            n = len(self.delivery_points) + 1  # 包括配送中心
            x = model.addVars(self.num_vehicles, n, n, vtype=GRB.BINARY, name="x")
            t = model.addVars(self.num_vehicles, n, vtype=GRB.CONTINUOUS, name="t")
            
            # 目标函数：最小化总距离
            model.setObjective(
                gp.quicksum(
                    self.distances[i,j] * x[k,i,j]
                    for k in range(self.num_vehicles)
                    for i in range(n)
                    for j in range(n)
                    if i != j
                ),
                GRB.MINIMIZE
            )
            
            # 约束条件
            
            # 每个配送点必须被访问一次
            for j in range(1, n):  # 跳过配送中心
                model.addConstr(
                    gp.quicksum(x[k,i,j] for k in range(self.num_vehicles) for i in range(n) if i != j) == 1
                )
            
            # 流量守恒
            for k in range(self.num_vehicles):
                for h in range(n):
                    model.addConstr(
                        gp.quicksum(x[k,i,h] for i in range(n) if i != h) ==
                        gp.quicksum(x[k,h,j] for j in range(n) if j != h)
                    )
            
            # 时间窗口约束
            for k in range(self.num_vehicles):
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            model.addConstr(
                                t[k,j] >= t[k,i] + self.service_times[i] + self.distances[i,j] - 
                                (1 - x[k,i,j]) * 10000
                            )
            
            for k in range(self.num_vehicles):
                for i in range(1, n):  # 跳过配送中心
                    model.addConstr(t[k,i] >= self.time_windows[i-1][0])
                    model.addConstr(t[k,i] <= self.time_windows[i-1][1])
            
            # 求解
            model.optimize()
            
            # 检查求解状态
            if model.status == GRB.INFEASIBLE:
                print("警告：模型无可行解！")
                return None, None
            elif model.status == GRB.TIME_LIMIT:
                print("警告：达到时间限制，可能未找到最优解！")
            
            # 提取解
            routes = []
            for k in range(self.num_vehicles):
                route = []
                current = 0  # 从配送中心开始
                while True:
                    next_point = None
                    for j in range(n):
                        if j != current and x[k,current,j].x > 0.5:
                            next_point = j
                            break
                    if next_point is None or next_point == 0:  # 返回配送中心
                        break
                    route.append(next_point - 1)  # 转换为0基配送点索引
                    current = next_point
                if route:  # 只添加非空路线
                    routes.append(route)
            
            return routes, model.objVal
            
        except gp.GurobiError as e:
            print(f"Gurobi错误: {e}")
            return None, None
        except Exception as e:
            print(f"其他错误: {e}")
            return None, None
    
    def update_with_new_order(self, 
                            new_point: Tuple[float, float],
                            time_window: Tuple[float, float],
                            service_time: float) -> Tuple[List[List[int]], float]:
        """
        更新VRP，添加新的配送点
        
        Args:
            new_point: 新配送点坐标
            time_window: 新配送点的时间窗口
            service_time: 新配送点的服务时间
            
        Returns:
            更新后的路线和目标值
        """
        self.delivery_points.append(new_point)
        self.time_windows.append(time_window)
        self.service_times.append(service_time)
        self.distances = self._calculate_distance_matrix()
        return self.solve() 