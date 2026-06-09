import time
from dataclasses import dataclass

@dataclass
class Vehicle:
    progress_m: float
    lane_index: int
    speed_mps: float
    desired_speed_mps: float
    length_m: float

N = 1853
edge_vehicles = [Vehicle(i * 0.1, 0, 0.0, 10.0, 4.5) for i in range(N)]

start = time.time()
for v in edge_vehicles:
    front_same_lane = None
    for other in edge_vehicles:
        if other.lane_index == v.lane_index and other.progress_m > v.progress_m:
            if front_same_lane is None or other.progress_m < front_same_lane.progress_m:
                front_same_lane = other

    if front_same_lane is not None:
        gap = front_same_lane.progress_m - v.progress_m - front_same_lane.length_m
        safe_gap = 2.0 + v.speed_mps * 0.5
        
        if gap < safe_gap and front_same_lane.speed_mps < v.desired_speed_mps * 0.95:
            possible_lanes = [1]
            for target_lane in possible_lanes:
                front_target = None
                back_target = None
                for other in edge_vehicles:
                    if other.lane_index == target_lane:
                        if other.progress_m > v.progress_m:
                            if front_target is None or other.progress_m < front_target.progress_m:
                                front_target = other
                        elif other.progress_m < v.progress_m:
                            if back_target is None or other.progress_m > back_target.progress_m:
                                back_target = other
end = time.time()
print(f"Time taken: {end - start:.4f} seconds")
