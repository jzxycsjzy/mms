times = []
for i in [1,2,3,4]:
    file_path = f"log{i}.txt"
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if "solver time" in line:
            info = line.split(' ')
            idx = info.index("solver") - 1
            policy = info[idx]
            if "sr-replace" in policy:
                policy = "sr-replace"
            solver_time = float(info[info.index("time:") + 1])
            # print(policy, solver_time)
            times.append(solver_time)

def avg_time(time_list: list):
    return sum(time_list) / len(time_list)
print(avg_time(times))
