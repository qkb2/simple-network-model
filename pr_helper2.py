import time
import subprocess
import matplotlib.pyplot as plt
import csv

def run_noise(type_, num_layers):
    layer_sizes = create_layer_sizes(num_layers)
    
    args = [
        "D:\\studia\\Semestr 6\\Przetwarzanie równoległe\\laby zwierzu\\projekt\\inny\\Noise\\x64\\Debug\\Noise.exe",
        type_, str(num_layers)]
    args.extend(str(size) for size in layer_sizes)
    
    start_time = time.time()
    subprocess.run(args)
    end_time = time.time()
    
    return end_time - start_time

def create_layer_sizes(num_layers):
    third = num_layers // 3
    sizes = [100] * third + [200] * third + [100] * third
    # Padding with 100s if sizes are missing
    while len(sizes) < num_layers:
        sizes.append(100)
    return sizes

def generate_data():
    data = {}
    for type_ in ["none", "MP", "CUDA"]:
        input_data = []
        for num_layers in range(1000, 8000, 1000):
            time_taken = run_noise(type_, num_layers)
            input_data.append(time_taken)
            print(f"Method: {type_} Layers: {num_layers} Time: {time_taken}")
        data[type_] = input_data
    return data

def plot_data(data):
    plt.figure(figsize=(12, 6))
    markers = {'none': 'o', 'MP': '^', 'CUDA': 's'}
    for type_, input_data in data.items():
        marker = markers[type_]
        plt.plot(range(1000, 8000, 1000), input_data, marker=marker, linestyle='-', label=f"{type_.upper()}")
    plt.title("Execution Time for Different Types")
    plt.xlabel("Number of Layers")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.savefig("all_types_chart.png")

def plot_ratios(data):
    plt.figure(figsize=(12, 6))
    num_layers_range = list(range(1000, 8000, 1000))
    
    none_times = data["none"]
    mp_times = data["MP"]
    cuda_times = data["CUDA"]
    
    mp_ratios = [none/mp for none, mp in zip(none_times, mp_times)]
    cuda_ratios = [none/cuda for none, cuda in zip(none_times, cuda_times)]
    
    plt.plot(num_layers_range, mp_ratios, marker='^', linestyle='-', label="NONE/MP")
    plt.plot(num_layers_range, cuda_ratios, marker='s', linestyle='-', label="NONE/CUDA")
    
    plt.title("Execution Time Ratios NONE/MP and NONE/CUDA")
    plt.xlabel("Number of Layers")
    plt.ylabel("Execution Time Ratio")
    plt.legend()
    plt.savefig("ratios_chart.png")

def write_data_to_csv(data):
    with open("execution_times.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Type", "Number of Layers", "Time (s)"])
        
        for type_, input_data in data.items():
            for i, time_taken in enumerate(input_data):
                num_layers = (i + 1) * 1000
                writer.writerow([type_, num_layers, time_taken])

if __name__ == "__main__":
    data = generate_data()
    plot_data(data)
    plot_ratios(data)
    write_data_to_csv(data)
