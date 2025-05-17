import gc
import os
import statistics
import time
import matplotlib.pyplot as plt

import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_settings')
django.setup()

# Import refactored schema models
from models import (
    MemberListSchema,
    MemberListSimpleSchema,
    MemberListBaseSchema,
    MemberListBaseV1Schema,
)

# Mock Member class
class Member:
    def __init__(self, id, nickname):
        self.id = id
        self.nickname = nickname

# Mock MemberList class
class MemberList:
    def __init__(self, members=None):
        self.member_list = members or []

# Function to generate test data
def create_test_data(count: int):
    """
    Generate test dictionary data.
    """
    # 1. Object-type data (ORM simulation)
    members = [Member(i, f"User{i}") for i in range(1, count + 1)]
    member_lists = [MemberList(members[:i % 10 + 1]) for i in range(count)]
    
    # 2. Dictionary-type data
    dict_data = []
    for i in range(count):
        member_count = i % 10 + 1
        member_list_dict = {
            'member_list': [
                {'id': j + 1, 'nickname': f"User{j + 1}"} 
                for j in range(member_count)
            ]
        }
        dict_data.append(member_list_dict)
    
    return member_lists, dict_data

def run_benchmark(sample_size: int = 1000, iterations: int = 5) -> dict[str, any]:
    """
    Run a benchmark with specified sample size and iterations.
    
    Args:
        sample_size: Number of objects to use in the test
        iterations: Number of iterations to run
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"=================================================")
    print(f"Starting benchmark for sample size {sample_size}")
    print(f"=================================================")
    print(f"Running benchmark: sample size {sample_size}, {iterations} iterations")
    print(f"Generating test data...")

    # Prepare test data
    member_lists, test_data_dicts = create_test_data(sample_size)
    print(f"Number of objects created: {len(member_lists)}")
    print(f"Number of dictionaries created: {len(test_data_dicts)}")

    # Benchmarking variables
    base_model_times = []
    base_model_v1_times = []
    simple_schema_times = []
    schema_times = []

    # 여러 번 반복 실행
    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")

        # Garbage collection to free up memory
        gc.collect()

        # 1. BaseModel Test
        start_time = time.perf_counter()
        base_model_result = [MemberListBaseSchema.model_validate(data) for data in test_data_dicts]
        base_model_time = time.perf_counter() - start_time
        base_model_times.append(base_model_time)

        # Garbage collection to free up memory
        del base_model_result
        gc.collect()

        # 2.SimpleDjangoSchema Test
        start_time = time.perf_counter()
        simple_schema_result = [MemberListSimpleSchema.model_validate(member_list) for member_list in member_lists]
        simple_schema_time = time.perf_counter() - start_time
        simple_schema_times.append(simple_schema_time)

        # Garbage collection to free up memory
        del simple_schema_result
        gc.collect()

        # 3. Schema Test
        start_time = time.perf_counter()
        schema_result = [MemberListSchema.model_validate(data) for data in test_data_dicts]
        schema_time = time.perf_counter() - start_time
        schema_times.append(schema_time)

        # Garbage collection to free up memory
        del schema_result
        gc.collect()

        # 4. BaseModelV1 테스트
        start_time = time.perf_counter()
        base_model_v1_result = [MemberListBaseV1Schema.parse_obj(data) for data in test_data_dicts]
        base_model_v1_time = time.perf_counter() - start_time
        base_model_v1_times.append(base_model_v1_time)

        # print results
        print(
            f"  BaseModel: {base_model_time:.6f}s, SimpleDjangoSchema: {simple_schema_time:.6f}s, Schema: {schema_time:.6f}s, BaseModelV1: {base_model_v1_time:.6f}s")

        # 메모리 정리
        gc.collect()

    # 통계 계산
    schema_avg = statistics.mean(schema_times)
    simple_schema_avg = statistics.mean(simple_schema_times)
    base_model_avg = statistics.mean(base_model_times)
    base_model_v1_avg = statistics.mean(base_model_v1_times)

    schema_median = statistics.median(schema_times)
    simple_schema_median = statistics.median(simple_schema_times)
    base_model_median = statistics.median(base_model_times)
    base_model_v1_median = statistics.median(base_model_v1_times)

    # 성능 향상율 계산 (Schema 대비)
    simple_improvement_avg = ((schema_avg - simple_schema_avg) / schema_avg) * 100
    simple_improvement_median = ((schema_median - simple_schema_median) / schema_median) * 100

    base_improvement_avg = ((schema_avg - base_model_avg) / schema_avg) * 100
    base_improvement_median = ((schema_median - base_model_median) / schema_median) * 100

    base_model_v1_improvement_avg = ((schema_avg - base_model_v1_avg) / schema_avg) * 100
    base_model_v1_improvement_median = ((schema_median - base_model_v1_median) / schema_median) * 100

    # 결과 요약
    results = {
        'sample_size': sample_size,
        'iterations': iterations,
        'schema_times': schema_times,
        'simple_schema_times': simple_schema_times,
        'base_model_times': base_model_times,
        'base_model_v1_times': base_model_v1_times,
        'schema_avg': schema_avg,
        'simple_schema_avg': simple_schema_avg,
        'base_model_avg': base_model_avg,
        'base_model_v1_avg': base_model_v1_avg,
        'schema_median': schema_median,
        'simple_schema_median': simple_schema_median,
        'base_model_median': base_model_median,
        'base_model_v1_median': base_model_v1_median,
        'simple_improvement_avg': simple_improvement_avg,
        'simple_improvement_median': simple_improvement_median,
        'base_improvement_avg': base_improvement_avg,
        'base_improvement_median': base_improvement_median,
        'base_model_v1_improvement_avg': base_model_v1_improvement_avg,
        'base_model_v1_improvement_median': base_model_v1_improvement_median
    }

    return results

def print_results(results: dict[str, any]) -> None:
    """
    Print benchmark results in a formatted way.
    
    Args:
        results: Dictionary containing benchmark results
    """
    sample_size = results["sample_size"]
    iterations = results["iterations"]
    
    print("\n" + "=" * 70)
    print(f"Benchmark Results Summary (Sample Size: {sample_size}, Iterations: {iterations})")
    print("=" * 70)

    print("\n[Average Execution Time]")
    print(f"Schema:             {results['schema_avg']:.6f}s")
    print(f"SimpleDjangoSchema: {results['simple_schema_avg']:.6f}s")
    print(f"BaseModel (v2):   {results['base_model_avg']:.6f}s")
    print(f"BaseModel (v1):     {results['base_model_v1_avg']:.6f}s")

    print(f"SimpleDjangoSchema Performance Improvement: {results['simple_improvement_avg']:.2f}%")
    print(f"BaseModel (v2) Performance Improvement:   {results['base_improvement_avg']:.2f}%")
    print(f"BaseModel (v1) Performance Improvement:     {results['base_model_v1_improvement_avg']:.2f}%")

    print("\n[Median Execution Time]")
    print(f"Schema:             {results['schema_median']:.6f}s")
    print(f"SimpleDjangoSchema: {results['simple_schema_median']:.6f}s")
    print(f"BaseModel (v2):   {results['base_model_median']:.6f}s")
    print(f"BaseModel (v1):     {results['base_model_v1_median']:.6f}s")


    print(f"SimpleDjangoSchema Performance Improvement: {results['simple_improvement_median']:.2f}%")
    print(f"BaseModel (v2) Performance Improvement:   {results['base_improvement_median']:.2f}%")
    print(f"BaseModel (v1) Performance Improvement:     {results['base_model_v1_improvement_median']:.2f}%")

    print("\n[All Execution Times]")
    print("Schema Times:")
    for i, t in enumerate(results['schema_times']):
        print(f"  Run {i + 1}: {t:.6f}s")

    print("\nSimpleDjangoSchema Times:")
    for i, t in enumerate(results['simple_schema_times']):
        print(f"  Run {i + 1}: {t:.6f}s")

    print("\nBaseModel (v2) Times:")
    for i, t in enumerate(results['base_model_times']):
        print(f"  Run {i + 1}: {t:.6f}s")

    print("\nBaseModel (v1) Times:")
    for i, t in enumerate(results['base_model_v1_times']):
        print(f"  Run {i + 1}: {t:.6f}s")

    print("\n" + "=" * 70)

    # Identify best performance
    candidates = [
        ('Schema (v2)', results['schema_avg'], results['schema_median']),
        ('SimpleDjangoSchema (v2)', results['simple_schema_avg'], results['simple_schema_median']),
        ('BaseModel (v2)', results['base_model_avg'], results['base_model_median']),
        ('BaseModel (v1)', results['base_model_v1_avg'], results['base_model_v1_median'])
    ]

    best_avg = min(candidates, key=lambda x: x[1])
    best_median = min(candidates, key=lambda x: x[2])

    print("\n[Best Performance]")
    print(f"Based on Average: {best_avg[0]} ({best_avg[1]:.6f}s)")
    print(f"Based on Median: {best_median[0]} ({best_median[2]:.6f}s)")

    print("\n" + "=" * 70)

def visualize_benchmark_results(results_list):
    """
    Visualize benchmark results for different sample sizes.
    
    Args:
        results_list: List of benchmark results for different sample sizes
    """
    sample_sizes = [r['sample_size'] for r in results_list]
    
    # Extract average time data
    schema_avgs = [r['schema_avg'] for r in results_list]
    simple_schema_avgs = [r['simple_schema_avg'] for r in results_list]
    base_model_avgs = [r['base_model_avg'] for r in results_list]
    pydantic_v1_avgs = [r['base_model_v1_avg'] for r in results_list]
    
    # Check if BaseModel (orm) data is available
    include_orm = True
    base_model_orm_avgs = []
    for r in results_list:
        if r.get('base_model_orm_avg') is None:
            include_orm = False
            break
        base_model_orm_avgs.append(r['base_model_orm_avg'])
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 15))
    
    # 1. Execution time graph (top subplot)
    plt.subplot(2, 2, 1)
    plt.plot(sample_sizes, schema_avgs, 'o-', linewidth=2, label='Schema (v2)')
    plt.plot(sample_sizes, simple_schema_avgs, 's-', linewidth=2, label='SimpleDjangoSchema (v2)')
    plt.plot(sample_sizes, base_model_avgs, '^-', linewidth=2, label='BaseModel (v2)')
    plt.plot(sample_sizes, pydantic_v1_avgs, 'x-', linewidth=2, label='BaseModel (v1)')
    
    plt.xlabel('Sample Size (number of objects)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Average Execution Time by Sample Size')
    plt.grid(True)
    plt.legend()
    
    # 2. Execution time graph - Log scale (top-right subplot)
    plt.subplot(2, 2, 2)
    plt.plot(sample_sizes, schema_avgs, 'o-', linewidth=2, label='Schema (v2)')
    plt.plot(sample_sizes, simple_schema_avgs, 's-', linewidth=2, label='SimpleDjangoSchema (v2)')
    plt.plot(sample_sizes, base_model_avgs, '^-', linewidth=2, label='BaseModel (v2)')
    plt.plot(sample_sizes, pydantic_v1_avgs, 'x-', linewidth=2, label='BaseModel (v1)')
    
    plt.xlabel('Sample Size (number of objects)')
    plt.ylabel('Average Execution Time (seconds)')
    plt.title('Average Execution Time by Sample Size (Log Scale)')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    
    # 3. Performance comparison as bar graph for the largest sample size (bottom-right subplot)
    plt.subplot(2, 2, 4)
    
    # Get largest sample size results for bar graph
    max_sample_idx = sample_sizes.index(max(sample_sizes))
    max_sample_size = sample_sizes[max_sample_idx]
    
    # Calculate performance multipliers (how many times faster than Schema)
    schema_time = schema_avgs[max_sample_idx]
    simple_multiplier = schema_time / simple_schema_avgs[max_sample_idx]
    base_multiplier = schema_time / base_model_avgs[max_sample_idx]
    v1_multiplier = schema_time / pydantic_v1_avgs[max_sample_idx]
    
    labels = ['Schema (v2)', 'SimpleDjangoSchema (v2)', 'BaseModel (v2)', 'BaseModel (v1)']
    values = [1.0, simple_multiplier, base_multiplier, v1_multiplier]  # Schema is the baseline (1.0x)
    
    # Create bar chart
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'][:len(labels)]
    bars = plt.bar(labels, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}x',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xlabel('Schema Type')
    plt.ylabel('Performance Multiplier (x times faster)')
    plt.title(f'Performance Comparison at {max_sample_size} Objects\n(Higher is better)')
    plt.grid(axis='y')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def run_multiple_benchmarks(sample_sizes=[10, 50, 100, 200, 500], iterations=3):
    """
    Run benchmarks for multiple sample sizes and visualize the results.
    
    Args:
        sample_sizes: List of sample sizes to test
        iterations: Number of iterations to run for each benchmark
    """
    results_list = []
    
    for size in sample_sizes:
        print(f"\n{'='*50}")
        print(f"Starting benchmark for sample size {size}")
        print(f"{'='*50}")
        
        results = run_benchmark(sample_size=size, iterations=iterations)
        results_list.append(results)
        print_results(results)
    
    # Visualize results
    visualize_benchmark_results(results_list)
    
    return results_list

# Example execution:
if __name__ == "__main__":
    # Set sample sizes and iteration count
    SAMPLE_SIZES = [10, 30, 50, 100, 300, 500, 1000, 3000]  # Number of objects to test
    ITERATIONS = 10  # Number of times to repeat each test

    # Run benchmarks for multiple sample sizes
    benchmark_results = run_multiple_benchmarks(sample_sizes=SAMPLE_SIZES, iterations=ITERATIONS)
