import time
import timeit

from rag.models.llm import LLMModel


def benchmark_llm_model(
    model: LLMModel,
    prompts: list[str],
    num_runs: int = 5,
    max_tokens_list: list[int] = [50, 300],
    temperature_list: list[float] = [0.6, 1.0],
):
    """Benchmark the LLMModel with various configurations."""

    def run_benchmark(name: str, **generate_args):
        def wrapper():
            for prompt in prompts:
                model.generate(prompt, generate_args)

        # Warmup
        for _ in range(2):
            wrapper()

        # Timing
        time_taken = timeit.timeit(wrapper, number=num_runs) / num_runs
        avg_time_per_prompt = time_taken / len(prompts)
        print(
            f"{name}: {avg_time_per_prompt:.3f}s/prompt (avg over {num_runs} runs)"
        )
        return avg_time_per_prompt

    def benchmark_streaming():
        def wrapper():
            for prompt in prompts:
                list(model.generate_stream(prompt))  # Exhaust the generator

        # Warmup
        for _ in range(2):
            wrapper()

        time_taken = timeit.timeit(wrapper, number=num_runs) / num_runs
        avg_time_per_prompt = time_taken / len(prompts)
        print(
            f"Streaming: {avg_time_per_prompt:.3f}s/prompt (avg over {num_runs} runs)"
        )
        return avg_time_per_prompt

    print(f"\nBenchmarking with {len(prompts)} prompts")
    print(f"Prompt lengths: {[len(p) for p in prompts]} characters")

    results = {}

    # Benchmark different parameter combinations
    for max_tokens in max_tokens_list:
        for temperature in temperature_list:
            name = f"Gen {max_tokens}tok@{temperature}t"
            results[name] = run_benchmark(
                name,
                model_params={
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )

    # Benchmark streaming
    results["Streaming"] = benchmark_streaming()

    return results


if __name__ == "__main__":
    start = time.time()
    model = LLMModel()
    print(f"Model loaded in {time.time() - start}s")

    # Test prompts
    short_prompts = [
        "Explain quantum computing in simple terms",
        "What is the capital of France?",
    ]

    medium_prompts = [
        "Write a detailed comparison between Python and JavaScript for web development, considering factors like performance, ecosystem, and learning curve.",
        "Describe the process of photosynthesis in plants, including the light-dependent and light-independent reactions.",
    ] * 2  # 4 prompts

    long_prompts = [
        "Write a comprehensive essay about the history of artificial intelligence, covering its origins in the 1950s, key developments through the decades, current state-of-the-art techniques, and ethical considerations for future development. Include at least five major milestones and discuss the impact of AI on three different industries."
    ] * 2

    print("=== Short Prompts Benchmark ===")
    benchmark_llm_model(model, short_prompts)

    print("\n=== Medium Prompts Benchmark ===")
    benchmark_llm_model(model, medium_prompts)

    print("\n=== Long Prompts Benchmark ===")
    benchmark_llm_model(model, long_prompts)
