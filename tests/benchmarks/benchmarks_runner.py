from tests.benchmarks.report import generate_html_report
from tests.benchmarks.test_latency import run_latency_benchmarks


def run_all_benchmarks():
    latency_results = run_latency_benchmarks()
    print(f"latency results: {latency_results}")
    generate_html_report(latency_results)


if __name__ == "__main__":
    run_all_benchmarks()
