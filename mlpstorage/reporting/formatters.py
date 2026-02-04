"""
Validation message formatters for MLPerf Storage reporting.

This module provides formatters for validation results with clear
OPEN vs CLOSED messaging and actionable feedback for users.
"""

from typing import List, Dict, Any, Optional
from mlpstorage.config import PARAM_VALIDATION


class ValidationMessageFormatter:
    """Format validation results with clear OPEN vs CLOSED messaging."""

    # Terminal color codes for enhanced output
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize the formatter.

        Args:
            use_colors: Whether to use terminal colors in output.
        """
        self.use_colors = use_colors

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors and color in self.COLORS:
            return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
        return text

    def format_category_badge(self, category: PARAM_VALIDATION) -> str:
        """Generate a colored badge for the category."""
        if category == PARAM_VALIDATION.CLOSED:
            return self._color("[CLOSED]", 'green')
        elif category == PARAM_VALIDATION.OPEN:
            return self._color("[OPEN]", 'yellow')
        else:
            return self._color("[INVALID]", 'red')

    def format_category_summary(self, category: PARAM_VALIDATION, issues: List) -> str:
        """
        Generate a clear summary of why a run is in a particular category.

        Args:
            category: The validation category (CLOSED, OPEN, or INVALID).
            issues: List of Issue objects from validation.

        Returns:
            Formatted summary string.
        """
        if category == PARAM_VALIDATION.CLOSED:
            return self._format_closed_summary()
        elif category == PARAM_VALIDATION.OPEN:
            return self._format_open_summary(issues)
        else:
            return self._format_invalid_summary(issues)

    def _format_closed_summary(self) -> str:
        """Format summary for CLOSED category."""
        header = self._color("CLOSED SUBMISSION QUALIFIED", 'green')
        return f"""
{header}
This run meets all requirements for CLOSED division submission.
All parameters comply with the strict requirements.
"""

    def _format_open_summary(self, issues: List) -> str:
        """Format summary for OPEN category."""
        header = self._color("OPEN SUBMISSION ONLY", 'yellow')

        open_issues = [i for i in issues if i.validation == PARAM_VALIDATION.OPEN]
        reasons = []
        for issue in open_issues:
            param = self._color(issue.parameter, 'cyan')
            reasons.append(f"  - {param}: {issue.message}")
            if issue.expected:
                reasons.append(f"      Required: {issue.expected}")
            if issue.actual:
                reasons.append(f"      Current:  {issue.actual}")

        reasons_text = "\n".join(reasons) if reasons else "  (No specific issues logged)"

        return f"""
{header}
This run qualifies for OPEN submission only.

The following parameters do not meet CLOSED requirements:
{reasons_text}

To qualify for CLOSED submission, modify these parameters and re-run.
"""

    def _format_invalid_summary(self, issues: List) -> str:
        """Format summary for INVALID category."""
        header = self._color("INVALID - CANNOT BE SUBMITTED", 'red')

        invalid_issues = [i for i in issues if i.validation == PARAM_VALIDATION.INVALID]
        reasons = []
        for issue in invalid_issues:
            param = self._color(issue.parameter, 'cyan')
            reasons.append(f"  - {param}: {issue.message}")
            if issue.expected:
                reasons.append(f"      Expected: {issue.expected}")
            if issue.actual:
                reasons.append(f"      Actual:   {issue.actual}")

        reasons_text = "\n".join(reasons) if reasons else "  (No specific issues logged)"

        return f"""
{header}
This run is INVALID and cannot be submitted to any division.

The following critical issues must be resolved:
{reasons_text}

Fix these issues and re-run the benchmark.
"""

    def format_run_header(self, run_id: str, category: PARAM_VALIDATION,
                          benchmark_type: str, model: str, command: str = None) -> str:
        """
        Format a header for a single run result.

        Args:
            run_id: Unique identifier for the run.
            category: Validation category.
            benchmark_type: Type of benchmark (training, checkpointing, etc.).
            model: Model name.
            command: Optional command (run, datagen, etc.).

        Returns:
            Formatted header string.
        """
        badge = self.format_category_badge(category)
        header = self._color(run_id, 'bold')

        lines = [
            f"{badge} {header}",
            f"    Benchmark: {benchmark_type}",
            f"    Model: {model}",
        ]
        if command:
            lines.append(f"    Command: {command}")

        return "\n".join(lines)

    def format_metrics(self, metrics: Dict[str, Any]) -> str:
        """
        Format benchmark metrics for display.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            Formatted metrics string.
        """
        if not metrics:
            return "    No metrics available"

        lines = ["    Metrics:"]
        for metric, value in metrics.items():
            formatted_value = self._format_metric_value(metric, value)
            lines.append(f"      - {metric}: {formatted_value}")

        return "\n".join(lines)

    def _format_metric_value(self, metric: str, value: Any) -> str:
        """Format a single metric value appropriately."""
        if isinstance(value, (int, float)):
            if "percentage" in metric.lower() or "pct" in metric.lower():
                return f"{value:,.1f}%"
            elif "bytes" in metric.lower():
                return self._format_bytes(value)
            else:
                return f"{value:,.2f}"
        elif isinstance(value, (list, tuple)):
            formatted = [self._format_metric_value(metric, v) for v in value]
            return ", ".join(formatted)
        else:
            return str(value)

    def _format_bytes(self, value: float) -> str:
        """Format bytes value with appropriate unit."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if abs(value) < 1024.0:
                return f"{value:,.1f} {unit}"
            value /= 1024.0
        return f"{value:,.1f} PB"

    def format_issues_list(self, issues: List, show_all: bool = False) -> str:
        """
        Format a list of issues for display.

        Args:
            issues: List of Issue objects.
            show_all: If True, show all issues. If False, only show non-CLOSED issues.

        Returns:
            Formatted issues string.
        """
        if not issues:
            return "    No issues found"

        filtered_issues = issues if show_all else [
            i for i in issues if i.validation != PARAM_VALIDATION.CLOSED
        ]

        if not filtered_issues:
            return "    No actionable issues"

        lines = ["    Issues:"]
        for issue in filtered_issues:
            validation = issue.validation.value.upper()
            color = 'green' if validation == 'CLOSED' else (
                'yellow' if validation == 'OPEN' else 'red'
            )
            badge = self._color(f"[{validation}]", color)
            lines.append(f"      {badge} {issue.parameter}: {issue.message}")

        return "\n".join(lines)


class ClosedRequirementsFormatter:
    """Format CLOSED submission requirements as checklists."""

    TRAINING_REQUIREMENTS = {
        'title': 'Training Benchmark CLOSED Requirements',
        'requirements': [
            'Dataset size >= 5x total cluster memory',
            'At least 500 steps per epoch',
            '5 complete runs required for submission',
            'Only allowed parameter overrides used',
            'No modifications to core workload parameters',
        ],
        'allowed_params': [
            'dataset.num_files_train',
            'dataset.num_subfolders_train',
            'dataset.data_folder',
            'reader.read_threads',
            'reader.computation_threads',
            'reader.transfer_size',
            'reader.odirect',
            'reader.prefetch_size',
            'checkpoint.checkpoint_folder',
            'storage.storage_type',
            'storage.storage_root',
        ],
    }

    CHECKPOINTING_REQUIREMENTS = {
        'title': 'Checkpointing Benchmark CLOSED Requirements',
        'requirements': [
            '10 checkpoint write operations total',
            '10 checkpoint read operations total',
            'Valid LLM model (llama3-8b, llama3-70b, llama3-405b)',
            'Only allowed parameter overrides used',
        ],
        'allowed_params': [
            'checkpoint.checkpoint_folder',
            'storage.storage_type',
            'storage.storage_root',
        ],
    }

    KVCACHE_REQUIREMENTS = {
        'title': 'KV Cache Benchmark Requirements (Preview)',
        'requirements': [
            'Minimum runtime of 30 seconds',
            'Valid model configuration',
            'At least 1 concurrent user',
            'Note: KV Cache is in preview and not yet accepted for CLOSED submissions',
        ],
        'allowed_params': [],
    }

    VECTORDB_REQUIREMENTS = {
        'title': 'VectorDB Benchmark Requirements (Preview)',
        'requirements': [
            'Minimum runtime of 30 seconds',
            'Valid collection configuration',
            'Database host and port accessible',
            'Note: VectorDB is in preview and not yet accepted for CLOSED submissions',
        ],
        'allowed_params': [],
    }

    @classmethod
    def get_requirements(cls, benchmark_type: str) -> Optional[Dict]:
        """Get requirements for a benchmark type."""
        requirements_map = {
            'training': cls.TRAINING_REQUIREMENTS,
            'checkpointing': cls.CHECKPOINTING_REQUIREMENTS,
            'kv_cache': cls.KVCACHE_REQUIREMENTS,
            'vector_database': cls.VECTORDB_REQUIREMENTS,
        }
        return requirements_map.get(benchmark_type)

    @classmethod
    def format_checklist(cls, benchmark_type: str) -> str:
        """
        Generate a checklist of CLOSED submission requirements.

        Args:
            benchmark_type: Type of benchmark.

        Returns:
            Formatted checklist string.
        """
        reqs = cls.get_requirements(benchmark_type)
        if not reqs:
            return f"No specific requirements defined for {benchmark_type}"

        lines = [
            f"\n{reqs['title']}",
            "=" * len(reqs['title']),
            "",
            "Requirements:",
        ]

        for req in reqs['requirements']:
            lines.append(f"  [ ] {req}")

        if reqs['allowed_params']:
            lines.append("")
            lines.append("Allowed Parameter Overrides (CLOSED):")
            for param in reqs['allowed_params']:
                lines.append(f"  - {param}")

        return "\n".join(lines)


class ReportSummaryFormatter:
    """Format report summaries."""

    def __init__(self, use_colors: bool = True):
        self.msg_formatter = ValidationMessageFormatter(use_colors=use_colors)

    def format_summary_header(self, total_runs: int, closed: int, open_: int, invalid: int) -> str:
        """Format the summary header with counts."""
        lines = [
            "",
            "=" * 70,
            "BENCHMARK VALIDATION REPORT",
            "=" * 70,
            "",
            f"Summary: {total_runs} runs analyzed",
            f"  {self.msg_formatter._color('CLOSED:', 'green')} {closed} runs",
            f"  {self.msg_formatter._color('OPEN:', 'yellow')} {open_} runs",
            f"  {self.msg_formatter._color('INVALID:', 'red')} {invalid} runs",
        ]
        return "\n".join(lines)

    def format_section_header(self, category: PARAM_VALIDATION, count: int) -> str:
        """Format a section header for a category."""
        if category == PARAM_VALIDATION.INVALID:
            title = "INVALID RUNS - These runs cannot be submitted"
            color = 'red'
        elif category == PARAM_VALIDATION.OPEN:
            title = "OPEN RUNS - These runs qualify for OPEN division only"
            color = 'yellow'
        else:
            title = "CLOSED RUNS - These runs qualify for CLOSED division"
            color = 'green'

        header = self.msg_formatter._color(title, color)
        return f"\n{'-' * 70}\n{header} ({count})\n{'-' * 70}"
