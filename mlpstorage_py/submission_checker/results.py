from .loader import SubmissionLogs


class ResultExporter:
    """Exports submission validation results to CSV format.

    The `ResultExporter` class collects validated submission data and
    exports it to a CSV file with standardized columns for MLPerf
    submission summaries. It handles both performance and power results,
    duplicating rows for power submissions with power-specific metrics.
    """
    def __init__(self, csv_path, config) -> None:
        """Initialize the result exporter.

        Sets up the CSV header columns and prepares for result collection.

        Args:
            csv_path (str): Path to the output CSV file.
            config (Config): Configuration helper for model mappings.
        """
        # Question: Is the final table defined?
        self.head = [
            "Public ID",
            "Organization",
            "Submission Name",
            "Description",
            "Type",
            "Access Protocol",
            "Availability",
            "RUs",
            "Integrated Client Storage",
            "Accelerator Type",
            "# Client Nodes",
            # TODO: Avoid hardcoding this
            # Training
            "3D-Unet - # Accel",
            "3D-Unet - Read B/W (GiB/s)",
            "ResNet-50 - # Accel",
            "ResNet-50 - Read B/W (GiB/s)",
            "CosmoFlow - # Accel",
            "CosmoFlow - Read B/W (GiB/s)",
            # Checkpointing
            "8B - Write B/W (GiB/s)",
            "8B - Read B/W (GiB/s)",
            "70B - Write B/W (GiB/s)",
            "70B - Read B/W (GiB/s)",
            "405B - Write B/W (GiB/s)",
            "405B - Read B/W (GiB/s)",
            "1T - Write B/W (GiB/s)",
            "1T - Read B/W (GiB/s)",
        ]
        self.rows = []
        self.csv_path = csv_path
        self.config = config

    def add_result(self, submission_logs: SubmissionLogs):
        """Add a validated submission result to the export queue.

        Extracts relevant fields from submission logs and system JSON,
        formats them into a CSV row, and appends to the rows list. For
        power submissions, adds an additional row with power metrics.

        Args:
            submission_logs (SubmissionLogs): Validated submission data
                and metadata.
        """
        row = {key: "" for key in self.head}
        # TODO: extract values from submission logs
        self.rows.append(row.copy())

    def export_row(self, row: dict):
        """Write a single result row to the CSV file.

        Formats the row dictionary into a quoted CSV line and appends it
        to the output file.

        Args:
            row (dict): Result row data keyed by column headers.
        """
        values = [f'"{row.get(key, "")}"' for key in self.head]
        csv_row = ",".join(values) + "\n"
        with open(self.csv_path, "+a") as csv:
            csv.write(csv_row)

    def export(self):
        """Export all accumulated results to the CSV file.

        Writes the header row first, then iterates through all collected
        rows, exporting each one.
        """
        csv_header = ",".join(self.head) + "\n"
        with open(self.csv_path, "w") as csv:
            csv.write(csv_header)
        for row in self.rows:
            self.export_row(row)
