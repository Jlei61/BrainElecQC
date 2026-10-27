"""Linxi task entry point; numerical work remains in the BrainElecQC core."""

# Linxi initializes its numerical runtime before NumPy and the QC core.
from linxi.processor import DefaultProcessor, PROCESS_STAGES, register_as_linxi_processor

from .adapter import normalize_options, process_file


@register_as_linxi_processor(stage=PROCESS_STAGES.QUALITY)
class BrainElecQCProcessor(DefaultProcessor):
    """Produce an immutable EEG QC bundle from run_state.input_path.

    Parameters are validated on construction. Linxi's toy-recording dry run
    skips file access; actual runs use the original file calibration and clock.
    """

    VALIDATION_SKIP = True

    def __init__(self, qc_config=None, figures=True, title="EEG QC",
                 display_start_seconds=None, display_duration_seconds=15,
                 recording_info=None, name=None):
        super().__init__(name)
        self.options = normalize_options(qc_config, figures, title,
            display_start_seconds, display_duration_seconds, recording_info)

    def _process(self, context):
        if context.run_state.validating:
            return context

        # A failed second invocation must never expose the first run as current.
        context.put_metric("brainelecqc", None)
        phase = "import"

        def progress(value):
            nonlocal phase
            phase = value
            context.put_metric("brainelecqc_status", {"status": "running", "phase": phase})

        progress(phase)
        try:
            if context.run_state.failed:
                raise ValueError("Retry with a fresh LinxiContext using the same input_path and work_dir; "
                                 "set incomplete_workdir_strategy='resume' in TaskRunner.run")
            summary = process_file(context.run_state.input_path,
                context.run_state.work_dir, self.options, progress=progress)
        except BaseException as exc:
            context.put_metric("brainelecqc_status", {
                "status": "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                "phase": phase, "error": f"{type(exc).__name__}: {exc}"})
            raise
        context.put_metric("brainelecqc", summary)
        context.put_metric("brainelecqc_status", {"status": "completed", "phase": "complete",
                                                 "run_id": summary["run_id"]})
        return context
