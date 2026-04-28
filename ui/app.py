from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, Static
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual import work
from rich.markup import escape

import router

class RouterApp(App):
    """Prompt router TUI."""

    TITLE = "prompt-router"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    CSS = """
    VerticalScroll {
        height: 1fr;
        padding: 0 1;
    }
    .entry {
        padding-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="log")
        yield Input(placeholder="Type a prompt and press Enter…")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        self._append(Static("[dim]prompt-router ready. Ctrl+C to quit.[/dim]"))

    def _append(self, widget: Static) -> None:
        log = self.query_one(VerticalScroll)
        log.mount(widget)
        log.scroll_end(animate=False)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()
        self._handle_prompt(prompt)

    @work(thread=True)
    def _handle_prompt(self, prompt: str) -> None:
        self.call_from_thread(
            self._append,
            Static(f"[bold]>>>[/bold] {escape(prompt)}", classes="entry"),
        )

        status = Static("[dim]classifying…[/dim]")
        self.call_from_thread(self._append, status)

        try:
            destination, reason, response, energy_meta = router.route(prompt)

            _CLOUD_HIGH_THRESHOLD_MWH = 2000

            if destination == "claude":
                inf = energy_meta["inference_wh"] * 1000
                tr_mid = energy_meta["training_wh"] * 1000
                tr_low = energy_meta["training_low"] * 1000
                tr_high = energy_meta["training_high"] * 1000
                tokens = energy_meta["tokens"]

                if inf >= _CLOUD_HIGH_THRESHOLD_MWH:
                    indicator = "🔥"
                    colour = "red"
                else:
                    indicator = "⚡"
                    colour = "yellow"

                label = f"[bold {colour}]cloud-based agent {indicator}[/bold {colour}]"
                summary = f"{label} [dim]— {escape(reason)} — [{colour}]{inf:.4f} mWh estimated[/{colour}][/dim]"
                tooltip = (
                    f"Inference:  {inf:.4f} mWh\n"
                    f"  {tokens} tokens × 1.54 Wh/1k × PUE 1.2\n"
                    f"\n"
                    f"Training share (amortised):  {tr_mid:.4f} mWh midpoint\n"
                    f"  range {tr_low:.5f} – {tr_high:.4f} mWh\n"
                    f"\n"
                    f"Sources: Brookings 2025, IEA 2025, Patterson et al. 2021\n"
                    f"All figures are estimates - Anthropic does not publish per-query energy data."
                )

            else:
                dur = energy_meta["duration_s"]
                label = "[bold green]local agent 🌿[/bold green]"

                if energy_meta["measured_wh"] is not None:
                    mwh = energy_meta["measured_wh"] * 1000
                    idle = energy_meta["idle_w"]
                    avg = energy_meta["avg_w"]
                    peak = energy_meta["peak_w"]
                    n = energy_meta["sample_count"]

                    summary = f"{label} [dim]- {escape(reason)} - [green]{mwh:.4f} mWh marginal[/green][/dim]"
                    tooltip = (
                        f"Measured via GPU power sensor ({n} samples over {dur:.1f}s)\n"
                        f"\n"
                        f"Idle:   {idle:.1f} W\n"
                        f"Avg:    {avg:.1f} W  (used for energy calculation)\n"
                        f"Peak:   {peak:.1f} W\n"
                        f"\n"
                        f"Marginal: ({avg:.1f} - {idle:.1f}) W × {dur:.1f}s / 3600 = {mwh:.4f} mWh"
                    )
                else:
                    summary = f"{label} [dim]- {escape(reason)} - [green]power unavailable[/green][/dim]"
                    tooltip = (
                        f"No compatible power sensor found on this system.\n"
                        f"Duration: {dur:.1f}s"
                    )

            summary_widget = Static(summary, classes="entry")
            summary_widget.tooltip = tooltip

            self.call_from_thread(status.remove)
            self.call_from_thread(self._append, summary_widget)
            self.call_from_thread(
                self._append,
                Static(escape(response), classes="entry"),
            )

        except Exception as e:
            self.call_from_thread(status.remove)
            self.call_from_thread(
                self._append,
                Static(f"[red]error:[/red] {escape(str(e))}", classes="entry"),
            )
