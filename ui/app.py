from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, RichLog
from textual.binding import Binding
from textual import work

import router


class RouterApp(App):
    """Prompt router TUI."""

    TITLE = "prompt-router"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield RichLog(id="log", wrap=True, markup=True)
        yield Input(placeholder="Type a prompt and press Enter…")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(Input).focus()
        log = self.query_one(RichLog)
        log.write("[dim]prompt-router ready. Ctrl+C to quit.[/dim]")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return
        event.input.clear()
        self._handle_prompt(prompt)

    @work(thread=True)
    def _handle_prompt(self, prompt: str) -> None:
        log = self.query_one(RichLog)
        self.call_from_thread(log.write, f"\n[bold]>[/bold] {prompt}")
        self.call_from_thread(log.write, "[dim]classifying…[/dim]")

        try:
            destination, reason, response = router.route(prompt)

            if destination == "claude":
                label = "[bold red]claude[/bold red]"
            else:
                label = "[bold cyan]ollama[/bold cyan]"

            self.call_from_thread(log.write, f"{label} [dim]— {reason}[/dim]")
            self.call_from_thread(log.write, response)

        except Exception as e:
            self.call_from_thread(log.write, f"[red]error:[/red] {e}")
