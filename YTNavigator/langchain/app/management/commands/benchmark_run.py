"""Headless benchmark runner for the chat agent.

Runs a JSONL question set through the agent graph without the web stack:
no HTTP, no auth, a fresh conversation thread per question (or per scenario),
LangSmith collection disabled, and a structured trace per question
(route, tool calls, per-LLM-call latency and tokens, fallback events).

Usage:
    python manage.py benchmark_run benchmark/questions.jsonl \
        --channel <CHANNEL_ID> -o benchmark/results/langgraph.jsonl

Results conform to benchmark/schemas.py:make_result and are scored with
``python benchmark/evaluate.py``.
"""

import asyncio
import json
import re
import time
import uuid
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.management.base import (
    BaseCommand,
    CommandError,
)

from benchmark.schemas import (
    load_questions,
    make_result,
)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


class Command(BaseCommand):
    """Run the chat agent headlessly over a question set and write structured results."""

    help = (
        "Run the chat agent headlessly over a JSONL question set and write one "
        "structured result record per question (route, answer, tool calls, latency, "
        "tokens, fallback events). Score the output with benchmark/evaluate.py."
    )

    def add_arguments(self, parser):
        """Register CLI arguments."""
        parser.add_argument("questions", help="Path to the questions JSONL file (see benchmark/schemas.py)")
        parser.add_argument(
            "--channel",
            default=None,
            help="Channel ID to benchmark against. Defaults to the only channel in the database; "
            "required when several channels exist.",
        )
        parser.add_argument(
            "--output",
            "-o",
            default=None,
            help="Results JSONL path (default: benchmark/results/<framework>-<run_id>.jsonl)",
        )
        parser.add_argument(
            "--framework",
            default="langgraph",
            help="Label stored in each result record, used to name runs in evaluation (default: langgraph)",
        )
        parser.add_argument(
            "--keep-threads",
            action="store_true",
            help="Keep the benchmark conversation checkpoints in the database after the run "
            "(by default they are deleted so runs leave no state behind)",
        )

    def handle(self, *args, **options):
        """Entry point: force benchmark mode, then run the async loop."""
        # Force benchmark mode for this process regardless of the environment:
        # disables LangSmith example collection inside AgentGraph.
        settings.BENCHMARK_MODE = True
        asyncio.run(self._run(options))

    async def _run(self, options):
        """Run all questions and write results incrementally."""
        # Imported here so Django and settings are fully set up first.
        from app.models import (
            Channel,
            User,
        )
        from app.services.agent.main_graph import AgentGraph
        from app.services.agent.trace import (
            BenchmarkCallbackHandler,
            start_trace,
            stop_trace,
        )

        try:
            questions = load_questions(options["questions"])
        except (OSError, ValueError) as e:
            raise CommandError(str(e))
        if not questions:
            raise CommandError(f"No questions found in {options['questions']}")

        channel = await self._get_channel(Channel, options["channel"])
        user = await self._get_user(User)

        run_id = uuid.uuid4().hex[:8]
        framework = options["framework"]
        output_path = Path(options["output"] or f"benchmark/results/{framework}-{run_id}.jsonl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.stdout.write(
            f"Run {run_id}: {len(questions)} questions against channel '{channel.id}' "
            f"(router={settings.INSTANT_LLM}, agent={settings.POWERFUL_LLM}) -> {output_path}"
        )

        graph = AgentGraph()
        await graph.setup()

        thread_ids = set()
        error_count = 0
        try:
            with output_path.open("w", encoding="utf-8") as out:
                for index, question in enumerate(questions):
                    # A scenario shares one thread across its questions (multi-turn);
                    # otherwise every question gets an isolated conversation.
                    thread_id = f"bench-{run_id}-{question.scenario_id or question.id}"
                    thread_ids.add(thread_id)

                    handler = BenchmarkCallbackHandler()
                    trace = start_trace()
                    status, error, result = "ok", None, None
                    started = time.perf_counter()
                    try:
                        result = await graph.invoke(
                            question.question,
                            channel,
                            user,
                            thread_id=thread_id,
                            callbacks=[handler],
                        )
                    except Exception as e:
                        status, error = "error", str(e)
                        error_count += 1
                    latency_s = round(time.perf_counter() - started, 3)
                    stop_trace()

                    record = self._build_record(
                        run_id=run_id,
                        framework=framework,
                        question=question,
                        thread_id=thread_id,
                        status=status,
                        error=error,
                        result=result,
                        handler=handler,
                        fallback_events=trace.events,
                        latency_s=latency_s,
                    )
                    out.write(json.dumps(record, default=str) + "\n")
                    out.flush()

                    detail = f"route={record['route']}" if status == "ok" else error
                    self.stdout.write(
                        f"  [{index + 1}/{len(questions)}] {question.id}: {status} "
                        f"({latency_s}s, {record['total_tokens']} tokens) {detail}"
                    )
        finally:
            if not options["keep_threads"]:
                await self._clear_threads(graph, thread_ids)
            if graph.conn_pool is not None:
                await graph.conn_pool.close()

        self.stdout.write(
            self.style.SUCCESS(
                f"Done: {len(questions) - error_count}/{len(questions)} completed, "
                f"{error_count} errors. Results: {output_path}"
            )
        )
        self.stdout.write(f"Score with: python benchmark/evaluate.py {output_path} --questions {options['questions']}")

    def _build_record(
        self,
        *,
        run_id,
        framework,
        question,
        thread_id,
        status,
        error,
        result,
        handler,
        fallback_events,
        latency_s,
    ):
        """Convert one graph invocation into a canonical result record."""
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
        )

        from app.schemas import AgentOutput
        from app.services.agent.main_graph import AgentGraph

        route = None
        answer_raw = None
        answer_text = None
        answer_parsed = False
        cited_video_ids = []
        tool_calls = []

        if result is not None:
            router_results = result.get("router_results")
            route = router_results.answer if router_results else None

            answer_raw = AgentGraph.extract_response(result)
            if answer_raw:
                try:
                    parsed = AgentOutput.model_validate_json(answer_raw)
                    answer_parsed = True
                    answer_text = _HTML_TAG_RE.sub("", parsed.placeholder or "").strip()
                    cited_video_ids = [v.id for v in parsed.videos or [] if v.id]
                except Exception:
                    answer_text = answer_raw

            # Only count tool calls from this turn: checkpointed threads (scenarios)
            # replay the whole conversation in `messages`.
            messages = result.get("messages", [])
            turn_start = 0
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage) and messages[i].content == question.question:
                    turn_start = i
                    break
            for message in messages[turn_start:]:
                if isinstance(message, AIMessage):
                    for tool_call in getattr(message, "tool_calls", None) or []:
                        tool_calls.append({"name": tool_call.get("name"), "args": tool_call.get("args")})

        tokens = handler.total_tokens()
        return make_result(
            run_id=run_id,
            framework=framework,
            question=question,
            thread_id=thread_id,
            status=status,
            error=error,
            route=route,
            answer_text=answer_text,
            answer_raw=answer_raw,
            answer_parsed=answer_parsed,
            cited_video_ids=cited_video_ids,
            tool_calls=tool_calls,
            llm_calls=handler.llm_calls,
            fallback_events=fallback_events,
            latency_s=latency_s,
            prompt_tokens=tokens["prompt_tokens"],
            completion_tokens=tokens["completion_tokens"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def _get_channel(self, Channel, channel_id):
        """Resolve the channel to benchmark against."""

        @sync_to_async
        def resolve():
            if channel_id:
                channel = Channel.objects.filter(id=channel_id).first()
                if channel is None:
                    available = list(Channel.objects.values_list("id", flat=True))
                    raise CommandError(f"Channel '{channel_id}' not found. Available: {available or 'none'}")
                return channel
            channels = list(Channel.objects.all()[:2])
            if not channels:
                raise CommandError(
                    "No channels in the database. Scan a channel first (or load a snapshot with "
                    "'python manage.py benchmark_snapshot load <file>')."
                )
            if len(channels) > 1:
                available = list(Channel.objects.values_list("id", flat=True))
                raise CommandError(f"Multiple channels found, pass --channel. Available: {available}")
            return channels[0]

        return await resolve()

    async def _get_user(self, User):
        """Get or create the synthetic user benchmark runs execute as."""

        @sync_to_async
        def resolve():
            user, _created = User.objects.get_or_create(
                username="benchmark",
                defaults={"email": "benchmark@localhost"},
            )
            return user

        return await resolve()

    async def _clear_threads(self, graph, thread_ids):
        """Delete the benchmark conversation checkpoints created by this run."""
        if not thread_ids:
            return
        try:
            async with graph.conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    await conn.execute(
                        f"DELETE FROM {table} WHERE thread_id = ANY(%s)",
                        (list(thread_ids),),
                    )
            self.stdout.write(f"Cleaned up {len(thread_ids)} benchmark thread(s)")
        except Exception as e:
            self.stderr.write(f"Warning: failed to clean up benchmark threads: {e}")
