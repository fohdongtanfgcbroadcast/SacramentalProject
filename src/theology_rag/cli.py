"""Typer 기반 CLI."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from theology_rag import generate as _generate
from theology_rag import ingest as _ingest
from theology_rag import retrieve as _retrieve
from theology_rag.config import CHROMA_DIR, DATA_ROOT, METADATA_DIR

app = typer.Typer(add_completion=False, help="신학 문헌 RAG 시스템")
console = Console()


@app.command("ingest")
def ingest_cmd(author: str = typer.Argument(..., help="저자 키. 예: moltmann")) -> None:
    metadata_path = METADATA_DIR / f"{author}.yaml"
    if not metadata_path.exists():
        console.print(f"[red]메타데이터 파일이 없습니다: {metadata_path}[/red]")
        raise typer.Exit(1)
    raw_dir = DATA_ROOT / "raw" / author
    if not raw_dir.exists():
        console.print(f"[red]원본 PDF 디렉터리가 없습니다: {raw_dir}[/red]")
        raise typer.Exit(1)
    _ingest.ingest_author(author, DATA_ROOT, metadata_path, CHROMA_DIR)
    console.print(f"[green]✓ {author} 인제스트 완료[/green]")


@app.command("search")
def search_cmd(query: str = typer.Argument(...), author: str = typer.Option("moltmann"), top_k: int = typer.Option(5)) -> None:
    hits = _retrieve.search(query, author, CHROMA_DIR, top_k=top_k)
    if not hits:
        console.print("[yellow]검색 결과 없음[/yellow]")
        return
    table = Table(title=f'검색: "{query}" (top-{top_k})')
    table.add_column("#", justify="right")
    table.add_column("저작")
    table.add_column("p.", justify="right")
    table.add_column("dist", justify="right")
    table.add_column("발췌", overflow="fold")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        preview = h["text"][:200].replace("\n", " ")
        table.add_row(str(i), str(m.get("title", "?")), str(m.get("page", "?")), f"{h['distance']:.3f}", preview + ("…" if len(h["text"]) > 200 else ""))
    console.print(table)


@app.command("ask")
def ask_cmd(question: str = typer.Argument(...), author: str = typer.Option("moltmann"), top_k: int = typer.Option(5)) -> None:
    console.print(f"[dim]검색 중... (author={author}, top-{top_k})[/dim]")
    hits = _retrieve.search(question, author, CHROMA_DIR, top_k=top_k)
    if not hits:
        console.print("[yellow]검색 결과 없음[/yellow]")
        raise typer.Exit(1)
    console.print(f"[dim]참고 발췌 {len(hits)}개[/dim]")
    console.print("\n[dim]Claude 응답 생성 중...[/dim]\n")
    answer, usage = _generate.ask(question, hits)
    console.print(Panel(answer, title="답변", border_style="green"))
    console.print(f"[dim]토큰: 입력 {usage['input_tokens']} / 출력 {usage['output_tokens']} / 캐시 읽기 {usage.get('cache_read_input_tokens', 0)}[/dim]")


if __name__ == "__main__":
    app()
