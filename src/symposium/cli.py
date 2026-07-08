"""Typer 기반 CLI."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from symposium import ingest as _ingest
from symposium import retrieve as _retrieve
from symposium.config import CHROMA_DIR, DATA_ROOT, METADATA_DIR

app = typer.Typer(add_completion=False, help="Symposium — 신학 문헌 RAG 플랫폼")
console = Console()


def _build_prompt(question: str, hits: list[dict]) -> str:
    """검색 결과와 질문을 하나의 프롬프트로 조합."""
    parts = [
        "당신은 기독교 조직신학 전문 연구 조수입니다. 아래 참고 자료에 근거해 질문에 답하세요.",
        "각 주장 뒤에 출처를 [저작명, p.페이지] 형식으로 표기하고,",
        "중요한 신학 용어는 원어(독일어/영어/라틴어)를 병기하세요.\n",
        "# 참고 자료\n",
    ]
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        source = f"[{m.get('title', '?')}, p.{m.get('page', '?')}]"
        parts.append(f"### 발췌 {i} {source}\n{hit['text']}\n")
    parts.append(f"# 질문\n\n{question}")
    return "\n".join(parts)


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
def ask_cmd(
    question: str = typer.Argument(...),
    author: str = typer.Option("moltmann"),
    top_k: int = typer.Option(5),
) -> None:
    """검색 결과 + 질문을 클립보드에 복사. Claude Code나 claude.ai에 붙여넣어 사용."""
    console.print(f"[dim]검색 중... (author={author}, top-{top_k})[/dim]")
    hits = _retrieve.search(question, author, CHROMA_DIR, top_k=top_k)
    if not hits:
        console.print("[yellow]검색 결과 없음[/yellow]")
        raise typer.Exit(1)
    console.print(f"[dim]참고 발췌 {len(hits)}개[/dim]")

    prompt = _build_prompt(question, hits)
    import subprocess
    subprocess.run(["pbcopy"], input=prompt.encode("utf-8"), check=True)
    console.print("[green]✓ 프롬프트가 클립보드에 복사되었습니다.[/green]")
    console.print("[dim]Claude Code 또는 claude.ai에 붙여넣기 하세요.[/dim]")


if __name__ == "__main__":
    app()
