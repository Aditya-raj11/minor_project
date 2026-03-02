"""
view_attendance.py — Attendance Viewer & Reports
==================================================
View, filter, and export attendance records.
"""

import sqlite3
import os
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box
from rich.columns import Columns
from rich.text import Text

console = Console()
DB_PATH = "database/attendance.db"

def get_conn():
    return sqlite3.connect(DB_PATH)

# ─────────────────────────────────────────────
# QUERIES
# ─────────────────────────────────────────────

def fetch_attendance(date_from=None, date_to=None, user_code=None):
    conn = get_conn()
    c = conn.cursor()

    query = """
        SELECT u.name, u.user_code, a.date, a.time, a.status
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE 1=1
    """
    params = []

    if date_from:
        query += " AND a.date >= ?"
        params.append(date_from)
    if date_to:
        query += " AND a.date <= ?"
        params.append(date_to)
    if user_code:
        query += " AND u.user_code = ?"
        params.append(user_code.upper())

    query += " ORDER BY a.date DESC, a.time DESC"

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return rows

def fetch_summary(date_from=None, date_to=None):
    """Returns per-user attendance count."""
    conn = get_conn()
    c = conn.cursor()

    query = """
        SELECT u.name, u.user_code, COUNT(a.id) as days_present
        FROM users u
        LEFT JOIN attendance a ON u.id = a.user_id
    """
    params = []
    if date_from or date_to:
        query += " WHERE 1=1"
        if date_from:
            query += " AND a.date >= ?"
            params.append(date_from)
        if date_to:
            query += " AND a.date <= ?"
            params.append(date_to)

    query += " GROUP BY u.id ORDER BY days_present DESC"

    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return rows

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def display_attendance_table(rows, title="Attendance Records"):
    if not rows:
        console.print("[yellow]  No records found for the selected filter.[/]")
        return

    table = Table(
        title=title,
        box=box.ROUNDED,
        border_style="bright_blue",
        show_lines=False,
        header_style="bold cyan"
    )
    table.add_column("#",        style="dim",        width=4)
    table.add_column("Name",     style="bold white",  min_width=18)
    table.add_column("ID",       style="cyan",        min_width=12)
    table.add_column("Date",     style="white",       min_width=12)
    table.add_column("Time",     style="green",       min_width=10)
    table.add_column("Status",   style="bold green",  min_width=10)

    for i, (name, code, date, time, status) in enumerate(rows, 1):
        status_str = f"[bold green]✓ {status}[/]" if status == "PRESENT" else f"[red]✗ {status}[/]"
        table.add_row(str(i), name, code, date, time, status_str)

    console.print("\n")
    console.print(table)
    console.print(f"\n  [dim]Total records: {len(rows)}[/]\n")

def display_summary_table(rows, date_from=None, date_to=None):
    if not rows:
        console.print("[yellow]  No users found.[/]")
        return

    period = f"{date_from or 'All'} → {date_to or 'All'}"
    table = Table(
        title=f"Attendance Summary  [{period}]",
        box=box.ROUNDED,
        border_style="bright_magenta",
        header_style="bold magenta"
    )
    table.add_column("Name",          style="bold white",  min_width=18)
    table.add_column("ID",            style="cyan",        min_width=12)
    table.add_column("Days Present",  style="bold green",  min_width=14)

    for name, code, days in rows:
        table.add_row(name, code, str(days))

    console.print("\n")
    console.print(table)
    console.print("\n")

# ─────────────────────────────────────────────
# CSV EXPORT
# ─────────────────────────────────────────────

def export_to_csv(rows, filename=None):
    if not rows:
        console.print("[yellow]  No data to export.[/]")
        return

    if not filename:
        filename = f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    os.makedirs("exports", exist_ok=True)
    filepath = os.path.join("exports", filename)

    with open(filepath, "w") as f:
        f.write("Name,User Code,Date,Time,Status\n")
        for name, code, date, time, status in rows:
            f.write(f"{name},{code},{date},{time},{status}\n")

    console.print(f"  [bold green]✓ Exported {len(rows)} records to:[/] [cyan]{filepath}[/]\n")

# ─────────────────────────────────────────────
# MENU
# ─────────────────────────────────────────────

def view_attendance():
    console.print("\n[bold blue]═══ View Attendance ═══[/]\n")
    console.print("  [dim]Filter options (leave blank for all):[/]\n")

    date_from = Prompt.ask(
        "  [cyan]From date[/] [dim](YYYY-MM-DD or leave blank)[/]",
        default=""
    ).strip() or None

    date_to = Prompt.ask(
        "  [cyan]To date[/]   [dim](YYYY-MM-DD or leave blank)[/]",
        default=""
    ).strip() or None

    user_code = Prompt.ask(
        "  [cyan]User ID[/]    [dim](leave blank for all)[/]",
        default=""
    ).strip() or None

    rows = fetch_attendance(date_from, date_to, user_code)
    display_attendance_table(rows)

    if rows:
        export = Prompt.ask("  [yellow]Export to CSV?[/] [dim](y/n)[/]", default="n")
        if export.lower() == "y":
            export_to_csv(rows)


def view_summary():
    console.print("\n[bold blue]═══ Attendance Summary ═══[/]\n")

    date_from = Prompt.ask(
        "  [cyan]From date[/] [dim](leave blank for all)[/]",
        default=""
    ).strip() or None

    date_to = Prompt.ask(
        "  [cyan]To date[/]   [dim](leave blank for all)[/]",
        default=""
    ).strip() or None

    rows = fetch_summary(date_from, date_to)
    display_summary_table(rows, date_from, date_to)


def list_users():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT name, user_code, registered_at FROM users ORDER BY registered_at")
    rows = c.fetchall()
    conn.close()

    if not rows:
        console.print("\n[yellow]  No registered users found.[/]\n")
        return

    table = Table(
        title="Registered Users",
        box=box.ROUNDED,
        border_style="cyan",
        header_style="bold cyan"
    )
    table.add_column("#",               style="dim",        width=4)
    table.add_column("Name",            style="bold white", min_width=18)
    table.add_column("ID / User Code",  style="cyan",       min_width=14)
    table.add_column("Registered At",   style="dim green",  min_width=20)

    for i, (name, code, reg_at) in enumerate(rows, 1):
        table.add_row(str(i), name, code, reg_at)

    console.print("\n")
    console.print(table)
    console.print(f"\n  [dim]Total registered: {len(rows)}[/]\n")
