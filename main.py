"""
Smart Attendance System - Main Menu
====================================
Entry point for the entire system.
"""

import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()

def print_banner():
    banner = """
███████╗███╗   ███╗ █████╗ ██████╗ ████████╗
██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝
███████╗██╔████╔██║███████║██████╔╝   ██║   
╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║   
███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║   
╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
    ATTENDANCE SYSTEM — Powered by ArcFace + MTCNN
    """
    console.print(Panel(Text(banner, style="bold cyan"), border_style="bright_blue"))

def show_menu():
    table = Table(box=box.ROUNDED, border_style="bright_blue", show_header=False, padding=(0, 2))
    table.add_column("Option", style="bold yellow", width=6)
    table.add_column("Action", style="white")
    table.add_column("Description", style="dim")

    table.add_row("[ 1 ]", "Register New User",      "Scan face with multi-angle capture + train model")
    table.add_row("[ 2 ]", "Start Attendance",        "Live webcam recognition & auto mark attendance")
    table.add_row("[ 3 ]", "View Attendance",         "Browse, filter & export attendance records")
    table.add_row("[ 4 ]", "List Registered Users",   "See all registered users in the system")
    table.add_row("[ 5 ]", "Delete User",             "Remove a user and their face data")
    table.add_row("[ 0 ]", "Exit",                    "Quit the application")

    console.print("\n")
    console.print(table)
    console.print("\n")

def main():
    os.makedirs("face_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("database", exist_ok=True)

    while True:
        print_banner()
        show_menu()

        choice = console.input("[bold yellow]  ➤  Enter your choice: [/]").strip()

        if choice == "1":
            from register import register_user
            register_user()
        elif choice == "2":
            from recognize import start_attendance
            start_attendance()
        elif choice == "3":
            from view_attendance import view_attendance
            view_attendance()
        elif choice == "4":
            from view_attendance import list_users
            list_users()
        elif choice == "5":
            from register import delete_user
            delete_user()
        elif choice == "0":
            console.print("\n[bold green]  Goodbye! 👋[/]\n")
            sys.exit(0)
        else:
            console.print("[bold red]  ✗ Invalid choice. Try again.[/]\n")

if __name__ == "__main__":
    main()
