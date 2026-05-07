"""
frontend/github_push_ui.py — GitHub Push helper tab
======================================================
A guided way to push the RAG Studio code to GitHub without leaving the
Streamlit app. Two modes coexist:

  1. COPY-PASTE COMMANDS — every Git command is shown in a code block
     ready to be copied and run in the user's own terminal. This is the
     "teach the user Git" path.

  2. RUN BUTTONS — for users who'd rather not touch the terminal,
     buttons execute the same commands via subprocess and show the
     output. This is the "just-do-it" path.

WHY BOTH?
  The student should *learn* what Git is doing — but during a 3-hour
  class it's also fine for them to click and watch.

SAFETY
  We never run anything destructive without explicit confirmation, and
  we always work in the project's own directory (PROJECT_ROOT from
  config.py).
"""

import subprocess
from pathlib import Path
from typing import Tuple

import streamlit as st

from backend.config import PROJECT_ROOT


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _run(cmd: list, cwd: Path = PROJECT_ROOT) -> Tuple[int, str, str]:
    """
    Run a shell command and return (returncode, stdout, stderr).

    - Always runs in the project root (no surprise location).
    - Captures both streams so we can show them in the UI.
    - 60s timeout — Git operations should never take longer.
    """
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 60 seconds."
    except FileNotFoundError as e:
        return -1, "", f"Command not found: {e}"
    except Exception as e:
        return -1, "", f"Unexpected error: {e}"


def _show_output(rc: int, stdout: str, stderr: str) -> None:
    """Render the result of a subprocess call."""
    if rc == 0:
        st.success("✅ Command succeeded")
    else:
        st.error(f"❌ Command failed (return code {rc})")
    if stdout.strip():
        st.markdown("**stdout:**")
        st.code(stdout.strip(), language="text")
    if stderr.strip():
        st.markdown("**stderr:**")
        st.code(stderr.strip(), language="text")


def _git_installed() -> bool:
    rc, _, _ = _run(["git", "--version"])
    return rc == 0


def _is_repo() -> bool:
    rc, _, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    return rc == 0


# ════════════════════════════════════════════════════════════
# Render
# ════════════════════════════════════════════════════════════

def render_github_push():
    """Top-level renderer for the GitHub Push tab."""

    st.markdown("### 🚀 Push to GitHub")
    st.caption(
        "Two ways to ship this project to GitHub: copy-paste the commands "
        "into your terminal (recommended for learning), or click the buttons "
        "to run them through Streamlit."
    )

    # ─── Pre-flight ──────────────────────────────────────
    if not _git_installed():
        st.error(
            "❌ **Git is not installed** on this machine.\n\n"
            "- macOS: `brew install git`\n"
            "- Ubuntu/Debian: `sudo apt install git`\n"
            "- Windows: https://git-scm.com/download/win"
        )
        return

    repo_status = "✅ Already a Git repository" if _is_repo() \
                   else "⚪ Not a Git repository yet"
    st.caption(f"📁 Project: `{PROJECT_ROOT}` · {repo_status}")

    st.markdown("---")

    # ─── 1. Configure Git identity ───────────────────────
    st.markdown("#### 1️⃣  Set your Git identity (one-time)")
    col_n, col_e = st.columns(2)
    with col_n:
        git_name = st.text_input("Git user.name", placeholder="Mario Rossi")
    with col_e:
        git_email = st.text_input("Git user.email", placeholder="mario@example.com")

    cmd_identity = (
        f'git config --global user.name "{git_name or "YOUR NAME"}"\n'
        f'git config --global user.email "{git_email or "YOUR EMAIL"}"'
    )
    st.code(cmd_identity, language="bash")
    if st.button("▶️ Set identity", key="btn_identity"):
        if not git_name or not git_email:
            st.warning("Fill in both name and email first.")
        else:
            rc1, o1, e1 = _run(["git", "config", "--global", "user.name", git_name])
            rc2, o2, e2 = _run(["git", "config", "--global", "user.email", git_email])
            _show_output(max(rc1, rc2), o1 + o2, e1 + e2)

    st.markdown("---")

    # ─── 2. Initialize repository ────────────────────────
    st.markdown("#### 2️⃣  Initialize the repository (skip if already done)")
    st.code("git init\ngit branch -M main", language="bash")
    if st.button("▶️ Run `git init`", key="btn_init",
                 disabled=_is_repo(),
                 help="Disabled if this is already a repo."):
        rc1, o1, e1 = _run(["git", "init"])
        rc2, o2, e2 = _run(["git", "branch", "-M", "main"])
        _show_output(max(rc1, rc2), o1 + o2, e1 + e2)

    st.markdown("---")

    # ─── 3. Stage and commit ─────────────────────────────
    st.markdown("#### 3️⃣  Stage all files and commit")
    commit_msg = st.text_input(
        "Commit message",
        value="Initial commit: RAG Studio",
    )
    st.code(
        f'git add .\n'
        f'git commit -m "{commit_msg}"',
        language="bash",
    )
    if st.button("▶️ `git add . && git commit`", key="btn_commit"):
        rc1, o1, e1 = _run(["git", "add", "."])
        rc2, o2, e2 = _run(["git", "commit", "-m", commit_msg])
        _show_output(max(rc1, rc2), o1 + o2, e1 + e2)

    st.markdown("---")

    # ─── 4. Connect to remote ────────────────────────────
    st.markdown("#### 4️⃣  Connect to your GitHub repo")
    st.caption(
        "First, create an empty repo on github.com. Then paste its URL below "
        "(HTTPS or SSH — both work)."
    )
    remote_url = st.text_input(
        "GitHub remote URL",
        placeholder="https://github.com/USERNAME/rag-studio.git",
    )
    cmd_remote = (
        f'git remote add origin {remote_url or "https://github.com/USERNAME/REPO.git"}\n'
        f'# If you already added a remote and want to change it:\n'
        f'# git remote set-url origin {remote_url or "https://github.com/USERNAME/REPO.git"}'
    )
    st.code(cmd_remote, language="bash")
    if st.button("▶️ Add remote", key="btn_remote"):
        if not remote_url:
            st.warning("Enter your GitHub repo URL first.")
        else:
            # Try `add` first; if it fails because the remote exists, switch to set-url.
            rc, o, e = _run(["git", "remote", "add", "origin", remote_url])
            if rc != 0 and "already exists" in (e or "").lower():
                rc, o, e = _run(["git", "remote", "set-url", "origin", remote_url])
            _show_output(rc, o, e)

    st.markdown("---")

    # ─── 5. Push ─────────────────────────────────────────
    st.markdown("#### 5️⃣  Push to GitHub")
    st.code("git push -u origin main", language="bash")
    st.warning(
        "⚠️ The first `git push` requires authentication. If a credential "
        "manager isn't already set up, the push will hang. In that case, "
        "run it from your terminal so you can enter credentials interactively."
    )
    if st.button("▶️ `git push`", key="btn_push", type="primary"):
        rc, o, e = _run(["git", "push", "-u", "origin", "main"])
        _show_output(rc, o, e)

    st.markdown("---")

    # ─── 6. Status & log (for sanity checking) ───────────
    st.markdown("#### 🔍 Repository status")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 git status", key="btn_status"):
            rc, o, e = _run(["git", "status"])
            _show_output(rc, o, e)
    with col2:
        if st.button("📜 git log (last 5)", key="btn_log"):
            rc, o, e = _run(["git", "log", "--oneline", "-5"])
            _show_output(rc, o, e)

    # ─── Cheat sheet (always visible at the bottom) ──────
    with st.expander("📚 Cheat sheet — full sequence in one block"):
        full = (
            f'# Run these IN ORDER from inside the project folder.\n'
            f'cd "{PROJECT_ROOT}"\n\n'
            f'# 1. Identity (one-time)\n'
            f'git config --global user.name "{git_name or "YOUR NAME"}"\n'
            f'git config --global user.email "{git_email or "YOUR EMAIL"}"\n\n'
            f'# 2. Initialize\n'
            f'git init\n'
            f'git branch -M main\n\n'
            f'# 3. Commit\n'
            f'git add .\n'
            f'git commit -m "{commit_msg}"\n\n'
            f'# 4. Remote\n'
            f'git remote add origin {remote_url or "https://github.com/USERNAME/REPO.git"}\n\n'
            f'# 5. Push\n'
            f'git push -u origin main\n'
        )
        st.code(full, language="bash")
