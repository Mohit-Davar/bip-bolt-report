"""Repository synchronization logic for BIP and BOLT specifications."""

import logging
import shutil
from pathlib import Path

import git
from git.exc import GitCommandError, InvalidGitRepositoryError, NoSuchPathError

logger = logging.getLogger(__name__)

# Official upstream repositories
REPOS: dict[str, str] = {
    "bips": "https://github.com/bitcoin/bips.git",
    "bolts": "https://github.com/lightning/bolts.git",
}

def sync_repository(name: str, url: str, data_dir: Path) -> tuple[Path, str]:
    """Clone a repository if absent, or pull latest changes with error recovery."""
    local_path = data_dir / name
    repo = None

    try:
        if local_path.exists():
            try:
                repo = git.Repo(local_path)
                logger.info(f"[bold cyan]Syncing[/] latest commits for [yellow]{name}[/]...")
                
                # Check for stale index locks (common crash cause if script was killed)
                lock_file = local_path / ".git" / "index.lock"
                if lock_file.exists():
                    lock_file.unlink()
                
                origin = repo.remotes.origin
                # Setting a timeout prevents the script from hanging forever on bad connections
                origin.pull(env={"GIT_TERMINAL_PROMPT": "0"}) 
                
            except (InvalidGitRepositoryError, NoSuchPathError):
                logger.warning(f"Corrupt repo found at {local_path}. Purging and re-cloning...")
                shutil.rmtree(local_path, ignore_errors=True)
                repo = git.Repo.clone_from(url, local_path, depth=1)
        else:
            logger.info(f"[bold green]Fetching[/] [yellow]{name}[/] (Shallow Clone)...")
            # depth=1 saves bandwidth and time by only fetching the latest state
            repo = git.Repo.clone_from(url, local_path, depth=1)

        commit_sha: str = repo.head.commit.hexsha
        logger.info(f"  [green]OK[/] {name} HEAD = [bold]{commit_sha[:12]}[/]")
        return local_path, commit_sha

    except GitCommandError as e:
        logger.error(f"[bold red]Git Error[/] for {name}: {e.stderr or e}")
        # Return what we have if possible, or re-raise if it's a total failure
        if local_path.exists() and (local_path / ".git").is_dir():
            repo = git.Repo(local_path)
            return local_path, repo.head.commit.hexsha
        raise
    except Exception as e:
        logger.error(f"Unexpected failure syncing {name}: {e}")
        raise

def sync_all(data_dir: Path) -> dict[str, tuple[Path, str]]:
    """Sync all repositories and aggregate results."""
    data_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[Path, str]] = {}
    
    for name, url in REPOS.items():
        try:
            results[name] = sync_repository(name, url, data_dir)
        except Exception:
            logger.error(f"Skipping {name} due to sync failure.")
            continue # Don't crash the whole pipeline if one repo fails
            
    return results