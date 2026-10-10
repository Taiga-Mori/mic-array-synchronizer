import argparse
import sys
from pathlib import Path

import streamlit.web.cli as stcli


def streamlit_run() -> None:
    parser = argparse.ArgumentParser(prog="mic-array-synchronizer")
    parser.add_argument("--port", type=int, default=None, help="GUIのポート番号（省略時は8501から空いているポートを自動で使用）")
    parser.add_argument(
        "--address",
        default="localhost",
        help="待ち受けるアドレス（標準はlocalhostのみ。SSH越しに使うときは ssh -L でポートを転送する）",
    )
    args = parser.parse_args()

    src = str(Path(__file__).resolve().parent / "gui.py")
    sys.argv = ["streamlit", "run", src, "--global.developmentMode=false", f"--server.address={args.address}"]
    if args.port is not None:
        sys.argv.append(f"--server.port={args.port}")
    sys.exit(stcli.main())


if __name__ == "__main__":
    streamlit_run()
