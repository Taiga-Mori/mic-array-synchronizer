import argparse
import os

from web_server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="mic-array-synchronizer")
    parser.add_argument("--port", type=int, default=None, help="GUIのポート番号（省略時は8501から空いているポートを自動で使用）")
    parser.add_argument(
        "--address",
        default="localhost",
        help="待ち受けるアドレス（標準はlocalhostのみ。SSH越しに使うときは ssh -L でポートを転送する）",
    )
    parser.add_argument("--no-browser", action="store_true", help="起動時にブラウザを開かない")
    args = parser.parse_args()
    over_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    serve(args.address, args.port, open_browser=not (args.no_browser or over_ssh))


if __name__ == "__main__":
    main()
