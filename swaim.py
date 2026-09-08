# swaim.py

import argparse

from model import ask, chat, train
from settings import BASE


def main():

    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="SwAIm: an AI designed to solve the problem of forgetting what land looks like.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Parse the train subcommand.
    train_parser = sub.add_parser("train", help="Train SwAIm.")
    train_parser.add_argument("--base", default=BASE)
    train_parser.set_defaults(fn=train)

    # Parse the chat subcommand.
    chat_parser = sub.add_parser("chat", help="Chat back and forth with SwAIm.")
    chat_parser.set_defaults(fn=chat)

    # Parse the ask subcommand.
    ask_parser = sub.add_parser("ask", help="Ask SwAIm a specific question.")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(fn=ask)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
