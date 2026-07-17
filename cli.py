import json
import sys

from .config import ConfigError
from .parser import JSONExtractionError
from .responder import triage_message


def main() -> None:
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
    else:
        print("Enter customer message (Ctrl-D to submit):")
        message = sys.stdin.read().strip()

    if not message:
        print("No message provided.")
        sys.exit(1)

    try:
        result = triage_message(message)
    except ConfigError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except JSONExtractionError as e:
        print(f"Could not get a valid structured response from the model: {e}")
        sys.exit(2)

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
